"""Background retry scheduler for pending_retry callback_logs.

Spec: webhook-callback § Requirement "重试策略：指数退避 + 最大次数" — runs as
      an independent task, scans ``status='pending_retry' AND next_retry_at <= now``
      and re-sends using the stored ``request_body`` (no re-evaluation of
      trigger / re-render of payload — see impl-worker delta spec).
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import UTC, datetime
from typing import Any

import httpx
from isales_common.enums import CallbackStatus
from isales_common.models.callback_config import CallbackConfig
from isales_common.models.callback_log import CallbackLog
from isales_common.utils.crypto import decrypt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from isales_worker.callbacks import (
    _classify_response,
    _decide_retry,
    _sign_request,
)
from isales_worker.settings import Settings

logger = logging.getLogger(__name__)


async def _retry_one(
    session: Any,
    http: httpx.AsyncClient,
    *,
    log: CallbackLog,
    config: CallbackConfig,
    default_timeout: float,
) -> None:
    body = log.request_body or ""
    secret = decrypt(config.signing_secret) if config.signing_secret else ""
    timestamp = int(time.time())
    headers = dict(config.headers or {})
    headers.update(_sign_request(secret, body, timestamp))
    timeout = float(config.timeout_seconds or default_timeout)

    try:
        resp = await http.request(
            (config.method or "POST").upper(),
            config.url,
            headers=headers,
            content=body,
            timeout=timeout,
        )
    except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPError) as exc:
        log.attempt_at = datetime.now(UTC)
        log.error_message = f"http_error: {exc!r}"
        log.status = _decide_retry(log, dict(config.retry_policy or {}))
        return

    log.attempt_at = datetime.now(UTC)
    log.response_code = resp.status_code
    log.response_body = resp.text[:1000]
    log.error_message = None

    classified = _classify_response(resp.status_code)
    if classified == CallbackStatus.SUCCESS:
        log.status = CallbackStatus.SUCCESS
        log.next_retry_at = None
        return
    if classified == CallbackStatus.FAILED_HTTP_4XX:
        # 4xx in retry path is unusual (most retried calls were 5xx); log
        # warn but stop retrying — spec says 4xx never retries.
        logger.warning(
            "callback_retry_4xx callback_log_id=%d status=%d",
            log.id, resp.status_code,
        )
        log.status = CallbackStatus.FAILED_HTTP_4XX
        log.next_retry_at = None
        return
    log.status = _decide_retry(log, dict(config.retry_policy or {}))


async def tick(
    sessionmaker: async_sessionmaker[Any],
    http: httpx.AsyncClient,
    settings: Settings,
    *,
    now: datetime | None = None,
) -> int:
    """Run one pass over pending_retry. Returns the count of logs touched."""

    when = now or datetime.now(UTC)
    async with sessionmaker() as session:
        stmt = (
            select(CallbackLog)
            .where(CallbackLog.status == CallbackStatus.PENDING_RETRY)
            .where(CallbackLog.next_retry_at.is_not(None))
            .where(CallbackLog.next_retry_at <= when)
            .order_by(CallbackLog.next_retry_at.asc())
            .limit(settings.worker_retry_batch_size)
        )
        logs = list((await session.execute(stmt)).scalars().all())

        if not logs:
            return 0

        # Fetch configs in one query
        config_ids = {log.callback_config_id for log in logs}
        cfgs_rows = (
            await session.execute(
                select(CallbackConfig).where(CallbackConfig.id.in_(config_ids))
            )
        ).scalars().all()
        cfg_by_id = {c.id: c for c in cfgs_rows}

        for log in logs:
            cfg = cfg_by_id.get(log.callback_config_id)
            if cfg is None:
                logger.warning(
                    "callback_retry_missing_config callback_log_id=%d", log.id
                )
                log.status = CallbackStatus.EXHAUSTED
                log.error_message = "callback_config_missing"
                continue
            try:
                await _retry_one(
                    session,
                    http,
                    log=log,
                    config=cfg,
                    default_timeout=settings.webhook_default_timeout_seconds,
                )
            except Exception:
                logger.exception("callback_retry_error callback_log_id=%d", log.id)

        await session.commit()
        return len(logs)


async def retry_loop(
    sessionmaker: async_sessionmaker[Any],
    http: httpx.AsyncClient,
    settings: Settings,
) -> None:
    while True:
        try:
            await tick(sessionmaker, http, settings)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("retry_loop_tick_failed")
        await asyncio.sleep(settings.worker_retry_tick_interval)
