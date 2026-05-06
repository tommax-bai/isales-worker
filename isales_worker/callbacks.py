"""process_callbacks — webhook trigger / payload / signing / dispatch.

Spec: webhook-callback (full); goal-achievement § Requirement "通话结束后回调匹配".
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
from isales_common.enums import CallbackStatus
from isales_common.models.call_record import CallRecord
from isales_common.models.call_summary import CallSummary
from isales_common.models.callback_config import CallbackConfig
from isales_common.models.callback_log import CallbackLog
from isales_common.models.lead import Lead
from isales_common.utils.crypto import decrypt
from jinja2 import StrictUndefined, TemplateError, UndefinedError
from jinja2.exceptions import SecurityError, TemplateSyntaxError
from jinja2.sandbox import SandboxedEnvironment
from json_logic import jsonLogic
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

_jinja_env = SandboxedEnvironment(undefined=StrictUndefined)


class RenderError(Exception):
    """Trigger evaluation or payload rendering failed permanently (no retry)."""


def _evaluate_trigger(trigger: dict[str, Any], ctx: dict[str, Any]) -> bool:
    """Evaluate a JsonLogic trigger. Any exception → False (callers decide
    whether to record failed_render based on the original exception type)."""

    try:
        return bool(jsonLogic(trigger, ctx))
    except Exception:
        logger.exception("trigger_eval_error trigger=%r", trigger)
        return False


def _build_eval_context(
    *,
    call_record: CallRecord,
    call_summary: CallSummary | None,
    lead: Lead,
    hangup_cause: str | None = None,
) -> dict[str, Any]:
    """Build the JsonLogic / Jinja2 evaluation context.

    Field set MUST match webhook-callback spec § "trigger 可引用字段范围":
    goal_achieved, goal_type, extracted.*, lead.*, call.*.

    ``hangup_cause`` comes from the inbound CallEnded message (CallRecord
    itself doesn't carry one — only the transcript's hangup event does).
    """

    if call_summary is not None:
        goal_achieved = bool(call_summary.goal_achieved)
        goal_type = call_summary.goal_type or ""
        extracted = dict(call_summary.extracted_fields or {})
    else:
        goal_achieved = False
        goal_type = ""
        extracted = {}

    return {
        "goal_achieved": goal_achieved,
        "goal_type": goal_type,
        "extracted": extracted,
        "lead": {
            "id": lead.id,
            "name": lead.name,
            "phone": lead.phone,
            "source": lead.source,
            "status": str(lead.status),
            "custom_data": dict(lead.custom_data or {}),
        },
        "call": {
            "id": call_record.id,
            "duration": call_record.duration,
            "started_at": call_record.started_at.isoformat() if call_record.started_at else None,
            "ended_at": call_record.ended_at.isoformat() if call_record.ended_at else None,
            "transfer_status": str(call_record.transfer_status),
            "hangup_cause": hangup_cause,
            "caller_id": call_record.caller_id,
            "campaign_id": call_record.campaign_id,
        },
        "call_summary": {
            "summary_text": call_summary.summary_text if call_summary else None,
        },
    }


def _render_payload(template_str: str, ctx: dict[str, Any]) -> str:
    try:
        tmpl = _jinja_env.from_string(template_str)
        return tmpl.render(**ctx)
    except (UndefinedError, SecurityError, TemplateSyntaxError, TemplateError) as exc:
        raise RenderError(f"jinja2_render_failed: {exc!s}") from exc
    except Exception as exc:
        raise RenderError(f"jinja2_unexpected: {exc!r}") from exc


def _sign_request(secret: str, body: str, timestamp: int) -> dict[str, str]:
    payload = f"{timestamp}.{body}".encode()
    digest = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    return {
        "X-Isales-Signature": f"sha256={digest}",
        "X-Isales-Timestamp": str(timestamp),
        "Content-Type": "application/json",
    }


def _classify_response(status_code: int) -> CallbackStatus:
    if 200 <= status_code < 300:
        return CallbackStatus.SUCCESS
    if 400 <= status_code < 500:
        return CallbackStatus.FAILED_HTTP_4XX
    return CallbackStatus.FAILED_HTTP_5XX


def _retry_interval_seconds(retry_policy: dict[str, Any], attempt_idx: int) -> int:
    intervals = list(retry_policy.get("intervals_seconds") or [])
    if not intervals:
        return 60
    idx = min(attempt_idx, len(intervals) - 1)
    return int(intervals[idx])


def _max_attempts(retry_policy: dict[str, Any]) -> int:
    return int(retry_policy.get("max_attempts") or 0)


async def _send_http(
    http: httpx.AsyncClient,
    *,
    url: str,
    method: str,
    headers: dict[str, str],
    body: str,
    timeout: float,  # noqa: ASYNC109 — passes through to httpx, not asyncio
) -> tuple[int | None, str | None, str | None]:
    """Returns (status_code, response_body_excerpt, error_message)."""

    try:
        resp = await http.request(
            method.upper() or "POST",
            url,
            headers=headers,
            content=body,
            timeout=timeout,
        )
    except (httpx.TimeoutException, httpx.NetworkError) as exc:
        return None, None, f"http_error: {exc!r}"
    except httpx.HTTPError as exc:
        return None, None, f"http_error: {exc!r}"
    return resp.status_code, resp.text[:1000], None


async def dispatch_callback(
    session: AsyncSession,
    http: httpx.AsyncClient,
    *,
    callback_config: CallbackConfig,
    ctx: dict[str, Any],
    call_record_id: int,
    default_timeout: float,
) -> None:
    """Evaluate trigger → render → sign → HTTP → write callback_log.

    Triggers that don't match are silently dropped (no log written) per
    design.md Decision 4 — keeps callback_log from blowing up.
    """

    if not _evaluate_trigger(callback_config.trigger, ctx):
        return

    log = CallbackLog(
        callback_config_id=callback_config.id,
        call_record_id=call_record_id,
        status=CallbackStatus.PENDING,
        retry_count=0,
    )
    session.add(log)
    await session.flush()

    try:
        body = _render_payload(callback_config.payload_template, ctx)
    except RenderError as exc:
        log.status = CallbackStatus.FAILED_RENDER
        log.error_message = str(exc)
        log.attempt_at = datetime.now(UTC)
        await session.flush()
        logger.warning(
            "callback_failed_render callback_config_id=%d call_record_id=%d err=%s",
            callback_config.id, call_record_id, exc,
        )
        return

    log.request_body = body

    secret_ct = callback_config.signing_secret
    secret = decrypt(secret_ct) if secret_ct else ""
    timestamp = int(time.time())
    headers = dict(callback_config.headers or {})
    headers.update(_sign_request(secret, body, timestamp))

    timeout = float(callback_config.timeout_seconds or default_timeout)
    status_code, resp_body, err = await _send_http(
        http,
        url=callback_config.url,
        method=callback_config.method,
        headers=headers,
        body=body,
        timeout=timeout,
    )
    log.attempt_at = datetime.now(UTC)
    log.response_code = status_code
    log.response_body = resp_body

    if err is not None or status_code is None:
        log.error_message = err
        retry_policy = dict(callback_config.retry_policy or {})
        next_retry = _decide_retry(log, retry_policy)
        log.status = next_retry
        await session.flush()
        return

    classified = _classify_response(status_code)
    if classified == CallbackStatus.SUCCESS:
        log.status = CallbackStatus.SUCCESS
        await session.flush()
        return

    if classified == CallbackStatus.FAILED_HTTP_4XX:
        log.status = CallbackStatus.FAILED_HTTP_4XX
        await session.flush()
        return

    # 5xx
    retry_policy = dict(callback_config.retry_policy or {})
    log.status = _decide_retry(log, retry_policy)
    await session.flush()


def _decide_retry(log: CallbackLog, retry_policy: dict[str, Any]) -> CallbackStatus:
    """Decide whether to schedule a retry or mark exhausted.

    Mutates ``log.retry_count`` and ``log.next_retry_at`` as appropriate.
    Returns the new status to assign.
    """

    max_attempts = _max_attempts(retry_policy)
    next_retry_count = log.retry_count + 1
    if max_attempts > 0 and next_retry_count >= max_attempts:
        log.retry_count = next_retry_count
        log.next_retry_at = None
        return CallbackStatus.EXHAUSTED

    log.retry_count = next_retry_count
    interval = _retry_interval_seconds(retry_policy, next_retry_count - 1)
    log.next_retry_at = datetime.now(UTC) + timedelta(seconds=interval)
    return CallbackStatus.PENDING_RETRY


async def process_callbacks(
    session: AsyncSession,
    http: httpx.AsyncClient,
    *,
    call_record_id: int,
    default_timeout: float,
    hangup_cause: str | None = None,
) -> int:
    """Run all enabled callback_configs for the call's campaign concurrently.

    Returns the count of configs evaluated (not necessarily dispatched).
    """

    call = await session.get(CallRecord, call_record_id)
    if call is None:
        logger.warning("callbacks_no_call_record call_record_id=%d", call_record_id)
        return 0

    lead = await session.get(Lead, call.lead_id)
    if lead is None:
        logger.warning(
            "callbacks_no_lead call_record_id=%d lead_id=%d", call_record_id, call.lead_id
        )
        return 0

    summary = (
        await session.execute(
            select(CallSummary).where(CallSummary.call_record_id == call_record_id)
        )
    ).scalar_one_or_none()

    ctx = _build_eval_context(
        call_record=call,
        call_summary=summary,
        lead=lead,
        hangup_cause=hangup_cause,
    )

    configs = list(
        (
            await session.execute(
                select(CallbackConfig)
                .where(CallbackConfig.campaign_id == call.campaign_id)
                .where(CallbackConfig.enabled.is_(True))
            )
        )
        .scalars()
        .all()
    )

    if not configs:
        return 0

    await asyncio.gather(
        *[
            dispatch_callback(
                session,
                http,
                callback_config=cfg,
                ctx=ctx,
                call_record_id=call_record_id,
                default_timeout=default_timeout,
            )
            for cfg in configs
        ],
        return_exceptions=False,
    )
    return len(configs)


# Helpers exposed for retry_loop reuse.
__all__ = [
    "RenderError",
    "_build_eval_context",
    "_classify_response",
    "_decide_retry",
    "_evaluate_trigger",
    "_max_attempts",
    "_render_payload",
    "_retry_interval_seconds",
    "_sign_request",
    "dispatch_callback",
    "process_callbacks",
]


def build_eval_context(
    *,
    call_record: CallRecord,
    call_summary: CallSummary | None,
    lead: Lead,
    hangup_cause: str | None = None,
) -> dict[str, Any]:
    return _build_eval_context(
        call_record=call_record,
        call_summary=call_summary,
        lead=lead,
        hangup_cause=hangup_cause,
    )


def evaluate_trigger(trigger: dict[str, Any], ctx: dict[str, Any]) -> bool:
    return _evaluate_trigger(trigger, ctx)


def render_payload(template_str: str, ctx: dict[str, Any]) -> str:
    return _render_payload(template_str, ctx)


def sign_request(secret: str, body: str, timestamp: int) -> dict[str, str]:
    return _sign_request(secret, body, timestamp)
