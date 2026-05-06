"""CallEnded queue consumer.

Reads ``engine:worker:call-ended``, validates schema_version, and runs
summarize → callbacks → lead_state in order. Failures within one stage
are logged but don't abort the rest of the pipeline (so lead state still
flips even if the LLM blew up).

Spec: service-communication § 通信通道矩阵 (engine→worker);
      message-contract § Scenario "consumer 校验 schema_version".
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
from typing import Any

import httpx
from isales_common.schemas.messages import CallEnded
from pydantic import ValidationError
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import async_sessionmaker

from isales_worker.callbacks import process_callbacks
from isales_worker.lead_state import apply_lead_state
from isales_worker.llm.base import LLMProvider
from isales_worker.settings import Settings
from isales_worker.summarize import summarize_call

logger = logging.getLogger(__name__)

CALL_ENDED_QUEUE = "engine:worker:call-ended"
DLQ = "worker:dlq"
SUPPORTED_SCHEMA_VERSIONS = {1}


def _peek_schema_version(raw: str) -> int | None:
    try:
        parsed = _json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(parsed, dict):
        return None
    sv = parsed.get("schema_version")
    return sv if isinstance(sv, int) else None


async def _process_call_ended(
    msg: CallEnded,
    *,
    sessionmaker: async_sessionmaker[Any],
    http: httpx.AsyncClient,
    provider: LLMProvider,
    settings: Settings,
) -> None:
    cid = msg.call_record_id

    # Stage 1: summarize (own session/txn).
    try:
        async with sessionmaker() as session:
            await summarize_call(session, cid, provider)
            await session.commit()
    except Exception:
        logger.exception("summarize_failed call_record_id=%d", cid)

    # Stage 2: callbacks (own session/txn — webhook is slow path,
    # design.md Decision 11 keeps it out of the lead-state txn).
    try:
        async with sessionmaker() as session:
            await process_callbacks(
                session,
                http,
                call_record_id=cid,
                default_timeout=settings.webhook_default_timeout_seconds,
                hangup_cause=msg.hangup_cause.value,
            )
            await session.commit()
    except Exception:
        logger.exception("process_callbacks_failed call_record_id=%d", cid)

    # Stage 3: lead state write-back (own session/txn).
    try:
        async with sessionmaker() as session:
            await apply_lead_state(session, cid, hangup_cause=msg.hangup_cause)
            await session.commit()
    except Exception:
        logger.exception("apply_lead_state_failed call_record_id=%d", cid)


async def _handle_raw(
    raw: str,
    *,
    redis: Redis[Any],
    sessionmaker: async_sessionmaker[Any],
    http: httpx.AsyncClient,
    provider: LLMProvider,
    settings: Settings,
) -> None:
    sv = _peek_schema_version(raw)
    if sv is None or sv not in SUPPORTED_SCHEMA_VERSIONS:
        logger.warning("callend_dlq schema_version=%s raw_len=%d", sv, len(raw))
        await redis.lpush(DLQ, raw)
        return
    try:
        msg = CallEnded.model_validate_json(raw)
    except ValidationError as exc:
        logger.warning("callend_dlq validation errors=%s", exc.errors()[:3])
        await redis.lpush(DLQ, raw)
        return

    logger.info(
        "callend_received call_record_id=%d hangup_cause=%s message_id=%s",
        msg.call_record_id, msg.hangup_cause, msg.message_id,
    )
    await _process_call_ended(
        msg,
        sessionmaker=sessionmaker,
        http=http,
        provider=provider,
        settings=settings,
    )


async def callend_loop(
    *,
    redis: Redis[Any],
    sessionmaker: async_sessionmaker[Any],
    http: httpx.AsyncClient,
    provider: LLMProvider,
    settings: Settings,
) -> None:
    while True:
        try:
            popped = await redis.blpop([CALL_ENDED_QUEUE], timeout=1)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("callend_blpop_error")
            await asyncio.sleep(1)
            continue

        if popped is None:
            continue
        _key, raw = popped
        try:
            await _handle_raw(
                raw,
                redis=redis,
                sessionmaker=sessionmaker,
                http=http,
                provider=provider,
                settings=settings,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("callend_handle_error raw_len=%d", len(raw))
