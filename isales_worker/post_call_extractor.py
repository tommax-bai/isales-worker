"""Post-call structured-extraction consumer (pipeline-stream-and-referee).

BLPOPs ``isales:extract`` (engine LPUSHes one task per ended call), runs the
extractor LLM over the call's transcript snapshot, and writes
``call_record.extracted`` + ``extract_status='done'``. On failure it writes
``extract_status='failed'`` + ``extract_error`` and does NOT re-enqueue
(anti-avalanche; ops re-runs via SQL on ``extract_status='failed'``).

Spec: service-communication § "isales:extract 队列消息 schema" / "队列消息容错";
ai-pipeline § "post-call extractor 异步抽取".
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from isales_common.models import CallRecord, PromptVersion
from isales_common.redis_keys import EXTRACT_QUEUE
from redis.asyncio import Redis
from sqlalchemy import update
from sqlalchemy.ext.asyncio import async_sessionmaker

from isales_worker.llm.base import LLMProvider

logger = logging.getLogger(__name__)


async def _load_extractor_prompt(
    sessionmaker: async_sessionmaker[Any], prompt_version_id: int
) -> str:
    """Load the extractor system prompt by prompt_version id (empty if absent)."""
    if not prompt_version_id:
        return ""
    async with sessionmaker() as db:
        pv = await db.get(PromptVersion, prompt_version_id)
        return pv.content if pv else ""


async def _mark_done(
    sessionmaker: async_sessionmaker[Any], cid: int, extracted: dict[str, Any]
) -> None:
    async with sessionmaker() as db:
        await db.execute(
            update(CallRecord)
            .where(CallRecord.id == cid)
            .values(extracted=extracted, extract_status="done", extract_error=None)
        )
        await db.commit()


async def _mark_failed(
    sessionmaker: async_sessionmaker[Any], cid: int, reason: str
) -> None:
    async with sessionmaker() as db:
        await db.execute(
            update(CallRecord)
            .where(CallRecord.id == cid)
            .values(extract_status="failed", extract_error=reason[:500])
        )
        await db.commit()


async def handle_extract_task(
    raw: str,
    *,
    sessionmaker: async_sessionmaker[Any],
    provider: LLMProvider,
) -> None:
    """Process one ``isales:extract`` task. Never raises (logs + marks failed)."""
    try:
        payload = json.loads(raw)
    except (ValueError, TypeError):
        logger.warning("extract_task_bad_json raw_len=%d", len(raw))
        return
    if not isinstance(payload, dict):
        logger.warning("extract_task_not_object")
        return

    cid = payload.get("call_record_id")
    if not isinstance(cid, int):
        logger.warning("extract_task_missing_call_record_id")
        return

    transcript = payload.get("transcript_snapshot") or []
    pv_id = payload.get("extractor_prompt_version_id") or 0

    try:
        prompt = await _load_extractor_prompt(sessionmaker, int(pv_id))
        extracted = await provider.extract(transcript, prompt)
        if not isinstance(extracted, dict):
            raise TypeError(f"extractor returned {type(extracted).__name__}, expected dict")
        await _mark_done(sessionmaker, cid, extracted)
        logger.info("extract_done call_record_id=%d fields=%d", cid, len(extracted))
    except Exception as exc:  # noqa: BLE001 — mark failed, do not re-enqueue
        logger.exception("extract_failed call_record_id=%d", cid)
        try:
            await _mark_failed(sessionmaker, cid, f"{type(exc).__name__}: {exc}")
        except Exception:  # noqa: BLE001
            logger.exception("extract_mark_failed_update_failed call_record_id=%d", cid)


async def extract_loop(
    *,
    redis: Redis[Any],
    sessionmaker: async_sessionmaker[Any],
    provider: LLMProvider,
) -> None:
    """BLPOP loop over ``isales:extract`` until cancelled."""
    while True:
        try:
            popped = await redis.blpop([EXTRACT_QUEUE], timeout=1)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("extract_blpop_error")
            await asyncio.sleep(1)
            continue

        if popped is None:
            continue
        _key, raw = popped
        try:
            await handle_extract_task(raw, sessionmaker=sessionmaker, provider=provider)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("extract_handle_error raw_len=%d", len(raw))
