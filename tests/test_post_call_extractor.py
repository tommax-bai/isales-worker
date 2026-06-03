"""Tests for the post_call_extractor consumer (pipeline-stream-and-referee)."""

from __future__ import annotations

import json
from typing import Any

import pytest
from isales_common.models.call_record import CallRecord
from isales_common.models.campaign import Campaign
from isales_common.models.lead import Lead

from isales_worker.llm import MockProvider
from isales_worker.llm.base import LLMProvider, SummaryResult
from isales_worker.post_call_extractor import handle_extract_task


async def _seed_call(sessionmaker_) -> int:  # type: ignore[no-untyped-def]
    async with sessionmaker_() as session:
        camp = Campaign(name="C")
        session.add(camp)
        await session.flush()
        lead = Lead(campaign_id=camp.id, phone="138")
        session.add(lead)
        await session.flush()
        cr = CallRecord(lead_id=lead.id, campaign_id=camp.id, transcript=[])
        session.add(cr)
        await session.flush()
        cid = cr.id
        await session.commit()
    return cid


def _task(cid: int) -> str:
    return json.dumps(
        {
            "call_record_id": cid,
            "transcript_snapshot": [
                {"role": "assistant", "text": "您好", "ts_ms": 0},
                {"role": "user", "text": "我想了解一下", "ts_ms": 1},
            ],
            "extractor_role_config_id": 7,
            "extractor_prompt_version_id": 0,
        }
    )


class _BoomProvider(MockProvider):
    async def extract(self, transcript_snapshot, extractor_prompt):  # type: ignore[override]
        raise RuntimeError("llm down")


class _BadShapeProvider(LLMProvider):
    async def summarize(self, transcript, extraction_fields) -> SummaryResult:
        return SummaryResult(summary_text="")

    async def extract(self, transcript_snapshot, extractor_prompt) -> Any:  # type: ignore[override]
        return "not a dict"


@pytest.mark.asyncio(loop_scope="session")
async def test_extract_success_writes_extracted(sessionmaker_) -> None:  # type: ignore[no-untyped-def]
    cid = await _seed_call(sessionmaker_)
    await handle_extract_task(_task(cid), sessionmaker=sessionmaker_, provider=MockProvider())
    async with sessionmaker_() as session:
        rec = await session.get(CallRecord, cid)
        assert rec is not None
        assert rec.extract_status == "done"
        assert rec.extracted == {"customer_intent": "我想了解一下", "n_turns": 2}
        assert rec.extract_error is None


@pytest.mark.asyncio(loop_scope="session")
async def test_extract_llm_failure_marks_failed(sessionmaker_) -> None:  # type: ignore[no-untyped-def]
    cid = await _seed_call(sessionmaker_)
    await handle_extract_task(_task(cid), sessionmaker=sessionmaker_, provider=_BoomProvider())
    async with sessionmaker_() as session:
        rec = await session.get(CallRecord, cid)
        assert rec.extract_status == "failed"
        assert "llm down" in (rec.extract_error or "")


@pytest.mark.asyncio(loop_scope="session")
async def test_extract_bad_shape_marks_failed(sessionmaker_) -> None:  # type: ignore[no-untyped-def]
    cid = await _seed_call(sessionmaker_)
    await handle_extract_task(_task(cid), sessionmaker=sessionmaker_, provider=_BadShapeProvider())
    async with sessionmaker_() as session:
        rec = await session.get(CallRecord, cid)
        assert rec.extract_status == "failed"


@pytest.mark.asyncio(loop_scope="session")
async def test_extract_bad_json_is_dropped(sessionmaker_) -> None:  # type: ignore[no-untyped-def]
    # Should not raise; nothing to assert beyond no exception.
    await handle_extract_task("not-json", sessionmaker=sessionmaker_, provider=MockProvider())


@pytest.mark.asyncio(loop_scope="session")
async def test_extract_missing_call_record_id_dropped(sessionmaker_) -> None:  # type: ignore[no-untyped-def]
    await handle_extract_task(
        json.dumps({"transcript_snapshot": []}),
        sessionmaker=sessionmaker_,
        provider=MockProvider(),
    )
