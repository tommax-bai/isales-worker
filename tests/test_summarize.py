"""Tests for summarize_call (mock LLM provider)."""

from __future__ import annotations

import pytest
from isales_common.models.call_record import CallRecord
from isales_common.models.call_summary import CallSummary
from isales_common.models.campaign import Campaign
from isales_common.models.lead import Lead
from sqlalchemy import select

from isales_worker.llm import MockProvider
from isales_worker.summarize import summarize_call


def _transcript(*, goal_achieved: bool, goal_type: str | None = None) -> list[dict]:
    return [
        {"type": "greeting", "ts": 0, "text": "hello"},
        {"type": "user_speech", "ts": 1, "text": "yes"},
        {"type": "bot_speech", "ts": 2, "text": "great",
         "goal_achieved": goal_achieved, "goal_type": goal_type or "",
         "extracted": {"intent": "ok"}},
    ]


@pytest.mark.asyncio(loop_scope="session")
async def test_summarize_writes_call_summary(sessionmaker_) -> None:  # type: ignore[no-untyped-def]
    async with sessionmaker_() as session:
        camp = Campaign(name="C", extraction_fields=[{"name": "intent"}])
        session.add(camp)
        await session.flush()
        lead = Lead(campaign_id=camp.id, phone="138")
        session.add(lead)
        await session.flush()
        cr = CallRecord(
            lead_id=lead.id,
            campaign_id=camp.id,
            transcript=_transcript(goal_achieved=True, goal_type="appointment"),
        )
        session.add(cr)
        await session.flush()
        cr_id = cr.id

        summary = await summarize_call(session, cr_id, MockProvider())
        await session.commit()

    assert summary is not None
    assert summary.goal_achieved is True
    assert summary.goal_type == "appointment"
    assert summary.extracted_fields == {"intent": "ok"}
    assert summary.summary_text and "用户" in summary.summary_text


@pytest.mark.asyncio(loop_scope="session")
async def test_summarize_idempotent(sessionmaker_) -> None:  # type: ignore[no-untyped-def]
    """Calling summarize twice for the same call_record returns the existing row."""

    async with sessionmaker_() as session:
        camp = Campaign(name="C2")
        session.add(camp)
        await session.flush()
        lead = Lead(campaign_id=camp.id, phone="139")
        session.add(lead)
        await session.flush()
        cr = CallRecord(
            lead_id=lead.id,
            campaign_id=camp.id,
            transcript=_transcript(goal_achieved=False),
        )
        session.add(cr)
        await session.flush()
        cr_id = cr.id

        s1 = await summarize_call(session, cr_id, MockProvider())
        await session.commit()
        s2 = await summarize_call(session, cr_id, MockProvider())
        await session.commit()

    assert s1 is not None and s2 is not None
    assert s1.id == s2.id

    # Only one call_summary row should exist
    async with sessionmaker_() as session:
        count = (
            await session.execute(
                select(CallSummary).where(CallSummary.call_record_id == cr_id)
            )
        ).all()
    assert len(count) == 1


@pytest.mark.asyncio(loop_scope="session")
async def test_summarize_no_call_record_returns_none(sessionmaker_) -> None:  # type: ignore[no-untyped-def]
    async with sessionmaker_() as session:
        result = await summarize_call(session, 99999, MockProvider())
    assert result is None
