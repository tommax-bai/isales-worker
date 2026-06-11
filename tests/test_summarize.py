"""Tests for summarize_call (mock LLM provider)."""

from __future__ import annotations

import pytest
from isales_common.models.call_record import CallRecord
from isales_common.models.call_summary import CallSummary
from isales_common.models.campaign import Campaign
from isales_common.models.lead import Lead
from sqlalchemy import select

from isales_worker.llm import MockProvider
from isales_worker.llm.mock import _last_role_markers
from isales_worker.summarize import summarize_call


def _transcript(*, goal_achieved: bool, goal_type: str | None = None) -> list[dict]:
    # Engine-shaped transcript: AI turns are ``ai_reply`` events carrying the goal
    # markers (not the gone ``bot_speech``). When achieved, the engine also writes a
    # standalone ``goal_achieved`` milestone event after the closing ai_reply.
    evts: list[dict] = [
        {"type": "greeting", "ts": 0, "text": "hello"},
        {"type": "user_speech", "ts": 1, "text": "yes"},
        {"type": "ai_reply", "ts": 2, "text": "great", "turn_id": 1,
         "goal_achieved": goal_achieved, "goal_type": goal_type or "",
         "extracted": {"intent": "ok"}, "is_wrap_up": False},
    ]
    if goal_achieved:
        evts.append({"type": "goal_achieved", "ts": 3,
                     "goal_type": goal_type or "", "extracted": {"intent": "ok"}})
    return evts


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


# ── _last_role_markers: pure reads against engine-shaped transcripts ──────────
# Regression for the bot_speech→ai_reply drift (fix-goal-achievement-pipeline).


def test_markers_read_from_ai_reply() -> None:
    transcript = [
        {"type": "greeting", "ts": 0, "text": "hi"},
        {"type": "ai_reply", "ts": 1, "text": "约一下吧", "turn_id": 1,
         "goal_achieved": True, "goal_type": "intent_confirmed", "extracted": {}},
        {"type": "goal_achieved", "ts": 2, "goal_type": "intent_confirmed", "extracted": {}},
    ]
    achieved, gtype, _ = _last_role_markers(transcript)
    assert achieved is True
    assert gtype == "intent_confirmed"


def test_markers_latch_through_wrap_up_replies() -> None:
    # After achievement the engine appends WRAPPING_UP ai_reply events with
    # goal_achieved=False — achievement MUST latch, not be overwritten by them.
    transcript = [
        {"type": "ai_reply", "ts": 1, "text": "约一下吧",
         "goal_achieved": True, "goal_type": "intent_confirmed", "extracted": {}},
        {"type": "goal_achieved", "ts": 2, "goal_type": "intent_confirmed", "extracted": {}},
        {"type": "ai_reply", "ts": 3, "text": "那不打扰了",
         "goal_achieved": False, "goal_type": "", "extracted": {}, "is_wrap_up": True},
        {"type": "ai_reply", "ts": 4, "text": "再见",
         "goal_achieved": False, "goal_type": "", "extracted": {}, "is_wrap_up": True},
    ]
    achieved, gtype, _ = _last_role_markers(transcript)
    assert achieved is True
    assert gtype == "intent_confirmed"


def test_markers_not_achieved() -> None:
    transcript = [
        {"type": "ai_reply", "ts": 1, "text": "考虑下",
         "goal_achieved": False, "goal_type": "", "extracted": {}},
    ]
    achieved, gtype, _ = _last_role_markers(transcript)
    assert achieved is False
    assert gtype is None


def test_markers_ignore_legacy_bot_speech() -> None:
    # The fixed bug: bot_speech was the ONLY type read. We now read ai_reply only;
    # there is no real bot_speech data in prod (engine never wrote it), so honoring
    # it would be a pointless fallback layer. Confirm it is ignored.
    transcript = [
        {"type": "bot_speech", "ts": 1, "text": "ok",
         "goal_achieved": True, "goal_type": "appointment", "extracted": {}},
    ]
    achieved, gtype, _ = _last_role_markers(transcript)
    assert achieved is False
    assert gtype is None
