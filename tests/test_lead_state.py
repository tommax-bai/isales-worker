"""Tests for the retry-followup decision matrix + apply_lead_state guard."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from isales_common.enums import HangupCause, LeadStatus
from isales_common.models.call_record import CallRecord
from isales_common.models.call_summary import CallSummary
from isales_common.models.campaign import Campaign
from isales_common.models.lead import Lead

from isales_worker.lead_state import apply_lead_state, decide_next


def _campaign(
    *,
    retry_intervals=(60, 240, 1440),
    retry_max=3,
    follow_up_days=3,
    follow_up_max=2,
):  # type: ignore[no-untyped-def]
    return SimpleNamespace(
        id=1,
        retry_intervals=list(retry_intervals),
        retry_max_count=retry_max,
        follow_up_interval_days=follow_up_days,
        follow_up_max_count=follow_up_max,
    )


def _lead(*, retry_count: int = 0, follow_up_count: int = 0):  # type: ignore[no-untyped-def]
    return SimpleNamespace(
        id=100,
        retry_count=retry_count,
        follow_up_count=follow_up_count,
        status=LeadStatus.CALLING,
    )


def _call(*, transcript: list[dict] | None = None):  # type: ignore[no-untyped-def]
    return SimpleNamespace(
        id=200,
        transcript=transcript or [],
        ended_at=datetime(2026, 5, 6, 12, 0, tzinfo=UTC),
    )


def _summary(*, goal_achieved: bool = False, goal_type: str | None = None):  # type: ignore[no-untyped-def]
    return SimpleNamespace(
        goal_achieved=goal_achieved,
        goal_type=goal_type,
        extracted_fields={},
        summary_text="x",
    )


def test_retry_path_under_max(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    now = datetime(2026, 5, 6, 12, 0, tzinfo=UTC)
    decision = decide_next(
        lead=_lead(retry_count=0),
        call_record=_call(),
        call_summary=_summary(),
        campaign=_campaign(retry_max=3, retry_intervals=(60, 240, 1440)),
        hangup_cause=HangupCause.NO_ANSWER,
        now=now,
    )
    assert decision.status == LeadStatus.RETRYING
    assert decision.retry_count_delta == 1
    assert decision.next_call_at == now + timedelta(minutes=60)


def test_retry_path_at_max_becomes_failed() -> None:
    decision = decide_next(
        lead=_lead(retry_count=2),  # next would be 3 == max
        call_record=_call(),
        call_summary=_summary(),
        campaign=_campaign(retry_max=3),
        hangup_cause=HangupCause.NO_ANSWER,
    )
    assert decision.status == LeadStatus.FAILED


def test_call_rejected_becomes_failed_no_retry() -> None:
    decision = decide_next(
        lead=_lead(),
        call_record=_call(),
        call_summary=_summary(),
        campaign=_campaign(),
        hangup_cause=HangupCause.CALL_REJECTED,
    )
    assert decision.status == LeadStatus.FAILED


def test_normal_clearing_with_goal_completed() -> None:
    decision = decide_next(
        lead=_lead(),
        call_record=_call(),
        call_summary=_summary(goal_achieved=True),
        campaign=_campaign(),
        hangup_cause=HangupCause.NORMAL_CLEARING,
    )
    assert decision.status == LeadStatus.COMPLETED


def test_normal_clearing_no_goal_following_up() -> None:
    cr = _call()
    decision = decide_next(
        lead=_lead(follow_up_count=0),
        call_record=cr,
        call_summary=_summary(goal_achieved=False),
        campaign=_campaign(follow_up_max=3, follow_up_days=2),
        hangup_cause=HangupCause.NORMAL_CLEARING,
    )
    assert decision.status == LeadStatus.FOLLOWING_UP
    assert decision.follow_up_count_delta == 1
    assert decision.next_call_at == cr.ended_at + timedelta(days=2)


def test_normal_clearing_no_goal_follow_up_exhausted() -> None:
    decision = decide_next(
        lead=_lead(follow_up_count=2),  # next would be 3 == max
        call_record=_call(),
        call_summary=_summary(goal_achieved=False),
        campaign=_campaign(follow_up_max=3),
        hangup_cause=HangupCause.NORMAL_CLEARING,
    )
    assert decision.status == LeadStatus.FOLLOW_UP_EXHAUSTED


def test_normal_clearing_no_goal_zero_follow_up_max_exhausted() -> None:
    """If campaign disables follow-up entirely (max=0), → exhausted."""

    decision = decide_next(
        lead=_lead(),
        call_record=_call(),
        call_summary=_summary(goal_achieved=False),
        campaign=_campaign(follow_up_max=0),
        hangup_cause=HangupCause.NORMAL_CLEARING,
    )
    assert decision.status == LeadStatus.FOLLOW_UP_EXHAUSTED


def test_marked_for_handoff_transferred() -> None:
    decision = decide_next(
        lead=_lead(),
        call_record=_call(),
        call_summary=_summary(),
        campaign=_campaign(),
        hangup_cause=HangupCause.MARKED_FOR_HANDOFF,
    )
    assert decision.status == LeadStatus.TRANSFERRED


def test_do_not_call_via_goal_type_overrides() -> None:
    decision = decide_next(
        lead=_lead(),
        call_record=_call(),
        call_summary=_summary(goal_type="do_not_call"),
        campaign=_campaign(),
        hangup_cause=HangupCause.NORMAL_CLEARING,
    )
    assert decision.status == LeadStatus.DO_NOT_CALL
    assert decision.clear_next_call_at is True


def test_do_not_call_via_transcript_event_overrides() -> None:
    decision = decide_next(
        lead=_lead(),
        call_record=_call(transcript=[{"type": "do_not_call_marked"}]),
        call_summary=_summary(),
        campaign=_campaign(),
        hangup_cause=HangupCause.NO_ANSWER,  # would normally retry
    )
    assert decision.status == LeadStatus.DO_NOT_CALL
    assert decision.clear_next_call_at is True


def test_retry_intervals_clamp_to_last() -> None:
    now = datetime(2026, 5, 6, 12, 0, tzinfo=UTC)
    decision = decide_next(
        lead=_lead(retry_count=5),  # next=6, beyond intervals length
        call_record=_call(),
        call_summary=_summary(),
        campaign=_campaign(retry_max=20, retry_intervals=(60, 240, 1440)),
        hangup_cause=HangupCause.NO_ANSWER,
        now=now,
    )
    # Index = min(5, 2) = 2 → 1440 minutes
    assert decision.next_call_at == now + timedelta(minutes=1440)


# ─── apply_lead_state row-guard ────────────────────────────────────────────


@pytest.mark.asyncio(loop_scope="session")
async def test_apply_lead_state_writes_when_calling(sessionmaker_) -> None:  # type: ignore[no-untyped-def]
    async with sessionmaker_() as session:
        camp = Campaign(name="C", retry_intervals=[60], retry_max_count=3,
                        follow_up_interval_days=3, follow_up_max_count=2)
        session.add(camp)
        await session.flush()
        lead = Lead(
            campaign_id=camp.id, phone="138",
            status=LeadStatus.CALLING, retry_count=0, follow_up_count=0,
        )
        session.add(lead)
        await session.flush()
        cr = CallRecord(lead_id=lead.id, campaign_id=camp.id,
                        ended_at=datetime.now(UTC))
        session.add(cr)
        await session.flush()
        cs = CallSummary(call_record_id=cr.id, summary_text="x", goal_achieved=True)
        session.add(cs)
        await session.commit()
        cr_id = cr.id
        lead_id = lead.id

    async with sessionmaker_() as session:
        ok = await apply_lead_state(
            session, cr_id, hangup_cause=HangupCause.NORMAL_CLEARING
        )
        await session.commit()
    assert ok is True

    async with sessionmaker_() as session:
        lead2 = await session.get(Lead, lead_id)
    assert lead2.status == LeadStatus.COMPLETED


@pytest.mark.asyncio(loop_scope="session")
async def test_apply_lead_state_blocked_when_not_calling(sessionmaker_) -> None:  # type: ignore[no-untyped-def]
    """Row guard: lead.status='new' should NOT be touched by worker."""

    async with sessionmaker_() as session:
        camp = Campaign(name="C2", retry_intervals=[60], retry_max_count=3)
        session.add(camp)
        await session.flush()
        lead = Lead(
            campaign_id=camp.id, phone="139",
            status=LeadStatus.NEW,  # NOT calling
            retry_count=0, follow_up_count=0,
        )
        session.add(lead)
        await session.flush()
        cr = CallRecord(lead_id=lead.id, campaign_id=camp.id,
                        ended_at=datetime.now(UTC))
        session.add(cr)
        await session.commit()
        cr_id = cr.id
        lead_id = lead.id

    async with sessionmaker_() as session:
        ok = await apply_lead_state(
            session, cr_id, hangup_cause=HangupCause.NO_ANSWER
        )
        await session.commit()
    assert ok is False

    async with sessionmaker_() as session:
        lead2 = await session.get(Lead, lead_id)
    assert lead2.status == LeadStatus.NEW  # unchanged
