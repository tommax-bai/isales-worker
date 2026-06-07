"""update_lead_state — decide + write the lead's next status.

Spec: retry-followup § Requirement "lead 状态机", "scheduler 调度数据流"
      (worker write-back scenarios), "重试间隔与上限", "跟进间隔与上限",
      ""勿打" 识别（双重 OR 关系）".

Priority of rule matching (high to low):
  1. do_not_call (goal_type='do_not_call' OR transcript do_not_call_marked)
  2. marked_for_handoff (hangup_cause)
  3. retry-bucket hangup_cause + retry-count check
  4. completed/follow_up bucket (normal hangup_cause + goal_achieved)
  5. fallback → failed
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from isales_common.enums import HangupCause, LeadStatus
from isales_common.models.call_record import CallRecord
from isales_common.models.call_summary import CallSummary
from isales_common.models.campaign import Campaign
from isales_common.models.lead import Lead
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

RETRY_HANGUP_CAUSES = {
    HangupCause.NO_ANSWER,
    HangupCause.USER_BUSY,
    HangupCause.NETWORK_OUT_OF_ORDER,
    HangupCause.TEMPORARY_FAILURE,
}

# Hangup causes that close the call cleanly enough that we look at goal_achieved.
NORMAL_HANGUP_CAUSES = {
    HangupCause.NORMAL_CLEARING,
    HangupCause.WRAP_UP_COMPLETED,
    HangupCause.SILENCE_MAX_REACHED,
    HangupCause.USER_HANGUP,
    HangupCause.NO_PROGRESS_TIMEOUT,
    HangupCause.MANUAL_HANGUP,
}

DO_NOT_CALL_TRANSCRIPT_TYPE = "do_not_call_marked"


@dataclass(frozen=True)
class LeadStateUpdate:
    status: LeadStatus
    retry_count_delta: int = 0
    follow_up_count_delta: int = 0
    next_call_at: datetime | None = None
    clear_next_call_at: bool = False
    last_hangup_cause: str | None = None


def _has_do_not_call_marker(
    call_summary: CallSummary | None,
    transcript: list[dict[str, Any]],
) -> bool:
    if call_summary is not None and (call_summary.goal_type or "").lower() == "do_not_call":
        return True
    return any(
        evt.get("type") == DO_NOT_CALL_TRANSCRIPT_TYPE for evt in transcript
    )


def _retry_interval_minutes(intervals_minutes: list[Any], retry_count: int) -> int:
    """retry-followup spec: ``retry_intervals[retry_count - 1]``; clamp to last."""

    if not intervals_minutes:
        return 60  # safe fallback when campaign has no retry_intervals configured
    idx = min(max(retry_count - 1, 0), len(intervals_minutes) - 1)
    return int(intervals_minutes[idx])


def decide_next(
    *,
    lead: Lead,
    call_record: CallRecord,
    call_summary: CallSummary | None,
    campaign: Campaign,
    hangup_cause: str | HangupCause | None,
    transcript: list[dict[str, Any]] | None = None,
    now: datetime | None = None,
) -> LeadStateUpdate:
    """Pure decision function — no DB writes.

    ``hangup_cause`` comes from the inbound CallEnded message; CallRecord
    itself has no hangup_cause column in v0.1.2.
    """

    transcript = transcript if transcript is not None else (call_record.transcript or [])
    when = now or datetime.now(UTC)
    cause: HangupCause | None
    if isinstance(hangup_cause, HangupCause):
        cause = hangup_cause
    else:
        try:
            cause = HangupCause(hangup_cause) if hangup_cause else None
        except ValueError:
            cause = None

    # Priority 1: do-not-call
    if _has_do_not_call_marker(call_summary, transcript):
        return LeadStateUpdate(
            status=LeadStatus.DO_NOT_CALL,
            clear_next_call_at=True,
            last_hangup_cause=str(cause) if cause else None,
        )

    # Priority 2: human handoff
    if cause == HangupCause.MARKED_FOR_HANDOFF:
        return LeadStateUpdate(
            status=LeadStatus.TRANSFERRED,
            last_hangup_cause=cause.value,
        )

    # Priority 3: retry bucket
    if cause in RETRY_HANGUP_CAUSES:
        next_retry_count = lead.retry_count + 1
        if next_retry_count >= campaign.retry_max_count:
            return LeadStateUpdate(
                status=LeadStatus.FAILED,
                retry_count_delta=1,
                last_hangup_cause=cause.value,
            )
        interval = _retry_interval_minutes(campaign.retry_intervals, next_retry_count)
        return LeadStateUpdate(
            status=LeadStatus.RETRYING,
            retry_count_delta=1,
            next_call_at=when + timedelta(minutes=interval),
            last_hangup_cause=cause.value,
        )

    # Priority 4: rejected user → failed (no retry, no follow-up)
    if cause == HangupCause.CALL_REJECTED:
        return LeadStateUpdate(
            status=LeadStatus.FAILED,
            last_hangup_cause=cause.value,
        )

    # Priority 4b: AI proactively hung up via the referee gating verdict
    # (engine-tools-multidialogue-gating; tool:hangup). retry-followup spec
    # § "REFEREE_HANGUP 归入不自动重拨终态": NO auto-redial — neither retry nor
    # follow-up. Explicit branch (not the catch-all fall-through) so the intent
    # is enforced by code, not by ordering. Honors goal_achieved on the rare
    # path where the AI closed after meeting the goal.
    if cause == HangupCause.REFEREE_HANGUP:
        goal_done = bool(call_summary.goal_achieved) if call_summary is not None else False
        return LeadStateUpdate(
            status=LeadStatus.COMPLETED if goal_done else LeadStatus.FAILED,
            last_hangup_cause=cause.value,
        )

    # Priority 5: normal hangup — branch on goal_achieved
    goal_achieved = bool(call_summary.goal_achieved) if call_summary is not None else False
    if cause in NORMAL_HANGUP_CAUSES or cause is None:
        if goal_achieved:
            return LeadStateUpdate(
                status=LeadStatus.COMPLETED,
                last_hangup_cause=cause.value if cause else None,
            )

        # Not achieved → follow-up bucket if configured
        max_follow = int(campaign.follow_up_max_count or 0)
        next_follow_count = lead.follow_up_count + 1
        if max_follow <= 0 or next_follow_count >= max_follow:
            return LeadStateUpdate(
                status=LeadStatus.FOLLOW_UP_EXHAUSTED,
                follow_up_count_delta=1 if max_follow > 0 else 0,
                last_hangup_cause=cause.value if cause else None,
            )
        interval_days = int(campaign.follow_up_interval_days or 0)
        ended_at = call_record.ended_at or when
        return LeadStateUpdate(
            status=LeadStatus.FOLLOWING_UP,
            follow_up_count_delta=1,
            next_call_at=ended_at + timedelta(days=max(interval_days, 0)),
            last_hangup_cause=cause.value if cause else None,
        )

    # Fallback (unrecognized cause): failed
    return LeadStateUpdate(
        status=LeadStatus.FAILED,
        last_hangup_cause=cause.value if cause else None,
    )


async def apply_lead_state(
    session: AsyncSession,
    call_record_id: int,
    *,
    hangup_cause: str | HangupCause | None,
    now: datetime | None = None,
) -> bool:
    """Compute + write next lead state. Returns True iff the row was updated.

    Uses ``WHERE status='calling'`` as a row-level guard so we never clobber
    a lead that scheduler hasn't claimed yet (or that another worker already
    moved). rowcount=0 is logged WARN but is not an error.
    """

    call = await session.get(CallRecord, call_record_id)
    if call is None:
        logger.warning("lead_state_no_call_record call_record_id=%d", call_record_id)
        return False
    lead = await session.get(Lead, call.lead_id)
    if lead is None:
        logger.warning(
            "lead_state_no_lead call_record_id=%d lead_id=%d",
            call_record_id, call.lead_id,
        )
        return False
    campaign = await session.get(Campaign, call.campaign_id)
    if campaign is None:
        logger.warning("lead_state_no_campaign campaign_id=%d", call.campaign_id)
        return False

    summary = None
    from isales_common.models.call_summary import CallSummary as _CS
    from sqlalchemy import select

    summary = (
        await session.execute(
            select(_CS).where(_CS.call_record_id == call_record_id)
        )
    ).scalar_one_or_none()

    decision = decide_next(
        lead=lead,
        call_record=call,
        call_summary=summary,
        campaign=campaign,
        hangup_cause=hangup_cause,
        now=now,
    )

    values: dict[str, Any] = {"status": decision.status}
    if decision.retry_count_delta:
        values["retry_count"] = lead.retry_count + decision.retry_count_delta
    if decision.follow_up_count_delta:
        values["follow_up_count"] = lead.follow_up_count + decision.follow_up_count_delta
    if decision.clear_next_call_at:
        values["next_call_at"] = None
    elif decision.next_call_at is not None:
        values["next_call_at"] = decision.next_call_at
    if decision.last_hangup_cause is not None:
        values["last_hangup_cause"] = decision.last_hangup_cause

    stmt = (
        update(Lead)
        .where(Lead.id == lead.id)
        .where(Lead.status == LeadStatus.CALLING)
        .values(**values)
    )
    result = await session.execute(stmt)
    rowcount = result.rowcount or 0  # type: ignore[attr-defined]
    if rowcount == 0:
        logger.warning(
            "lead_state_guard_blocked lead_id=%d current_status=%s want=%s",
            lead.id, lead.status, decision.status,
        )
        return False
    logger.info(
        "lead_state_updated lead_id=%d new_status=%s next_call_at=%s",
        lead.id, decision.status,
        decision.next_call_at.isoformat() if decision.next_call_at else None,
    )
    return True
