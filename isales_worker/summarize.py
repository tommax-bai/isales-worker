"""summarize_call: generate call_summary from transcript + last-round markers.

Spec: goal-achievement § Scenario "worker 不重复判定";
      webhook-callback § Scenario "summarize_call 完成后派发".
"""

from __future__ import annotations

import logging

from isales_common.models.call_record import CallRecord
from isales_common.models.call_summary import CallSummary
from isales_common.models.campaign import Campaign
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from isales_worker.llm.base import LLMProvider

logger = logging.getLogger(__name__)


async def summarize_call(
    session: AsyncSession,
    call_record_id: int,
    provider: LLMProvider,
) -> CallSummary | None:
    """Idempotent: if a CallSummary already exists for this call_record_id,
    return the existing row (unique constraint on call_record_id).
    """

    existing = (
        await session.execute(
            select(CallSummary).where(CallSummary.call_record_id == call_record_id)
        )
    ).scalar_one_or_none()
    if existing is not None:
        logger.info("summarize_skip_existing call_record_id=%d", call_record_id)
        return existing

    call = await session.get(CallRecord, call_record_id)
    if call is None:
        logger.warning("summarize_no_call_record call_record_id=%d", call_record_id)
        return None
    campaign = await session.get(Campaign, call.campaign_id)
    extraction_fields = list(campaign.extraction_fields) if campaign is not None else []

    result = await provider.summarize(call.transcript or [], extraction_fields)

    summary = CallSummary(
        call_record_id=call_record_id,
        summary_text=result.summary_text,
        extracted_fields=result.extracted_fields,
        goal_achieved=result.goal_achieved,
        goal_type=result.goal_type,
    )
    session.add(summary)
    await session.flush()
    logger.info(
        "summarize_done call_record_id=%d goal_achieved=%s goal_type=%s",
        call_record_id,
        result.goal_achieved,
        result.goal_type,
    )
    return summary
