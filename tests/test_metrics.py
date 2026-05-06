"""Tests for aggregate_metrics — Redis Hash output."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from isales_common.enums import CallStatus
from isales_common.models.call_record import CallRecord
from isales_common.models.call_summary import CallSummary
from isales_common.models.campaign import Campaign
from isales_common.models.lead import Lead

from isales_worker.metrics import METRIC_KEY_PREFIX, tick


@pytest.mark.asyncio(loop_scope="session")
async def test_metrics_aggregates_per_campaign(sessionmaker_, redis_client) -> None:  # type: ignore[no-untyped-def]
    now = datetime.now(UTC)

    async with sessionmaker_() as session:
        camp_a = Campaign(name="A")
        camp_b = Campaign(name="B")
        session.add_all([camp_a, camp_b])
        await session.flush()

        for camp_id, count, answered_count, achieved_count in [
            (camp_a.id, 5, 4, 2),
            (camp_b.id, 3, 1, 0),
        ]:
            for i in range(count):
                lead = Lead(campaign_id=camp_id, phone=f"138000{camp_id:02d}{i:02d}")
                session.add(lead)
                await session.flush()
                cr = CallRecord(
                    lead_id=lead.id,
                    campaign_id=camp_id,
                    status=CallStatus.END,
                    started_at=(now - timedelta(hours=1)) if i < answered_count else None,
                    ended_at=now - timedelta(hours=1, seconds=-30),
                    duration=30,
                )
                session.add(cr)
                await session.flush()
                session.add(CallSummary(
                    call_record_id=cr.id,
                    summary_text="s",
                    goal_achieved=(i < achieved_count),
                ))
        await session.commit()
        a_id, b_id = camp_a.id, camp_b.id

    n = await tick(sessionmaker_, redis_client)
    assert n == 2

    a_hash = await redis_client.hgetall(f"{METRIC_KEY_PREFIX}{a_id}")
    assert int(a_hash["total_calls"]) == 5
    assert int(a_hash["answered"]) == 4
    assert int(a_hash["goal_achieved"]) == 2

    b_hash = await redis_client.hgetall(f"{METRIC_KEY_PREFIX}{b_id}")
    assert int(b_hash["total_calls"]) == 3
    assert int(b_hash["answered"]) == 1
    assert int(b_hash["goal_achieved"]) == 0

    # cleanup
    await redis_client.delete(f"{METRIC_KEY_PREFIX}{a_id}")
    await redis_client.delete(f"{METRIC_KEY_PREFIX}{b_id}")


@pytest.mark.asyncio(loop_scope="session")
async def test_metrics_excludes_old_data(sessionmaker_, redis_client) -> None:  # type: ignore[no-untyped-def]
    """Records older than 7 days should not be counted."""

    now = datetime.now(UTC)
    async with sessionmaker_() as session:
        camp = Campaign(name="C")
        session.add(camp)
        await session.flush()
        lead = Lead(campaign_id=camp.id, phone="1380000")
        session.add(lead)
        await session.flush()
        # Old record (10 days back)
        old = CallRecord(
            lead_id=lead.id, campaign_id=camp.id,
            status=CallStatus.END,
            started_at=now - timedelta(days=10),
            ended_at=now - timedelta(days=10, seconds=-30),
            duration=30,
            created_at=now - timedelta(days=10),
        )
        session.add(old)
        await session.commit()
        camp_id = camp.id

    await tick(sessionmaker_, redis_client)
    h = await redis_client.hgetall(f"{METRIC_KEY_PREFIX}{camp_id}")
    # No row for this campaign within window → no key
    assert h == {}
