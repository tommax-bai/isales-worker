"""aggregate_metrics — periodic 7-day rollup into Redis Hash.

Writes per-campaign aggregates into ``isales:metrics:7d:{campaign_id}`` so
the v1 dashboard can read in O(1). The /analytics/* endpoints in
isales-api remain the source of truth (they SQL-aggregate on demand);
this is a fast-path cache only.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from isales_common.enums import CallStatus
from isales_common.models.call_record import CallRecord
from isales_common.models.call_summary import CallSummary
from redis.asyncio import Redis
from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from isales_worker.settings import Settings

logger = logging.getLogger(__name__)

METRIC_KEY_PREFIX = "isales:metrics:7d:"


async def tick(
    sessionmaker: async_sessionmaker[Any],
    redis: Redis[Any],
    *,
    now: datetime | None = None,
) -> int:
    when = now or datetime.now(UTC)
    since = when - timedelta(days=7)

    async with sessionmaker() as session:
        # answered = call has non-null started_at; goal_achieved comes from
        # call_summary.goal_achieved.
        stmt = (
            select(
                CallRecord.campaign_id,
                func.count(CallRecord.id).label("total"),
                func.sum(case((CallRecord.started_at.is_not(None), 1), else_=0)).label("answered"),
                func.sum(case((CallSummary.goal_achieved.is_(True), 1), else_=0)).label(
                    "goal_achieved"
                ),
                func.coalesce(func.avg(CallRecord.duration), 0).label("avg_duration"),
            )
            .join(
                CallSummary,
                CallSummary.call_record_id == CallRecord.id,
                isouter=True,
            )
            .where(CallRecord.status == CallStatus.END)
            .where(CallRecord.created_at >= since)
            .group_by(CallRecord.campaign_id)
        )
        rows = (await session.execute(stmt)).all()

    written = 0
    for row in rows:
        campaign_id = row.campaign_id
        await redis.hset(
            f"{METRIC_KEY_PREFIX}{campaign_id}",
            mapping={
                "total_calls": int(row.total or 0),
                "answered": int(row.answered or 0),
                "goal_achieved": int(row.goal_achieved or 0),
                "avg_duration": float(row.avg_duration or 0),
                "updated_at": when.isoformat(),
            },
        )
        written += 1
    return written


async def metrics_loop(
    sessionmaker: async_sessionmaker[Any],
    redis: Redis[Any],
    settings: Settings,
) -> None:
    while True:
        try:
            n = await tick(sessionmaker, redis)
            logger.info("metrics_tick_done campaigns=%d", n)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("metrics_tick_failed")
        await asyncio.sleep(settings.worker_metrics_tick_interval)
