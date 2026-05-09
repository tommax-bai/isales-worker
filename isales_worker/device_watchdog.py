"""Device-status watchdog: marks long-stale modems offline.

Spec: device-hardware § modem-controller 心跳与失联探测 — every 30s, flip
``device.status`` to ``offline`` for any row whose ``last_seen_at`` is
older than 120 s. The ``WHERE status != 'offline'`` guard makes the update
idempotent and prevents racing with the udev-triggered direct PATCH.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta

from isales_common.enums import DeviceStatus
from isales_common.models import Device
from sqlalchemy import update
from sqlalchemy.ext.asyncio import async_sessionmaker

logger = logging.getLogger(__name__)


WATCHDOG_INTERVAL_SECONDS = 30
STALE_THRESHOLD_SECONDS = 120


async def run_watchdog_once(
    sm: async_sessionmaker,
    *,
    threshold_seconds: int = STALE_THRESHOLD_SECONDS,
) -> int:
    """Mark all stale rows offline. Returns the number of rows updated."""

    cutoff = datetime.now(tz=UTC) - timedelta(seconds=threshold_seconds)
    stmt = (
        update(Device)
        .where(
            Device.last_seen_at.is_not(None),
            Device.last_seen_at < cutoff,
            Device.status != DeviceStatus.OFFLINE,
        )
        .values(status=DeviceStatus.OFFLINE)
    )
    async with sm() as session:
        result = await session.execute(stmt)
        await session.commit()
    rowcount = int(result.rowcount or 0)
    if rowcount:
        logger.info(
            "watchdog_marked_offline",
            extra={"rowcount": rowcount, "threshold_s": threshold_seconds},
        )
    return rowcount


async def watchdog_loop(
    sm: async_sessionmaker,
    *,
    interval_seconds: float = WATCHDOG_INTERVAL_SECONDS,
    threshold_seconds: int = STALE_THRESHOLD_SECONDS,
    iterations: int | None = None,
) -> None:
    """Run the watchdog forever (or for ``iterations`` if set, for tests)."""

    completed = 0
    while iterations is None or completed < iterations:
        try:
            await run_watchdog_once(sm, threshold_seconds=threshold_seconds)
        except Exception:
            logger.exception("watchdog_failed")
        completed += 1
        if iterations is not None and completed >= iterations:
            return
        await asyncio.sleep(interval_seconds)
