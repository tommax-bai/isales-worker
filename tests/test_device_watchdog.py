"""Watchdog tests — flips long-stale devices to offline, idempotent."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from isales_common.enums import DeviceStatus
from isales_common.models import Device
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from isales_worker.device_watchdog import run_watchdog_once, watchdog_loop


@pytest.mark.asyncio(loop_scope="session")
async def test_watchdog_marks_stale_idle_device_offline(
    clean_engine: AsyncEngine,
) -> None:
    sm = async_sessionmaker(clean_engine, expire_on_commit=False)
    stale = datetime.now(tz=UTC) - timedelta(seconds=180)
    async with sm() as session:
        dev = Device(
            name="stale",
            imei="imei-stale",
            status=DeviceStatus.IDLE,
            last_seen_at=stale,
        )
        session.add(dev)
        await session.commit()
        device_id = dev.id

    rowcount = await run_watchdog_once(sm)
    assert rowcount == 1

    async with sm() as session:
        d = await session.get(Device, device_id)
        assert d is not None
        assert d.status == DeviceStatus.OFFLINE


@pytest.mark.asyncio(loop_scope="session")
async def test_watchdog_leaves_recent_devices_alone(
    clean_engine: AsyncEngine,
) -> None:
    sm = async_sessionmaker(clean_engine, expire_on_commit=False)
    recent = datetime.now(tz=UTC) - timedelta(seconds=30)
    async with sm() as session:
        dev = Device(
            name="fresh",
            imei="imei-fresh",
            status=DeviceStatus.IDLE,
            last_seen_at=recent,
        )
        session.add(dev)
        await session.commit()
        device_id = dev.id

    rowcount = await run_watchdog_once(sm)
    assert rowcount == 0

    async with sm() as session:
        d = await session.get(Device, device_id)
        assert d is not None
        assert d.status == DeviceStatus.IDLE


@pytest.mark.asyncio(loop_scope="session")
async def test_watchdog_is_idempotent_for_already_offline(
    clean_engine: AsyncEngine,
) -> None:
    sm = async_sessionmaker(clean_engine, expire_on_commit=False)
    long_ago = datetime.now(tz=UTC) - timedelta(seconds=600)
    async with sm() as session:
        dev = Device(
            name="dead",
            imei="imei-dead",
            status=DeviceStatus.OFFLINE,
            last_seen_at=long_ago,
        )
        session.add(dev)
        await session.commit()

    # Two consecutive runs both report 0 rowcount.
    assert await run_watchdog_once(sm) == 0
    assert await run_watchdog_once(sm) == 0


@pytest.mark.asyncio(loop_scope="session")
async def test_watchdog_skips_rows_with_no_last_seen_at(
    clean_engine: AsyncEngine,
) -> None:
    sm = async_sessionmaker(clean_engine, expire_on_commit=False)
    async with sm() as session:
        dev = Device(name="never-seen", imei="i-never", status=DeviceStatus.UNKNOWN)
        session.add(dev)
        await session.commit()
        device_id = dev.id

    rowcount = await run_watchdog_once(sm)
    assert rowcount == 0

    async with sm() as session:
        d = await session.get(Device, device_id)
        assert d is not None
        assert d.status == DeviceStatus.UNKNOWN


@pytest.mark.asyncio(loop_scope="session")
async def test_watchdog_loop_runs_n_iterations(
    clean_engine: AsyncEngine,
) -> None:
    sm = async_sessionmaker(clean_engine, expire_on_commit=False)
    # No devices — loop should still terminate cleanly.
    await watchdog_loop(sm, interval_seconds=0, iterations=3)
