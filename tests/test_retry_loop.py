"""Tests for callback retry scheduler."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx
import pytest
from isales_common.enums import CallbackStatus
from isales_common.models.call_record import CallRecord
from isales_common.models.callback_config import CallbackConfig
from isales_common.models.callback_log import CallbackLog
from isales_common.models.campaign import Campaign
from isales_common.models.lead import Lead
from isales_common.utils.crypto import encrypt

from isales_worker.retry_loop import tick
from isales_worker.settings import Settings


def _settings() -> Settings:
    import os
    os.environ.setdefault("ISALES_DATABASE_URL", "postgresql+asyncpg://test/test")
    os.environ.setdefault("ISALES_REDIS_URL", "redis://localhost:6379/0")
    return Settings()


async def _seed_pending_retry(
    sessionmaker_,  # type: ignore[no-untyped-def]
    *,
    url: str,
    retry_policy: dict,
    retry_count: int = 1,
    next_retry_offset: int = -10,
) -> int:
    async with sessionmaker_() as session:
        camp = Campaign(name="C")
        session.add(camp)
        await session.flush()
        lead = Lead(campaign_id=camp.id, phone="138")
        session.add(lead)
        await session.flush()
        cr = CallRecord(lead_id=lead.id, campaign_id=camp.id)
        session.add(cr)
        await session.flush()
        cfg = CallbackConfig(
            campaign_id=camp.id, name="cb",
            trigger={}, url=url, payload_template="{}",
            retry_policy=retry_policy,
            signing_secret=encrypt("secret"),
            timeout_seconds=5,
        )
        session.add(cfg)
        await session.flush()
        log = CallbackLog(
            callback_config_id=cfg.id,
            call_record_id=cr.id,
            status=CallbackStatus.PENDING_RETRY,
            request_body='{"x": 1}',
            retry_count=retry_count,
            next_retry_at=datetime.now(UTC) + timedelta(seconds=next_retry_offset),
        )
        session.add(log)
        await session.commit()
        return log.id


@pytest.mark.asyncio(loop_scope="session")
async def test_retry_2xx_marks_success(sessionmaker_) -> None:  # type: ignore[no-untyped-def]
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200)

    log_id = await _seed_pending_retry(
        sessionmaker_,
        url="https://hook.test",
        retry_policy={"intervals_seconds": [60, 300, 1800], "max_attempts": 3},
    )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        n = await tick(sessionmaker_, http, _settings())
    assert n == 1
    async with sessionmaker_() as session:
        log = await session.get(CallbackLog, log_id)
    assert log.status == CallbackStatus.SUCCESS
    assert log.next_retry_at is None


@pytest.mark.asyncio(loop_scope="session")
async def test_retry_5xx_increments_and_reschedules(sessionmaker_) -> None:  # type: ignore[no-untyped-def]
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    log_id = await _seed_pending_retry(
        sessionmaker_,
        url="https://hook.test",
        retry_policy={"intervals_seconds": [60, 300, 1800], "max_attempts": 5},
        retry_count=1,
    )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        await tick(sessionmaker_, http, _settings())

    async with sessionmaker_() as session:
        log = await session.get(CallbackLog, log_id)
    assert log.status == CallbackStatus.PENDING_RETRY
    assert log.retry_count == 2
    # interval should be intervals[1] = 300s; allow generous skew
    assert log.next_retry_at is not None
    delta = (log.next_retry_at - datetime.now(UTC)).total_seconds()
    assert 290 < delta < 310


@pytest.mark.asyncio(loop_scope="session")
async def test_retry_max_attempts_exhausts(sessionmaker_) -> None:  # type: ignore[no-untyped-def]
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    log_id = await _seed_pending_retry(
        sessionmaker_,
        url="https://hook.test",
        retry_policy={"intervals_seconds": [60], "max_attempts": 3},
        retry_count=2,  # next attempt → 3, hits max
    )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        await tick(sessionmaker_, http, _settings())

    async with sessionmaker_() as session:
        log = await session.get(CallbackLog, log_id)
    assert log.status == CallbackStatus.EXHAUSTED
    assert log.retry_count == 3
    assert log.next_retry_at is None


@pytest.mark.asyncio(loop_scope="session")
async def test_retry_4xx_in_retry_path_marks_failed_4xx(sessionmaker_) -> None:  # type: ignore[no-untyped-def]
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(400)

    log_id = await _seed_pending_retry(
        sessionmaker_,
        url="https://hook.test",
        retry_policy={"intervals_seconds": [60], "max_attempts": 5},
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        await tick(sessionmaker_, http, _settings())

    async with sessionmaker_() as session:
        log = await session.get(CallbackLog, log_id)
    assert log.status == CallbackStatus.FAILED_HTTP_4XX
    assert log.next_retry_at is None


@pytest.mark.asyncio(loop_scope="session")
async def test_retry_intervals_clamped_to_last(sessionmaker_) -> None:  # type: ignore[no-untyped-def]
    """retry_count well past intervals length → uses last interval."""

    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    # intervals only has [60]; we'll be on retry_count=4 → next_attempt=5 → max=10
    log_id = await _seed_pending_retry(
        sessionmaker_,
        url="https://hook.test",
        retry_policy={"intervals_seconds": [60], "max_attempts": 10},
        retry_count=4,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        await tick(sessionmaker_, http, _settings())

    async with sessionmaker_() as session:
        log = await session.get(CallbackLog, log_id)
    assert log.status == CallbackStatus.PENDING_RETRY
    delta = (log.next_retry_at - datetime.now(UTC)).total_seconds()
    assert 50 < delta < 70


@pytest.mark.asyncio(loop_scope="session")
async def test_retry_skips_future_next_retry_at(sessionmaker_) -> None:  # type: ignore[no-untyped-def]
    """Logs whose next_retry_at is in the future should NOT be touched."""

    def handler(_req: httpx.Request) -> httpx.Response:
        raise AssertionError("HTTP should not be called for future retries")

    log_id = await _seed_pending_retry(
        sessionmaker_,
        url="https://hook.test",
        retry_policy={"intervals_seconds": [60], "max_attempts": 5},
        next_retry_offset=600,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        n = await tick(sessionmaker_, http, _settings())
    assert n == 0

    async with sessionmaker_() as session:
        log = await session.get(CallbackLog, log_id)
    assert log.status == CallbackStatus.PENDING_RETRY  # unchanged
