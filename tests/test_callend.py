"""End-to-end tests for CallEnded queue consumer."""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest
from isales_common.enums import CallbackStatus, CallStatus, HangupCause, LeadStatus
from isales_common.models.call_record import CallRecord
from isales_common.models.call_summary import CallSummary
from isales_common.models.callback_config import CallbackConfig
from isales_common.models.callback_log import CallbackLog
from isales_common.models.campaign import Campaign
from isales_common.models.lead import Lead
from isales_common.schemas.messages import CallEnded
from isales_common.utils.crypto import encrypt
from sqlalchemy import select

from isales_worker.callend import DLQ, _handle_raw
from isales_worker.llm import MockProvider
from isales_worker.settings import Settings


def _settings() -> Settings:
    import os
    os.environ.setdefault("ISALES_DATABASE_URL", "postgresql+asyncpg://test/test")
    os.environ.setdefault("ISALES_REDIS_URL", "redis://localhost:6379/0")
    return Settings()


def _ok_handler(_req: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"ok": True})


@pytest.mark.asyncio(loop_scope="session")
async def test_full_pipeline_normal_clearing_with_goal(  # type: ignore[no-untyped-def]
    sessionmaker_, redis_client
) -> None:
    """Inject a CallEnded → call_summary written, callback fires, lead → completed."""

    async with sessionmaker_() as session:
        camp = Campaign(
            name="C", retry_intervals=[60], retry_max_count=3,
            follow_up_interval_days=3, follow_up_max_count=2,
        )
        session.add(camp)
        await session.flush()
        lead = Lead(
            campaign_id=camp.id, phone="13800001000",
            status=LeadStatus.CALLING, retry_count=0, follow_up_count=0,
        )
        session.add(lead)
        await session.flush()
        transcript = [
            {"type": "greeting", "ts": 0, "text": "hi"},
            {"type": "user_speech", "ts": 1, "text": "yes"},
            {"type": "bot_speech", "ts": 2, "text": "ok",
             "goal_achieved": True, "goal_type": "appointment",
             "extracted": {"intent": "yes"}},
        ]
        cr = CallRecord(
            lead_id=lead.id, campaign_id=camp.id,
            status=CallStatus.END,
            started_at=datetime.now(UTC),
            ended_at=datetime.now(UTC),
            duration=10,
            transcript=transcript,
            prompt_versions={},
        )
        session.add(cr)
        await session.flush()
        cfg = CallbackConfig(
            campaign_id=camp.id, name="cb",
            trigger={"==": [{"var": "goal_achieved"}, True]},
            url="https://hook.test/cb",
            payload_template='{"lead_id": "{{ lead.id }}"}',
            retry_policy={"intervals_seconds": [60], "max_attempts": 3},
            signing_secret=encrypt("supersecret"),
            timeout_seconds=5,
            enabled=True,
        )
        session.add(cfg)
        await session.commit()
        cr_id, lead_id = cr.id, lead.id

    msg = CallEnded(
        call_record_id=cr_id,
        hangup_cause=HangupCause.NORMAL_CLEARING,
        ended_at=datetime.now(UTC),
    )

    async with httpx.AsyncClient(transport=httpx.MockTransport(_ok_handler)) as http:
        await _handle_raw(
            msg.model_dump_json(),
            redis=redis_client,
            sessionmaker=sessionmaker_,
            http=http,
            provider=MockProvider(),
            settings=_settings(),
        )

    # Verify all three tables
    async with sessionmaker_() as session:
        cs = (
            await session.execute(select(CallSummary).where(CallSummary.call_record_id == cr_id))
        ).scalar_one()
        log = (await session.execute(select(CallbackLog))).scalar_one()
        lead2 = await session.get(Lead, lead_id)

    assert cs.goal_achieved is True
    assert cs.goal_type == "appointment"
    assert log.status == CallbackStatus.SUCCESS
    assert lead2.status == LeadStatus.COMPLETED


@pytest.mark.asyncio(loop_scope="session")
async def test_unsupported_schema_version_lands_in_dlq(  # type: ignore[no-untyped-def]
    sessionmaker_, redis_client
) -> None:
    bad_raw = (
        '{"schema_version": 99, "call_record_id": 1, '
        '"hangup_cause": "normal_clearing", '
        '"ended_at": "2026-05-06T12:00:00Z"}'
    )

    async with httpx.AsyncClient(transport=httpx.MockTransport(_ok_handler)) as http:
        await _handle_raw(
            bad_raw,
            redis=redis_client,
            sessionmaker=sessionmaker_,
            http=http,
            provider=MockProvider(),
            settings=_settings(),
        )

    assert await redis_client.llen(DLQ) == 1
    raw = await redis_client.lrange(DLQ, 0, 0)
    assert raw[0] == bad_raw

    # Cleanup
    await redis_client.delete(DLQ)


@pytest.mark.asyncio(loop_scope="session")
async def test_malformed_json_lands_in_dlq(sessionmaker_, redis_client) -> None:  # type: ignore[no-untyped-def]
    async with httpx.AsyncClient(transport=httpx.MockTransport(_ok_handler)) as http:
        await _handle_raw(
            "garbage",
            redis=redis_client,
            sessionmaker=sessionmaker_,
            http=http,
            provider=MockProvider(),
            settings=_settings(),
        )
    assert await redis_client.llen(DLQ) == 1
    await redis_client.delete(DLQ)
