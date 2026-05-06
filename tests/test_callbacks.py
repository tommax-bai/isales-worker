"""Tests for process_callbacks: trigger / payload / signing / HTTP / status."""

from __future__ import annotations

import hashlib
import hmac
import json

import httpx
import pytest
from isales_common.enums import CallbackStatus
from isales_common.models.call_record import CallRecord
from isales_common.models.call_summary import CallSummary
from isales_common.models.callback_config import CallbackConfig
from isales_common.models.callback_log import CallbackLog
from isales_common.models.campaign import Campaign
from isales_common.models.lead import Lead
from isales_common.utils.crypto import encrypt
from sqlalchemy import select

from isales_worker.callbacks import (
    RenderError,
    evaluate_trigger,
    process_callbacks,
    render_payload,
    sign_request,
)


def test_evaluate_trigger_simple_match() -> None:
    trig = {"and": [
        {"==": [{"var": "goal_achieved"}, True]},
        {"==": [{"var": "goal_type"}, "appointment"]},
    ]}
    ctx = {"goal_achieved": True, "goal_type": "appointment"}
    assert evaluate_trigger(trig, ctx) is True


def test_evaluate_trigger_no_match() -> None:
    trig = {"==": [{"var": "goal_achieved"}, True]}
    assert evaluate_trigger(trig, {"goal_achieved": False}) is False


def test_evaluate_trigger_invalid_returns_false() -> None:
    trig = {"unknown_op": [1]}
    assert evaluate_trigger(trig, {}) is False


def test_render_payload_jinja_tojson() -> None:
    tmpl = '{"lead_id": "{{ lead.id }}", "summary": {{ summary | tojson }}}'
    out = render_payload(tmpl, {"lead": {"id": 7}, "summary": "ok"})
    assert json.loads(out) == {"lead_id": "7", "summary": "ok"}


def test_render_payload_undefined_raises() -> None:
    tmpl = '{"x": "{{ extracted.nonexistent }}"}'
    with pytest.raises(RenderError):
        render_payload(tmpl, {"extracted": {}})


def test_render_payload_security_error_raises() -> None:
    tmpl = "{{ ''.__class__ }}"
    with pytest.raises(RenderError):
        render_payload(tmpl, {})


def test_sign_request_format() -> None:
    headers = sign_request("supersecret", "{}", 1717000000)
    expected = hmac.new(
        b"supersecret", b"1717000000.{}", hashlib.sha256
    ).hexdigest()
    assert headers["X-Isales-Signature"] == f"sha256={expected}"
    assert headers["X-Isales-Timestamp"] == "1717000000"
    assert headers["Content-Type"] == "application/json"


async def _seed_for_dispatch(
    sessionmaker_,  # type: ignore[no-untyped-def]
    *,
    url: str,
    trigger: dict,
    template: str,
    retry_policy: dict | None = None,
    secret: str = "shh",
) -> tuple[int, int, int]:
    async with sessionmaker_() as session:
        camp = Campaign(name="C", retry_intervals=[60, 300, 1800], retry_max_count=3)
        session.add(camp)
        await session.flush()
        lead = Lead(campaign_id=camp.id, phone="13800000001")
        session.add(lead)
        await session.flush()
        cr = CallRecord(lead_id=lead.id, campaign_id=camp.id, duration=10)
        session.add(cr)
        await session.flush()

        cs = CallSummary(
            call_record_id=cr.id,
            summary_text="hi",
            extracted_fields={"intent": "yes"},
            goal_achieved=True,
            goal_type="appointment",
        )
        session.add(cs)

        cfg = CallbackConfig(
            campaign_id=camp.id,
            name="cb",
            trigger=trigger,
            url=url,
            method="POST",
            payload_template=template,
            retry_policy=retry_policy or {"intervals_seconds": [60, 300], "max_attempts": 3},
            signing_secret=encrypt(secret),
            timeout_seconds=5,
            enabled=True,
        )
        session.add(cfg)
        await session.commit()
        return cr.id, cfg.id, lead.id


def _client(handler):  # type: ignore[no-untyped-def]
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.asyncio(loop_scope="session")
async def test_process_callbacks_success_writes_success_log(sessionmaker_) -> None:  # type: ignore[no-untyped-def]
    received: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        received["url"] = str(req.url)
        received["body"] = req.content.decode()
        received["sig"] = req.headers.get("x-isales-signature")
        return httpx.Response(200, json={"ok": True})

    cr_id, cfg_id, lead_id = await _seed_for_dispatch(
        sessionmaker_,
        url="https://hook.test/cb",
        trigger={"==": [{"var": "goal_achieved"}, True]},
        template='{"lead_id": "{{ lead.id }}"}',
    )
    async with _client(handler) as http, sessionmaker_() as session:
        n = await process_callbacks(
            session, http, call_record_id=cr_id,
            default_timeout=5.0, hangup_cause="normal_clearing",
        )
        await session.commit()
    assert n == 1

    async with sessionmaker_() as session:
        log = (await session.execute(select(CallbackLog))).scalar_one()
    assert log.status == CallbackStatus.SUCCESS
    assert log.response_code == 200
    assert log.request_body == f'{{"lead_id": "{lead_id}"}}'
    assert received["sig"].startswith("sha256=")


@pytest.mark.asyncio(loop_scope="session")
async def test_trigger_no_match_writes_no_log(sessionmaker_) -> None:  # type: ignore[no-untyped-def]
    def handler(_req: httpx.Request) -> httpx.Response:
        raise AssertionError("HTTP must not be called when trigger doesn't match")

    cr_id, _, _ = await _seed_for_dispatch(
        sessionmaker_,
        url="https://hook.test/cb",
        trigger={"==": [{"var": "goal_achieved"}, False]},  # ctx has True → no match
        template='{"x": 1}',
    )
    async with _client(handler) as http, sessionmaker_() as session:
        n = await process_callbacks(
            session, http, call_record_id=cr_id,
            default_timeout=5.0, hangup_cause="normal_clearing",
        )
        await session.commit()
    assert n == 1  # one config evaluated, but trigger missed

    async with sessionmaker_() as session:
        logs = (await session.execute(select(CallbackLog))).scalars().all()
    assert len(logs) == 0


@pytest.mark.asyncio(loop_scope="session")
async def test_render_failure_marks_failed_render(sessionmaker_) -> None:  # type: ignore[no-untyped-def]
    cr_id, _, _ = await _seed_for_dispatch(
        sessionmaker_,
        url="https://hook.test/cb",
        trigger={"==": [{"var": "goal_achieved"}, True]},
        template='{"x": "{{ extracted.nonexistent }}"}',
    )

    def handler(_req: httpx.Request) -> httpx.Response:
        raise AssertionError("HTTP must not be called when render fails")

    async with _client(handler) as http, sessionmaker_() as session:
        await process_callbacks(
            session, http, call_record_id=cr_id,
            default_timeout=5.0, hangup_cause="normal_clearing",
        )
        await session.commit()

    async with sessionmaker_() as session:
        log = (await session.execute(select(CallbackLog))).scalar_one()
    assert log.status == CallbackStatus.FAILED_RENDER
    assert "jinja2_render_failed" in (log.error_message or "")


@pytest.mark.asyncio(loop_scope="session")
async def test_http_400_marks_failed_4xx_no_retry(sessionmaker_) -> None:  # type: ignore[no-untyped-def]
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "bad"})

    cr_id, _, _ = await _seed_for_dispatch(
        sessionmaker_,
        url="https://hook.test/cb",
        trigger={"==": [{"var": "goal_achieved"}, True]},
        template='{"x": 1}',
    )
    async with _client(handler) as http, sessionmaker_() as session:
        await process_callbacks(
            session, http, call_record_id=cr_id,
            default_timeout=5.0, hangup_cause="normal_clearing",
        )
        await session.commit()

    async with sessionmaker_() as session:
        log = (await session.execute(select(CallbackLog))).scalar_one()
    assert log.status == CallbackStatus.FAILED_HTTP_4XX
    assert log.next_retry_at is None
    assert log.retry_count == 0


@pytest.mark.asyncio(loop_scope="session")
async def test_http_500_schedules_retry(sessionmaker_) -> None:  # type: ignore[no-untyped-def]
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    cr_id, _, _ = await _seed_for_dispatch(
        sessionmaker_,
        url="https://hook.test/cb",
        trigger={"==": [{"var": "goal_achieved"}, True]},
        template='{"x": 1}',
        retry_policy={"intervals_seconds": [60, 300, 1800], "max_attempts": 3},
    )
    async with _client(handler) as http, sessionmaker_() as session:
        await process_callbacks(
            session, http, call_record_id=cr_id,
            default_timeout=5.0, hangup_cause="normal_clearing",
        )
        await session.commit()

    async with sessionmaker_() as session:
        log = (await session.execute(select(CallbackLog))).scalar_one()
    assert log.status == CallbackStatus.PENDING_RETRY
    assert log.retry_count == 1
    assert log.next_retry_at is not None


@pytest.mark.asyncio(loop_scope="session")
async def test_http_network_error_schedules_retry(sessionmaker_) -> None:  # type: ignore[no-untyped-def]
    def handler(_req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("network down")

    cr_id, _, _ = await _seed_for_dispatch(
        sessionmaker_,
        url="https://hook.test/cb",
        trigger={"==": [{"var": "goal_achieved"}, True]},
        template='{"x": 1}',
    )
    async with _client(handler) as http, sessionmaker_() as session:
        await process_callbacks(
            session, http, call_record_id=cr_id,
            default_timeout=5.0, hangup_cause="normal_clearing",
        )
        await session.commit()

    async with sessionmaker_() as session:
        log = (await session.execute(select(CallbackLog))).scalar_one()
    assert log.status == CallbackStatus.PENDING_RETRY
    assert log.error_message and "http_error" in log.error_message
