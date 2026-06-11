"""Inject a synthetic CallEnded into engine:worker:call-ended for stage-3 dev verification.

WARNING: dev/test only. Not exposed via [project.scripts]; invoke with
``python -m scripts.fake_call_end``.

Optionally seeds a lead + call_record (with a mock transcript) and flips
lead.status to 'calling' so the worker's lead_state row-guard passes.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from isales_common.enums import CallStatus, HangupCause, LeadStatus
from isales_common.models.call_record import CallRecord
from isales_common.models.campaign import Campaign
from isales_common.models.lead import Lead
from isales_common.schemas.messages import CallEnded
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from isales_worker.callend import CALL_ENDED_QUEUE
from isales_worker.redis_client import get_redis

logger = logging.getLogger("fake_call_end")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inject a synthetic CallEnded (dev only).")
    p.add_argument("--db-url", required=True)
    p.add_argument("--redis-url", required=True)
    p.add_argument("--campaign-id", type=int, required=True)
    p.add_argument("--lead-id", type=int, default=None,
                   help="reuse an existing lead; otherwise a new one is created")
    p.add_argument("--hangup-cause", required=True,
                   choices=[c.value for c in HangupCause])
    p.add_argument("--goal-achieved", default="false",
                   choices=("true", "false"))
    p.add_argument("--goal-type", default=None)
    p.add_argument("--do-not-call", action="store_true",
                   help="add a do_not_call_marked transcript event")
    return p.parse_args()


def _build_transcript(
    *,
    goal_achieved: bool,
    goal_type: str | None,
    do_not_call: bool,
    hangup_cause: str,
) -> list[dict[str, Any]]:
    transcript: list[dict[str, Any]] = [
        {"type": "greeting", "ts": 0, "text": "您好，这里是 iSales 测试。"},
        {"type": "user_speech", "ts": 1500, "text": "你好。"},
        {"type": "ai_reply", "ts": 3000, "text": "请问您方便沟通吗？", "turn_id": 1,
         "goal_achieved": False, "goal_type": "", "extracted": {}, "is_wrap_up": False},
        {"type": "user_speech", "ts": 6000, "text": "可以。"},
        {
            "type": "ai_reply",
            "ts": 8000,
            "turn_id": 2,
            "text": "好的，请问您有兴趣预约一次详细介绍吗？",
            "goal_achieved": goal_achieved,
            "goal_type": goal_type or "",
            "extracted": {"intent": "interested" if goal_achieved else "neutral"},
            "is_wrap_up": False,
        },
    ]
    if goal_achieved:
        transcript.append({
            "type": "goal_achieved", "ts": 8500, "goal_type": goal_type or "",
            "extracted": {"intent": "interested"},
        })
    if do_not_call:
        transcript.append({"type": "do_not_call_marked", "ts": 9500, "reason": "keyword"})
    transcript.append({"type": "hangup", "ts": 10000, "cause": hangup_cause})
    return transcript


async def _run() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    goal_achieved = args.goal_achieved == "true"

    engine = create_async_engine(args.db_url, future=True)
    sm = async_sessionmaker(engine, expire_on_commit=False)
    redis = get_redis(args.redis_url)

    try:
        async with sm() as session:
            campaign = await session.get(Campaign, args.campaign_id)
            if campaign is None:
                raise SystemExit(f"campaign_id={args.campaign_id} not found in DB")

            lead_id = args.lead_id
            if lead_id is None:
                lead = Lead(
                    campaign_id=campaign.id,
                    phone="13800009999",
                    name="fake-lead",
                    status=LeadStatus.NEW,
                    next_call_at=datetime.now(UTC) - timedelta(seconds=10),
                )
                session.add(lead)
                await session.flush()
                lead_id = lead.id
                logger.info("seeded_lead lead_id=%d", lead_id)
            else:
                lead = await session.get(Lead, lead_id)
                if lead is None:
                    raise SystemExit(f"lead_id={lead_id} not found in DB")

            transcript = _build_transcript(
                goal_achieved=goal_achieved,
                goal_type=args.goal_type,
                do_not_call=args.do_not_call,
                hangup_cause=args.hangup_cause,
            )
            now = datetime.now(UTC)
            cr = CallRecord(
                lead_id=lead_id,
                campaign_id=campaign.id,
                caller_id="13900000001",
                status=CallStatus.END,
                started_at=now - timedelta(seconds=10),
                ended_at=now,
                duration=10,
                transcript=transcript,
                prompt_versions={},
            )
            session.add(cr)
            await session.flush()
            cr_id = cr.id

            # Flip lead.status to 'calling' so worker's row guard passes
            await session.execute(
                update(Lead).where(Lead.id == lead_id).values(status=LeadStatus.CALLING)
            )
            await session.commit()
            logger.info("seeded_call_record call_record_id=%d", cr_id)

        # Verify lead.status now (sanity)
        async with sm() as session:
            cur = await session.execute(select(Lead.status).where(Lead.id == lead_id))
            cur_status = cur.scalar_one()
            logger.info("lead_status_pre_publish lead_id=%d status=%s", lead_id, cur_status)

        msg = CallEnded(
            call_record_id=cr_id,
            hangup_cause=HangupCause(args.hangup_cause),
            ended_at=now,
        )
        await redis.lpush(CALL_ENDED_QUEUE, msg.model_dump_json())
        logger.info(
            "published call_record_id=%d message_id=%s queue=%s",
            cr_id, msg.message_id, CALL_ENDED_QUEUE,
        )
    finally:
        await redis.close()
        await engine.dispose()


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
