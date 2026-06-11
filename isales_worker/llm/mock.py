"""Mock LLM provider — concatenates transcript text, no real LLM call.

Used for stage 3 verification: the real role/judge/polish LLMs land in
engine stage 4-5. The mock follows the same contract (transcript-driven
summary + carry-over of structured markers).
"""

from __future__ import annotations

from typing import Any

from isales_worker.llm.base import LLMProvider, SummaryResult

_SUMMARY_LIMIT = 200


def _last_role_markers(transcript: list[dict[str, Any]]) -> tuple[bool, str | None, dict[str, Any]]:
    """Pull goal_achieved / goal_type / extracted from the engine's transcript.

    Goal achievement is a *latch*: once a closing turn fires it (an ``ai_reply``
    with ``goal_achieved=True`` or a standalone ``goal_achieved`` milestone event),
    the call achieved its goal regardless of the later WRAPPING_UP replies — which
    the engine writes as ``ai_reply`` with ``goal_achieved=False`` (run_loop
    ``_gated_wrap_up_turn``). So we scan the whole transcript and keep the last
    achieved marker, rather than trusting the final ``ai_reply``.

    Markers ride on ``ai_reply`` events + a standalone ``goal_achieved`` event
    (transcript spec § 事件类型枚举; goal-achievement spec § "worker 从 ai_reply
    事件读取 goal 标记"). The pre-canonical ``bot_speech`` name (impl-worker,
    2026-05; canonicalised to ``ai_reply`` by fix-transcript-schema-drift) is gone
    — engine never writes it, so we MUST NOT read it. Read miss → fail-safe to
    not-achieved rather than raising.
    """

    achieved: tuple[str | None, dict[str, Any]] | None = None
    last_extracted: dict[str, Any] = {}
    for evt in transcript:
        etype = evt.get("type")
        if etype == "goal_achieved":
            achieved = (evt.get("goal_type") or None, dict(evt.get("extracted") or {}))
        elif etype == "ai_reply":
            if "extracted" in evt:
                last_extracted = dict(evt.get("extracted") or {})
            if evt.get("goal_achieved"):
                achieved = (evt.get("goal_type") or None, dict(evt.get("extracted") or {}))
    if achieved is not None:
        return True, achieved[0], achieved[1]
    return False, None, last_extracted


def _stitch_text(transcript: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for evt in transcript:
        kind = evt.get("type")
        text = evt.get("text")
        if not text:
            continue
        if kind == "user_speech":
            chunks.append(f"用户：{text}")
        elif kind == "ai_reply":
            # AI side rides on ``ai_reply`` events (not the gone ``bot_speech``);
            # reading the old name dropped every bot turn from the summary text.
            chunks.append(f"机器人：{text}")
        elif kind == "greeting":
            chunks.append(f"开场：{text}")
    out = " / ".join(chunks)
    if len(out) > _SUMMARY_LIMIT:
        out = out[: _SUMMARY_LIMIT - 1] + "…"
    return out


class MockProvider(LLMProvider):
    async def summarize(
        self,
        transcript: list[dict[str, Any]],
        extraction_fields: list[Any],
    ) -> SummaryResult:
        goal_achieved, goal_type, extracted = _last_role_markers(transcript)
        # Filter extracted by campaign-declared extraction_fields (best-effort;
        # extraction_fields shape is JSONB list of {name, ...} per data-model).
        if extraction_fields:
            allowed = {
                f["name"] for f in extraction_fields
                if isinstance(f, dict) and "name" in f
            }
            if allowed:
                extracted = {k: v for k, v in extracted.items() if k in allowed}
        return SummaryResult(
            summary_text=_stitch_text(transcript),
            extracted_fields=extracted,
            goal_achieved=goal_achieved,
            goal_type=goal_type,
        )

    async def extract(
        self,
        transcript_snapshot: list[dict[str, Any]],
        extractor_prompt: str,
    ) -> dict[str, Any]:
        """Deterministic stub extraction: pull the first user utterance as the
        customer's intent and count the turns. Real LLM extraction lands with
        the real provider (stage 5)."""
        first_user = next(
            (
                t.get("text", "")
                for t in transcript_snapshot
                if t.get("role") == "user" and t.get("text")
            ),
            "",
        )
        return {
            "customer_intent": first_user,
            "n_turns": len(transcript_snapshot),
        }
