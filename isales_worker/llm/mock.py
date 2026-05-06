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
    """Walk transcript backwards; pull goal_achieved / goal_type / extracted
    off the most recent bot_speech event that carries them.

    Engine writes those markers onto ``bot_speech`` events (see goal-achievement
    spec). Older events may not have them at all (e.g. fillers, greeting).
    """

    for evt in reversed(transcript):
        if evt.get("type") != "bot_speech":
            continue
        if "goal_achieved" not in evt and "goal_type" not in evt and "extracted" not in evt:
            continue
        return (
            bool(evt.get("goal_achieved", False)),
            evt.get("goal_type") or None,
            dict(evt.get("extracted") or {}),
        )
    return False, None, {}


def _stitch_text(transcript: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for evt in transcript:
        kind = evt.get("type")
        text = evt.get("text")
        if not text:
            continue
        if kind == "user_speech":
            chunks.append(f"用户：{text}")
        elif kind == "bot_speech":
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
