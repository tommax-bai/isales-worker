"""LLM provider ABC for summarize_call.

Spec: goal-achievement § Scenario "worker 不重复判定" — worker MUST NOT
      trigger an independent goal-achieved LLM call. The role LLM in the
      engine already wrote ``goal_achieved`` / ``goal_type`` / ``extracted``
      into the last bot_speech transcript event; summarize_call only
      generates the summary text and copies those fields verbatim.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SummaryResult:
    summary_text: str
    extracted_fields: dict[str, Any] = field(default_factory=dict)
    goal_achieved: bool = False
    goal_type: str | None = None


class LLMProvider(ABC):
    @abstractmethod
    async def summarize(
        self,
        transcript: list[dict[str, Any]],
        extraction_fields: list[Any],
    ) -> SummaryResult:
        """Produce a summary + carry the structured markers from the
        engine's last role_llm output (which lives in the transcript)."""

    @abstractmethod
    async def extract(
        self,
        transcript_snapshot: list[dict[str, Any]],
        extractor_prompt: str,
    ) -> dict[str, Any]:
        """Run the post-call extractor LLM over a call's transcript.

        ``transcript_snapshot`` is the engine-pushed dialog history
        (``[{role, text, ts_ms}]``); ``extractor_prompt`` is the campaign's
        extractor system prompt (with a ``{{transcript}}`` placeholder rendered
        by the caller, or rendered here). Returns the structured ``extracted``
        dict written to ``call_record.extracted``. Raises on hard failure so the
        consumer marks ``extract_status='failed'`` (pipeline-stream-and-referee
        § post-call extractor).
        """
