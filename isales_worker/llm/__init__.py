"""LLM provider abstraction. v1 ships only the mock provider."""

from isales_worker.llm.base import LLMProvider, SummaryResult
from isales_worker.llm.mock import MockProvider

__all__ = ["LLMProvider", "MockProvider", "SummaryResult", "build_provider"]


def build_provider(name: str) -> LLMProvider:
    if name == "mock":
        return MockProvider()
    raise NotImplementedError(
        f"LLM provider '{name}' not implemented in stage 3B; "
        "real providers (openai/alibaba/...) land in stage 5"
    )
