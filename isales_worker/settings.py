"""Process-level settings loaded from environment variables."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ISALES_", case_sensitive=False)

    database_url: str = Field(alias="ISALES_DATABASE_URL")
    redis_url: str = Field(alias="ISALES_REDIS_URL")
    fernet_key: str = Field(alias="ISALES_FERNET_KEY")
    worker_llm_provider: str = Field(default="mock", alias="ISALES_WORKER_LLM_PROVIDER")
    webhook_default_timeout_seconds: float = Field(
        default=10.0, alias="ISALES_WEBHOOK_DEFAULT_TIMEOUT_SECONDS"
    )
    worker_retry_tick_interval: int = Field(
        default=60, alias="ISALES_WORKER_RETRY_TICK_INTERVAL"
    )
    worker_retry_batch_size: int = Field(default=50, alias="ISALES_WORKER_RETRY_BATCH_SIZE")
    worker_metrics_tick_interval: int = Field(
        default=60, alias="ISALES_WORKER_METRICS_TICK_INTERVAL"
    )


def load_settings() -> Settings:
    return Settings()
