"""isales-worker entrypoint — wires CallEnded consumer + retry + metrics loops."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal

import httpx

from isales_worker.callend import callend_loop
from isales_worker.db import get_engine, get_sessionmaker
from isales_worker.llm import build_provider
from isales_worker.metrics import metrics_loop
from isales_worker.post_call_extractor import extract_loop
from isales_worker.redis_client import get_redis
from isales_worker.retry_loop import retry_loop
from isales_worker.settings import load_settings

logger = logging.getLogger(__name__)


async def _main() -> None:
    settings = load_settings()
    engine = get_engine(settings.database_url)
    sessionmaker = get_sessionmaker(engine)
    redis = get_redis(settings.redis_url)
    http = httpx.AsyncClient()
    provider = build_provider(settings.worker_llm_provider)

    stop_event = asyncio.Event()

    def _request_stop() -> None:
        logger.info("shutdown_signal received")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, _request_stop)

    callend_task = asyncio.create_task(
        callend_loop(
            redis=redis,
            sessionmaker=sessionmaker,
            http=http,
            provider=provider,
            settings=settings,
        ),
        name="callend_loop",
    )
    retry_task = asyncio.create_task(
        retry_loop(sessionmaker, http, settings),
        name="retry_loop",
    )
    metrics_task = asyncio.create_task(
        metrics_loop(sessionmaker, redis, settings),
        name="metrics_loop",
    )
    extract_task = asyncio.create_task(
        extract_loop(redis=redis, sessionmaker=sessionmaker, provider=provider),
        name="extract_loop",
    )

    logger.info("isales_worker_started")
    try:
        await stop_event.wait()
    finally:
        for task in (callend_task, retry_task, metrics_task, extract_task):
            task.cancel()
        for task in (callend_task, retry_task, metrics_task, extract_task):
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("task_shutdown_error name=%s", task.get_name())

        await http.aclose()
        await redis.close()
        await engine.dispose()
        logger.info("isales_worker_stopped")


def run() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    asyncio.run(_main())


if __name__ == "__main__":
    run()
