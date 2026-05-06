# isales-worker

Post-call async post-processor for the iSales platform (stage 3B).

Consumes `engine:worker:call-ended` and runs:

1. `summarize_call` — generate `call_summary` (mock LLM in v1; transcript-driven)
2. `process_callbacks` — evaluate JsonLogic triggers, render Jinja2 sandbox
   payloads, sign with HMAC-SHA256, dispatch HTTP, write `callback_log`
3. `update_lead_state` — apply the retry-followup decision matrix and
   write back `lead.status` (with row-level guard `WHERE status='calling'`)

Plus two background tasks:

- **retry_loop** — every `ISALES_WORKER_RETRY_TICK_INTERVAL` seconds, re-sends
  `pending_retry` callback_logs (reuses stored `request_body` so trigger /
  payload semantics stay frozen at the original send moment)
- **metrics_loop** — periodic 7-day rollup written to Redis Hash
  `isales:metrics:7d:{campaign_id}` for v1 dashboard

## Environment

| Variable | Default | Purpose |
|---|---|---|
| `ISALES_DATABASE_URL` | (required) | postgres async URL |
| `ISALES_REDIS_URL` | (required) | redis URL |
| `ISALES_FERNET_KEY` | (required) | symmetric key for `signing_secret`; MUST match isales-api |
| `ISALES_WORKER_LLM_PROVIDER` | `mock` | LLM backend; v1 only `mock` |
| `ISALES_WEBHOOK_DEFAULT_TIMEOUT_SECONDS` | `10` | per-request timeout when callback_config doesn't override |
| `ISALES_WORKER_RETRY_TICK_INTERVAL` | `60` | retry scheduler tick |
| `ISALES_WORKER_RETRY_BATCH_SIZE` | `50` | max pending_retry rows per tick |
| `ISALES_WORKER_METRICS_TICK_INTERVAL` | `60` | metrics rollup tick |
| `TZ` | (deployment) | server timezone |

DB migrations run from isales-common's alembic — worker does not own them.

## Run

    pip install -e '.[dev]'
    isales-worker

## Dev verification

isales-engine isn't built yet (stage 4). Inject a synthetic `CallEnded`
to exercise the full pipeline end-to-end:

    python -m scripts.fake_call_end \
      --db-url postgresql+asyncpg://localhost/isales_dev \
      --redis-url redis://localhost:6379/0 \
      --campaign-id 1 \
      --hangup-cause normal_clearing \
      --goal-achieved true

The script seeds a lead + call_record with a mock transcript, sets
`lead.status='calling'` (so the worker's row guard passes), and `LPUSH`es
the `CallEnded` message. Watch worker logs and inspect the DB:

    select status, summary_text from call_summary order by id desc limit 1;
    select status, retry_count, response_code from callback_log order by id desc;
    select id, status, retry_count, follow_up_count, next_call_at from lead order by id desc limit 1;

## Tests

    pytest

Real Postgres + Redis (skip if unreachable). HTTP path uses
`httpx.MockTransport`.
