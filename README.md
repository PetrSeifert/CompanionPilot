# CompanionPilot

CompanionPilot is a Rust-first AI orchestrator for Discord chat with long/short-term memory, tool execution, and Railway-ready deployment.

Workspace layout:

- `apps/companionpilot`: executable app crate
- `crates/companionpilot-core`: reusable core library crate

## Implemented v1 baseline

- Discord text event ingestion (`serenity`)
- Orchestrator pipeline with explicit interfaces
- Model abstraction (`OpenRouterProvider`, `MockModelProvider`)
- Memory abstraction (`PostgresMemoryStore`, `InMemoryMemoryStore`)
- Tool runtime with Tavily web search support
- Built-in `current_datetime` tool for UTC date/time grounding
- Built-in `spotify_playing_status` tool for current Spotify playback
- HTTP API for health and chat (`axum`)
- Reply timing telemetry (planner/tools/model/memory stage durations)
- Local dev infra (`docker-compose` with Postgres + Redis)
- Railway deployment entry (`railway.json`)

## Quick start

1. Copy env file:

```bash
cp .env.example .env
```

2. Start local data services:

```bash
docker compose up -d
```

3. Apply migrations:

```bash
psql postgres://postgres:postgres@localhost:5432/companionpilot -f migrations/0001_init.sql
psql postgres://postgres:postgres@localhost:5432/companionpilot -f migrations/0002_chat_messages.sql
psql postgres://postgres:postgres@localhost:5432/companionpilot -f migrations/0003_tool_call_logs.sql
psql postgres://postgres:postgres@localhost:5432/companionpilot -f migrations/0004_planner_decision_logs.sql
```

4. Run the service:

```bash
cargo run -p companionpilot
```

5. Test health endpoint:

```bash
curl http://localhost:8080/health
```

6. Test chat endpoint:

```bash
curl -X POST http://localhost:8080/chat \
  -H "content-type: application/json" \
  -d '{"user_id":"demo","content":"my name is Petr"}'
```

## Discord usage

- Set `DISCORD_TOKEN` in `.env`.
- Mention the bot or DM it.
- CompanionPilot decides tool usage automatically from a unified planner decision.
- For time-sensitive requests, planner can call `current_datetime` before `web_search`.
- For Spotify playback requests, planner can call `spotify_playing_status`.
- Web search is used when the planner determines external facts are required.
- Memory storage is model-driven (no memory command prefix required); corrections can overwrite prior facts.
- Short-term memory is injected from recent channel turns, even when no long-term fact is stored.

## Model provider selection

CompanionPilot supports provider routing through environment variables:

- `MODEL_PROVIDER=auto` (default): use OpenRouter if configured, else mock.
- `MODEL_PROVIDER=openrouter`: force OpenRouter.
- `MODEL_PROVIDER=mock`: force mock provider.

OpenRouter settings:

- `OPENROUTER_API_KEY`
- `OPENROUTER_MODEL` (example: `anthropic/claude-3.5-sonnet`, `meta-llama/llama-3.1-70b-instruct`)
- `OPENROUTER_REFERER` (optional but recommended)
- `OPENROUTER_TITLE` (optional app label)

## Notes

- If `OPENROUTER_API_KEY` is missing (or provider is `mock`), the app uses the mock model provider.
- If `DATABASE_URL` is missing, memory uses in-process storage.
- If `TAVILY_API_KEY` is missing, planner-selected `web_search` calls return a configuration error.
- HTTP endpoints are currently unauthenticated. Add auth before exposing to untrusted users.

## Search diagnostics

To inspect search behavior in logs, set:

```bash
RUST_LOG=companionpilot=debug,companionpilot_core=debug,info
```

Then look for:

- `tool call selected by unified planner` (tool + args selected)
- `tool call completed` (tool finished)
- `tavily web search start` / `tavily web search success` (actual Tavily call path)
- `planner fallback: running without tools and without memory write` (planner failure fallback)
- `reply completed` (per-message timing summary)
- `slow reply detected` / `slow Discord reply detected` (slow-path warnings, threshold 30s)
