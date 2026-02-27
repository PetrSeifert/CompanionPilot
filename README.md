# CompanionPilot

CompanionPilot is a Rust-first AI orchestrator for Discord chat with long/short-term memory, tool execution, and Railway-ready deployment.

## Implemented v1 baseline

- Discord text event ingestion (`serenity`)
- Orchestrator pipeline with explicit interfaces
- Model abstraction (`OpenRouterProvider`, `MockModelProvider`)
- Memory abstraction (`PostgresMemoryStore`, `InMemoryMemoryStore`)
- Tool runtime with Tavily web search support
- HTTP API for health, chat, and dashboard data (`axum`)
- Built-in web dashboard at `/app` for users/memory/chats
- Dashboard now includes tool-call history (query, result/error, source)
- Dashboard now includes planner decision history (search/memory decisions)
- Dashboard supports manual memory fact add/update/delete and conversation message deletion/clear
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
cargo run
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

7. Open the dashboard:

```bash
http://localhost:8080/app
```

## Discord usage

- Set `DISCORD_TOKEN` in `.env`.
- Mention the bot or DM it.
- Use `/search <query>` in Discord as a manual search override.
- CompanionPilot can also decide to run web search automatically for time-sensitive/factual questions.
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
- If `TAVILY_API_KEY` is missing, `/search` returns a configuration error.
- Dashboard endpoints are currently unauthenticated. Add auth before exposing to untrusted users.

## Search diagnostics

To inspect search behavior in logs, set:

```bash
RUST_LOG=companionpilot=debug,info
```

Then look for:

- `web search selected` (decision + query + source)
- `web search tool completed` (search finished)
- `tavily web search start` / `tavily web search success` (actual Tavily call path)
- `web search skipped` (why it was not used)
