# CompanionPilot

CompanionPilot is a Rust-first AI orchestrator for Discord chat with long/short-term memory, tool execution, and Railway-ready deployment.

## Implemented v1 baseline

- Discord text event ingestion (`serenity`)
- Orchestrator pipeline with explicit interfaces
- Model abstraction (`OpenAiProvider`, `MockModelProvider`)
- Memory abstraction (`PostgresMemoryStore`, `InMemoryMemoryStore`)
- Tool runtime with Tavily web search support
- HTTP API for health and local chat testing (`axum`)
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

3. Apply migration:

```bash
psql postgres://postgres:postgres@localhost:5432/companionpilot -f migrations/0001_init.sql
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

## Discord usage

- Set `DISCORD_TOKEN` in `.env`.
- Mention the bot or DM it.
- Use `/search <query>` in Discord to trigger tool execution.

## Notes

- If `OPENAI_API_KEY` is missing, the app falls back to a mock model.
- If `DATABASE_URL` is missing, memory uses in-process storage.
- If `TAVILY_API_KEY` is missing, `/search` returns a configuration error.
