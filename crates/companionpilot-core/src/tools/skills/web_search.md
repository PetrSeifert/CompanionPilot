# web_search tool skill

Purpose:
- Retrieve external factual information and recent/current updates from the web.

Args schema:
- `{ "query": "string", "max_results": 1..10 }`
- `query` is required and must be non-empty after trimming.
- `max_results` defaults to 5 when omitted.

When to use:
- Unknown factual claims.
- Current events, news, prices, weather, schedules, regulations, recommendations.

Query construction guidance:
- Write concrete, searchable queries with key entities and time anchors.
- For latest/time-sensitive asks, include timeframe hints (for example month/year).
- Avoid vague single-word queries when user intent is specific.

Result strategy:
- Favor precision over volume; increase `max_results` only when breadth is needed.
- Keep results grounded in returned evidence.
