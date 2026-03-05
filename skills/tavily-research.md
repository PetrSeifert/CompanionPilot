---
id: tavily-research
title: Tavily Research
description: Comprehensive research grounded in web data with explicit citations. Use when you need multi-source synthesis — comparisons, current events, market analysis, detailed reports.
tags: [tavily, research, strategy]
---

# Research

Conduct comprehensive research on any topic with automatic source gathering, analysis, and response generation with citations.

## Quick Start

### Using the `web_research` tool (preferred)

```json
// Quick research
{
  "tool_name": "web_research",
  "arguments": {
    "input": "quantum computing trends",
    "reasoning": "Research quantum computing developments"
  }
}

// Comprehensive analysis with pro model
{
  "tool_name": "web_research",
  "arguments": {
    "input": "AI agents comparison: LangChain vs CrewAI vs AutoGen",
    "model": "pro",
    "reasoning": "Deep comparison of AI agent frameworks"
  }
}
```

Research can take 30-120 seconds. The tool handles polling automatically.

### Alternative: Using the script

```bash
./skills/scripts/tavily-research.sh '{"input": "quantum computing trends"}'
./skills/scripts/tavily-research.sh '{"input": "AI agents comparison", "model": "pro"}'
```

### Alternative: Using curl

```bash
# Start research task
curl -s --request POST \
  --url https://api.tavily.com/research \
  --header "Authorization: Bearer $TAVILY_API_KEY" \
  --header 'Content-Type: application/json' \
  --data '{"input": "quantum computing trends", "model": "pro"}'

# Poll for results (using request_id from response)
curl -s --request GET \
  --url "https://api.tavily.com/research/<request_id>" \
  --header "Authorization: Bearer $TAVILY_API_KEY"
```

## API Reference

### Endpoint

```
POST https://api.tavily.com/research
GET  https://api.tavily.com/research/<request_id>
```

### Request Body

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `input` | string | Required | Research topic or question |
| `model` | string | `"auto"` | `"mini"`, `"pro"`, `"auto"` |

### Model Selection

| Model | Use Case | Speed |
|-------|----------|-------|
| `mini` | Single topic, targeted research | ~30s |
| `pro` | Comprehensive multi-angle analysis | ~60-120s |
| `auto` | API chooses based on complexity | Varies |

**Rule of thumb:** "what does X do?" -> mini. "X vs Y vs Z" or "best way to..." -> pro.

### Response Format

```json
{
  "status": "completed",
  "content": "# Research Report\n\n...",
  "sources": [
    { "url": "https://example.com", "title": "Source Title", "citation": "[1]" }
  ],
  "response_time": 45.2
}
```

## Multi-Step Research Strategy

When using `web_search` calls instead of the research API:

1. **Broad scan** — general query, `max_results: 8-10`
2. **Targeted follow-up** — refine using terms from step 1, `search_depth: "advanced"`
3. **Verification** — search with different keywords, `include_domains` for primary sources

## Tips

- **Be specific in prompts** — include known details: target market, competitors, constraints
- **Share prior context** — include what you already know to avoid repetition
- **Use `topic: "news"` in web_search** for evolving stories
- **Use `topic: "finance"` in web_search** for market/financial data
- **Cross-reference claims** across sources; prefer primary sources over aggregators
