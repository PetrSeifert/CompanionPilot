---
id: tavily-best-practices
title: Tavily Best Practices
description: Build production-ready Tavily integrations with best practices for web search, content extraction, crawling, and research workflows.
tags: [tavily, best-practices, web]
---

# Tavily Best Practices

Tavily is a search API designed for LLMs, enabling AI applications to access real-time web data.

## Choosing the Right Method

| Need | Method | Tool |
|------|--------|------|
| Web search results | Search | `web_search` tool |
| Content from specific URLs | Extract | `web_extract` tool |
| Content from entire site | Crawl | `web_crawl` tool |
| URL discovery from site | Map | `curl` to `api.tavily.com/map` |
| End-to-end research with AI synthesis | Research | `web_research` tool |

## Query Optimization

- **Keep queries under 400 characters** — think search query, not prompt
- **Be specific**: `"Python 3.12 match statement syntax"` > `"how to use match in Python"`
- **Include context**: add version numbers, years, or product names
- **Avoid filler**: drop words like "how to", "what is", "please find"
- **Break complex queries into sub-queries** — better results than one massive query

## Search Depth Selection

| Depth | Latency | Relevance | Use Case |
|-------|---------|-----------|----------|
| `ultra-fast` | Lowest | Lower | Real-time chat, simple fact lookups |
| `fast` | Low | Good | Need chunks but latency matters |
| `basic` | Medium | High | General-purpose, balanced |
| `advanced` | Higher | Highest | Complex/technical, precision matters |

## Parameter Selection

- Start with defaults (`basic` depth, `general` topic, 5 results)
- Use `topic: "news"` only for current events — it filters out non-news sources
- Use `topic: "finance"` for market/financial data
- Apply `time_range` when freshness matters; omit for evergreen topics
- Use domain filters sparingly — over-restricting can return zero results

## Result Handling

- Always cite URLs when presenting search findings
- Content snippets are truncated — treat them as previews
- Filter by `score` (0-1) to get highest relevance results
- If results are sparse, retry with broader query or fewer domain restrictions

## Common Patterns

| Goal | Approach |
|------|----------|
| Quick fact check | `web_search` with `fast` depth, 3 results |
| In-depth research | `web_search` with `advanced`, 10+ results, then follow-up |
| Latest news | `web_search` with `topic: "news"`, `time_range: "day"` |
| Financial data | `web_search` with `topic: "finance"` |
| Official docs | `web_search` with `include_domains`, `advanced` depth |
| Avoid noise | `web_search` with `exclude_domains` for social/forum sites |
| Extract specific page | `web_extract` tool with URL + `query` for targeted chunks |
| Download docs site | `web_crawl` tool with `instructions` + `chunks_per_source` |
| Discover site URLs | `curl` to Map API, then extract from discovered URLs |

## Multi-Step Research Pattern

```json
{
  "tool_name": "execute_program",
  "arguments": {
    "steps": [
      { "step_id": "broad", "tool_name": "web_search", "arguments": { "query": "topic overview", "max_results": 8 } },
      { "step_id": "deep", "tool_name": "web_search", "arguments": { "query": "topic specific detail", "search_depth": "advanced", "max_results": 5 } },
      { "step_id": "verify", "tool_name": "web_search", "arguments": { "query": "topic counter evidence", "include_domains": ["edu", "gov"], "max_results": 3 } }
    ]
  }
}
```

## Scripts Reference

All scripts require `TAVILY_API_KEY` environment variable.

| Script | Usage |
|--------|-------|
| `./skills/scripts/tavily-search.sh` | `'{"query": "...", "max_results": 10}'` |
| `./skills/scripts/tavily-extract.sh` | `'{"urls": ["..."], "query": "..."}'` |
| `./skills/scripts/tavily-crawl.sh` | `'{"url": "...", "max_depth": 2}'` |
| `./skills/scripts/tavily-research.sh` | `'{"input": "...", "model": "pro"}'` |
