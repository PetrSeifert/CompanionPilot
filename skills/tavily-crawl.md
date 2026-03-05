---
id: tavily-crawl
title: Tavily Crawl & Map
description: Crawl websites and save pages as markdown, or map site structure to discover URLs. Use when you need to download documentation, knowledge bases, or web content.
tags: [tavily, crawl, map, site]
---

# Crawl & Map

Crawl websites to extract content from multiple pages, or map site structure to discover URLs.

## Quick Start

### Using the `web_crawl` tool (preferred)

```json
// Basic crawl
{
  "tool_name": "web_crawl",
  "arguments": {
    "url": "https://docs.example.com",
    "reasoning": "Crawl documentation site"
  }
}

// Deeper crawl with limits
{
  "tool_name": "web_crawl",
  "arguments": {
    "url": "https://docs.example.com",
    "max_depth": 2,
    "limit": 50,
    "reasoning": "Deep crawl of docs"
  }
}

// Focused crawl with semantic instructions
{
  "tool_name": "web_crawl",
  "arguments": {
    "url": "https://docs.example.com",
    "instructions": "Find API documentation",
    "reasoning": "Crawl for API docs"
  }
}

// Path-filtered crawl
{
  "tool_name": "web_crawl",
  "arguments": {
    "url": "https://example.com",
    "max_depth": 2,
    "select_paths": ["/docs/.*", "/api/.*"],
    "exclude_paths": ["/blog/.*"],
    "reasoning": "Crawl only docs and API sections"
  }
}
```

### Alternative: Using the script

```bash
./skills/scripts/tavily-crawl.sh '{"url": "https://docs.example.com"}'
./skills/scripts/tavily-crawl.sh '{"url": "https://docs.example.com", "max_depth": 2, "limit": 50}'
```

### Alternative: Using curl

```bash
# Crawl
curl -s --request POST \
  --url https://api.tavily.com/crawl \
  --header "Authorization: Bearer $TAVILY_API_KEY" \
  --header 'Content-Type: application/json' \
  --data '{"url": "https://docs.example.com", "max_depth": 2, "limit": 20}'

# Map (URL discovery only — faster than crawl)
curl -s --request POST \
  --url https://api.tavily.com/map \
  --header "Authorization: Bearer $TAVILY_API_KEY" \
  --header 'Content-Type: application/json' \
  --data '{"url": "https://docs.example.com", "max_depth": 2}'
```

## Crawl API Reference

### Endpoint

```
POST https://api.tavily.com/crawl
```

### Request Body

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `url` | string | Required | Root URL to begin crawling |
| `max_depth` | integer | 1 | Levels deep to crawl (1-5) |
| `max_breadth` | integer | 20 | Links per page (1-500) |
| `limit` | integer | 50 | Total pages cap |
| `instructions` | string | null | Natural language guidance for focus |
| `chunks_per_source` | integer | 3 | Chunks per page (1-5, requires instructions) |
| `extract_depth` | string | `"basic"` | `"basic"` or `"advanced"` |
| `format` | string | `"markdown"` | `"markdown"` or `"text"` |
| `select_paths` | array | null | Regex patterns to include |
| `exclude_paths` | array | null | Regex patterns to exclude |
| `select_domains` | array | null | Regex for domains to include |
| `exclude_domains` | array | null | Regex for domains to exclude |
| `allow_external` | boolean | true | Include external domain links |
| `timeout` | float | 150 | Max wait (10-150 seconds) |

### Crawl Response

```json
{
  "base_url": "https://docs.example.com",
  "results": [
    { "url": "https://docs.example.com/page", "raw_content": "# Page Title\n\nContent..." }
  ],
  "response_time": 12.5
}
```

## Map API Reference

### Endpoint

```
POST https://api.tavily.com/map
```

Same navigation params as crawl (`max_depth`, `max_breadth`, `limit`, path/domain filters).
Returns URLs only (no content) — faster than crawl.

### Map Response

```json
{
  "base_url": "https://docs.example.com",
  "results": ["https://docs.example.com/api/auth", "https://docs.example.com/guides/quickstart"],
  "response_time": 3.1
}
```

## Depth vs Performance

| Depth | Typical Pages | Time |
|-------|---------------|------|
| 1 | 10-50 | Seconds |
| 2 | 50-500 | Minutes |
| 3 | 500-5000 | Many minutes |

## Tips

- **Always use `instructions` + `chunks_per_source` for agentic workflows** — prevents context explosion
- **Use Map first** to understand site structure before full crawl
- **Start conservative** (`max_depth=1`, `limit=20`) and scale up
- **Use path patterns** to focus on relevant sections
- **Always set a `limit`** to prevent runaway crawls
