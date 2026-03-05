---
id: tavily-search
title: Tavily Web Search
description: Search the web using Tavily's LLM-optimized search API. Returns relevant results with content snippets, scores, and metadata.
tags: [tavily, search, web]
---

# Search

Search the web and get relevant results optimized for LLM consumption.

## Quick Start

### Using the web_search tool (preferred)

The `web_search` tool calls Tavily search directly. No setup needed.

```json
{
  "tool_name": "web_search",
  "arguments": {
    "query": "latest developments in quantum computing",
    "max_results": 10,
    "search_depth": "advanced"
  }
}
```

### Using the script

```bash
./skills/scripts/tavily-search.sh '{"query": "AI news", "time_range": "week", "max_results": 10}'
```

**More examples:**
```bash
# Domain-filtered search
./skills/scripts/tavily-search.sh '{"query": "machine learning", "include_domains": ["arxiv.org", "github.com"], "search_depth": "advanced"}'

# News search
./skills/scripts/tavily-search.sh '{"query": "tech industry layoffs", "topic": "news", "time_range": "day"}'

# Finance search
./skills/scripts/tavily-search.sh '{"query": "NVIDIA earnings Q4", "topic": "finance"}'
```

### Using curl

```bash
curl -s --request POST \
  --url https://api.tavily.com/search \
  --header "Authorization: Bearer $TAVILY_API_KEY" \
  --header 'Content-Type: application/json' \
  --data '{"query": "latest developments in quantum computing", "max_results": 5}'
```

## API Reference

### Endpoint

```
POST https://api.tavily.com/search
```

### Request Body

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `query` | string | Required | Search query (keep under 400 chars) |
| `max_results` | integer | 5 | Maximum results (0-20) |
| `search_depth` | string | `"basic"` | `"ultra-fast"`, `"fast"`, `"basic"`, `"advanced"` |
| `topic` | string | `"general"` | `"general"`, `"news"`, `"finance"` |
| `time_range` | string | null | `"day"`, `"week"`, `"month"`, `"year"` |
| `include_domains` | array | [] | Domains to include (max 300) |
| `exclude_domains` | array | [] | Domains to exclude (max 150) |
| `include_answer` | bool | false | Generate LLM answer summary |
| `include_raw_content` | bool/string | false | Include full page content (`true`, `"markdown"`, `"text"`) |
| `include_images` | bool | false | Include image results |

### Response Format

```json
{
  "query": "latest developments in quantum computing",
  "results": [
    {
      "title": "Page Title",
      "url": "https://example.com/page",
      "content": "Extracted text snippet...",
      "score": 0.85
    }
  ],
  "response_time": 1.2
}
```

## Search Depth

| Depth | Latency | Relevance | Content Type |
|-------|---------|-----------|--------------|
| `ultra-fast` | Lowest | Lower | NLP summary |
| `fast` | Low | Good | Chunks |
| `basic` | Medium | High | NLP summary |
| `advanced` | Higher | Highest | Chunks |

**When to use each:**
- `ultra-fast`: Real-time chat, simple fact lookups
- `fast`: Need chunks but latency matters
- `basic`: General-purpose, balanced
- `advanced`: Precision matters, complex/technical topics

## Tips

- **Keep queries under 400 characters** — think search query, not prompt
- **Break complex queries into sub-queries** — better results than one massive query
- **Use `include_domains`** to focus on trusted sources
- **Use `time_range`** for recent information
- **Filter by `score`** (0-1) to get highest relevance results
