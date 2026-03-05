---
id: tavily-extract
title: Tavily Extract
description: Extract content from specific URLs using Tavily's extraction API. Returns clean markdown/text from web pages. Use when you have specific URLs and need their content.
tags: [tavily, extract, content]
---

# Extract

Extract clean content from specific URLs. Ideal when you know which pages you want content from.

## Quick Start

### Using the `web_extract` tool (preferred)

```json
// Single URL
{
  "tool_name": "web_extract",
  "arguments": {
    "urls": ["https://example.com/article"],
    "reasoning": "Extract article content"
  }
}

// Multiple URLs with query focus
{
  "tool_name": "web_extract",
  "arguments": {
    "urls": ["https://example.com/docs", "https://example.com/api"],
    "query": "authentication API",
    "reasoning": "Extract auth documentation"
  }
}

// Advanced extraction for JS-heavy pages
{
  "tool_name": "web_extract",
  "arguments": {
    "urls": ["https://app.example.com"],
    "extract_depth": "advanced",
    "reasoning": "Extract JS-rendered page content"
  }
}
```

### Alternative: Using the script

```bash
./skills/scripts/tavily-extract.sh '{"urls": ["https://example.com/article"]}'
./skills/scripts/tavily-extract.sh '{"urls": ["https://example.com/docs"], "query": "authentication API", "chunks_per_source": 3}'
```

### Alternative: Using curl

```bash
curl -s --request POST \
  --url https://api.tavily.com/extract \
  --header "Authorization: Bearer $TAVILY_API_KEY" \
  --header 'Content-Type: application/json' \
  --data '{
    "urls": ["https://example.com/article", "https://example.com/docs"],
    "query": "API authentication",
    "chunks_per_source": 3
  }'
```

## API Reference

### Endpoint

```
POST https://api.tavily.com/extract
```

### Request Body

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `urls` | array | Required | URLs to extract (max 20) |
| `query` | string | null | Reranks chunks by relevance |
| `chunks_per_source` | integer | 3 | Chunks per URL (1-5, requires query) |
| `extract_depth` | string | `"basic"` | `"basic"` or `"advanced"` (for JS pages) |
| `format` | string | `"markdown"` | `"markdown"` or `"text"` |
| `include_images` | boolean | false | Include image URLs |
| `timeout` | float | varies | Max wait (1-60 seconds) |

### Response Format

```json
{
  "results": [
    {
      "url": "https://example.com/article",
      "raw_content": "# Article Title\n\nContent..."
    }
  ],
  "failed_results": [
    { "url": "https://example.com/broken", "error": "Timeout" }
  ],
  "response_time": 2.3
}
```

## Extract Depth

| Depth | When to Use |
|-------|-------------|
| `basic` | Simple text extraction, faster |
| `advanced` | Dynamic/JS-rendered pages, tables, structured data |

**Fallback strategy:** Try `basic` first. If content is missing, retry with `advanced`.

## Tips

- **Max 20 URLs per request** — batch larger lists
- **Use `query` + `chunks_per_source`** to get only relevant content
- **Try `basic` first**, fall back to `advanced` if content is missing
- **Set longer `timeout`** for slow pages (up to 60s)
- **Check `failed_results`** for URLs that couldn't be extracted
