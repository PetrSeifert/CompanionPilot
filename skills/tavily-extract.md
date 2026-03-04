---
id: tavily-extract
title: Tavily Content Extraction
description: Reference for extracting structured content from specific URLs using Tavily.
tags: [tavily, extract, content]
---

# Content extraction

Tavily offers an extract API for pulling structured content from specific URLs. CompanionPilot does not yet expose a dedicated extract tool.

## Workaround with web_search

To approximate extraction from a known domain:

1. Use `include_domains` to restrict results to the target site
2. Set `search_depth: "advanced"` for richer content snippets
3. Use a specific query matching the page content

```json
{
  "tool_name": "web_search",
  "arguments": {
    "query": "installation guide getting started",
    "include_domains": ["docs.example.com"],
    "search_depth": "advanced",
    "max_results": 3
  }
}
```

## When extraction is needed

- Pulling documentation from a known URL
- Summarizing an article the user shared
- Getting structured data from a product page

## Limitations of the workaround

- Cannot target a single exact URL, only a domain
- Content snippets are truncated — may miss full page context
- Results depend on Tavily's index coverage of the domain
