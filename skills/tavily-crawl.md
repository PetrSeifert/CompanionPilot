---
id: tavily-crawl
title: Tavily Site Crawling
description: Reference for crawling multiple pages from a site using Tavily search.
tags: [tavily, crawl, site]
---

# Site crawling

Tavily offers a crawl API for systematically scanning multiple pages on a site. CompanionPilot does not yet expose a dedicated crawl tool.

## Workaround with multiple web_search calls

Simulate crawling by issuing several searches scoped to one domain:

1. Use `include_domains` to lock to the target site
2. Vary the query to cover different sections or topics
3. Chain with `execute_program` for ordered execution

```json
{
  "tool_name": "execute_program",
  "arguments": {
    "steps": [
      { "step_id": "overview", "tool_name": "web_search", "arguments": { "query": "site overview features", "include_domains": ["example.com"], "max_results": 5 } },
      { "step_id": "api", "tool_name": "web_search", "arguments": { "query": "API reference endpoints", "include_domains": ["example.com"], "max_results": 5 } },
      { "step_id": "pricing", "tool_name": "web_search", "arguments": { "query": "pricing plans", "include_domains": ["example.com"], "max_results": 3 } }
    ]
  }
}
```

## Limitations

- Cannot follow internal links or discover pages automatically
- Each search returns independent results — no deduplication across steps
- Coverage depends on Tavily's index of the domain
