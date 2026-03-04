---
id: tavily-search
title: Tavily Web Search
description: Optimize web_search tool usage with query crafting, parameter selection, and domain filtering.
tags: [tavily, search, web]
---

# web_search usage

Use `web_search` for real-time factual queries. Craft concise, keyword-rich queries.

## Parameters

| Param | Default | When to use |
|---|---|---|
| `query` | (required) | Keywords, not full sentences |
| `max_results` | 5 | Raise to 10-15 for broad topics, lower to 2-3 for precise lookups |
| `search_depth` | basic | Use `advanced` for complex/technical topics needing deeper results |
| `topic` | general | Use `news` for current events, press releases, breaking stories |
| `time_range` | (none) | `day` for breaking news, `week`/`month` for recent developments |
| `include_domains` | (none) | Lock to trusted sources, e.g. `["docs.python.org","stackoverflow.com"]` |
| `exclude_domains` | (none) | Filter out low-quality or paywalled sites |

## Query tips

- Lead with the most specific noun: `"Rust async trait 2024"` not `"how do async traits work in Rust"`
- Add version/year when relevant: `"React 19 server components"`
- For comparisons, name both: `"PostgreSQL vs MySQL JSON performance"`
- Use quotes inside the query for exact phrases: `"segmentation fault" mmap`

## Domain filtering examples

- Official docs only: `include_domains: ["docs.rs","doc.rust-lang.org"]`
- Exclude social media: `exclude_domains: ["reddit.com","twitter.com","x.com"]`
