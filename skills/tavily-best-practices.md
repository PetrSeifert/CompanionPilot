---
id: tavily-best-practices
title: Tavily Best Practices
description: General best practices for web search query optimization, parameter selection, and result handling.
tags: [tavily, best-practices, web]
---

# Best practices

## Query optimization

- **Be specific**: `"Python 3.12 match statement syntax"` > `"how to use match in Python"`
- **Include context**: add version numbers, years, or product names
- **Avoid filler**: drop words like "how to", "what is", "please find"
- **Use natural keywords**: the query is passed to a search engine, not a chat model

## Parameter selection

- Start with defaults (`basic` depth, `general` topic, 5 results)
- Escalate to `advanced` depth only when basic returns insufficient detail
- Use `topic: "news"` only for current events — it filters out non-news sources
- Apply `time_range` when freshness matters; omit it for evergreen topics
- Use domain filters sparingly — over-restricting can return zero results

## Result handling

- The summary field gives a quick answer; individual results provide sources
- Always cite URLs when presenting search findings to the user
- Content snippets are truncated at 300 chars — treat them as previews
- If results look sparse, retry with broader query or fewer domain restrictions

## Common patterns

| Goal | Approach |
|---|---|
| Quick fact check | `basic`, 3 results, specific query |
| In-depth research | `advanced`, 10+ results, then follow-up searches |
| Latest news | `topic: "news"`, `time_range: "day"` or `"week"` |
| Official docs | `include_domains` with doc sites, `advanced` depth |
| Avoid noise | `exclude_domains` with social/forum sites |
