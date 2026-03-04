---
id: tavily-research
title: Tavily Multi-Step Research
description: Chain multiple web_search calls for deep research with verification and synthesis.
tags: [tavily, research, strategy]
---

# Multi-step research strategy

When a question requires depth, use multiple `web_search` calls in sequence.

## Broad-then-narrow pattern

1. **Broad scan** — `web_search` with a general query, `max_results: 8-10`
2. **Targeted follow-up** — refine the query using terms from step 1, `search_depth: "advanced"`
3. **Verification** — search for contradicting or confirming evidence with different keywords

## Using execute_program for chained searches

```json
{
  "tool_name": "execute_program",
  "arguments": {
    "steps": [
      { "step_id": "broad", "tool_name": "web_search", "arguments": { "query": "topic overview", "max_results": 8 } },
      { "step_id": "deep", "tool_name": "web_search", "arguments": { "query": "topic specific detail", "search_depth": "advanced", "max_results": 5 } }
    ]
  }
}
```

## Verification techniques

- Search the same fact with different phrasing
- Use `include_domains` to check primary sources (official docs, .gov, .edu)
- Use `topic: "news"` to find the latest reporting on evolving stories

## Synthesis

After gathering results, cross-reference claims across sources. Flag conflicting information and prefer primary sources over aggregators.
