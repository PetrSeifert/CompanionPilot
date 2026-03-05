#!/bin/bash
# Tavily Search API script
# Usage: ./tavily-search.sh '{"query": "your search query", ...}'
# Example: ./tavily-search.sh '{"query": "AI news", "time_range": "week", "max_results": 10}'

set -e

JSON_INPUT="$1"

if [ -z "$JSON_INPUT" ]; then
    echo "Usage: ./tavily-search.sh '<json>'"
    echo ""
    echo "Required:"
    echo "  query: string - Search query (keep under 400 chars)"
    echo ""
    echo "Optional:"
    echo "  search_depth: \"ultra-fast\", \"fast\", \"basic\" (default), \"advanced\""
    echo "  topic: \"general\" (default), \"news\", \"finance\""
    echo "  max_results: 1-20 (default: 5)"
    echo "  time_range: \"day\", \"week\", \"month\", \"year\""
    echo "  include_domains: [\"domain1.com\", \"domain2.com\"] (max 300)"
    echo "  exclude_domains: [\"domain1.com\", \"domain2.com\"] (max 150)"
    echo "  include_answer: true/false"
    echo "  include_raw_content: true/false/\"markdown\"/\"text\""
    echo "  include_images: true/false"
    echo ""
    echo "Example:"
    echo "  ./tavily-search.sh '{\"query\": \"latest AI trends\", \"time_range\": \"week\"}'"
    exit 1
fi

if [ -z "$TAVILY_API_KEY" ]; then
    echo "Error: TAVILY_API_KEY environment variable is not set"
    echo "Get your API key at https://tavily.com"
    exit 1
fi

# Validate JSON
if ! echo "$JSON_INPUT" | jq empty 2>/dev/null; then
    echo "Error: Invalid JSON input"
    exit 1
fi

# Check for required query field
if ! echo "$JSON_INPUT" | jq -e '.query' >/dev/null 2>&1; then
    echo "Error: 'query' field is required"
    exit 1
fi

# Call Tavily Search API
curl -s --request POST \
    --url https://api.tavily.com/search \
    --header "Authorization: Bearer $TAVILY_API_KEY" \
    --header 'Content-Type: application/json' \
    --data "$JSON_INPUT" | jq '.'
