#!/bin/bash
# Tavily Crawl API script
# Usage: ./tavily-crawl.sh '{"url": "https://example.com", ...}'
# Example: ./tavily-crawl.sh '{"url": "https://docs.example.com", "max_depth": 2, "limit": 20}'

set -e

JSON_INPUT="$1"

if [ -z "$JSON_INPUT" ]; then
    echo "Usage: ./tavily-crawl.sh '<json>'"
    echo ""
    echo "Required:"
    echo "  url: string - Root URL to begin crawling"
    echo ""
    echo "Optional:"
    echo "  max_depth: 1-5 (default: 1) - Levels deep to crawl"
    echo "  max_breadth: integer (default: 20) - Links per page"
    echo "  limit: integer (default: 50) - Total pages cap"
    echo "  instructions: string - Natural language guidance for semantic focus"
    echo "  chunks_per_source: 1-5 (default: 3, requires instructions)"
    echo "  extract_depth: \"basic\" (default), \"advanced\""
    echo "  format: \"markdown\" (default), \"text\""
    echo "  select_paths: [\"regex1\", \"regex2\"] - Paths to include"
    echo "  exclude_paths: [\"regex1\", \"regex2\"] - Paths to exclude"
    echo "  select_domains: [\"regex1\"] - Domains to include"
    echo "  exclude_domains: [\"regex1\"] - Domains to exclude"
    echo "  allow_external: true/false (default: true)"
    echo "  timeout: 10-150 seconds (default: 150)"
    echo ""
    echo "Example:"
    echo "  ./tavily-crawl.sh '{\"url\": \"https://docs.example.com\", \"max_depth\": 2}'"
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

# Check for required url field
if ! echo "$JSON_INPUT" | jq -e '.url' >/dev/null 2>&1; then
    echo "Error: 'url' field is required"
    exit 1
fi

# Call Tavily Crawl API
curl -s --request POST \
    --url https://api.tavily.com/crawl \
    --header "Authorization: Bearer $TAVILY_API_KEY" \
    --header 'Content-Type: application/json' \
    --data "$JSON_INPUT" | jq '.'
