#!/bin/bash
# Tavily Research API script
# Usage: ./tavily-research.sh '{"input": "your research query", ...}'
# Example: ./tavily-research.sh '{"input": "quantum computing trends", "model": "pro"}'

set -e

JSON_INPUT="$1"

if [ -z "$JSON_INPUT" ]; then
    echo "Usage: ./tavily-research.sh '<json>'"
    echo ""
    echo "Required:"
    echo "  input: string - The research topic or question"
    echo ""
    echo "Optional:"
    echo "  model: \"mini\" (default), \"pro\", \"auto\""
    echo "    - mini: Targeted, efficient research for narrow questions"
    echo "    - pro: Comprehensive, multi-agent research for complex topics"
    echo "    - auto: Automatically selects based on query complexity"
    echo ""
    echo "Example:"
    echo "  ./tavily-research.sh '{\"input\": \"AI agent frameworks comparison\", \"model\": \"pro\"}'"
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

# Check for required input field
if ! echo "$JSON_INPUT" | jq -e '.input' >/dev/null 2>&1; then
    echo "Error: 'input' field is required"
    exit 1
fi

INPUT=$(echo "$JSON_INPUT" | jq -r '.input')
MODEL=$(echo "$JSON_INPUT" | jq -r '.model // "mini"')

echo "Researching: $INPUT (model: $MODEL)" >&2
echo "This may take 30-120 seconds..." >&2

# Call Tavily Research API
RESPONSE=$(curl -s --request POST \
    --url https://api.tavily.com/research \
    --header "Authorization: Bearer $TAVILY_API_KEY" \
    --header 'Content-Type: application/json' \
    --data "$JSON_INPUT")

# Check if we got a request_id for polling
REQUEST_ID=$(echo "$RESPONSE" | jq -r '.request_id // empty' 2>/dev/null)

if [ -z "$REQUEST_ID" ]; then
    echo "$RESPONSE" | jq '.'
    exit 0
fi

echo "Request ID: $REQUEST_ID — polling for results..." >&2

# Poll until completed
while true; do
    sleep 10
    RESULT=$(curl -s --request GET \
        --url "https://api.tavily.com/research/$REQUEST_ID" \
        --header "Authorization: Bearer $TAVILY_API_KEY")

    STATUS=$(echo "$RESULT" | jq -r '.status // empty' 2>/dev/null)

    if [ "$STATUS" = "completed" ] || [ "$STATUS" = "failed" ]; then
        echo "$RESULT" | jq '.'
        break
    fi

    echo "Status: $STATUS ..." >&2
done
