use serde_json::json;

use crate::model::ModelToolDefinition;

pub(super) fn build_native_tool_definitions() -> Vec<ModelToolDefinition> {
    vec![
        ModelToolDefinition {
            name: "current_datetime".to_owned(),
            description: "Get current UTC date/time for time-sensitive answers.".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation of why this tool is needed for the user's request."
                    }
                },
                "required": ["reasoning"],
                "additionalProperties": false
            }),
        },
        ModelToolDefinition {
            name: "run_terminal_command".to_owned(),
            description: "Run a local terminal command.".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "command": { "type": "string", "minLength": 1 },
                    "args": {
                        "oneOf": [
                            {
                                "type": "array",
                                "items": { "type": "string" },
                                "minItems": 1
                            },
                            {
                                "type": "string",
                                "minLength": 1
                            }
                        ]
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation of why this tool is needed for the user's request."
                    }
                },
                "required": ["reasoning"],
                // Restrict keys to command/args so provider models don't invent unsupported fields.
                "anyOf": [
                    { "required": ["command"] },
                    { "required": ["args"] }
                ],
                "additionalProperties": false
            }),
        },
        ModelToolDefinition {
            name: "web_search".to_owned(),
            description: "Search the web for current factual information.".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" },
                    "max_results": { "type": "integer", "minimum": 1, "maximum": 20 },
                    "search_depth": {
                        "type": "string",
                        "enum": ["basic", "advanced", "fast", "ultra-fast"],
                        "description": "Search depth: 'fast'/'basic' are quick, 'advanced' is slower but more thorough, 'ultra-fast' is lowest latency."
                    },
                    "topic": {
                        "type": "string",
                        "enum": ["general", "news", "finance"],
                        "description": "Category filter for search results."
                    },
                    "time_range": {
                        "type": "string",
                        "enum": ["day", "week", "month", "year", "d", "w", "m", "y"],
                        "description": "Filter results by recency."
                    },
                    "include_domains": {
                        "type": "array",
                        "items": { "type": "string" },
                        "maxItems": 10,
                        "description": "Only include results from these domains."
                    },
                    "exclude_domains": {
                        "type": "array",
                        "items": { "type": "string" },
                        "maxItems": 10,
                        "description": "Exclude results from these domains."
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation of why this tool is needed for the user's request."
                    }
                },
                "required": ["query", "reasoning"],
                "additionalProperties": false
            }),
        },
        ModelToolDefinition {
            name: "web_extract".to_owned(),
            description: "Extract clean content from specific URLs.".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "urls": {
                        "type": "array",
                        "items": { "type": "string" },
                        "minItems": 1,
                        "maxItems": 20,
                        "description": "URLs to extract content from (max 20)."
                    },
                    "query": {
                        "type": "string",
                        "description": "Optional query to rerank extracted chunks by relevance."
                    },
                    "extract_depth": {
                        "type": "string",
                        "enum": ["basic", "advanced"],
                        "description": "'basic' for simple pages, 'advanced' for JS-rendered pages."
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation of why this tool is needed for the user's request."
                    }
                },
                "required": ["urls", "reasoning"],
                "additionalProperties": false
            }),
        },
        ModelToolDefinition {
            name: "web_crawl".to_owned(),
            description: "Crawl a website to extract content from multiple pages.".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Root URL to begin crawling."
                    },
                    "max_depth": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5,
                        "description": "Levels deep to crawl (1-5)."
                    },
                    "max_breadth": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 500,
                        "description": "Max links to follow per page (1-500)."
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 500,
                        "description": "Total pages cap (1-500)."
                    },
                    "instructions": {
                        "type": "string",
                        "description": "Natural language guidance to focus the crawl."
                    },
                    "extract_depth": {
                        "type": "string",
                        "enum": ["basic", "advanced"],
                        "description": "'basic' for simple pages, 'advanced' for JS-rendered pages."
                    },
                    "select_paths": {
                        "type": "array",
                        "items": { "type": "string" },
                        "maxItems": 20,
                        "description": "Regex patterns for paths to include."
                    },
                    "exclude_paths": {
                        "type": "array",
                        "items": { "type": "string" },
                        "maxItems": 20,
                        "description": "Regex patterns for paths to exclude."
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation of why this tool is needed for the user's request."
                    }
                },
                "required": ["url", "reasoning"],
                "additionalProperties": false
            }),
        },
        ModelToolDefinition {
            name: "web_research".to_owned(),
            description: "Conduct comprehensive research on a topic with automatic source gathering and AI synthesis.".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "Research topic or question."
                    },
                    "model": {
                        "type": "string",
                        "enum": ["mini", "pro", "auto"],
                        "description": "'mini' for quick single-topic, 'pro' for comprehensive multi-angle, 'auto' lets API choose."
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation of why this tool is needed for the user's request."
                    }
                },
                "required": ["input", "reasoning"],
                "additionalProperties": false
            }),
        },
        ModelToolDefinition {
            name: "discord_voice_join".to_owned(),
            description: "Join user's Discord voice channel.".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "channel_id": { "type": "string" },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation of why this tool is needed for the user's request."
                    }
                },
                "required": ["reasoning"],
                "additionalProperties": false
            }),
        },
        ModelToolDefinition {
            name: "discord_voice_leave".to_owned(),
            description: "Leave Discord voice channel.".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation of why this tool is needed for the user's request."
                    }
                },
                "required": ["reasoning"],
                "additionalProperties": false
            }),
        },
        ModelToolDefinition {
            name: "store_memory".to_owned(),
            description: "Persist a durable user fact into memory.".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "key": { "type": "string" },
                    "value": { "type": "string" },
                    "confidence": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation of why this tool is needed for the user's request."
                    }
                },
                "required": ["key", "value", "reasoning"],
                "additionalProperties": false
            }),
        },
        ModelToolDefinition {
            name: "execute_program".to_owned(),
            description: "Execute a dependent sequence of tool calls where later steps can reference earlier outputs.".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 10,
                        "items": {
                            "type": "object",
                            "properties": {
                                "step_id": {
                                    "type": "string",
                                    "pattern": "^[a-zA-Z_][a-zA-Z0-9_]*$"
                                },
                                "tool_name": { "type": "string", "minLength": 1 },
                                "arguments": { "type": "object" }
                            },
                            "required": ["step_id", "tool_name", "arguments"],
                            "additionalProperties": false
                        }
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation of why this tool is needed for the user's request."
                    }
                },
                "required": ["steps", "reasoning"],
                "additionalProperties": false
            }),
        },
    ]
}
