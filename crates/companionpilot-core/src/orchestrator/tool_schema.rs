use serde_json::json;

use crate::model::ModelToolDefinition;

pub(super) fn build_native_tool_definitions() -> Vec<ModelToolDefinition> {
    vec![
        ModelToolDefinition {
            name: "current_datetime".to_owned(),
            description: "Get current UTC date/time for time-sensitive answers.".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }),
        },
        ModelToolDefinition {
            name: "cli".to_owned(),
            description: "Execute local spogo CLI commands for Spotify operations.".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "command": { "type": "string" },
                    "args": {
                        "oneOf": [
                            { "type": "array", "items": { "type": "string" } },
                            { "type": "string" }
                        ]
                    }
                },
                // Allow additional keys for forward-compatible CLI wrappers while sanitizer enforces safety.
                "additionalProperties": true
            }),
        },
        ModelToolDefinition {
            name: "web_search".to_owned(),
            description: "Search the web for current factual information.".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" },
                    "max_results": { "type": "integer", "minimum": 1, "maximum": 10 }
                },
                "required": ["query"],
                "additionalProperties": false
            }),
        },
        ModelToolDefinition {
            name: "discord_voice_join".to_owned(),
            description: "Join user's Discord voice channel.".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "channel_id": { "type": "string" }
                },
                "additionalProperties": false
            }),
        },
        ModelToolDefinition {
            name: "discord_voice_leave".to_owned(),
            description: "Leave Discord voice channel.".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": {},
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
                    "confidence": { "type": "number", "minimum": 0.0, "maximum": 1.0 }
                },
                "required": ["key", "value"],
                "additionalProperties": false
            }),
        },
    ]
}
