use chrono::Utc;
use serde_json::{Value, json};
use tracing::debug;

use crate::{
    model::ModelToolCall,
    tools::sanitize_cli_invocation_args,
    types::{MemoryFact, ToolCall},
};

#[derive(Debug, Clone)]
pub(super) struct SanitizedToolCall {
    pub(super) call_id: String,
    pub(super) call: ToolCall,
}

pub(super) fn sanitize_native_tool_calls(raw_calls: Vec<ModelToolCall>) -> Vec<SanitizedToolCall> {
    raw_calls
        .into_iter()
        .filter_map(sanitize_native_tool_call)
        .collect()
}

fn sanitize_native_tool_call(raw_call: ModelToolCall) -> Option<SanitizedToolCall> {
    let call = match raw_call.name.as_str() {
        "current_datetime" => ToolCall {
            tool_name: "current_datetime".to_owned(),
            args: json!({}),
        },
        "cli" => {
            let args = sanitize_cli_invocation_args(&raw_call.arguments)?;
            ToolCall {
                tool_name: "cli".to_owned(),
                args,
            }
        }
        "web_search" => {
            let query = raw_call
                .arguments
                .get("query")
                .and_then(Value::as_str)
                .map(str::trim)
                .unwrap_or_default();
            if query.is_empty() {
                debug!("dropping native web_search call with empty query");
                return None;
            }

            let max_results = raw_call
                .arguments
                .get("max_results")
                .and_then(Value::as_u64)
                .unwrap_or(5)
                .clamp(1, 10);

            ToolCall {
                tool_name: "web_search".to_owned(),
                args: json!({
                    "query": query,
                    "max_results": max_results,
                }),
            }
        }
        "discord_voice_join" => {
            let channel_id = raw_call
                .arguments
                .get("channel_id")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty());
            let args = match channel_id {
                Some(channel_id) => json!({ "channel_id": channel_id }),
                None => json!({}),
            };
            ToolCall {
                tool_name: "discord_voice_join".to_owned(),
                args,
            }
        }
        "discord_voice_leave" => ToolCall {
            tool_name: "discord_voice_leave".to_owned(),
            args: json!({}),
        },
        "store_memory" => {
            let key = raw_call
                .arguments
                .get("key")
                .and_then(Value::as_str)
                .map(sanitize_memory_key)
                .unwrap_or_default();
            let value = raw_call
                .arguments
                .get("value")
                .and_then(Value::as_str)
                .map(clean_memory_value)
                .unwrap_or_default();
            if key.is_empty() || value.is_empty() {
                debug!("dropping native store_memory call with invalid key/value");
                return None;
            }

            let confidence = raw_call
                .arguments
                .get("confidence")
                .and_then(Value::as_f64)
                .unwrap_or(0.8)
                .clamp(0.0, 1.0);

            ToolCall {
                tool_name: "store_memory".to_owned(),
                args: json!({
                    "key": key,
                    "value": value,
                    "confidence": confidence
                }),
            }
        }
        other => {
            debug!(tool_name = other, "dropping unknown native tool call");
            return None;
        }
    };

    Some(SanitizedToolCall {
        call_id: raw_call.id,
        call,
    })
}

pub(super) fn memory_fact_from_store_memory_args(args: &Value) -> Option<MemoryFact> {
    let key = args
        .get("key")
        .and_then(Value::as_str)
        .map(sanitize_memory_key)?;
    let value = args
        .get("value")
        .and_then(Value::as_str)
        .map(clean_memory_value)?;
    if key.is_empty() || value.is_empty() {
        return None;
    }

    let confidence = args
        .get("confidence")
        .and_then(Value::as_f64)
        .unwrap_or(0.8)
        .clamp(0.0, 1.0);

    Some(MemoryFact {
        key,
        value,
        confidence: confidence as f32,
        source: "model_store_memory".to_owned(),
        updated_at: Utc::now(),
    })
}

pub(super) fn sanitize_memory_key(raw: &str) -> String {
    let mut normalized = raw
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect::<String>();

    while normalized.contains("__") {
        normalized = normalized.replace("__", "_");
    }

    normalized.trim_matches('_').to_owned()
}

pub(super) fn clean_memory_value(value: &str) -> String {
    value
        .trim()
        .trim_matches(|character: char| character == '"' || character == '\'')
        .trim()
        .to_owned()
}
