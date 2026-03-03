use chrono::Utc;
use serde_json::{Value, json};
use tracing::debug;

use crate::{
    tools::sanitize_cli_invocation_args,
    types::{MemoryFact, ToolCall},
};

use super::contracts::{MemoryDecision, PlannedMemory, PlannedToolCall};

pub(super) fn sanitize_planned_tool_calls(planned_calls: Vec<PlannedToolCall>) -> Vec<ToolCall> {
    planned_calls
        .into_iter()
        .filter_map(sanitize_planned_tool_call)
        .collect()
}

fn sanitize_planned_tool_call(planned_call: PlannedToolCall) -> Option<ToolCall> {
    match planned_call.tool_name.as_str() {
        "current_datetime" => Some(ToolCall {
            tool_name: "current_datetime".to_owned(),
            args: json!({}),
        }),
        "cli" => {
            if let Some(args) = sanitize_cli_invocation_args(&planned_call.args) {
                Some(ToolCall {
                    tool_name: "cli".to_owned(),
                    args,
                })
            } else {
                debug!("dropping planner cli call with invalid args");
                None
            }
        }
        "web_search" => {
            let query = planned_call
                .args
                .get("query")
                .and_then(Value::as_str)
                .map(str::trim)
                .unwrap_or("");
            if query.is_empty() {
                debug!("dropping planner web_search call with empty query");
                return None;
            }

            let max_results = planned_call
                .args
                .get("max_results")
                .and_then(Value::as_u64)
                .unwrap_or(5)
                .clamp(1, 10);

            Some(ToolCall {
                tool_name: "web_search".to_owned(),
                args: json!({
                    "query": query,
                    "max_results": max_results
                }),
            })
        }
        "discord_voice_join" => {
            let channel_id = planned_call
                .args
                .get("channel_id")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty());
            let args = match channel_id {
                Some(channel_id) => json!({ "channel_id": channel_id }),
                None => json!({}),
            };
            Some(ToolCall {
                tool_name: "discord_voice_join".to_owned(),
                args,
            })
        }
        "discord_voice_leave" => Some(ToolCall {
            tool_name: "discord_voice_leave".to_owned(),
            args: json!({}),
        }),
        other => {
            debug!(tool_name = other, "dropping unknown planner tool call");
            None
        }
    }
}

pub(super) fn enforce_datetime_planning_boundary(tool_calls: Vec<ToolCall>) -> Vec<ToolCall> {
    let has_datetime = tool_calls
        .iter()
        .any(|call| call.tool_name == "current_datetime");
    let has_non_datetime = tool_calls
        .iter()
        .any(|call| call.tool_name != "current_datetime");
    if !has_datetime || !has_non_datetime {
        return tool_calls;
    }

    debug!(
        total_calls = tool_calls.len(),
        "deferring non-datetime tools to follow-up round because current_datetime was requested"
    );
    let datetime_call = tool_calls
        .into_iter()
        .find(|call| call.tool_name == "current_datetime")
        .expect("checked current_datetime presence above");
    vec![datetime_call]
}

pub(super) fn memory_decision_from_plan(plan: PlannedMemory) -> MemoryDecision {
    if !plan.store {
        return MemoryDecision::Skip {
            reason: "planner_no_store",
        };
    }

    let key = sanitize_memory_key(&plan.key);
    let value = clean_memory_value(&plan.value);
    if key.is_empty() || value.is_empty() {
        return MemoryDecision::Skip {
            reason: "planner_invalid_fact",
        };
    }

    MemoryDecision::Store {
        fact: MemoryFact {
            key,
            value,
            confidence: plan.confidence.clamp(0.0, 1.0),
            source: "user_message".to_owned(),
            updated_at: Utc::now(),
        },
        rationale: "model_planner",
    }
}

pub(super) fn memory_payload(memory: &MemoryDecision) -> Value {
    match memory {
        MemoryDecision::Store { fact, .. } => json!({
            "store": true,
            "key": fact.key,
            "value": fact.value,
            "confidence": fact.confidence
        }),
        MemoryDecision::Skip { reason } => json!({
            "store": false,
            "reason": reason
        }),
    }
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
