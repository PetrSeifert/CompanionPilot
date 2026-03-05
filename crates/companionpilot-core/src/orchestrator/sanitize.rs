use std::collections::HashSet;

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
    pub(super) reasoning: Option<String>,
}

pub(super) fn sanitize_native_tool_calls(raw_calls: Vec<ModelToolCall>) -> Vec<SanitizedToolCall> {
    raw_calls
        .into_iter()
        .filter_map(sanitize_native_tool_call)
        .collect()
}

fn sanitize_native_tool_call(raw_call: ModelToolCall) -> Option<SanitizedToolCall> {
    let reasoning = raw_call
        .arguments
        .get("reasoning")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToOwned::to_owned);

    let call = match raw_call.name.as_str() {
        "current_datetime" => ToolCall {
            tool_name: "current_datetime".to_owned(),
            args: json!({}),
        },
        "run_terminal_command" | "cli" => {
            let args = sanitize_cli_invocation_args(&raw_call.arguments)?;
            ToolCall {
                tool_name: "run_terminal_command".to_owned(),
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
                .clamp(1, 20);

            let mut args = json!({
                "query": query,
                "max_results": max_results,
            });

            if let Some(depth) = raw_call
                .arguments
                .get("search_depth")
                .and_then(Value::as_str)
            {
                if matches!(depth, "basic" | "advanced" | "fast" | "ultra-fast") {
                    args["search_depth"] = json!(depth);
                }
            }

            if let Some(topic) = raw_call.arguments.get("topic").and_then(Value::as_str) {
                if matches!(topic, "general" | "news" | "finance") {
                    args["topic"] = json!(topic);
                }
            }

            if let Some(range) = raw_call
                .arguments
                .get("time_range")
                .and_then(Value::as_str)
            {
                if matches!(range, "day" | "week" | "month" | "year" | "d" | "w" | "m" | "y") {
                    args["time_range"] = json!(range);
                }
            }

            let extract_domain_list = |key: &str| -> Option<Vec<String>> {
                let arr = raw_call.arguments.get(key)?.as_array()?;
                let domains: Vec<String> = arr
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .take(10)
                    .map(ToOwned::to_owned)
                    .collect();
                if domains.is_empty() { None } else { Some(domains) }
            };

            if let Some(domains) = extract_domain_list("include_domains") {
                args["include_domains"] = json!(domains);
            }
            if let Some(domains) = extract_domain_list("exclude_domains") {
                args["exclude_domains"] = json!(domains);
            }

            ToolCall {
                tool_name: "web_search".to_owned(),
                args,
            }
        }
        "web_extract" => {
            let urls: Vec<String> = raw_call
                .arguments
                .get("urls")
                .and_then(Value::as_array)
                .map(|arr| {
                    arr.iter()
                        .filter_map(Value::as_str)
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                        .take(20)
                        .map(ToOwned::to_owned)
                        .collect()
                })
                .unwrap_or_default();
            if urls.is_empty() {
                debug!("dropping native web_extract call with empty urls");
                return None;
            }

            let mut args = json!({ "urls": urls });

            if let Some(query) = raw_call
                .arguments
                .get("query")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
            {
                args["query"] = json!(query);
            }

            if let Some(depth) = raw_call
                .arguments
                .get("extract_depth")
                .and_then(Value::as_str)
            {
                if matches!(depth, "basic" | "advanced") {
                    args["extract_depth"] = json!(depth);
                }
            }

            ToolCall {
                tool_name: "web_extract".to_owned(),
                args,
            }
        }
        "web_crawl" => {
            let url = raw_call
                .arguments
                .get("url")
                .and_then(Value::as_str)
                .map(str::trim)
                .unwrap_or_default();
            if url.is_empty() {
                debug!("dropping native web_crawl call with empty url");
                return None;
            }

            let mut args = json!({ "url": url });

            if let Some(depth) = raw_call.arguments.get("max_depth").and_then(Value::as_u64) {
                args["max_depth"] = json!(depth.clamp(1, 5));
            }
            if let Some(breadth) = raw_call
                .arguments
                .get("max_breadth")
                .and_then(Value::as_u64)
            {
                args["max_breadth"] = json!(breadth.clamp(1, 500));
            }
            if let Some(limit) = raw_call.arguments.get("limit").and_then(Value::as_u64) {
                args["limit"] = json!(limit.clamp(1, 500));
            }

            if let Some(instructions) = raw_call
                .arguments
                .get("instructions")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
            {
                args["instructions"] = json!(instructions);
            }

            if let Some(depth) = raw_call
                .arguments
                .get("extract_depth")
                .and_then(Value::as_str)
            {
                if matches!(depth, "basic" | "advanced") {
                    args["extract_depth"] = json!(depth);
                }
            }

            let extract_path_list = |key: &str| -> Option<Vec<String>> {
                let arr = raw_call.arguments.get(key)?.as_array()?;
                let paths: Vec<String> = arr
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .take(20)
                    .map(ToOwned::to_owned)
                    .collect();
                if paths.is_empty() { None } else { Some(paths) }
            };

            if let Some(paths) = extract_path_list("select_paths") {
                args["select_paths"] = json!(paths);
            }
            if let Some(paths) = extract_path_list("exclude_paths") {
                args["exclude_paths"] = json!(paths);
            }

            ToolCall {
                tool_name: "web_crawl".to_owned(),
                args,
            }
        }
        "web_research" => {
            let input = raw_call
                .arguments
                .get("input")
                .and_then(Value::as_str)
                .map(str::trim)
                .unwrap_or_default();
            if input.is_empty() {
                debug!("dropping native web_research call with empty input");
                return None;
            }

            let mut args = json!({ "input": input });

            if let Some(model) = raw_call
                .arguments
                .get("model")
                .and_then(Value::as_str)
            {
                if matches!(model, "mini" | "pro" | "auto") {
                    args["model"] = json!(model);
                }
            }

            ToolCall {
                tool_name: "web_research".to_owned(),
                args,
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
        "execute_program" => {
            let Some(steps) = raw_call.arguments.get("steps").and_then(Value::as_array) else {
                debug!("dropping execute_program call with missing steps array");
                return None;
            };
            if steps.is_empty() || steps.len() > 10 {
                debug!(
                    step_count = steps.len(),
                    "dropping execute_program call with invalid step count"
                );
                return None;
            }

            let mut seen_step_ids = HashSet::new();
            let mut prior_step_ids = HashSet::new();
            let mut sanitized_steps = Vec::with_capacity(steps.len());

            for step in steps {
                let Some(step_obj) = step.as_object() else {
                    debug!("dropping execute_program call with non-object step");
                    return None;
                };

                let step_id = step_obj
                    .get("step_id")
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .unwrap_or_default();
                if step_id.is_empty() || !is_valid_step_id(step_id) {
                    debug!(
                        step_id,
                        "dropping execute_program call with invalid step_id"
                    );
                    return None;
                }
                if !seen_step_ids.insert(step_id.to_owned()) {
                    debug!(
                        step_id,
                        "dropping execute_program call with duplicate step_id"
                    );
                    return None;
                }

                let tool_name = step_obj
                    .get("tool_name")
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .unwrap_or_default();
                if tool_name.is_empty() {
                    debug!(
                        step_id,
                        "dropping execute_program call with empty tool_name"
                    );
                    return None;
                }
                if tool_name == "execute_program" {
                    debug!(
                        step_id,
                        "dropping execute_program call with nested execute_program step"
                    );
                    return None;
                }

                let Some(arguments_obj) = step_obj.get("arguments").and_then(Value::as_object)
                else {
                    debug!(
                        step_id,
                        "dropping execute_program call with non-object step arguments"
                    );
                    return None;
                };
                let arguments = Value::Object(arguments_obj.clone());
                let references = extract_step_references(&arguments);
                for reference in references {
                    if !prior_step_ids.contains(&reference) {
                        debug!(
                            step_id,
                            reference,
                            "dropping execute_program call with invalid/forward step reference"
                        );
                        return None;
                    }
                }

                prior_step_ids.insert(step_id.to_owned());
                sanitized_steps.push(json!({
                    "step_id": step_id,
                    "tool_name": tool_name,
                    "arguments": arguments
                }));
            }

            ToolCall {
                tool_name: "execute_program".to_owned(),
                args: json!({
                    "steps": sanitized_steps
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
        reasoning,
    })
}

pub(super) fn is_valid_step_id(id: &str) -> bool {
    let mut chars = id.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !(first.is_ascii_alphabetic() || first == '_') {
        return false;
    }
    chars.all(|character| character.is_ascii_alphanumeric() || character == '_')
}

pub(super) fn extract_step_references(value: &Value) -> Vec<String> {
    match value {
        Value::String(text) => {
            let mut references = Vec::new();
            for (start, _) in text.match_indices("${") {
                let rest = &text[(start + 2)..];
                if let Some(end) = rest.find('}') {
                    let step_id = &rest[..end];
                    if !step_id.is_empty() {
                        references.push(step_id.to_owned());
                    }
                }
            }
            references
        }
        Value::Array(items) => items
            .iter()
            .flat_map(extract_step_references)
            .collect::<Vec<_>>(),
        Value::Object(object) => object
            .values()
            .flat_map(extract_step_references)
            .collect::<Vec<_>>(),
        _ => Vec::new(),
    }
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
