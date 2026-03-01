use async_trait::async_trait;
use serde_json::json;

use super::{ModelProvider, ModelRequest};

#[derive(Debug, Default)]
pub struct MockModelProvider;

#[async_trait]
impl ModelProvider for MockModelProvider {
    async fn complete(&self, request: ModelRequest) -> anyhow::Result<String> {
        if request
            .system_prompt
            .contains("You are the unified planner for CompanionPilot.")
        {
            let memory = if let Some(name) = extract_name(&request.user_prompt) {
                json!({
                    "store": true,
                    "key": "name",
                    "value": name,
                    "confidence": 0.96
                })
            } else if let Some(game) = extract_game(&request.user_prompt) {
                json!({
                    "store": true,
                    "key": "favorite_game",
                    "value": game,
                    "confidence": 0.84
                })
            } else {
                json!({
                    "store": false,
                    "key": "",
                    "value": "",
                    "confidence": 0.0
                })
            };

            let mut tool_calls = Vec::new();
            if let Some(query) = extract_search_query(&request.user_prompt) {
                tool_calls.push(json!({
                    "tool_name": "web_search",
                    "args": {
                        "query": query,
                        "max_results": 5
                    }
                }));
            }
            if extract_join_voice(&request.user_prompt) {
                tool_calls.push(json!({
                    "tool_name": "discord_voice_join",
                    "args": {}
                }));
            }
            if extract_listen_voice_turn(&request.user_prompt) {
                tool_calls.push(json!({
                    "tool_name": "discord_voice_listen_turn",
                    "args": {}
                }));
            }
            if extract_leave_voice(&request.user_prompt) {
                tool_calls.push(json!({
                    "tool_name": "discord_voice_leave",
                    "args": {}
                }));
            }

            return Ok(json!({
                "tool_calls": tool_calls,
                "memory": memory,
                "rationale": "mock_unified_planner"
            })
            .to_string());
        }

        Ok(format!(
            "CompanionPilot mock reply.\n\nSystem: {}\n\nUser: {}",
            request.system_prompt, request.user_prompt
        ))
    }
}

fn extract_name(input: &str) -> Option<String> {
    let lowered = input.to_lowercase();
    if let Some(index) = lowered.find("name is ") {
        return Some(input[index + "name is ".len()..].trim().to_owned());
    }
    if let Some(index) = lowered.find("it's ") {
        return Some(
            input[index + "it's ".len()..]
                .trim()
                .trim_end_matches('.')
                .to_owned(),
        );
    }
    None
}

fn extract_game(input: &str) -> Option<String> {
    let lowered = input.to_lowercase();
    lowered
        .find("i play ")
        .map(|index| input[index + "i play ".len()..].trim().to_owned())
}

fn extract_search_query(input: &str) -> Option<String> {
    let lowered = input.to_lowercase();

    let query = if let Some(index) = lowered.find("search the web for ") {
        input[index + "search the web for ".len()..].trim()
    } else if let Some(index) = lowered.find("look up ") {
        input[index + "look up ".len()..].trim()
    } else {
        return None;
    };

    let query = query
        .trim_matches(|character: char| !character.is_alphanumeric() && !character.is_whitespace())
        .trim();
    if query.is_empty() {
        None
    } else {
        Some(query.to_owned())
    }
}

fn extract_join_voice(input: &str) -> bool {
    let lowered = input.to_lowercase();
    lowered.contains("join voice")
        || lowered.contains("join the voice")
        || lowered.contains("connect to voice")
}

fn extract_listen_voice_turn(input: &str) -> bool {
    let lowered = input.to_lowercase();
    lowered.contains("listen in voice")
        || lowered.contains("listen now in voice")
        || lowered.contains("voice turn")
}

fn extract_leave_voice(input: &str) -> bool {
    let lowered = input.to_lowercase();
    lowered.contains("leave voice") || lowered.contains("disconnect from voice")
}
