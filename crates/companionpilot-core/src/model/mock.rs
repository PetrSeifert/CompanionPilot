use async_trait::async_trait;
use serde_json::json;

use super::{
    ModelMessage, ModelMessageRole, ModelProvider, ModelRequest, ModelToolCall, ModelTurnRequest,
    ModelTurnResponse,
};

#[derive(Debug, Default)]
pub struct MockModelProvider;

#[async_trait]
impl ModelProvider for MockModelProvider {
    async fn complete(&self, request: ModelRequest) -> anyhow::Result<String> {
        if request
            .system_prompt
            .contains("You are the skill selector for CompanionPilot.")
        {
            return Ok(json!({
                "selected_skills": [],
                "rationale": "mock_selector_default"
            })
            .to_string());
        }

        Ok(format!(
            "CompanionPilot mock reply.\n\nSystem: {}\n\nUser: {}",
            request.system_prompt, request.user_prompt
        ))
    }

    async fn complete_turn(&self, request: ModelTurnRequest) -> anyhow::Result<ModelTurnResponse> {
        let user_input = latest_user_input(&request.messages);
        let tool_messages = request
            .messages
            .iter()
            .filter(|message| message.role == ModelMessageRole::Tool)
            .collect::<Vec<_>>();

        if user_input.contains("find a final answer using tools") {
            if has_tool_output(&tool_messages, "result:beta") {
                return Ok(ModelTurnResponse {
                    assistant_text: "Final answer from native tool loop.".to_owned(),
                    tool_calls: Vec::new(),
                });
            }

            if has_tool_output(&tool_messages, "result:alpha") {
                return Ok(ModelTurnResponse {
                    assistant_text: String::new(),
                    tool_calls: vec![tool_call(
                        "call-beta",
                        "web_search",
                        json!({ "query": "beta", "max_results": 2 }),
                    )],
                });
            }

            return Ok(ModelTurnResponse {
                assistant_text: String::new(),
                tool_calls: vec![tool_call(
                    "call-alpha",
                    "web_search",
                    json!({ "query": "alpha", "max_results": 3 }),
                )],
            });
        }

        if !tool_messages.is_empty() {
            let joined = tool_messages
                .iter()
                .map(|message| message.content.as_str())
                .collect::<Vec<_>>()
                .join("\n\n");
            return Ok(ModelTurnResponse {
                assistant_text: format!("Tool summary:\n{joined}"),
                tool_calls: Vec::new(),
            });
        }

        let mut tool_calls = Vec::new();
        if let Some(name) = extract_name(&user_input) {
            tool_calls.push(tool_call(
                "call-store-name",
                "store_memory",
                json!({
                    "key": "name",
                    "value": name,
                    "confidence": 0.96
                }),
            ));
        }
        if let Some(game) = extract_game(&user_input) {
            tool_calls.push(tool_call(
                "call-store-game",
                "store_memory",
                json!({
                    "key": "favorite_game",
                    "value": game,
                    "confidence": 0.84
                }),
            ));
        }
        if let Some(query) = extract_search_query(&user_input) {
            tool_calls.push(tool_call(
                "call-web-search",
                "web_search",
                json!({
                    "query": query,
                    "max_results": 5
                }),
            ));
        }
        if let Some(query) = extract_spotify_query(&user_input) {
            tool_calls.push(tool_call(
                "call-run-terminal-command",
                "run_terminal_command",
                json!({
                    "command": format!("spogo search track {query}")
                }),
            ));
        }
        if extract_join_voice(&user_input) {
            tool_calls.push(tool_call(
                "call-join-voice",
                "discord_voice_join",
                json!({}),
            ));
        }
        if extract_leave_voice(&user_input) {
            tool_calls.push(tool_call(
                "call-leave-voice",
                "discord_voice_leave",
                json!({}),
            ));
        }

        if !tool_calls.is_empty() {
            return Ok(ModelTurnResponse {
                assistant_text: String::new(),
                tool_calls,
            });
        }

        Ok(ModelTurnResponse {
            assistant_text: format!("CompanionPilot mock reply to: {user_input}"),
            tool_calls: Vec::new(),
        })
    }
}

fn latest_user_input(messages: &[ModelMessage]) -> String {
    messages
        .iter()
        .rev()
        .find(|message| message.role == ModelMessageRole::User)
        .map(|message| message.content.clone())
        .unwrap_or_default()
}

fn has_tool_output(messages: &[&ModelMessage], needle: &str) -> bool {
    messages.iter().any(|message| {
        message
            .content
            .to_lowercase()
            .contains(&needle.to_lowercase())
    })
}

fn tool_call(id: &str, name: &str, arguments: serde_json::Value) -> ModelToolCall {
    ModelToolCall {
        id: id.to_owned(),
        name: name.to_owned(),
        arguments,
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

fn extract_spotify_query(input: &str) -> Option<String> {
    let lowered = input.to_lowercase();

    let query = if let Some(index) = lowered.find("search spotify for ") {
        input[index + "search spotify for ".len()..].trim()
    } else if let Some(index) = lowered.find("spotify search for ") {
        input[index + "spotify search for ".len()..].trim()
    } else if let Some(index) = lowered.find("find on spotify ") {
        input[index + "find on spotify ".len()..].trim()
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

fn extract_leave_voice(input: &str) -> bool {
    let lowered = input.to_lowercase();
    lowered.contains("leave voice") || lowered.contains("disconnect from voice")
}
