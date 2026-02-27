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
            .contains("You are a tool router for CompanionPilot.")
        {
            return Ok(json!({
                "use_search": false,
                "query": ""
            })
            .to_string());
        }

        if request
            .system_prompt
            .contains("You are a memory router for CompanionPilot.")
        {
            if let Some(name) = extract_name(&request.user_prompt) {
                return Ok(json!({
                    "store": true,
                    "key": "name",
                    "value": name,
                    "confidence": 0.96
                })
                .to_string());
            }
            if let Some(game) = extract_game(&request.user_prompt) {
                return Ok(json!({
                    "store": true,
                    "key": "favorite_game",
                    "value": game,
                    "confidence": 0.84
                })
                .to_string());
            }

            return Ok(json!({
                "store": false,
                "key": "",
                "value": "",
                "confidence": 0.0
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
