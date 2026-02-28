use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{ModelProvider, ModelRequest};

#[derive(Debug, Clone)]
pub struct OpenRouterProvider {
    client: Client,
    api_key: String,
    model: String,
    referer: Option<String>,
    title: Option<String>,
}

impl OpenRouterProvider {
    pub fn new(
        api_key: String,
        model: String,
        referer: Option<String>,
        title: Option<String>,
    ) -> Self {
        Self {
            client: Client::new(),
            api_key,
            model,
            referer,
            title,
        }
    }
}

#[derive(Debug, Serialize)]
struct ChatCompletionRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMessage<'a>>,
}

#[derive(Debug, Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatChoiceMessage,
}

#[derive(Debug, Deserialize)]
struct ChatChoiceMessage {
    content: Value,
}

#[async_trait]
impl ModelProvider for OpenRouterProvider {
    async fn complete(&self, request: ModelRequest) -> anyhow::Result<String> {
        let payload = ChatCompletionRequest {
            model: &self.model,
            messages: vec![
                ChatMessage {
                    role: "system",
                    content: &request.system_prompt,
                },
                ChatMessage {
                    role: "user",
                    content: &request.user_prompt,
                },
            ],
        };

        let mut builder = self
            .client
            .post("https://openrouter.ai/api/v1/chat/completions")
            .bearer_auth(&self.api_key)
            .json(&payload);

        if let Some(referer) = &self.referer {
            builder = builder.header("HTTP-Referer", referer);
        }
        if let Some(title) = &self.title {
            builder = builder.header("X-Title", title);
        }

        let response = builder
            .send()
            .await?
            .error_for_status()?
            .json::<ChatCompletionResponse>()
            .await?;

        let content = response
            .choices
            .first()
            .and_then(|choice| extract_message_content(&choice.message.content))
            .ok_or_else(|| anyhow::anyhow!("model returned no choices"))?;

        Ok(content)
    }
}

fn extract_message_content(content: &Value) -> Option<String> {
    if let Some(text) = content.as_str() {
        return Some(text.to_owned());
    }

    let array = content.as_array()?;
    let joined = array
        .iter()
        .filter_map(|item| item.get("text").and_then(Value::as_str))
        .collect::<Vec<_>>()
        .join("");

    if joined.is_empty() {
        None
    } else {
        Some(joined)
    }
}
