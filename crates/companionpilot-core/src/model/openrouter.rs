use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::{
    ModelMessage, ModelProvider, ModelRequest, ModelToolCall, ModelToolDefinition,
    ModelTurnRequest, ModelTurnResponse,
};

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

    pub async fn complete_with_model(
        &self,
        model: &str,
        request: ModelRequest,
    ) -> anyhow::Result<String> {
        let payload = ChatCompletionRequest {
            model: model.to_owned(),
            messages: vec![
                ChatCompletionMessage {
                    role: "system".to_owned(),
                    content: request.system_prompt,
                    name: None,
                    tool_call_id: None,
                    tool_calls: Vec::new(),
                },
                ChatCompletionMessage {
                    role: "user".to_owned(),
                    content: request.user_prompt,
                    name: None,
                    tool_call_id: None,
                    tool_calls: Vec::new(),
                },
            ],
            tools: Vec::new(),
            tool_choice: None,
        };

        let response = self.post_chat_completion(payload).await?;
        let choice = response
            .choices
            .first()
            .ok_or_else(|| anyhow::anyhow!("model returned no choices"))?;
        let content = extract_message_content(choice.message.content.as_ref())
            .ok_or_else(|| anyhow::anyhow!("model returned no text content"))?;
        Ok(content)
    }

    pub async fn complete_turn_with_model(
        &self,
        model: &str,
        request: ModelTurnRequest,
    ) -> anyhow::Result<ModelTurnResponse> {
        let mut messages = Vec::with_capacity(request.messages.len() + 1);
        messages.push(ChatCompletionMessage {
            role: "system".to_owned(),
            content: request.system_prompt,
            name: None,
            tool_call_id: None,
            tool_calls: Vec::new(),
        });
        for message in request.messages {
            messages.push(serialize_model_message(message)?);
        }

        let payload = ChatCompletionRequest {
            model: model.to_owned(),
            messages,
            tools: request
                .tools
                .iter()
                .map(serialize_tool_definition)
                .collect::<Vec<_>>(),
            tool_choice: Some("auto"),
        };

        let response = self.post_chat_completion(payload).await?;
        let choice = response
            .choices
            .first()
            .ok_or_else(|| anyhow::anyhow!("model returned no choices"))?;
        let assistant_text =
            extract_message_content(choice.message.content.as_ref()).unwrap_or_default();
        let tool_calls = parse_tool_calls(&choice.message.tool_calls)?;

        Ok(ModelTurnResponse {
            assistant_text,
            tool_calls,
        })
    }

    async fn post_chat_completion(
        &self,
        payload: ChatCompletionRequest,
    ) -> anyhow::Result<ChatCompletionResponse> {
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
        Ok(response)
    }
}

fn serialize_model_message(message: ModelMessage) -> anyhow::Result<ChatCompletionMessage> {
    let mut tool_calls = Vec::with_capacity(message.tool_calls.len());
    for call in message.tool_calls {
        tool_calls.push(ChatCompletionToolCall {
            id: call.id,
            call_type: "function".to_owned(),
            function: ChatCompletionFunctionCall {
                name: call.name,
                arguments: serde_json::to_string(&call.arguments)?,
            },
        });
    }

    Ok(ChatCompletionMessage {
        role: message.role.as_str().to_owned(),
        content: message.content,
        name: message.name,
        tool_call_id: message.tool_call_id,
        tool_calls,
    })
}

fn serialize_tool_definition(definition: &ModelToolDefinition) -> ChatCompletionToolDefinition {
    ChatCompletionToolDefinition {
        tool_type: "function".to_owned(),
        function: ChatCompletionToolFunction {
            name: definition.name.clone(),
            description: definition.description.clone(),
            parameters: definition.parameters.clone(),
        },
    }
}

fn parse_tool_calls(raw_calls: &[ChatCompletionToolCall]) -> anyhow::Result<Vec<ModelToolCall>> {
    if raw_calls.is_empty() {
        return Ok(Vec::new());
    }

    let mut parsed = Vec::new();
    for call in raw_calls {
        if call.call_type != "function" {
            continue;
        }
        let arguments = parse_tool_call_arguments(&call.function.arguments)?;
        parsed.push(ModelToolCall {
            id: call.id.clone(),
            name: call.function.name.clone(),
            arguments,
        });
    }

    Ok(parsed)
}

fn parse_tool_call_arguments(raw: &str) -> anyhow::Result<Value> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(json!({}));
    }

    match serde_json::from_str::<Value>(trimmed) {
        Ok(value) => Ok(value),
        Err(error) => Err(anyhow::anyhow!(
            "failed to parse tool arguments as JSON: {error}"
        )),
    }
}

#[derive(Debug, Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatCompletionMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<ChatCompletionToolDefinition>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<&'static str>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct ChatCompletionMessage {
    role: String,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    tool_calls: Vec<ChatCompletionToolCall>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct ChatCompletionToolCall {
    id: String,
    #[serde(rename = "type")]
    call_type: String,
    function: ChatCompletionFunctionCall,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct ChatCompletionFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Debug, Serialize)]
struct ChatCompletionToolDefinition {
    #[serde(rename = "type")]
    tool_type: String,
    function: ChatCompletionToolFunction,
}

#[derive(Debug, Serialize)]
struct ChatCompletionToolFunction {
    name: String,
    description: String,
    parameters: Value,
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
    content: Option<Value>,
    #[serde(default)]
    tool_calls: Vec<ChatCompletionToolCall>,
}

#[async_trait]
impl ModelProvider for OpenRouterProvider {
    async fn complete(&self, request: ModelRequest) -> anyhow::Result<String> {
        self.complete_with_model(&self.model, request).await
    }

    async fn complete_turn(&self, request: ModelTurnRequest) -> anyhow::Result<ModelTurnResponse> {
        self.complete_turn_with_model(&self.model, request).await
    }
}

fn extract_message_content(content: Option<&Value>) -> Option<String> {
    let content = content?;
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
