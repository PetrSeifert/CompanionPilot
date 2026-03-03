mod mock;
mod openrouter;
mod runtime;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub use mock::MockModelProvider;
pub use openrouter::OpenRouterProvider;
pub use runtime::RuntimeModelProvider;

#[derive(Debug, Clone)]
pub struct ModelRequest {
    pub system_prompt: String,
    pub user_prompt: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ModelMessageRole {
    System,
    User,
    Assistant,
    Tool,
}

impl ModelMessageRole {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMessage {
    pub role: ModelMessageRole,
    pub content: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
    #[serde(default)]
    pub tool_calls: Vec<ModelToolCall>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelTurnRequest {
    pub system_prompt: String,
    pub messages: Vec<ModelMessage>,
    pub tools: Vec<ModelToolDefinition>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelTurnResponse {
    #[serde(default)]
    pub assistant_text: String,
    #[serde(default)]
    pub tool_calls: Vec<ModelToolCall>,
}

#[async_trait]
pub trait ModelProvider: Send + Sync {
    async fn complete(&self, request: ModelRequest) -> anyhow::Result<String>;
    async fn complete_turn(&self, request: ModelTurnRequest) -> anyhow::Result<ModelTurnResponse>;
}
