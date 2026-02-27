use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserIdentity {
    pub discord_user_id: String,
    pub guild_id: String,
    pub aliases: Vec<String>,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageCtx {
    pub message_id: String,
    pub user_id: String,
    pub guild_id: String,
    pub channel_id: String,
    pub content: String,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFact {
    pub key: String,
    pub value: String,
    pub confidence: f32,
    pub source: String,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryContext {
    pub summary: Option<String>,
    pub recent_messages: Vec<String>,
    pub facts: Vec<MemoryFact>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub tool_name: String,
    pub args: serde_json::Value,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OrchestratorReply {
    pub text: String,
    pub citations: Vec<String>,
    pub tool_calls: Vec<ToolCall>,
    pub safety_flags: Vec<String>,
}
