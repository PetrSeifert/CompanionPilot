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

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ChatRole {
    User,
    Assistant,
}

impl ChatRole {
    pub fn as_str(self) -> &'static str {
        match self {
            ChatRole::User => "user",
            ChatRole::Assistant => "assistant",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessageRecord {
    pub id: String,
    pub user_id: String,
    pub guild_id: String,
    pub channel_id: String,
    pub role: ChatRole,
    pub content: String,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserDashboardSummary {
    pub user_id: String,
    pub fact_count: i64,
    pub message_count: i64,
    pub last_activity: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallRecord {
    pub user_id: String,
    pub guild_id: String,
    pub channel_id: String,
    pub tool_name: String,
    pub source: String,
    pub args_json: String,
    pub result_text: String,
    pub citations: Vec<String>,
    pub success: bool,
    pub error: Option<String>,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlannerDecisionRecord {
    pub user_id: String,
    pub guild_id: String,
    pub channel_id: String,
    pub planner: String,
    pub decision: String,
    pub rationale: String,
    pub payload_json: String,
    pub success: bool,
    pub error: Option<String>,
    pub timestamp: DateTime<Utc>,
}
