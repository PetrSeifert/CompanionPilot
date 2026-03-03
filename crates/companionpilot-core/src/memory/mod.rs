mod in_memory;
mod postgres;

use async_trait::async_trait;

use crate::types::{
    ChatMessageRecord, MemoryContext, MemoryFact, MessageLatencyRecord,
    OrchestrationDecisionRecord, ToolCallRecord, UserDashboardSummary,
};

pub use in_memory::InMemoryMemoryStore;
pub use postgres::PostgresMemoryStore;

#[async_trait]
pub trait MemoryStore: Send + Sync {
    async fn load_context(
        &self,
        user_id: &str,
        guild_id: &str,
        channel_id: &str,
    ) -> anyhow::Result<MemoryContext>;

    async fn upsert_fact(&self, user_id: &str, fact: MemoryFact) -> anyhow::Result<()>;

    async fn search_relevant(
        &self,
        user_id: &str,
        query: &str,
        k: usize,
    ) -> anyhow::Result<Vec<MemoryFact>>;

    async fn list_facts(&self, user_id: &str, limit: usize) -> anyhow::Result<Vec<MemoryFact>>;

    async fn delete_fact(&self, user_id: &str, key: &str) -> anyhow::Result<bool>;

    async fn record_chat_message(&self, message: ChatMessageRecord) -> anyhow::Result<()>;

    async fn list_chat_messages(
        &self,
        user_id: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<ChatMessageRecord>>;

    async fn delete_chat_message(&self, user_id: &str, message_id: &str) -> anyhow::Result<bool>;

    async fn clear_chat_messages(&self, user_id: &str) -> anyhow::Result<u64>;

    async fn clear_tool_calls(&self, user_id: &str) -> anyhow::Result<u64>;

    async fn clear_facts(&self, user_id: &str) -> anyhow::Result<u64>;

    async fn clear_orchestration_decisions(&self, user_id: &str) -> anyhow::Result<u64>;

    async fn list_users(&self, limit: usize) -> anyhow::Result<Vec<UserDashboardSummary>>;

    async fn record_tool_call(&self, tool_call: ToolCallRecord) -> anyhow::Result<()>;

    async fn list_tool_calls(
        &self,
        user_id: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<ToolCallRecord>>;

    async fn record_orchestration_decision(
        &self,
        decision: OrchestrationDecisionRecord,
    ) -> anyhow::Result<()>;

    async fn list_orchestration_decisions(
        &self,
        user_id: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<OrchestrationDecisionRecord>>;

    async fn record_message_latency(&self, latency: MessageLatencyRecord) -> anyhow::Result<()>;

    async fn list_message_latencies(
        &self,
        user_id: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<MessageLatencyRecord>>;

    async fn clear_message_latencies(&self, user_id: &str) -> anyhow::Result<u64>;
}
