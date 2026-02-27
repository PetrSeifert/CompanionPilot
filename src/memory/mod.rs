mod in_memory;
mod postgres;

use async_trait::async_trait;

use crate::types::{
    ChatMessageRecord, MemoryContext, MemoryFact, PlannerDecisionRecord, ToolCallRecord,
    UserDashboardSummary,
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

    async fn record_chat_message(&self, message: ChatMessageRecord) -> anyhow::Result<()>;

    async fn list_chat_messages(
        &self,
        user_id: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<ChatMessageRecord>>;

    async fn list_users(&self, limit: usize) -> anyhow::Result<Vec<UserDashboardSummary>>;

    async fn record_tool_call(&self, tool_call: ToolCallRecord) -> anyhow::Result<()>;

    async fn list_tool_calls(
        &self,
        user_id: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<ToolCallRecord>>;

    async fn record_planner_decision(&self, decision: PlannerDecisionRecord) -> anyhow::Result<()>;

    async fn list_planner_decisions(
        &self,
        user_id: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<PlannerDecisionRecord>>;
}
