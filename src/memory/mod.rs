mod in_memory;
mod postgres;

use async_trait::async_trait;

use crate::types::{MemoryContext, MemoryFact};

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
}
