use std::{collections::HashMap, sync::Arc};

use async_trait::async_trait;
use tokio::sync::RwLock;

use crate::types::{MemoryContext, MemoryFact};

use super::MemoryStore;

#[derive(Debug, Default)]
pub struct InMemoryMemoryStore {
    facts: Arc<RwLock<HashMap<String, Vec<MemoryFact>>>>,
    summaries: Arc<RwLock<HashMap<String, String>>>,
}

#[async_trait]
impl MemoryStore for InMemoryMemoryStore {
    async fn load_context(
        &self,
        user_id: &str,
        _guild_id: &str,
        _channel_id: &str,
    ) -> anyhow::Result<MemoryContext> {
        let facts = self
            .facts
            .read()
            .await
            .get(user_id)
            .cloned()
            .unwrap_or_default();
        let summary = self.summaries.read().await.get(user_id).cloned();

        Ok(MemoryContext {
            summary,
            recent_messages: Vec::new(),
            facts,
        })
    }

    async fn upsert_fact(&self, user_id: &str, fact: MemoryFact) -> anyhow::Result<()> {
        let mut facts = self.facts.write().await;
        let user_facts = facts.entry(user_id.to_owned()).or_default();

        if let Some(existing) = user_facts.iter_mut().find(|item| item.key == fact.key) {
            *existing = fact;
        } else {
            user_facts.push(fact);
        }

        Ok(())
    }

    async fn search_relevant(
        &self,
        user_id: &str,
        query: &str,
        k: usize,
    ) -> anyhow::Result<Vec<MemoryFact>> {
        let facts = self.facts.read().await;
        let mut matches = facts
            .get(user_id)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .filter(|fact| {
                fact.key.to_lowercase().contains(&query.to_lowercase())
                    || fact.value.to_lowercase().contains(&query.to_lowercase())
            })
            .collect::<Vec<_>>();

        matches.truncate(k);
        Ok(matches)
    }
}
