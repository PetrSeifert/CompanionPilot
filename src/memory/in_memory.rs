use std::{
    collections::HashMap,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use async_trait::async_trait;
use chrono::Utc;
use tokio::sync::RwLock;

use crate::types::{
    ChatMessageRecord, MemoryContext, MemoryFact, PlannerDecisionRecord, ToolCallRecord,
    UserDashboardSummary,
};

use super::MemoryStore;

#[derive(Debug)]
pub struct InMemoryMemoryStore {
    facts: Arc<RwLock<HashMap<String, Vec<MemoryFact>>>>,
    summaries: Arc<RwLock<HashMap<String, String>>>,
    chats: Arc<RwLock<HashMap<String, Vec<ChatMessageRecord>>>>,
    tool_calls: Arc<RwLock<HashMap<String, Vec<ToolCallRecord>>>>,
    planner_decisions: Arc<RwLock<HashMap<String, Vec<PlannerDecisionRecord>>>>,
    chat_seq: AtomicU64,
}

impl Default for InMemoryMemoryStore {
    fn default() -> Self {
        Self {
            facts: Arc::new(RwLock::new(HashMap::new())),
            summaries: Arc::new(RwLock::new(HashMap::new())),
            chats: Arc::new(RwLock::new(HashMap::new())),
            tool_calls: Arc::new(RwLock::new(HashMap::new())),
            planner_decisions: Arc::new(RwLock::new(HashMap::new())),
            chat_seq: AtomicU64::new(1),
        }
    }
}

#[async_trait]
impl MemoryStore for InMemoryMemoryStore {
    async fn load_context(
        &self,
        user_id: &str,
        guild_id: &str,
        channel_id: &str,
    ) -> anyhow::Result<MemoryContext> {
        let facts = self
            .facts
            .read()
            .await
            .get(user_id)
            .cloned()
            .unwrap_or_default();
        let summary = self.summaries.read().await.get(user_id).cloned();
        let recent_messages = self
            .chats
            .read()
            .await
            .get(user_id)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .filter(|message| message.guild_id == guild_id && message.channel_id == channel_id)
            .rev()
            .take(8)
            .map(|message| format!("{}: {}", message.role.as_str(), message.content))
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>();

        Ok(MemoryContext {
            summary,
            recent_messages,
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

    async fn delete_fact(&self, user_id: &str, key: &str) -> anyhow::Result<bool> {
        let mut facts = self.facts.write().await;
        let Some(user_facts) = facts.get_mut(user_id) else {
            return Ok(false);
        };
        let initial_len = user_facts.len();
        user_facts.retain(|fact| fact.key != key);
        Ok(user_facts.len() != initial_len)
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

    async fn list_facts(&self, user_id: &str, limit: usize) -> anyhow::Result<Vec<MemoryFact>> {
        let mut facts = self
            .facts
            .read()
            .await
            .get(user_id)
            .cloned()
            .unwrap_or_default();
        facts.sort_by_key(|fact| std::cmp::Reverse(fact.updated_at));
        facts.truncate(limit);
        Ok(facts)
    }

    async fn record_chat_message(&self, message: ChatMessageRecord) -> anyhow::Result<()> {
        let user_id = message.user_id.clone();
        let mut chats = self.chats.write().await;
        let mut message = message;
        if message.id.is_empty() {
            let id = self.chat_seq.fetch_add(1, Ordering::Relaxed);
            message.id = format!("local-{id}");
        }
        chats.entry(user_id).or_default().push(message);
        Ok(())
    }

    async fn list_chat_messages(
        &self,
        user_id: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<ChatMessageRecord>> {
        let mut messages = self
            .chats
            .read()
            .await
            .get(user_id)
            .cloned()
            .unwrap_or_default();
        messages.sort_by_key(|message| message.timestamp);
        if messages.len() > limit {
            let start = messages.len().saturating_sub(limit);
            messages = messages.split_off(start);
        }
        Ok(messages)
    }

    async fn delete_chat_message(&self, user_id: &str, message_id: &str) -> anyhow::Result<bool> {
        let mut chats = self.chats.write().await;
        let Some(user_chats) = chats.get_mut(user_id) else {
            return Ok(false);
        };
        let initial_len = user_chats.len();
        user_chats.retain(|message| message.id != message_id);
        Ok(user_chats.len() != initial_len)
    }

    async fn clear_chat_messages(&self, user_id: &str) -> anyhow::Result<u64> {
        let mut chats = self.chats.write().await;
        let removed = chats
            .remove(user_id)
            .map(|list| list.len() as u64)
            .unwrap_or(0);
        Ok(removed)
    }

    async fn clear_tool_calls(&self, user_id: &str) -> anyhow::Result<u64> {
        let mut tool_calls = self.tool_calls.write().await;
        let removed = tool_calls
            .remove(user_id)
            .map(|list| list.len() as u64)
            .unwrap_or(0);
        Ok(removed)
    }

    async fn clear_planner_decisions(&self, user_id: &str) -> anyhow::Result<u64> {
        let mut decisions = self.planner_decisions.write().await;
        let removed = decisions
            .remove(user_id)
            .map(|list| list.len() as u64)
            .unwrap_or(0);
        Ok(removed)
    }

    async fn list_users(&self, limit: usize) -> anyhow::Result<Vec<UserDashboardSummary>> {
        let facts = self.facts.read().await;
        let chats = self.chats.read().await;

        let mut users = chats
            .iter()
            .map(|(user_id, messages)| {
                let last_activity = messages
                    .iter()
                    .map(|message| message.timestamp)
                    .max()
                    .unwrap_or_else(Utc::now);
                let message_count = messages.len() as i64;
                let fact_count = facts.get(user_id).map_or(0_i64, |f| f.len() as i64);

                UserDashboardSummary {
                    user_id: user_id.clone(),
                    fact_count,
                    message_count,
                    last_activity,
                }
            })
            .collect::<Vec<_>>();

        for (user_id, memory_facts) in facts.iter() {
            if users.iter().any(|entry| entry.user_id == *user_id) {
                continue;
            }
            users.push(UserDashboardSummary {
                user_id: user_id.clone(),
                fact_count: memory_facts.len() as i64,
                message_count: 0,
                last_activity: memory_facts
                    .iter()
                    .map(|fact| fact.updated_at)
                    .max()
                    .unwrap_or_else(Utc::now),
            });
        }

        users.sort_by_key(|entry| std::cmp::Reverse(entry.last_activity));
        users.truncate(limit);
        Ok(users)
    }

    async fn record_tool_call(&self, tool_call: ToolCallRecord) -> anyhow::Result<()> {
        let user_id = tool_call.user_id.clone();
        let mut tool_calls = self.tool_calls.write().await;
        tool_calls.entry(user_id).or_default().push(tool_call);
        Ok(())
    }

    async fn list_tool_calls(
        &self,
        user_id: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<ToolCallRecord>> {
        let mut calls = self
            .tool_calls
            .read()
            .await
            .get(user_id)
            .cloned()
            .unwrap_or_default();
        calls.sort_by_key(|call| call.timestamp);
        if calls.len() > limit {
            let start = calls.len().saturating_sub(limit);
            calls = calls.split_off(start);
        }
        Ok(calls)
    }

    async fn record_planner_decision(&self, decision: PlannerDecisionRecord) -> anyhow::Result<()> {
        let user_id = decision.user_id.clone();
        let mut decisions = self.planner_decisions.write().await;
        decisions.entry(user_id).or_default().push(decision);
        Ok(())
    }

    async fn list_planner_decisions(
        &self,
        user_id: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<PlannerDecisionRecord>> {
        let mut decisions = self
            .planner_decisions
            .read()
            .await
            .get(user_id)
            .cloned()
            .unwrap_or_default();
        decisions.sort_by_key(|decision| decision.timestamp);
        if decisions.len() > limit {
            let start = decisions.len().saturating_sub(limit);
            decisions = decisions.split_off(start);
        }
        Ok(decisions)
    }
}
