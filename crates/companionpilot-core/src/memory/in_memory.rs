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
    ChatMessageRecord, MemoryContext, MemoryFact, MessageLatencyRecord, PlannerDecisionRecord,
    ToolCallRecord, UserDashboardSummary,
};

use super::MemoryStore;

const MAX_FACTS_PER_USER: usize = 512;
const MAX_CHAT_MESSAGES_PER_USER: usize = 2_048;
const MAX_TOOL_CALLS_PER_USER: usize = 1_024;
const MAX_PLANNER_DECISIONS_PER_USER: usize = 1_024;
const MAX_MESSAGE_LATENCIES_PER_USER: usize = 2_048;

#[derive(Debug)]
pub struct InMemoryMemoryStore {
    facts: Arc<RwLock<HashMap<String, Vec<MemoryFact>>>>,
    summaries: Arc<RwLock<HashMap<String, String>>>,
    chats: Arc<RwLock<HashMap<String, Vec<ChatMessageRecord>>>>,
    tool_calls: Arc<RwLock<HashMap<String, Vec<ToolCallRecord>>>>,
    planner_decisions: Arc<RwLock<HashMap<String, Vec<PlannerDecisionRecord>>>>,
    message_latencies: Arc<RwLock<HashMap<String, Vec<MessageLatencyRecord>>>>,
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
            message_latencies: Arc::new(RwLock::new(HashMap::new())),
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
        prune_oldest_by(user_facts, MAX_FACTS_PER_USER, |entry| entry.updated_at);

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
        let entries = chats.entry(user_id).or_default();
        entries.push(message);
        prune_oldest_by(entries, MAX_CHAT_MESSAGES_PER_USER, |entry| entry.timestamp);
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

    async fn clear_facts(&self, user_id: &str) -> anyhow::Result<u64> {
        let mut facts = self.facts.write().await;
        let removed = facts
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
        let entries = tool_calls.entry(user_id).or_default();
        entries.push(tool_call);
        prune_oldest_by(entries, MAX_TOOL_CALLS_PER_USER, |entry| entry.timestamp);
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
        let entries = decisions.entry(user_id).or_default();
        entries.push(decision);
        prune_oldest_by(entries, MAX_PLANNER_DECISIONS_PER_USER, |entry| entry.timestamp);
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

    async fn record_message_latency(&self, latency: MessageLatencyRecord) -> anyhow::Result<()> {
        let user_id = latency.user_id.clone();
        let mut latencies = self.message_latencies.write().await;
        let entries = latencies.entry(user_id).or_default();
        if let Some(existing) = entries
            .iter_mut()
            .find(|entry| entry.message_id == latency.message_id)
        {
            existing.stt_ms = latency.stt_ms.or(existing.stt_ms);
            existing.tts_ms = latency.tts_ms.or(existing.tts_ms);
            existing.decision_ms = existing.decision_ms.max(latency.decision_ms);
            existing.tool_call_ms = existing.tool_call_ms.max(latency.tool_call_ms);
            existing.final_response_ms = existing.final_response_ms.max(latency.final_response_ms);
            existing.timestamp = existing.timestamp.max(latency.timestamp);
        } else {
            entries.push(latency);
        }
        prune_oldest_by(entries, MAX_MESSAGE_LATENCIES_PER_USER, |entry| entry.timestamp);
        Ok(())
    }

    async fn list_message_latencies(
        &self,
        user_id: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<MessageLatencyRecord>> {
        let mut latencies = self
            .message_latencies
            .read()
            .await
            .get(user_id)
            .cloned()
            .unwrap_or_default();
        latencies.sort_by_key(|latency| latency.timestamp);
        if latencies.len() > limit {
            let start = latencies.len().saturating_sub(limit);
            latencies = latencies.split_off(start);
        }
        Ok(latencies)
    }

    async fn clear_message_latencies(&self, user_id: &str) -> anyhow::Result<u64> {
        let mut latencies = self.message_latencies.write().await;
        let removed = latencies
            .remove(user_id)
            .map(|list| list.len() as u64)
            .unwrap_or(0);
        Ok(removed)
    }
}

fn prune_oldest_by<T, F>(entries: &mut Vec<T>, max_len: usize, mut timestamp_of: F)
where
    F: FnMut(&T) -> chrono::DateTime<chrono::Utc>,
{
    if entries.len() <= max_len {
        return;
    }
    entries.sort_by_key(|entry| timestamp_of(entry));
    let excess = entries.len().saturating_sub(max_len);
    entries.drain(0..excess);
}

#[cfg(test)]
mod tests {
    use chrono::{Duration, Utc};

    use crate::{
        memory::MemoryStore,
        types::{ChatMessageRecord, ChatRole, MemoryFact, MessageLatencyRecord},
    };

    use super::{InMemoryMemoryStore, MAX_CHAT_MESSAGES_PER_USER, MAX_FACTS_PER_USER};

    #[tokio::test]
    async fn record_chat_message_prunes_oldest_entries() {
        let store = InMemoryMemoryStore::default();
        let base = Utc::now();
        for idx in 0..(MAX_CHAT_MESSAGES_PER_USER + 3) {
            store
                .record_chat_message(ChatMessageRecord {
                    id: format!("m-{idx}"),
                    user_id: "u1".to_owned(),
                    guild_id: "g1".to_owned(),
                    channel_id: "c1".to_owned(),
                    role: ChatRole::User,
                    content: format!("message {idx}"),
                    timestamp: base + Duration::milliseconds(idx as i64),
                })
                .await
                .expect("chat message should record");
        }

        let messages = store
            .list_chat_messages("u1", usize::MAX)
            .await
            .expect("chat messages should list");
        assert_eq!(messages.len(), MAX_CHAT_MESSAGES_PER_USER);
        assert_eq!(messages.first().map(|entry| entry.id.as_str()), Some("m-3"));
    }

    #[tokio::test]
    async fn upsert_fact_prunes_oldest_entries() {
        let store = InMemoryMemoryStore::default();
        let base = Utc::now();
        for idx in 0..(MAX_FACTS_PER_USER + 2) {
            store
                .upsert_fact(
                    "u1",
                    MemoryFact {
                        key: format!("key-{idx}"),
                        value: format!("value-{idx}"),
                        confidence: 1.0,
                        source: "test".to_owned(),
                        updated_at: base + Duration::seconds(idx as i64),
                    },
                )
                .await
                .expect("fact should upsert");
        }

        let facts = store
            .list_facts("u1", usize::MAX)
            .await
            .expect("facts should list");
        assert_eq!(facts.len(), MAX_FACTS_PER_USER);
        assert!(!facts.iter().any(|entry| entry.key == "key-0"));
        assert!(!facts.iter().any(|entry| entry.key == "key-1"));
    }

    #[tokio::test]
    async fn clear_chat_messages_does_not_clear_message_latencies() {
        let store = InMemoryMemoryStore::default();
        let now = Utc::now();
        store
            .record_chat_message(ChatMessageRecord {
                id: "m1".to_owned(),
                user_id: "u1".to_owned(),
                guild_id: "g1".to_owned(),
                channel_id: "c1".to_owned(),
                role: ChatRole::User,
                content: "hello".to_owned(),
                timestamp: now,
            })
            .await
            .expect("chat message should record");
        store
            .record_message_latency(MessageLatencyRecord {
                user_id: "u1".to_owned(),
                guild_id: "g1".to_owned(),
                channel_id: "c1".to_owned(),
                message_id: "m1".to_owned(),
                stt_ms: Some(10),
                tts_ms: None,
                final_response_ms: 100,
                decision_ms: 20,
                tool_call_ms: 30,
                timestamp: now,
            })
            .await
            .expect("latency should record");

        store
            .clear_chat_messages("u1")
            .await
            .expect("chat messages should clear");
        let latencies = store
            .list_message_latencies("u1", usize::MAX)
            .await
            .expect("latencies should list");

        assert_eq!(latencies.len(), 1);
        assert_eq!(latencies[0].message_id, "m1");
    }
}
