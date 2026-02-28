use async_trait::async_trait;
use sqlx::{PgPool, postgres::PgPoolOptions};

use crate::types::{
    ChatMessageRecord, ChatRole, MemoryContext, MemoryFact, PlannerDecisionRecord, ToolCallRecord,
    UserDashboardSummary,
};

use super::MemoryStore;

#[derive(Debug, Clone)]
pub struct PostgresMemoryStore {
    pool: PgPool,
}

impl PostgresMemoryStore {
    pub async fn connect(database_url: &str) -> anyhow::Result<Self> {
        let pool = PgPoolOptions::new()
            .max_connections(10)
            .connect(database_url)
            .await?;
        Ok(Self { pool })
    }
}

#[async_trait]
impl MemoryStore for PostgresMemoryStore {
    async fn load_context(
        &self,
        user_id: &str,
        guild_id: &str,
        channel_id: &str,
    ) -> anyhow::Result<MemoryContext> {
        let facts =
            sqlx::query_as::<_, (String, String, f32, String, chrono::DateTime<chrono::Utc>)>(
                "SELECT key, value, confidence, source, updated_at
             FROM memory_facts
             WHERE user_id = $1
             ORDER BY updated_at DESC
             LIMIT 32",
            )
            .bind(user_id)
            .fetch_all(&self.pool)
            .await?
            .into_iter()
            .map(|(key, value, confidence, source, updated_at)| MemoryFact {
                key,
                value,
                confidence,
                source,
                updated_at,
            })
            .collect::<Vec<_>>();

        let summary = sqlx::query_as::<_, (String,)>(
            "SELECT summary
             FROM message_summaries
             WHERE user_id = $1 AND guild_id = $2 AND channel_id = $3
             ORDER BY updated_at DESC
             LIMIT 1",
        )
        .bind(user_id)
        .bind(guild_id)
        .bind(channel_id)
        .fetch_optional(&self.pool)
        .await?
        .map(|row| row.0);

        let recent_messages = sqlx::query_as::<_, (String, String)>(
            "SELECT role, content
             FROM chat_messages
             WHERE user_id = $1 AND guild_id = $2 AND channel_id = $3
             ORDER BY timestamp DESC
             LIMIT 8",
        )
        .bind(user_id)
        .bind(guild_id)
        .bind(channel_id)
        .fetch_all(&self.pool)
        .await?
        .into_iter()
        .rev()
        .map(|(role, content)| format!("{role}: {content}"))
        .collect::<Vec<_>>();

        Ok(MemoryContext {
            summary,
            recent_messages,
            facts,
        })
    }

    async fn upsert_fact(&self, user_id: &str, fact: MemoryFact) -> anyhow::Result<()> {
        sqlx::query(
            "INSERT INTO memory_facts (user_id, key, value, confidence, source, updated_at)
             VALUES ($1, $2, $3, $4, $5, $6)
             ON CONFLICT (user_id, key)
             DO UPDATE SET value = EXCLUDED.value, confidence = EXCLUDED.confidence, source = EXCLUDED.source, updated_at = EXCLUDED.updated_at",
        )
        .bind(user_id)
        .bind(fact.key)
        .bind(fact.value)
        .bind(fact.confidence)
        .bind(fact.source)
        .bind(fact.updated_at)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    async fn delete_fact(&self, user_id: &str, key: &str) -> anyhow::Result<bool> {
        let result = sqlx::query("DELETE FROM memory_facts WHERE user_id = $1 AND key = $2")
            .bind(user_id)
            .bind(key)
            .execute(&self.pool)
            .await?;
        Ok(result.rows_affected() > 0)
    }

    async fn search_relevant(
        &self,
        user_id: &str,
        query: &str,
        k: usize,
    ) -> anyhow::Result<Vec<MemoryFact>> {
        let query = format!("%{}%", query.to_lowercase());
        let limit = k as i64;

        let facts =
            sqlx::query_as::<_, (String, String, f32, String, chrono::DateTime<chrono::Utc>)>(
                "SELECT key, value, confidence, source, updated_at
             FROM memory_facts
             WHERE user_id = $1
               AND (LOWER(key) LIKE $2 OR LOWER(value) LIKE $2)
             ORDER BY updated_at DESC
             LIMIT $3",
            )
            .bind(user_id)
            .bind(query)
            .bind(limit)
            .fetch_all(&self.pool)
            .await?
            .into_iter()
            .map(|(key, value, confidence, source, updated_at)| MemoryFact {
                key,
                value,
                confidence,
                source,
                updated_at,
            })
            .collect();

        Ok(facts)
    }

    async fn list_facts(&self, user_id: &str, limit: usize) -> anyhow::Result<Vec<MemoryFact>> {
        let limit = limit as i64;

        let facts =
            sqlx::query_as::<_, (String, String, f32, String, chrono::DateTime<chrono::Utc>)>(
                "SELECT key, value, confidence, source, updated_at
                 FROM memory_facts
                 WHERE user_id = $1
                 ORDER BY updated_at DESC
                 LIMIT $2",
            )
            .bind(user_id)
            .bind(limit)
            .fetch_all(&self.pool)
            .await?
            .into_iter()
            .map(|(key, value, confidence, source, updated_at)| MemoryFact {
                key,
                value,
                confidence,
                source,
                updated_at,
            })
            .collect::<Vec<_>>();

        Ok(facts)
    }

    async fn record_chat_message(&self, message: ChatMessageRecord) -> anyhow::Result<()> {
        sqlx::query(
            "INSERT INTO chat_messages
             (user_id, guild_id, channel_id, role, content, timestamp)
             VALUES ($1, $2, $3, $4, $5, $6)",
        )
        .bind(message.user_id)
        .bind(message.guild_id)
        .bind(message.channel_id)
        .bind(message.role.as_str())
        .bind(message.content)
        .bind(message.timestamp)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    async fn list_chat_messages(
        &self,
        user_id: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<ChatMessageRecord>> {
        let limit = limit as i64;

        let mut messages = sqlx::query_as::<
            _,
            (
                i64,
                String,
                String,
                String,
                String,
                String,
                chrono::DateTime<chrono::Utc>,
            ),
        >(
            "SELECT id, user_id, guild_id, channel_id, role, content, timestamp
             FROM chat_messages
             WHERE user_id = $1
             ORDER BY timestamp DESC
             LIMIT $2",
        )
        .bind(user_id)
        .bind(limit)
        .fetch_all(&self.pool)
        .await?
        .into_iter()
        .map(
            |(id, user_id, guild_id, channel_id, role, content, timestamp)| ChatMessageRecord {
                id: id.to_string(),
                user_id,
                guild_id,
                channel_id,
                role: parse_role(&role),
                content,
                timestamp,
            },
        )
        .collect::<Vec<_>>();

        messages.reverse();
        Ok(messages)
    }

    async fn delete_chat_message(&self, user_id: &str, message_id: &str) -> anyhow::Result<bool> {
        let id = match message_id.parse::<i64>() {
            Ok(value) => value,
            Err(_) => return Ok(false),
        };
        let result = sqlx::query("DELETE FROM chat_messages WHERE user_id = $1 AND id = $2")
            .bind(user_id)
            .bind(id)
            .execute(&self.pool)
            .await?;
        Ok(result.rows_affected() > 0)
    }

    async fn clear_chat_messages(&self, user_id: &str) -> anyhow::Result<u64> {
        let result = sqlx::query("DELETE FROM chat_messages WHERE user_id = $1")
            .bind(user_id)
            .execute(&self.pool)
            .await?;
        Ok(result.rows_affected())
    }

    async fn clear_facts(&self, user_id: &str) -> anyhow::Result<u64> {
        let result = sqlx::query("DELETE FROM memory_facts WHERE user_id = $1")
            .bind(user_id)
            .execute(&self.pool)
            .await?;
        Ok(result.rows_affected())
    }

    async fn clear_tool_calls(&self, user_id: &str) -> anyhow::Result<u64> {
        let result = sqlx::query("DELETE FROM tool_call_logs WHERE user_id = $1")
            .bind(user_id)
            .execute(&self.pool)
            .await?;
        Ok(result.rows_affected())
    }

    async fn clear_planner_decisions(&self, user_id: &str) -> anyhow::Result<u64> {
        let result = sqlx::query("DELETE FROM planner_decision_logs WHERE user_id = $1")
            .bind(user_id)
            .execute(&self.pool)
            .await?;
        Ok(result.rows_affected())
    }

    async fn list_users(&self, limit: usize) -> anyhow::Result<Vec<UserDashboardSummary>> {
        let limit = limit as i64;

        let users = sqlx::query_as::<_, (String, i64, i64, chrono::DateTime<chrono::Utc>)>(
            "SELECT
                     u.user_id AS user_id,
                     COALESCE(f.fact_count, 0) AS fact_count,
                     COALESCE(m.message_count, 0) AS message_count,
                     u.last_activity AS last_activity
                 FROM (
                     SELECT user_id, MAX(last_activity) AS last_activity
                     FROM (
                         SELECT user_id, MAX(updated_at) AS last_activity
                         FROM memory_facts
                         GROUP BY user_id
                         UNION ALL
                         SELECT user_id, MAX(timestamp) AS last_activity
                         FROM chat_messages
                         GROUP BY user_id
                     ) activity
                     GROUP BY user_id
                 ) u
                 LEFT JOIN (
                     SELECT user_id, COUNT(*)::bigint AS fact_count
                     FROM memory_facts
                     GROUP BY user_id
                 ) f ON f.user_id = u.user_id
                 LEFT JOIN (
                     SELECT user_id, COUNT(*)::bigint AS message_count
                     FROM chat_messages
                     GROUP BY user_id
                 ) m ON m.user_id = u.user_id
                 ORDER BY u.last_activity DESC
                 LIMIT $1",
        )
        .bind(limit)
        .fetch_all(&self.pool)
        .await?
        .into_iter()
        .map(
            |(user_id, fact_count, message_count, last_activity)| UserDashboardSummary {
                user_id,
                fact_count,
                message_count,
                last_activity,
            },
        )
        .collect::<Vec<_>>();

        Ok(users)
    }

    async fn record_tool_call(&self, tool_call: ToolCallRecord) -> anyhow::Result<()> {
        sqlx::query(
            "INSERT INTO tool_call_logs
             (user_id, guild_id, channel_id, tool_name, source, args_json, result_text, citations_text, success, error, timestamp)
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)",
        )
        .bind(tool_call.user_id)
        .bind(tool_call.guild_id)
        .bind(tool_call.channel_id)
        .bind(tool_call.tool_name)
        .bind(tool_call.source)
        .bind(tool_call.args_json)
        .bind(tool_call.result_text)
        .bind(tool_call.citations.join("\n"))
        .bind(tool_call.success)
        .bind(tool_call.error)
        .bind(tool_call.timestamp)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    async fn list_tool_calls(
        &self,
        user_id: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<ToolCallRecord>> {
        let limit = limit as i64;
        let mut calls = sqlx::query_as::<
            _,
            (
                String,
                String,
                String,
                String,
                String,
                String,
                String,
                String,
                bool,
                Option<String>,
                chrono::DateTime<chrono::Utc>,
            ),
        >(
            "SELECT user_id, guild_id, channel_id, tool_name, source, args_json, result_text, citations_text, success, error, timestamp
             FROM tool_call_logs
             WHERE user_id = $1
             ORDER BY timestamp DESC
             LIMIT $2",
        )
        .bind(user_id)
        .bind(limit)
        .fetch_all(&self.pool)
        .await?
        .into_iter()
        .map(
            |(
                user_id,
                guild_id,
                channel_id,
                tool_name,
                source,
                args_json,
                result_text,
                citations_text,
                success,
                error,
                timestamp,
            )| ToolCallRecord {
                user_id,
                guild_id,
                channel_id,
                tool_name,
                source,
                args_json,
                result_text,
                citations: split_citations(&citations_text),
                success,
                error,
                timestamp,
            },
        )
        .collect::<Vec<_>>();

        calls.reverse();
        Ok(calls)
    }

    async fn record_planner_decision(&self, decision: PlannerDecisionRecord) -> anyhow::Result<()> {
        sqlx::query(
            "INSERT INTO planner_decision_logs
             (user_id, guild_id, channel_id, planner, decision, rationale, payload_json, success, error, timestamp)
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)",
        )
        .bind(decision.user_id)
        .bind(decision.guild_id)
        .bind(decision.channel_id)
        .bind(decision.planner)
        .bind(decision.decision)
        .bind(decision.rationale)
        .bind(decision.payload_json)
        .bind(decision.success)
        .bind(decision.error)
        .bind(decision.timestamp)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    async fn list_planner_decisions(
        &self,
        user_id: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<PlannerDecisionRecord>> {
        let limit = limit as i64;
        let mut decisions = sqlx::query_as::<
            _,
            (
                String,
                String,
                String,
                String,
                String,
                String,
                String,
                bool,
                Option<String>,
                chrono::DateTime<chrono::Utc>,
            ),
        >(
            "SELECT user_id, guild_id, channel_id, planner, decision, rationale, payload_json, success, error, timestamp
             FROM planner_decision_logs
             WHERE user_id = $1
             ORDER BY timestamp DESC
             LIMIT $2",
        )
        .bind(user_id)
        .bind(limit)
        .fetch_all(&self.pool)
        .await?
        .into_iter()
        .map(
            |(
                user_id,
                guild_id,
                channel_id,
                planner,
                decision,
                rationale,
                payload_json,
                success,
                error,
                timestamp,
            )| PlannerDecisionRecord {
                user_id,
                guild_id,
                channel_id,
                planner,
                decision,
                rationale,
                payload_json,
                success,
                error,
                timestamp,
            },
        )
        .collect::<Vec<_>>();

        decisions.reverse();
        Ok(decisions)
    }
}

fn parse_role(role: &str) -> ChatRole {
    match role {
        "assistant" => ChatRole::Assistant,
        _ => ChatRole::User,
    }
}

fn split_citations(raw: &str) -> Vec<String> {
    raw.split('\n')
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(|line| line.to_owned())
        .collect()
}
