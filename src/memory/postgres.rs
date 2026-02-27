use async_trait::async_trait;
use sqlx::{PgPool, postgres::PgPoolOptions};

use crate::types::{MemoryContext, MemoryFact};

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

        Ok(MemoryContext {
            summary,
            recent_messages: Vec::new(),
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
}
