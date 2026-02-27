use std::sync::Arc;

use chrono::Utc;
use serde_json::json;

use crate::{
    memory::MemoryStore,
    model::{ModelProvider, ModelRequest},
    safety::SafetyPolicy,
    tools::ToolExecutor,
    types::{ChatMessageRecord, ChatRole, MemoryFact, MessageCtx, OrchestratorReply, ToolCall},
};

pub struct DefaultChatOrchestrator {
    model: Arc<dyn ModelProvider>,
    memory: Arc<dyn MemoryStore>,
    tools: Arc<dyn ToolExecutor>,
    safety: SafetyPolicy,
}

impl DefaultChatOrchestrator {
    pub fn new(
        model: Arc<dyn ModelProvider>,
        memory: Arc<dyn MemoryStore>,
        tools: Arc<dyn ToolExecutor>,
        safety: SafetyPolicy,
    ) -> Self {
        Self {
            model,
            memory,
            tools,
            safety,
        }
    }

    pub async fn handle_message(&self, ctx: MessageCtx) -> anyhow::Result<OrchestratorReply> {
        self.memory
            .record_chat_message(ChatMessageRecord {
                user_id: ctx.user_id.clone(),
                guild_id: ctx.guild_id.clone(),
                channel_id: ctx.channel_id.clone(),
                role: ChatRole::User,
                content: ctx.content.clone(),
                timestamp: ctx.timestamp,
            })
            .await?;

        let safety_flags = self.safety.validate_user_message(&ctx.content);
        let memory_context = self
            .memory
            .load_context(&ctx.user_id, &ctx.guild_id, &ctx.channel_id)
            .await?;

        if let Some(query) = parse_search_command(&ctx.content) {
            let args = json!({
                "query": query,
                "max_results": 5
            });
            let tool_result = self.tools.execute("web_search", args.clone()).await?;
            let reply = OrchestratorReply {
                text: tool_result.text,
                citations: tool_result.citations,
                tool_calls: vec![ToolCall {
                    tool_name: "web_search".to_owned(),
                    args,
                }],
                safety_flags,
            };
            self.memory
                .record_chat_message(ChatMessageRecord {
                    user_id: ctx.user_id,
                    guild_id: ctx.guild_id,
                    channel_id: ctx.channel_id,
                    role: ChatRole::Assistant,
                    content: reply.text.clone(),
                    timestamp: Utc::now(),
                })
                .await?;
            return Ok(reply);
        }

        let system_prompt = build_system_prompt(&memory_context);
        let model_response = self
            .model
            .complete(ModelRequest {
                system_prompt,
                user_prompt: ctx.content.clone(),
            })
            .await?;

        if let Some(fact) = extract_memory_fact(&ctx.content) {
            self.memory.upsert_fact(&ctx.user_id, fact).await?;
        }

        let reply = OrchestratorReply {
            text: model_response,
            citations: Vec::new(),
            tool_calls: Vec::new(),
            safety_flags,
        };

        self.memory
            .record_chat_message(ChatMessageRecord {
                user_id: ctx.user_id,
                guild_id: ctx.guild_id,
                channel_id: ctx.channel_id,
                role: ChatRole::Assistant,
                content: reply.text.clone(),
                timestamp: Utc::now(),
            })
            .await?;

        Ok(reply)
    }
}

fn parse_search_command(content: &str) -> Option<&str> {
    let trimmed = content.trim();
    trimmed
        .strip_prefix("/search ")
        .map(str::trim)
        .filter(|query| !query.is_empty())
}

fn build_system_prompt(memory: &crate::types::MemoryContext) -> String {
    let mut sections = vec![
        "You are CompanionPilot, a helpful Discord AI companion.".to_owned(),
        "Keep replies concise and practical.".to_owned(),
    ];

    if let Some(summary) = &memory.summary {
        sections.push(format!("Conversation summary: {summary}"));
    }

    if !memory.facts.is_empty() {
        let lines = memory
            .facts
            .iter()
            .map(|fact| format!("{} = {}", fact.key, fact.value))
            .collect::<Vec<_>>()
            .join("; ");
        sections.push(format!("Known user facts: {lines}"));
    }

    sections.join("\n")
}

fn extract_memory_fact(input: &str) -> Option<MemoryFact> {
    let lower = input.to_lowercase();
    if let Some(name) = lower.strip_prefix("my name is ") {
        return Some(MemoryFact {
            key: "name".to_owned(),
            value: name.trim().to_owned(),
            confidence: 0.95,
            source: "user_message".to_owned(),
            updated_at: Utc::now(),
        });
    }
    if let Some(game) = lower.strip_prefix("i play ") {
        return Some(MemoryFact {
            key: "favorite_game".to_owned(),
            value: game.trim().to_owned(),
            confidence: 0.8,
            source: "user_message".to_owned(),
            updated_at: Utc::now(),
        });
    }
    None
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use chrono::Utc;
    use serde_json::json;

    use crate::{
        memory::{InMemoryMemoryStore, MemoryStore},
        model::MockModelProvider,
        safety::SafetyPolicy,
        tools::ToolRegistry,
        types::MessageCtx,
    };

    use super::DefaultChatOrchestrator;

    #[tokio::test]
    async fn persists_simple_name_fact() {
        let memory = Arc::new(InMemoryMemoryStore::default());
        let orchestrator = DefaultChatOrchestrator::new(
            Arc::new(MockModelProvider),
            memory.clone(),
            Arc::new(ToolRegistry::default()),
            SafetyPolicy::default(),
        );

        let _ = orchestrator
            .handle_message(MessageCtx {
                message_id: "1".into(),
                user_id: "u1".into(),
                guild_id: "g1".into(),
                channel_id: "c1".into(),
                content: "my name is petr".into(),
                timestamp: Utc::now(),
            })
            .await
            .expect("handle message should succeed");

        let facts = memory
            .search_relevant("u1", "name", 10)
            .await
            .expect("search should succeed");
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].key, "name");
    }

    #[tokio::test]
    async fn invokes_search_tool_on_command() {
        let memory = Arc::new(InMemoryMemoryStore::default());
        let orchestrator = DefaultChatOrchestrator::new(
            Arc::new(MockModelProvider),
            memory,
            Arc::new(ToolRegistry::default()),
            SafetyPolicy::default(),
        );

        let result = orchestrator
            .handle_message(MessageCtx {
                message_id: "2".into(),
                user_id: "u2".into(),
                guild_id: "g1".into(),
                channel_id: "c1".into(),
                content: "/search rust".into(),
                timestamp: Utc::now(),
            })
            .await;

        assert!(result.is_err());
        let err = result.err().map(|e| e.to_string()).unwrap_or_default();
        assert!(err.contains("web_search tool is not configured"));

        let _ = json!({});
    }
}
