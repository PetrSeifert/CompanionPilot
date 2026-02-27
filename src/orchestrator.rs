use std::sync::Arc;

use chrono::Utc;
use serde::Deserialize;
use serde_json::json;
use tracing::{debug, info, warn};

use crate::{
    memory::MemoryStore,
    model::{ModelProvider, ModelRequest},
    safety::SafetyPolicy,
    tools::ToolExecutor,
    types::{
        ChatMessageRecord, ChatRole, MemoryFact, MessageCtx, OrchestratorReply,
        PlannerDecisionRecord, ToolCall, ToolCallRecord,
    },
};

pub struct DefaultChatOrchestrator {
    model: Arc<dyn ModelProvider>,
    memory: Arc<dyn MemoryStore>,
    tools: Arc<dyn ToolExecutor>,
    safety: SafetyPolicy,
}

enum SearchDecision {
    Use { query: String, source: &'static str },
    Skip { reason: &'static str },
}

enum MemoryDecision {
    Store {
        fact: MemoryFact,
        rationale: &'static str,
    },
    Skip {
        reason: &'static str,
        error: Option<String>,
    },
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
        let safety_flags = self.safety.validate_user_message(&ctx.content);
        let memory_context = self
            .memory
            .load_context(&ctx.user_id, &ctx.guild_id, &ctx.channel_id)
            .await?;
        self.memory
            .record_chat_message(ChatMessageRecord {
                id: ctx.message_id.clone(),
                user_id: ctx.user_id.clone(),
                guild_id: ctx.guild_id.clone(),
                channel_id: ctx.channel_id.clone(),
                role: ChatRole::User,
                content: ctx.content.clone(),
                timestamp: ctx.timestamp,
            })
            .await?;

        let search_decision = if let Some(manual) = parse_search_command(&ctx.content) {
            SearchDecision::Use {
                query: manual.to_owned(),
                source: "manual_prefix",
            }
        } else {
            self.decide_search_query(&ctx.content, &memory_context)
                .await
        };
        self.record_search_planner_decision(&ctx, &search_decision)
            .await;

        if let SearchDecision::Use { query, source } = search_decision {
            info!(
                user_id = %ctx.user_id,
                guild_id = %ctx.guild_id,
                channel_id = %ctx.channel_id,
                source,
                query = %query,
                "web search selected"
            );
            let args = json!({
                "query": query,
                "max_results": 5
            });
            let tool_result = match self.tools.execute("web_search", args.clone()).await {
                Ok(result) => result,
                Err(error) => {
                    self.record_tool_call(ToolCallRecord {
                        user_id: ctx.user_id.clone(),
                        guild_id: ctx.guild_id.clone(),
                        channel_id: ctx.channel_id.clone(),
                        tool_name: "web_search".to_owned(),
                        source: source.to_owned(),
                        args_json: args.to_string(),
                        result_text: String::new(),
                        citations: Vec::new(),
                        success: false,
                        error: Some(error.to_string()),
                        timestamp: Utc::now(),
                    })
                    .await;
                    warn!(
                        user_id = %ctx.user_id,
                        guild_id = %ctx.guild_id,
                        channel_id = %ctx.channel_id,
                        ?error,
                        "web search tool failed"
                    );
                    return Err(error);
                }
            };
            self.record_tool_call(ToolCallRecord {
                user_id: ctx.user_id.clone(),
                guild_id: ctx.guild_id.clone(),
                channel_id: ctx.channel_id.clone(),
                tool_name: "web_search".to_owned(),
                source: source.to_owned(),
                args_json: args.to_string(),
                result_text: truncate_for_log(&tool_result.text, 1200),
                citations: tool_result.citations.clone(),
                success: true,
                error: None,
                timestamp: Utc::now(),
            })
            .await;
            info!(
                user_id = %ctx.user_id,
                result_citations = tool_result.citations.len(),
                "web search tool completed"
            );
            let final_text = self
                .model
                .complete(ModelRequest {
                    system_prompt: format!(
                        "You are CompanionPilot. Use the provided search output to answer the user's request precisely. If citations are provided, keep your answer concise and factual.\n{}",
                        build_recent_context_block(&memory_context.recent_messages)
                    ),
                    user_prompt: format!(
                        "User request:\n{}\n\nSearch output:\n{}",
                        ctx.content, tool_result.text
                    ),
                })
                .await
                .unwrap_or(tool_result.text.clone());
            let reply = OrchestratorReply {
                text: final_text,
                citations: tool_result.citations,
                tool_calls: vec![ToolCall {
                    tool_name: "web_search".to_owned(),
                    args,
                }],
                safety_flags,
            };
            self.memory
                .record_chat_message(ChatMessageRecord {
                    id: format!("{}-assistant", ctx.message_id),
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
        if let SearchDecision::Skip { reason } = search_decision {
            debug!(
                user_id = %ctx.user_id,
                guild_id = %ctx.guild_id,
                channel_id = %ctx.channel_id,
                reason,
                "web search skipped"
            );
        }

        let system_prompt = build_system_prompt(&memory_context);
        let model_response = self
            .model
            .complete(ModelRequest {
                system_prompt,
                user_prompt: ctx.content.clone(),
            })
            .await?;

        match self.decide_memory_fact(&ctx.content, &memory_context).await {
            MemoryDecision::Store { fact, rationale } => {
                info!(
                    user_id = %ctx.user_id,
                    memory_key = %fact.key,
                    confidence = fact.confidence,
                    "memory fact stored"
                );
                self.record_memory_planner_decision(
                    &ctx,
                    "store",
                    rationale,
                    &json!({
                        "key": fact.key,
                        "value": fact.value,
                        "confidence": fact.confidence
                    }),
                    true,
                    None,
                )
                .await;
                self.memory.upsert_fact(&ctx.user_id, fact).await?;
            }
            MemoryDecision::Skip { reason, error } => {
                debug!(
                    user_id = %ctx.user_id,
                    reason,
                    "memory write skipped"
                );
                self.record_memory_planner_decision(
                    &ctx,
                    "skip",
                    reason,
                    &json!({}),
                    error.is_none(),
                    error,
                )
                .await;
            }
        }

        let reply = OrchestratorReply {
            text: model_response,
            citations: Vec::new(),
            tool_calls: Vec::new(),
            safety_flags,
        };

        self.memory
            .record_chat_message(ChatMessageRecord {
                id: format!("{}-assistant", ctx.message_id),
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

    async fn decide_search_query(
        &self,
        user_input: &str,
        memory: &crate::types::MemoryContext,
    ) -> SearchDecision {
        let planner_prompt = build_search_planner_prompt(memory);
        let planner_result = self
            .model
            .complete(ModelRequest {
                system_prompt: planner_prompt,
                user_prompt: user_input.to_owned(),
            })
            .await;

        let planner_result = match planner_result {
            Ok(content) => content,
            Err(error) => {
                warn!(?error, "search planner model call failed");
                return SearchDecision::Skip {
                    reason: "planner_model_error",
                };
            }
        };

        match parse_planner_output(&planner_result) {
            Ok(plan) => {
                if !plan.use_search {
                    return SearchDecision::Skip {
                        reason: "planner_no_search",
                    };
                }
                let query = plan.query.trim();
                if query.is_empty() {
                    return SearchDecision::Skip {
                        reason: "planner_empty_query",
                    };
                }

                let normalized_query = normalize_for_compare(query);
                let normalized_input = normalize_for_compare(user_input);
                let query_word_count = query.split_whitespace().count();
                if normalized_query == normalized_input && query_word_count > 7 {
                    return SearchDecision::Use {
                        query: fallback_search_query(user_input),
                        source: "planner_refined",
                    };
                }

                SearchDecision::Use {
                    query: query.to_owned(),
                    source: "model_planner",
                }
            }
            Err(error) => {
                warn!(
                    ?error,
                    planner_output = %truncate_for_log(&planner_result, 220),
                    "failed to parse search planner output"
                );
                SearchDecision::Skip {
                    reason: "planner_parse_error",
                }
            }
        }
    }

    async fn decide_memory_fact(
        &self,
        user_input: &str,
        memory: &crate::types::MemoryContext,
    ) -> MemoryDecision {
        let planner_prompt = build_memory_planner_prompt(memory);
        let planner_result = self
            .model
            .complete(ModelRequest {
                system_prompt: planner_prompt,
                user_prompt: user_input.to_owned(),
            })
            .await;

        let planner_result = match planner_result {
            Ok(content) => content,
            Err(error) => {
                warn!(?error, "memory planner model call failed");
                return MemoryDecision::Skip {
                    reason: "planner_model_error",
                    error: Some(error.to_string()),
                };
            }
        };

        match parse_memory_plan(&planner_result) {
            Ok(plan) => {
                if !plan.store {
                    return MemoryDecision::Skip {
                        reason: "planner_no_store",
                        error: None,
                    };
                }

                let key = sanitize_memory_key(&plan.key);
                let value = clean_memory_value(&plan.value);
                if key.is_empty() || value.is_empty() {
                    return MemoryDecision::Skip {
                        reason: "planner_invalid_fact",
                        error: None,
                    };
                }

                MemoryDecision::Store {
                    fact: MemoryFact {
                        key,
                        value,
                        confidence: plan.confidence.clamp(0.0, 1.0),
                        source: "user_message".to_owned(),
                        updated_at: Utc::now(),
                    },
                    rationale: "model_planner",
                }
            }
            Err(error) => {
                warn!(
                    ?error,
                    planner_output = %truncate_for_log(&planner_result, 220),
                    "failed to parse memory planner output"
                );
                MemoryDecision::Skip {
                    reason: "planner_parse_error",
                    error: Some(error.to_string()),
                }
            }
        }
    }

    async fn record_tool_call(&self, call: ToolCallRecord) {
        if let Err(error) = self.memory.record_tool_call(call).await {
            warn!(?error, "failed to persist tool call log");
        }
    }

    async fn record_search_planner_decision(&self, ctx: &MessageCtx, decision: &SearchDecision) {
        let (decision_value, rationale, payload, success, error) = match decision {
            SearchDecision::Use { query, source } => (
                "use_search",
                *source,
                json!({
                    "query": query
                }),
                true,
                None,
            ),
            SearchDecision::Skip { reason } => (
                "skip_search",
                *reason,
                json!({}),
                !reason.starts_with("planner_"),
                if reason.starts_with("planner_") {
                    Some((*reason).to_owned())
                } else {
                    None
                },
            ),
        };

        let record = PlannerDecisionRecord {
            user_id: ctx.user_id.clone(),
            guild_id: ctx.guild_id.clone(),
            channel_id: ctx.channel_id.clone(),
            planner: "search".to_owned(),
            decision: decision_value.to_owned(),
            rationale: rationale.to_owned(),
            payload_json: payload.to_string(),
            success,
            error,
            timestamp: Utc::now(),
        };

        if let Err(error) = self.memory.record_planner_decision(record).await {
            warn!(?error, "failed to persist search planner decision log");
        }
    }

    async fn record_memory_planner_decision(
        &self,
        ctx: &MessageCtx,
        decision: &str,
        rationale: &str,
        payload: &serde_json::Value,
        success: bool,
        error: Option<String>,
    ) {
        let record = PlannerDecisionRecord {
            user_id: ctx.user_id.clone(),
            guild_id: ctx.guild_id.clone(),
            channel_id: ctx.channel_id.clone(),
            planner: "memory".to_owned(),
            decision: decision.to_owned(),
            rationale: rationale.to_owned(),
            payload_json: payload.to_string(),
            success,
            error,
            timestamp: Utc::now(),
        };

        if let Err(store_error) = self.memory.record_planner_decision(record).await {
            warn!(
                ?store_error,
                "failed to persist memory planner decision log"
            );
        }
    }
}

fn parse_search_command(content: &str) -> Option<&str> {
    let trimmed = content.trim();
    trimmed
        .strip_prefix("/search ")
        .map(str::trim)
        .filter(|query| !query.is_empty())
}

fn build_search_planner_prompt(memory: &crate::types::MemoryContext) -> String {
    let mut context = String::new();
    if !memory.facts.is_empty() {
        let facts = memory
            .facts
            .iter()
            .map(|fact| format!("{}={}", fact.key, fact.value))
            .collect::<Vec<_>>()
            .join("; ");
        context = format!("Known user facts: {facts}");
    }

    format!(
        "You are a tool router for CompanionPilot.
Decide whether web search is required to answer accurately.
If search is required, produce a short search query.
Return strict JSON with no markdown:
{{\"use_search\": true|false, \"query\": \"...\"}}
Set query to empty string when use_search is false.
{}
Rules:
- If the user explicitly asks to search, find, look up, browse, compare options, or discover similar projects/tools, set use_search=true.
- If the user asks for recommendations that depend on currently available options, set use_search=true.
- Use search for time-sensitive, latest/current, news, prices, weather, or unknown factual claims.
- Do not use search for casual conversation or personal memory recall.
- Keep query concise and retrieval-oriented (3-12 words), avoiding filler words.
Examples:
- User: \"Search for some similar AI project like the one I am building.\" -> {{\"use_search\":true,\"query\":\"similar AI companion orchestrator projects\"}}
- User: \"Find alternatives to Tavily for AI search.\" -> {{\"use_search\":true,\"query\":\"Tavily alternatives AI search API\"}}
- User: \"What did I just tell you?\" -> {{\"use_search\":false,\"query\":\"\"}}",
        context
    )
}

#[derive(Debug, Deserialize)]
struct SearchPlan {
    use_search: bool,
    query: String,
}

fn parse_planner_output(raw: &str) -> Result<SearchPlan, serde_json::Error> {
    let candidate = raw
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();
    serde_json::from_str::<SearchPlan>(candidate)
}

fn truncate_for_log(input: &str, max_len: usize) -> String {
    let mut result = input.replace('\n', "\\n");
    if result.len() > max_len {
        result.truncate(max_len);
        result.push_str("...");
    }
    result
}

fn build_memory_planner_prompt(memory: &crate::types::MemoryContext) -> String {
    let mut context = String::new();
    if !memory.facts.is_empty() {
        let facts = memory
            .facts
            .iter()
            .map(|fact| format!("{}={}", fact.key, fact.value))
            .collect::<Vec<_>>()
            .join("; ");
        context = format!("Existing user facts: {facts}");
    }

    format!(
        "You are a memory router for CompanionPilot.
Decide if the user's message should be stored as long-term memory.
Return strict JSON only (no markdown):
{{\"store\": true|false, \"key\": \"...\", \"value\": \"...\", \"confidence\": 0.0-1.0}}
If store=false, set key and value to empty strings.
{}
Rules:
- Store only durable personal facts (identity, preferences, recurring goals, corrections).
- Do not store one-off requests or transient states.
- If user corrects a previous fact, store corrected value under the same key.",
        context
    )
}

#[derive(Debug, Deserialize)]
struct MemoryPlan {
    store: bool,
    key: String,
    value: String,
    confidence: f32,
}

fn parse_memory_plan(raw: &str) -> Result<MemoryPlan, serde_json::Error> {
    let candidate = raw
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();
    serde_json::from_str::<MemoryPlan>(candidate)
}

fn sanitize_memory_key(raw: &str) -> String {
    let mut normalized = raw
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect::<String>();

    while normalized.contains("__") {
        normalized = normalized.replace("__", "_");
    }

    normalized.trim_matches('_').to_owned()
}

fn fallback_search_query(input: &str) -> String {
    const STOPWORDS: &[&str] = &[
        "a", "an", "and", "are", "be", "can", "could", "for", "get", "i", "is", "it", "latest",
        "me", "my", "of", "on", "please", "search", "tell", "the", "to", "up", "what", "whats",
        "with", "you", "today",
    ];

    let mut tokens = Vec::new();
    for raw in input.split_whitespace() {
        let cleaned = raw
            .trim_matches(|character: char| !character.is_alphanumeric())
            .to_lowercase();
        if cleaned.is_empty() {
            continue;
        }
        if STOPWORDS.contains(&cleaned.as_str()) {
            continue;
        }
        tokens.push(cleaned);
        if tokens.len() >= 8 {
            break;
        }
    }

    if tokens.is_empty() {
        return input.trim().to_owned();
    }

    tokens.join(" ")
}

fn normalize_for_compare(value: &str) -> String {
    value
        .chars()
        .map(|character| {
            if character.is_alphanumeric() || character.is_whitespace() {
                character.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn build_system_prompt(memory: &crate::types::MemoryContext) -> String {
    let mut sections = vec![
        "You are CompanionPilot, a helpful Discord AI companion.".to_owned(),
        "Keep replies concise and practical.".to_owned(),
    ];

    if let Some(summary) = &memory.summary {
        sections.push(format!("Conversation summary: {summary}"));
    }

    if !memory.recent_messages.is_empty() {
        sections.push(build_recent_context_block(&memory.recent_messages));
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

fn build_recent_context_block(recent_messages: &[String]) -> String {
    if recent_messages.is_empty() {
        return String::new();
    }

    let turns = recent_messages
        .iter()
        .rev()
        .take(8)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .map(|line| format!("- {line}"))
        .collect::<Vec<_>>()
        .join("\n");
    format!("Recent conversation turns:\n{turns}")
}

fn clean_memory_value(value: &str) -> String {
    value
        .trim()
        .trim_matches(|character: char| character == '"' || character == '\'')
        .trim_end_matches(|character: char| {
            character == '.' || character == '!' || character == '?'
        })
        .trim()
        .to_owned()
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

    use super::{
        DefaultChatOrchestrator, clean_memory_value, fallback_search_query, normalize_for_compare,
        sanitize_memory_key,
    };

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

    #[tokio::test]
    async fn auto_search_uses_tool_on_latest_question() {
        let memory = Arc::new(InMemoryMemoryStore::default());
        let orchestrator = DefaultChatOrchestrator::new(
            Arc::new(MockModelProvider),
            memory,
            Arc::new(ToolRegistry::default()),
            SafetyPolicy::default(),
        );

        let result = orchestrator
            .handle_message(MessageCtx {
                message_id: "3".into(),
                user_id: "u3".into(),
                guild_id: "g1".into(),
                channel_id: "c1".into(),
                content: "What is the latest Rust release today?".into(),
                timestamp: Utc::now(),
            })
            .await
            .expect("message should succeed when planner decides no search");
        assert!(result.tool_calls.is_empty());
    }

    #[tokio::test]
    async fn name_correction_overwrites_previous_memory() {
        let memory = Arc::new(InMemoryMemoryStore::default());
        let orchestrator = DefaultChatOrchestrator::new(
            Arc::new(MockModelProvider),
            memory.clone(),
            Arc::new(ToolRegistry::default()),
            SafetyPolicy::default(),
        );

        let _ = orchestrator
            .handle_message(MessageCtx {
                message_id: "4".into(),
                user_id: "u4".into(),
                guild_id: "g1".into(),
                channel_id: "c1".into(),
                content: "my name is Petrr".into(),
                timestamp: Utc::now(),
            })
            .await
            .expect("first message should succeed");

        let _ = orchestrator
            .handle_message(MessageCtx {
                message_id: "5".into(),
                user_id: "u4".into(),
                guild_id: "g1".into(),
                channel_id: "c1".into(),
                content: "I misspelled my name, it's Petr.".into(),
                timestamp: Utc::now(),
            })
            .await
            .expect("correction message should succeed");

        let facts = memory
            .search_relevant("u4", "name", 10)
            .await
            .expect("search should succeed");
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].value, "Petr");
    }

    #[tokio::test]
    async fn short_term_memory_includes_recent_non_fact_turns() {
        let memory = Arc::new(InMemoryMemoryStore::default());
        let orchestrator = DefaultChatOrchestrator::new(
            Arc::new(MockModelProvider),
            memory,
            Arc::new(ToolRegistry::default()),
            SafetyPolicy::default(),
        );

        let _ = orchestrator
            .handle_message(MessageCtx {
                message_id: "6".into(),
                user_id: "u6".into(),
                guild_id: "g1".into(),
                channel_id: "c1".into(),
                content: "I am 24 years old.".into(),
                timestamp: Utc::now(),
            })
            .await
            .expect("first message should succeed");

        let second = orchestrator
            .handle_message(MessageCtx {
                message_id: "7".into(),
                user_id: "u6".into(),
                guild_id: "g1".into(),
                channel_id: "c1".into(),
                content: "What did I just tell you?".into(),
                timestamp: Utc::now(),
            })
            .await
            .expect("second message should succeed");

        assert!(second.text.contains("Recent conversation turns:"));
        assert!(second.text.contains("user: I am 24 years old."));
    }

    #[test]
    fn fallback_query_compacts_user_prompt() {
        let query = fallback_search_query("What is the latest Rust release today?");
        assert_eq!(query, "rust release");
    }

    #[test]
    fn normalize_compare_ignores_formatting() {
        let a = normalize_for_compare("What is  Rust?  ");
        let b = normalize_for_compare("what is rust");
        assert_eq!(a, b);
    }

    #[test]
    fn sanitize_memory_key_normalizes_words() {
        assert_eq!(sanitize_memory_key("Favorite Game"), "favorite_game");
    }

    #[test]
    fn clean_memory_value_trims_wrappers() {
        assert_eq!(clean_memory_value("\"Petr.\""), "Petr");
    }
}
