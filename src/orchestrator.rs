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
    types::{ChatMessageRecord, ChatRole, MemoryFact, MessageCtx, OrchestratorReply, ToolCall},
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

        let search_decision = if let Some(manual) = parse_search_command(&ctx.content) {
            SearchDecision::Use {
                query: manual.to_owned(),
                source: "manual_prefix",
            }
        } else {
            self.decide_search_query(&ctx.content, &memory_context)
                .await
        };

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
            info!(
                user_id = %ctx.user_id,
                result_citations = tool_result.citations.len(),
                "web search tool completed"
            );
            let final_text = self
                .model
                .complete(ModelRequest {
                    system_prompt: "You are CompanionPilot. Use the provided search output to answer the user's request precisely. If citations are provided, keep your answer concise and factual.".to_owned(),
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

    async fn decide_search_query(
        &self,
        user_input: &str,
        memory: &crate::types::MemoryContext,
    ) -> SearchDecision {
        let heuristic_likely = needs_search_heuristic(user_input);

        let planner_prompt = build_search_planner_prompt(memory, heuristic_likely);
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
                if heuristic_likely {
                    return SearchDecision::Use {
                        query: fallback_search_query(user_input),
                        source: "heuristic_fallback",
                    };
                }
                return SearchDecision::Skip {
                    reason: "planner_model_error",
                };
            }
        };

        match parse_planner_output(&planner_result) {
            Ok(plan) => {
                if !plan.use_search {
                    if heuristic_likely {
                        return SearchDecision::Use {
                            query: fallback_search_query(user_input),
                            source: "heuristic_override",
                        };
                    }
                    return SearchDecision::Skip {
                        reason: "planner_no_search",
                    };
                }
                let query = plan.query.trim();
                if query.is_empty() {
                    if heuristic_likely {
                        return SearchDecision::Use {
                            query: fallback_search_query(user_input),
                            source: "heuristic_empty_query_fallback",
                        };
                    }
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
                if heuristic_likely {
                    return SearchDecision::Use {
                        query: fallback_search_query(user_input),
                        source: "heuristic_parse_fallback",
                    };
                }
                SearchDecision::Skip {
                    reason: "planner_parse_error",
                }
            }
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

fn needs_search_heuristic(content: &str) -> bool {
    let lowered = content.trim().to_lowercase();
    let triggers = [
        "latest", "today", "current", "news", "price", "release", "update", "weather", "search",
        "look up",
    ];

    triggers.iter().any(|item| lowered.contains(item))
}

fn build_search_planner_prompt(
    memory: &crate::types::MemoryContext,
    heuristic_likely: bool,
) -> String {
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
Heuristic likely_search={}
Rules:
- Use search for time-sensitive, latest/current, news, prices, weather, or unknown factual claims.
- Do not use search for casual conversation or personal memory recall.",
        context, heuristic_likely
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

    use super::{DefaultChatOrchestrator, fallback_search_query, normalize_for_compare};

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
            .await;

        assert!(result.is_err());
        let err = result.err().map(|e| e.to_string()).unwrap_or_default();
        assert!(err.contains("web_search tool is not configured"));
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
}
