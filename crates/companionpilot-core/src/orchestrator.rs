use std::sync::Arc;

use chrono::Utc;
use serde::{Deserialize, de::DeserializeOwned};
use serde_json::{Value, json};
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

const MAX_PLANNED_TOOL_CALLS: usize = 6;

pub struct DefaultChatOrchestrator {
    model: Arc<dyn ModelProvider>,
    memory: Arc<dyn MemoryStore>,
    tools: Arc<dyn ToolExecutor>,
    safety: SafetyPolicy,
}

enum UnifiedPlanDecision {
    UsePlan {
        tool_calls: Vec<ToolCall>,
        memory: MemoryDecision,
        rationale: String,
        payload: Value,
    },
    Fallback {
        reason: &'static str,
        error: Option<String>,
    },
}

enum MemoryDecision {
    Store {
        fact: MemoryFact,
        rationale: &'static str,
    },
    Skip {
        reason: &'static str,
    },
}

#[derive(Debug, Deserialize)]
struct UnifiedPlan {
    #[serde(default)]
    tool_calls: Vec<PlannedToolCall>,
    #[serde(default)]
    memory: PlannedMemory,
    #[serde(default)]
    rationale: String,
}

#[derive(Debug, Deserialize)]
struct PlannedToolCall {
    tool_name: String,
    #[serde(default)]
    args: Value,
}

#[derive(Debug, Default, Deserialize)]
struct PlannedMemory {
    #[serde(default)]
    store: bool,
    #[serde(default)]
    key: String,
    #[serde(default)]
    value: String,
    #[serde(default)]
    confidence: f32,
}

struct ExecutedToolOutput {
    tool_name: String,
    args: Value,
    text: String,
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
        self.handle_message_with_system_prompt_override(ctx, None)
            .await
    }

    pub async fn handle_message_with_system_prompt_override(
        &self,
        ctx: MessageCtx,
        system_prompt_override: Option<String>,
    ) -> anyhow::Result<OrchestratorReply> {
        let system_prompt_override = system_prompt_override
            .map(|prompt| prompt.trim().to_owned())
            .filter(|prompt| !prompt.is_empty());
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

        let planner_decision = self
            .decide_unified_plan(&ctx.content, &memory_context)
            .await;
        self.record_unified_planner_decision(&ctx, &planner_decision)
            .await;

        let (planned_tool_calls, memory_decision) = match planner_decision {
            UnifiedPlanDecision::UsePlan {
                tool_calls, memory, ..
            } => (tool_calls, memory),
            UnifiedPlanDecision::Fallback { reason, .. } => {
                debug!(
                    user_id = %ctx.user_id,
                    reason,
                    "planner fallback: running without tools and without memory write"
                );
                (
                    Vec::new(),
                    MemoryDecision::Skip {
                        reason: "planner_fallback",
                    },
                )
            }
        };

        let mut executed_tool_calls = Vec::new();
        let mut tool_outputs = Vec::new();
        let mut citations = Vec::new();

        for tool_call in planned_tool_calls {
            let args = tool_call.args.clone();
            info!(
                user_id = %ctx.user_id,
                guild_id = %ctx.guild_id,
                channel_id = %ctx.channel_id,
                tool_name = %tool_call.tool_name,
                args_json = %args,
                "tool call selected by unified planner"
            );

            let tool_result = match self.tools.execute(&tool_call.tool_name, args.clone()).await {
                Ok(result) => result,
                Err(error) => {
                    self.record_tool_call(ToolCallRecord {
                        user_id: ctx.user_id.clone(),
                        guild_id: ctx.guild_id.clone(),
                        channel_id: ctx.channel_id.clone(),
                        tool_name: tool_call.tool_name.clone(),
                        source: "unified_planner".to_owned(),
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
                        tool_name = %tool_call.tool_name,
                        ?error,
                        "tool call failed"
                    );
                    return Err(error);
                }
            };

            self.record_tool_call(ToolCallRecord {
                user_id: ctx.user_id.clone(),
                guild_id: ctx.guild_id.clone(),
                channel_id: ctx.channel_id.clone(),
                tool_name: tool_call.tool_name.clone(),
                source: "unified_planner".to_owned(),
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
                tool_name = %tool_call.tool_name,
                result_citations = tool_result.citations.len(),
                "tool call completed"
            );

            citations.extend(tool_result.citations);
            executed_tool_calls.push(ToolCall {
                tool_name: tool_call.tool_name.clone(),
                args: args.clone(),
            });
            tool_outputs.push(ExecutedToolOutput {
                tool_name: tool_call.tool_name,
                args,
                text: tool_result.text,
            });
        }

        let reply_text = if tool_outputs.is_empty() {
            self.model
                .complete(ModelRequest {
                    system_prompt: build_system_prompt(
                        &memory_context,
                        system_prompt_override.as_deref(),
                    ),
                    user_prompt: ctx.content.clone(),
                })
                .await?
        } else {
            let tool_output_block = format_tool_outputs(&tool_outputs);
            let custom_prompt_header = system_prompt_override
                .as_deref()
                .map(|prompt| format!("Custom system prompt override:\n{prompt}\n\n"))
                .unwrap_or_default();
            self.model
                .complete(ModelRequest {
                    system_prompt: format!(
                        "{}You are CompanionPilot. Use the provided tool outputs to answer the user's request precisely.\nNever say you cannot browse the web in this mode.\nNever output XML/JSON/pseudo tool-call markup.\nReturn only the final user-facing answer.\nIf citations are provided, keep your answer concise and factual.\n{}",
                        custom_prompt_header,
                        build_recent_context_block(&memory_context.recent_messages)
                    ),
                    user_prompt: format!(
                        "User request:\n{}\n\nTool outputs:\n{}",
                        ctx.content, tool_output_block
                    ),
                })
                .await
                .unwrap_or_else(|error| {
                    warn!(?error, "failed to synthesize final answer from tool outputs");
                    fallback_tool_output_text(&tool_outputs)
                })
        };

        match memory_decision {
            MemoryDecision::Store { fact, rationale } => {
                info!(
                    user_id = %ctx.user_id,
                    memory_key = %fact.key,
                    confidence = fact.confidence,
                    rationale,
                    "memory fact stored"
                );
                self.memory.upsert_fact(&ctx.user_id, fact).await?;
            }
            MemoryDecision::Skip { reason } => {
                debug!(
                    user_id = %ctx.user_id,
                    reason,
                    "memory write skipped"
                );
            }
        }

        let reply = OrchestratorReply {
            text: reply_text,
            citations: dedupe_citations(citations),
            tool_calls: executed_tool_calls,
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

    async fn decide_unified_plan(
        &self,
        user_input: &str,
        memory: &crate::types::MemoryContext,
    ) -> UnifiedPlanDecision {
        let planner_prompt = build_unified_planner_prompt(memory);
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
                warn!(?error, "unified planner model call failed");
                return UnifiedPlanDecision::Fallback {
                    reason: "planner_model_error",
                    error: Some(error.to_string()),
                };
            }
        };

        match parse_unified_plan(&planner_result) {
            Ok(plan) => {
                let tool_calls = sanitize_planned_tool_calls(plan.tool_calls);
                let memory = memory_decision_from_plan(plan.memory);
                let rationale = if plan.rationale.trim().is_empty() {
                    "model_planner".to_owned()
                } else {
                    plan.rationale.trim().to_owned()
                };

                let payload = json!({
                    "tool_calls": tool_calls,
                    "memory": memory_payload(&memory),
                    "rationale": rationale
                });

                UnifiedPlanDecision::UsePlan {
                    tool_calls,
                    memory,
                    rationale,
                    payload,
                }
            }
            Err(error) => {
                warn!(
                    ?error,
                    planner_output = %truncate_for_log(&planner_result, 220),
                    "failed to parse unified planner output"
                );
                UnifiedPlanDecision::Fallback {
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

    async fn record_unified_planner_decision(
        &self,
        ctx: &MessageCtx,
        decision: &UnifiedPlanDecision,
    ) {
        let (decision_value, rationale, payload, success, error) = match decision {
            UnifiedPlanDecision::UsePlan {
                rationale, payload, ..
            } => ("apply_plan", rationale.clone(), payload.clone(), true, None),
            UnifiedPlanDecision::Fallback { reason, error } => (
                "fallback_no_tools",
                (*reason).to_owned(),
                json!({}),
                false,
                error.clone(),
            ),
        };

        let record = PlannerDecisionRecord {
            user_id: ctx.user_id.clone(),
            guild_id: ctx.guild_id.clone(),
            channel_id: ctx.channel_id.clone(),
            planner: "unified".to_owned(),
            decision: decision_value.to_owned(),
            rationale,
            payload_json: payload.to_string(),
            success,
            error,
            timestamp: Utc::now(),
        };

        if let Err(store_error) = self.memory.record_planner_decision(record).await {
            warn!(
                ?store_error,
                "failed to persist unified planner decision log"
            );
        }
    }
}

fn build_unified_planner_prompt(memory: &crate::types::MemoryContext) -> String {
    let mut context_lines = Vec::new();
    if let Some(summary) = &memory.summary {
        context_lines.push(format!("Conversation summary: {summary}"));
    }

    if !memory.facts.is_empty() {
        let facts = memory
            .facts
            .iter()
            .map(|fact| format!("{}={}", fact.key, fact.value))
            .collect::<Vec<_>>()
            .join("; ");
        context_lines.push(format!("Known user facts: {facts}"));
    }

    if !memory.recent_messages.is_empty() {
        context_lines.push(build_recent_context_block(&memory.recent_messages));
    }

    let context_block = if context_lines.is_empty() {
        String::new()
    } else {
        format!("Context:\n{}\n", context_lines.join("\n"))
    };

    format!(
        "You are the unified planner for CompanionPilot.
Decide both tool usage and memory write for one user message.
Return strict JSON only (no markdown, no prose) with this exact schema:
{{
  \"tool_calls\": [{{\"tool_name\":\"...\",\"args\":{{...}}}}],
  \"memory\": {{
    \"store\": true|false,
    \"key\": \"...\",
    \"value\": \"...\",
    \"confidence\": 0.0-1.0
  }},
  \"rationale\": \"short reason\"
}}
Tool calls are executed sequentially in listed order.
There are no manual commands or manual overrides: all tool usage must come from this decision.
If no tool is needed, return an empty tool_calls array.
If memory should not be stored, set store=false and key/value to empty strings.
Store only durable personal facts (identity, preferences, recurring goals, corrections).
Do not store one-off requests or transient states.
Use web search for latest/current/news/prices/weather or unknown factual claims.
Tool inventory:
{}
{}",
        build_tool_inventory_for_planner(),
        context_block
    )
}

fn build_tool_inventory_for_planner() -> &'static str {
    r#"[
  {
    "tool_name": "web_search",
    "args_schema": {
      "query": "string (required, non-empty)",
      "max_results": "integer 1-10 (optional, default 5)"
    },
    "when_to_use": "Need external factual information, latest/current info, or web-sourced recommendations.",
    "when_not_to_use": "Casual chat, personal memory recall, or when the answer can be provided from context."
  }
]"#
}

fn parse_unified_plan(raw: &str) -> Result<UnifiedPlan, serde_json::Error> {
    parse_json_plan(raw)
}

fn sanitize_planned_tool_calls(planned_calls: Vec<PlannedToolCall>) -> Vec<ToolCall> {
    let mut sanitized_calls = Vec::new();

    for planned_call in planned_calls {
        if sanitized_calls.len() >= MAX_PLANNED_TOOL_CALLS {
            break;
        }
        match planned_call.tool_name.as_str() {
            "web_search" => {
                let query = planned_call
                    .args
                    .get("query")
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .unwrap_or("");
                if query.is_empty() {
                    debug!("dropping planner web_search call with empty query");
                    continue;
                }

                let max_results = planned_call
                    .args
                    .get("max_results")
                    .and_then(Value::as_u64)
                    .unwrap_or(5)
                    .clamp(1, 10);

                sanitized_calls.push(ToolCall {
                    tool_name: "web_search".to_owned(),
                    args: json!({
                        "query": query,
                        "max_results": max_results
                    }),
                });
            }
            other => {
                debug!(tool_name = other, "dropping unknown planner tool call");
            }
        }
    }

    sanitized_calls
}

fn memory_decision_from_plan(plan: PlannedMemory) -> MemoryDecision {
    if !plan.store {
        return MemoryDecision::Skip {
            reason: "planner_no_store",
        };
    }

    let key = sanitize_memory_key(&plan.key);
    let value = clean_memory_value(&plan.value);
    if key.is_empty() || value.is_empty() {
        return MemoryDecision::Skip {
            reason: "planner_invalid_fact",
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

fn memory_payload(memory: &MemoryDecision) -> Value {
    match memory {
        MemoryDecision::Store { fact, .. } => json!({
            "store": true,
            "key": fact.key,
            "value": fact.value,
            "confidence": fact.confidence
        }),
        MemoryDecision::Skip { reason } => json!({
            "store": false,
            "reason": reason
        }),
    }
}

fn truncate_for_log(input: &str, max_len: usize) -> String {
    let mut result = input.replace('\n', "\\n");
    if result.len() > max_len {
        result.truncate(max_len);
        result.push_str("...");
    }
    result
}

fn parse_json_plan<T: DeserializeOwned>(raw: &str) -> Result<T, serde_json::Error> {
    let candidate = raw
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();

    match serde_json::from_str::<T>(candidate) {
        Ok(plan) => Ok(plan),
        Err(original_error) => {
            if let Some(object_candidate) = extract_first_json_object(candidate) {
                serde_json::from_str::<T>(object_candidate).map_err(|_| original_error)
            } else {
                Err(original_error)
            }
        }
    }
}

fn extract_first_json_object(raw: &str) -> Option<&str> {
    let mut start_index: Option<usize> = None;
    let mut depth = 0usize;
    let mut in_string = false;
    let mut is_escaped = false;

    for (index, character) in raw.char_indices() {
        if start_index.is_none() {
            if character == '{' {
                start_index = Some(index);
                depth = 1;
            }
            continue;
        }

        if in_string {
            if is_escaped {
                is_escaped = false;
                continue;
            }
            match character {
                '\\' => is_escaped = true,
                '"' => in_string = false,
                _ => {}
            }
            continue;
        }

        match character {
            '"' => in_string = true,
            '{' => depth += 1,
            '}' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    let start = start_index?;
                    return Some(&raw[start..=index]);
                }
            }
            _ => {}
        }
    }

    None
}

fn format_tool_outputs(outputs: &[ExecutedToolOutput]) -> String {
    outputs
        .iter()
        .enumerate()
        .map(|(index, output)| {
            format!(
                "{}. Tool: {}\nArgs: {}\nOutput:\n{}",
                index + 1,
                output.tool_name,
                output.args,
                output.text
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn fallback_tool_output_text(outputs: &[ExecutedToolOutput]) -> String {
    outputs
        .iter()
        .map(|output| output.text.clone())
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn dedupe_citations(citations: Vec<String>) -> Vec<String> {
    let mut deduped = Vec::new();
    for citation in citations {
        if deduped.iter().any(|existing| existing == &citation) {
            continue;
        }
        deduped.push(citation);
    }
    deduped
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

const DEFAULT_SYSTEM_PROMPT_BASE: &str = "You are CompanionPilot, a helpful Discord AI companion.\nKeep replies concise and practical.\nNever emit XML/JSON/pseudo tool-call markup in normal replies.";

pub fn default_system_prompt_base() -> &'static str {
    DEFAULT_SYSTEM_PROMPT_BASE
}

fn build_system_prompt(
    memory: &crate::types::MemoryContext,
    override_prompt: Option<&str>,
) -> String {
    let mut sections = if let Some(prompt) = override_prompt {
        vec![prompt.to_owned()]
    } else {
        vec![DEFAULT_SYSTEM_PROMPT_BASE.to_owned()]
    };

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
        DefaultChatOrchestrator, PlannedToolCall, clean_memory_value, parse_unified_plan,
        sanitize_memory_key, sanitize_planned_tool_calls,
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
    async fn search_command_is_not_a_manual_override() {
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
            .await
            .expect("planner should be allowed to skip tool usage");

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
    fn sanitize_memory_key_normalizes_words() {
        assert_eq!(sanitize_memory_key("Favorite Game"), "favorite_game");
    }

    #[test]
    fn clean_memory_value_trims_wrappers() {
        assert_eq!(clean_memory_value("\"Petr.\""), "Petr");
    }

    #[test]
    fn parse_unified_plan_from_wrapped_json() {
        let raw = "Result:\n{\"tool_calls\":[],\"memory\":{\"store\":false,\"key\":\"\",\"value\":\"\",\"confidence\":0.0},\"rationale\":\"none\"}\nDone.";
        let plan = parse_unified_plan(raw).expect("wrapped JSON should parse");
        assert!(plan.tool_calls.is_empty());
        assert!(!plan.memory.store);
    }

    #[test]
    fn sanitize_planned_tool_calls_drops_unknown_and_limits_to_max() {
        let mut planned_calls = Vec::new();
        planned_calls.push(PlannedToolCall {
            tool_name: "unknown_tool".to_owned(),
            args: json!({}),
        });

        for index in 0..8 {
            planned_calls.push(PlannedToolCall {
                tool_name: "web_search".to_owned(),
                args: json!({
                    "query": format!("rust query {index}"),
                    "max_results": 5
                }),
            });
        }

        let sanitized = sanitize_planned_tool_calls(planned_calls);
        assert_eq!(sanitized.len(), 6);
        assert_eq!(sanitized[0].tool_name, "web_search");
        assert_eq!(sanitized[5].tool_name, "web_search");
    }
}
