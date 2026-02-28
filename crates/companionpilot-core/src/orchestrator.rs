use std::{sync::Arc, time::Instant};

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
        PlannerDecisionRecord, ReplyTimings, ToolCall, ToolCallRecord, ToolCallTiming,
    },
};

const MAX_PLANNED_TOOL_CALLS: usize = 6;
const MAX_TOOL_DECISION_ROUNDS: usize = 3;
const SLOW_REPLY_THRESHOLD_MS: u64 = 30_000;

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

enum ToolFollowupDecision {
    Final {
        answer: String,
        rationale: String,
        payload: Value,
    },
    UseTools {
        tool_calls: Vec<ToolCall>,
        rationale: String,
        payload: Value,
    },
    Fallback {
        reason: &'static str,
        error: Option<String>,
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

#[derive(Debug, Default, Deserialize)]
struct ToolFollowupPlan {
    #[serde(default)]
    action: String,
    #[serde(default)]
    final_answer: String,
    #[serde(default)]
    tool_calls: Vec<PlannedToolCall>,
    #[serde(default)]
    rationale: String,
}

struct ExecutedToolOutput {
    tool_name: String,
    args: Value,
    success: bool,
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
        let request_started_at = Instant::now();
        let system_prompt_override = system_prompt_override
            .map(|prompt| prompt.trim().to_owned())
            .filter(|prompt| !prompt.is_empty());
        let safety_flags = self.safety.validate_user_message(&ctx.content);

        let load_context_started_at = Instant::now();
        let memory_context = self
            .memory
            .load_context(&ctx.user_id, &ctx.guild_id, &ctx.channel_id)
            .await?;
        let load_context_ms = elapsed_ms(load_context_started_at);

        let record_user_message_started_at = Instant::now();
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
        let record_user_message_ms = elapsed_ms(record_user_message_started_at);

        let planner_started_at = Instant::now();
        let planner_decision = self
            .decide_unified_plan(&ctx.content, &memory_context)
            .await;
        let mut planner_ms = elapsed_ms(planner_started_at);
        self.record_unified_planner_decision(&ctx, &planner_decision)
            .await;

        let (mut pending_tool_calls, memory_decision) = match planner_decision {
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
        let mut tool_timings = Vec::new();
        let mut followup_reply_text: Option<String> = None;
        let mut tool_round = 0usize;

        loop {
            if pending_tool_calls.is_empty() {
                break;
            }

            tool_round += 1;
            let planner_source = if tool_round == 1 {
                "unified_planner"
            } else {
                "tool_followup"
            };
            self.execute_planned_tool_calls(
                &ctx,
                pending_tool_calls,
                planner_source,
                &mut executed_tool_calls,
                &mut tool_outputs,
                &mut citations,
                &mut tool_timings,
            )
            .await;

            if tool_round >= MAX_TOOL_DECISION_ROUNDS {
                debug!(
                    user_id = %ctx.user_id,
                    tool_round,
                    "tool planning rounds limit reached; forcing final synthesis"
                );
                break;
            }

            let followup_started_at = Instant::now();
            let followup_decision = self
                .decide_tool_followup(&ctx.content, &memory_context, &tool_outputs)
                .await;
            planner_ms = planner_ms.saturating_add(elapsed_ms(followup_started_at));
            self.record_tool_followup_decision(&ctx, tool_round, &followup_decision)
                .await;

            match followup_decision {
                ToolFollowupDecision::Final { answer, .. } => {
                    followup_reply_text = Some(answer);
                    break;
                }
                ToolFollowupDecision::UseTools { tool_calls, .. } => {
                    pending_tool_calls = tool_calls;
                }
                ToolFollowupDecision::Fallback { reason, .. } => {
                    debug!(
                        user_id = %ctx.user_id,
                        reason,
                        tool_round,
                        "tool follow-up planner fallback; forcing final synthesis"
                    );
                    break;
                }
            }
        }

        let tool_execution_ms = tool_timings.iter().fold(0u64, |total, timing| {
            total.saturating_add(timing.duration_ms)
        });

        let (reply_text, final_model_ms) = if let Some(answer) = followup_reply_text {
            (answer, 0)
        } else {
            let final_model_started_at = Instant::now();
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
            (reply_text, elapsed_ms(final_model_started_at))
        };

        let memory_write_started_at = Instant::now();
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
        let memory_write_ms = elapsed_ms(memory_write_started_at);

        let record_assistant_message_started_at = Instant::now();
        self.memory
            .record_chat_message(ChatMessageRecord {
                id: format!("{}-assistant", ctx.message_id),
                user_id: ctx.user_id.clone(),
                guild_id: ctx.guild_id.clone(),
                channel_id: ctx.channel_id.clone(),
                role: ChatRole::Assistant,
                content: reply_text.clone(),
                timestamp: Utc::now(),
            })
            .await?;
        let record_assistant_message_ms = elapsed_ms(record_assistant_message_started_at);

        let timings = ReplyTimings {
            total_ms: elapsed_ms(request_started_at),
            load_context_ms,
            record_user_message_ms,
            planner_ms,
            tool_execution_ms,
            final_model_ms,
            memory_write_ms,
            record_assistant_message_ms,
            tool_calls: tool_timings,
        };

        if timings.total_ms >= SLOW_REPLY_THRESHOLD_MS {
            warn!(
                user_id = %ctx.user_id,
                guild_id = %ctx.guild_id,
                channel_id = %ctx.channel_id,
                message_id = %ctx.message_id,
                total_ms = timings.total_ms,
                planner_ms = timings.planner_ms,
                tool_execution_ms = timings.tool_execution_ms,
                final_model_ms = timings.final_model_ms,
                memory_write_ms = timings.memory_write_ms,
                "slow reply detected"
            );
        } else {
            info!(
                user_id = %ctx.user_id,
                guild_id = %ctx.guild_id,
                channel_id = %ctx.channel_id,
                message_id = %ctx.message_id,
                total_ms = timings.total_ms,
                planner_ms = timings.planner_ms,
                tool_execution_ms = timings.tool_execution_ms,
                final_model_ms = timings.final_model_ms,
                memory_write_ms = timings.memory_write_ms,
                "reply completed"
            );
        }

        let reply = OrchestratorReply {
            text: reply_text,
            citations: dedupe_citations(citations),
            tool_calls: executed_tool_calls,
            safety_flags,
            timings,
        };

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
                let tool_calls = enforce_datetime_planning_boundary(sanitize_planned_tool_calls(
                    plan.tool_calls,
                ));
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

    async fn decide_tool_followup(
        &self,
        user_input: &str,
        memory: &crate::types::MemoryContext,
        tool_outputs: &[ExecutedToolOutput],
    ) -> ToolFollowupDecision {
        let planner_prompt = build_tool_followup_prompt(memory);
        let planner_result = self
            .model
            .complete(ModelRequest {
                system_prompt: planner_prompt,
                user_prompt: format!(
                    "User request:\n{}\n\nTool outputs so far:\n{}",
                    user_input,
                    format_tool_outputs(tool_outputs)
                ),
            })
            .await;

        let planner_result = match planner_result {
            Ok(content) => content,
            Err(error) => {
                warn!(?error, "tool follow-up planner model call failed");
                return ToolFollowupDecision::Fallback {
                    reason: "followup_model_error",
                    error: Some(error.to_string()),
                };
            }
        };

        match parse_tool_followup_plan(&planner_result) {
            Ok(plan) => {
                let rationale = if plan.rationale.trim().is_empty() {
                    "tool_followup_planner".to_owned()
                } else {
                    plan.rationale.trim().to_owned()
                };
                let action = plan.action.trim().to_ascii_lowercase();

                match action.as_str() {
                    "final" | "final_answer" => {
                        let answer = plan.final_answer.trim().to_owned();
                        if answer.is_empty() {
                            return ToolFollowupDecision::Fallback {
                                reason: "followup_empty_final",
                                error: Some(
                                    "follow-up planner returned empty final answer".to_owned(),
                                ),
                            };
                        }

                        ToolFollowupDecision::Final {
                            answer: answer.clone(),
                            rationale: rationale.clone(),
                            payload: json!({
                                "action": "final",
                                "final_answer": answer,
                                "rationale": rationale
                            }),
                        }
                    }
                    "tools" | "tool_calls" => {
                        let tool_calls = enforce_datetime_planning_boundary(
                            sanitize_planned_tool_calls(plan.tool_calls),
                        );
                        if tool_calls.is_empty() {
                            return ToolFollowupDecision::Fallback {
                                reason: "followup_empty_tools",
                                error: Some(
                                    "follow-up planner requested tools but produced none"
                                        .to_owned(),
                                ),
                            };
                        }

                        ToolFollowupDecision::UseTools {
                            payload: json!({
                                "action": "tools",
                                "tool_calls": &tool_calls,
                                "rationale": rationale.clone()
                            }),
                            rationale,
                            tool_calls,
                        }
                    }
                    _ => ToolFollowupDecision::Fallback {
                        reason: "followup_invalid_action",
                        error: Some(format!(
                            "follow-up planner returned unsupported action `{}`",
                            plan.action
                        )),
                    },
                }
            }
            Err(error) => {
                warn!(
                    ?error,
                    planner_output = %truncate_for_log(&planner_result, 220),
                    "failed to parse tool follow-up planner output"
                );
                ToolFollowupDecision::Fallback {
                    reason: "followup_parse_error",
                    error: Some(error.to_string()),
                }
            }
        }
    }

    async fn execute_planned_tool_calls(
        &self,
        ctx: &MessageCtx,
        planned_tool_calls: Vec<ToolCall>,
        source: &'static str,
        executed_tool_calls: &mut Vec<ToolCall>,
        tool_outputs: &mut Vec<ExecutedToolOutput>,
        citations: &mut Vec<String>,
        tool_timings: &mut Vec<ToolCallTiming>,
    ) {
        for tool_call in planned_tool_calls {
            let tool_started_at = Instant::now();
            let tool_name = tool_call.tool_name;
            let args = tool_call.args.clone();
            executed_tool_calls.push(ToolCall {
                tool_name: tool_name.clone(),
                args: args.clone(),
            });
            info!(
                user_id = %ctx.user_id,
                guild_id = %ctx.guild_id,
                channel_id = %ctx.channel_id,
                planner_source = source,
                tool_name = %tool_name,
                args_json = %args,
                "tool call selected by unified planner"
            );

            let tool_result = match self.tools.execute(&tool_name, args.clone()).await {
                Ok(result) => result,
                Err(error) => {
                    let error_text = error.to_string();
                    self.record_tool_call(ToolCallRecord {
                        user_id: ctx.user_id.clone(),
                        guild_id: ctx.guild_id.clone(),
                        channel_id: ctx.channel_id.clone(),
                        tool_name: tool_name.clone(),
                        source: source.to_owned(),
                        args_json: args.to_string(),
                        result_text: String::new(),
                        citations: Vec::new(),
                        success: false,
                        error: Some(error_text.clone()),
                        timestamp: Utc::now(),
                    })
                    .await;
                    let duration_ms = elapsed_ms(tool_started_at);
                    tool_timings.push(ToolCallTiming {
                        tool_name: tool_name.clone(),
                        duration_ms,
                        success: false,
                    });
                    warn!(
                        user_id = %ctx.user_id,
                        guild_id = %ctx.guild_id,
                        channel_id = %ctx.channel_id,
                        planner_source = source,
                        tool_name = %tool_name,
                        duration_ms,
                        ?error,
                        "tool call failed; continuing orchestration"
                    );
                    tool_outputs.push(ExecutedToolOutput {
                        tool_name,
                        args,
                        success: false,
                        text: error_text,
                    });
                    continue;
                }
            };

            self.record_tool_call(ToolCallRecord {
                user_id: ctx.user_id.clone(),
                guild_id: ctx.guild_id.clone(),
                channel_id: ctx.channel_id.clone(),
                tool_name: tool_name.clone(),
                source: source.to_owned(),
                args_json: args.to_string(),
                result_text: truncate_for_log(&tool_result.text, 1200),
                citations: tool_result.citations.clone(),
                success: true,
                error: None,
                timestamp: Utc::now(),
            })
            .await;

            let duration_ms = elapsed_ms(tool_started_at);
            tool_timings.push(ToolCallTiming {
                tool_name: tool_name.clone(),
                duration_ms,
                success: true,
            });
            info!(
                user_id = %ctx.user_id,
                planner_source = source,
                tool_name = %tool_name,
                duration_ms,
                result_citations = tool_result.citations.len(),
                "tool call completed"
            );

            citations.extend(tool_result.citations);
            tool_outputs.push(ExecutedToolOutput {
                tool_name,
                args,
                success: true,
                text: tool_result.text,
            });
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

        self.record_planner_decision(
            ctx,
            "unified",
            decision_value,
            rationale,
            payload,
            success,
            error,
        )
        .await;
    }

    async fn record_tool_followup_decision(
        &self,
        ctx: &MessageCtx,
        round: usize,
        decision: &ToolFollowupDecision,
    ) {
        let (decision_value, rationale, payload, success, error) = match decision {
            ToolFollowupDecision::Final {
                rationale, payload, ..
            } => (
                "final_answer",
                rationale.clone(),
                payload.clone(),
                true,
                None,
            ),
            ToolFollowupDecision::UseTools {
                rationale, payload, ..
            } => (
                "request_tools",
                rationale.clone(),
                payload.clone(),
                true,
                None,
            ),
            ToolFollowupDecision::Fallback { reason, error } => (
                "fallback_no_tools",
                (*reason).to_owned(),
                json!({}),
                false,
                error.clone(),
            ),
        };

        self.record_planner_decision(
            ctx,
            "tool_followup",
            decision_value,
            rationale,
            json!({
                "round": round,
                "decision": payload
            }),
            success,
            error,
        )
        .await;
    }

    async fn record_planner_decision(
        &self,
        ctx: &MessageCtx,
        planner: &str,
        decision: &str,
        rationale: String,
        payload: Value,
        success: bool,
        error: Option<String>,
    ) {
        let record = PlannerDecisionRecord {
            user_id: ctx.user_id.clone(),
            guild_id: ctx.guild_id.clone(),
            channel_id: ctx.channel_id.clone(),
            planner: planner.to_owned(),
            decision: decision.to_owned(),
            rationale,
            payload_json: payload.to_string(),
            success,
            error,
            timestamp: Utc::now(),
        };

        if let Err(store_error) = self.memory.record_planner_decision(record).await {
            warn!(
                ?store_error,
                planner, "failed to persist planner decision log"
            );
        }
    }
}

fn build_unified_planner_prompt(memory: &crate::types::MemoryContext) -> String {
    let context_block = build_planner_context_block(memory);

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
For time-sensitive requests, call current_datetime before web_search so queries and answers are anchored to real current time.
If current_datetime is needed, request only current_datetime in this decision and wait for its output before planning web_search.
Tool inventory:
{}
{}",
        build_tool_inventory_for_planner(),
        context_block
    )
}

fn build_tool_followup_prompt(memory: &crate::types::MemoryContext) -> String {
    let context_block = build_planner_context_block(memory);

    format!(
        "You are the tool follow-up planner for CompanionPilot.
Decide whether the current evidence is enough for a final user-facing answer, or whether more tool calls are needed.
Return strict JSON only (no markdown, no prose) with this exact schema:
{{
  \"action\": \"final\"|\"tools\",
  \"final_answer\": \"non-empty only when action=final\",
  \"tool_calls\": [{{\"tool_name\":\"...\",\"args\":{{...}}}}],
  \"rationale\": \"short reason\"
}}
If action=final, provide the complete final answer and return an empty tool_calls array.
If action=tools, final_answer must be empty and tool_calls must contain at least one valid call.
Only request tools when the current outputs are insufficient or conflicting.
For time-sensitive requests, prefer calling current_datetime before additional web_search calls.
If current_datetime is needed, call it alone first, then plan web_search in a later tool round.
Tool inventory:
{}
{}",
        build_tool_inventory_for_planner(),
        context_block
    )
}

fn build_planner_context_block(memory: &crate::types::MemoryContext) -> String {
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

    if context_lines.is_empty() {
        String::new()
    } else {
        format!("Context:\n{}\n", context_lines.join("\n"))
    }
}

fn build_tool_inventory_for_planner() -> &'static str {
    r#"[
  {
    "tool_name": "current_datetime",
    "args_schema": {},
    "when_to_use": "Need the exact current date/time before time-sensitive lookups or answers.",
    "when_not_to_use": "Question is timeless or explicitly historical."
  },
  {
    "tool_name": "spotify_playing_status",
    "args_schema": {},
    "when_to_use": "Need the user's currently playing Spotify track/status.",
    "when_not_to_use": "Question is unrelated to Spotify playback."
  },
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

fn parse_tool_followup_plan(raw: &str) -> Result<ToolFollowupPlan, serde_json::Error> {
    parse_json_plan(raw)
}

fn sanitize_planned_tool_calls(planned_calls: Vec<PlannedToolCall>) -> Vec<ToolCall> {
    let mut sanitized_calls = Vec::new();

    for planned_call in planned_calls {
        if sanitized_calls.len() >= MAX_PLANNED_TOOL_CALLS {
            break;
        }
        match planned_call.tool_name.as_str() {
            "current_datetime" => {
                sanitized_calls.push(ToolCall {
                    tool_name: "current_datetime".to_owned(),
                    args: json!({}),
                });
            }
            "spotify_playing_status" => {
                sanitized_calls.push(ToolCall {
                    tool_name: "spotify_playing_status".to_owned(),
                    args: json!({}),
                });
            }
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

fn enforce_datetime_planning_boundary(tool_calls: Vec<ToolCall>) -> Vec<ToolCall> {
    let has_datetime = tool_calls
        .iter()
        .any(|call| call.tool_name == "current_datetime");
    let has_non_datetime = tool_calls
        .iter()
        .any(|call| call.tool_name != "current_datetime");
    if !has_datetime || !has_non_datetime {
        return tool_calls;
    }

    debug!(
        total_calls = tool_calls.len(),
        "deferring non-datetime tools to follow-up round because current_datetime was requested"
    );
    let datetime_call = tool_calls
        .into_iter()
        .find(|call| call.tool_name == "current_datetime")
        .expect("checked current_datetime presence above");
    vec![datetime_call]
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

fn elapsed_ms(started_at: Instant) -> u64 {
    started_at
        .elapsed()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
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
            let (status, label) = if output.success {
                ("success", "Output")
            } else {
                ("error", "Error")
            };
            format!(
                "{}. Tool: {}\nArgs: {}\nStatus: {}\n{}:\n{}",
                index + 1,
                output.tool_name,
                output.args,
                status,
                label,
                output.text
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn fallback_tool_output_text(outputs: &[ExecutedToolOutput]) -> String {
    outputs
        .iter()
        .map(|output| {
            if output.success {
                format!("{} output:\n{}", output.tool_name, output.text)
            } else {
                format!("{} error:\n{}", output.tool_name, output.text)
            }
        })
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

    use async_trait::async_trait;
    use chrono::Utc;
    use serde_json::{Value, json};

    use crate::{
        memory::{InMemoryMemoryStore, MemoryStore},
        model::{MockModelProvider, ModelProvider, ModelRequest},
        safety::SafetyPolicy,
        tools::{ToolExecutor, ToolRegistry, ToolResult},
        types::{MessageCtx, ToolCall},
    };

    use super::{
        DefaultChatOrchestrator, PlannedToolCall, clean_memory_value,
        enforce_datetime_planning_boundary, parse_unified_plan, sanitize_memory_key,
        sanitize_planned_tool_calls,
    };

    #[derive(Debug, Default)]
    struct FollowupLoopModelProvider;

    #[async_trait]
    impl ModelProvider for FollowupLoopModelProvider {
        async fn complete(&self, request: ModelRequest) -> anyhow::Result<String> {
            if request
                .system_prompt
                .contains("You are the unified planner for CompanionPilot.")
            {
                return Ok(json!({
                    "tool_calls": [
                        {
                            "tool_name": "web_search",
                            "args": {
                                "query": "alpha",
                                "max_results": 3
                            }
                        }
                    ],
                    "memory": {
                        "store": false,
                        "key": "",
                        "value": "",
                        "confidence": 0.0
                    },
                    "rationale": "need first lookup"
                })
                .to_string());
            }

            if request
                .system_prompt
                .contains("You are the tool follow-up planner for CompanionPilot.")
            {
                if request.user_prompt.contains("result:alpha")
                    && !request.user_prompt.contains("result:beta")
                {
                    return Ok(json!({
                        "action": "tools",
                        "final_answer": "",
                        "tool_calls": [
                            {
                                "tool_name": "web_search",
                                "args": {
                                    "query": "beta",
                                    "max_results": 2
                                }
                            }
                        ],
                        "rationale": "need second lookup"
                    })
                    .to_string());
                }

                if request.user_prompt.contains("result:beta") {
                    return Ok(json!({
                        "action": "final",
                        "final_answer": "Final answer from follow-up planner.",
                        "tool_calls": [],
                        "rationale": "have enough evidence"
                    })
                    .to_string());
                }
            }

            Ok("fallback final synthesis".to_owned())
        }
    }

    #[derive(Debug, Default)]
    struct StubWebSearchToolExecutor;

    #[async_trait]
    impl ToolExecutor for StubWebSearchToolExecutor {
        async fn execute(&self, tool_name: &str, args: Value) -> anyhow::Result<ToolResult> {
            if tool_name != "web_search" {
                return Err(anyhow::anyhow!("unknown tool: {tool_name}"));
            }

            let query = args
                .get("query")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow::anyhow!("missing query arg"))?;

            Ok(ToolResult {
                text: format!("result:{query}"),
                citations: vec![format!("https://example.com/{query}")],
            })
        }
    }

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
    async fn tool_failure_is_included_in_final_synthesis_context() {
        let memory = Arc::new(InMemoryMemoryStore::default());
        let orchestrator = DefaultChatOrchestrator::new(
            Arc::new(MockModelProvider),
            memory.clone(),
            Arc::new(ToolRegistry::default()),
            SafetyPolicy::default(),
        );

        let result = orchestrator
            .handle_message(MessageCtx {
                message_id: "3".into(),
                user_id: "u3".into(),
                guild_id: "g1".into(),
                channel_id: "c1".into(),
                content: "search the web for rust async traits".into(),
                timestamp: Utc::now(),
            })
            .await
            .expect("tool failure should still synthesize a final answer");

        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].tool_name, "web_search");
        assert!(result.text.contains("Status: error"));
        assert!(result.text.contains("web_search tool is not configured"));
    }

    #[tokio::test]
    async fn followup_planner_can_run_multiple_tool_rounds_before_final_answer() {
        let memory = Arc::new(InMemoryMemoryStore::default());
        let orchestrator = DefaultChatOrchestrator::new(
            Arc::new(FollowupLoopModelProvider),
            memory,
            Arc::new(StubWebSearchToolExecutor),
            SafetyPolicy::default(),
        );

        let result = orchestrator
            .handle_message(MessageCtx {
                message_id: "3b".into(),
                user_id: "u3b".into(),
                guild_id: "g1".into(),
                channel_id: "c1".into(),
                content: "find a final answer using tools".into(),
                timestamp: Utc::now(),
            })
            .await
            .expect("follow-up planning loop should complete");

        assert_eq!(result.tool_calls.len(), 2);
        assert_eq!(result.tool_calls[0].tool_name, "web_search");
        assert_eq!(result.tool_calls[0].args["query"], "alpha");
        assert_eq!(result.tool_calls[1].tool_name, "web_search");
        assert_eq!(result.tool_calls[1].args["query"], "beta");
        assert_eq!(result.text, "Final answer from follow-up planner.");
        assert_eq!(result.citations.len(), 2);
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

    #[test]
    fn sanitize_planned_tool_calls_allows_current_datetime() {
        let planned_calls = vec![PlannedToolCall {
            tool_name: "current_datetime".to_owned(),
            args: json!({"ignored": true}),
        }];

        let sanitized = sanitize_planned_tool_calls(planned_calls);
        assert_eq!(sanitized.len(), 1);
        assert_eq!(sanitized[0].tool_name, "current_datetime");
        assert_eq!(sanitized[0].args, json!({}));
    }

    #[test]
    fn sanitize_planned_tool_calls_preserves_datetime_then_search_order() {
        let planned_calls = vec![
            PlannedToolCall {
                tool_name: "current_datetime".to_owned(),
                args: json!({}),
            },
            PlannedToolCall {
                tool_name: "web_search".to_owned(),
                args: json!({
                    "query": "current weather in berlin",
                    "max_results": 5
                }),
            },
        ];

        let sanitized = sanitize_planned_tool_calls(planned_calls);
        assert_eq!(sanitized.len(), 2);
        assert_eq!(sanitized[0].tool_name, "current_datetime");
        assert_eq!(sanitized[1].tool_name, "web_search");
        let query = sanitized[1].args["query"]
            .as_str()
            .expect("query should be a string");
        assert_eq!(query, "current weather in berlin");
    }

    #[test]
    fn sanitize_planned_tool_calls_allows_spotify_playing_status() {
        let planned_calls = vec![PlannedToolCall {
            tool_name: "spotify_playing_status".to_owned(),
            args: json!({"ignored": true}),
        }];

        let sanitized = sanitize_planned_tool_calls(planned_calls);
        assert_eq!(sanitized.len(), 1);
        assert_eq!(sanitized[0].tool_name, "spotify_playing_status");
        assert_eq!(sanitized[0].args, json!({}));
    }

    #[test]
    fn enforce_datetime_planning_boundary_runs_datetime_in_isolation() {
        let calls = vec![
            ToolCall {
                tool_name: "web_search".to_owned(),
                args: json!({"query": "major video game releases late 2024 early 2025", "max_results": 10}),
            },
            ToolCall {
                tool_name: "current_datetime".to_owned(),
                args: json!({}),
            },
        ];

        let bounded = enforce_datetime_planning_boundary(calls);
        assert_eq!(bounded.len(), 1);
        assert_eq!(bounded[0].tool_name, "current_datetime");
    }

    #[test]
    fn enforce_datetime_planning_boundary_keeps_non_datetime_plans_unchanged() {
        let calls = vec![ToolCall {
            tool_name: "web_search".to_owned(),
            args: json!({"query": "rust async traits", "max_results": 3}),
        }];

        let bounded = enforce_datetime_planning_boundary(calls.clone());
        assert_eq!(bounded.len(), calls.len());
        assert_eq!(bounded[0].tool_name, "web_search");
        assert_eq!(bounded[0].args, calls[0].args);
    }
}
