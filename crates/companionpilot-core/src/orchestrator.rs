use std::{sync::Arc, time::Instant};

use async_trait::async_trait;
use chrono::Utc;
use serde_json::json;
use tracing::{info, warn};

use crate::{
    memory::MemoryStore,
    model::{ModelMessage, ModelMessageRole, ModelProvider, ModelRequest, ModelTurnRequest},
    safety::SafetyPolicy,
    skills::SkillCatalog,
    tools::ToolExecutor,
    types::{
        ChatMessageRecord, ChatRole, MessageCtx, MessageLatencyRecord, OrchestratorReply,
        ReplyTimings,
    },
    voice::VoiceReplyOrchestrator,
};

mod contracts;
mod parse;
mod planners;
mod prompts;
mod sanitize;
mod telemetry;
mod tool_exec;
mod tool_schema;
mod util;

#[cfg(test)]
mod tests;

use contracts::{NativeTurnDecision, SkillSelectionDecision};
use prompts::build_native_agent_system_prompt;
use sanitize::sanitize_native_tool_calls;
use tool_exec::{dedupe_citations, fallback_tool_output_text};
use tool_schema::build_native_tool_definitions;
use util::elapsed_ms;

pub const SLOW_REPLY_THRESHOLD_MS: u64 = 30_000;
const MAX_NATIVE_TOOL_ROUNDS: usize = 6;

pub struct DefaultChatOrchestrator {
    model: Arc<dyn ModelProvider>,
    memory: Arc<dyn MemoryStore>,
    tools: Arc<dyn ToolExecutor>,
    skills: Arc<SkillCatalog>,
    safety: SafetyPolicy,
}

pub fn default_system_prompt_base() -> &'static str {
    prompts::default_system_prompt_base()
}

impl DefaultChatOrchestrator {
    pub fn new(
        model: Arc<dyn ModelProvider>,
        memory: Arc<dyn MemoryStore>,
        tools: Arc<dyn ToolExecutor>,
        skills: Arc<SkillCatalog>,
        safety: SafetyPolicy,
    ) -> Self {
        Self {
            model,
            memory,
            tools,
            skills,
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

        let skill_selector_started_at = Instant::now();
        let skill_selection_decision = self
            .decide_selected_skills(&ctx.content, &memory_context)
            .await;
        let skill_selector_ms = elapsed_ms(skill_selector_started_at);
        self.record_skill_selector_decision(&ctx, &skill_selection_decision, skill_selector_ms)
            .await;

        let selected_skill_ids = match &skill_selection_decision {
            SkillSelectionDecision::UseSelection {
                selected_skill_ids, ..
            } => selected_skill_ids.clone(),
            SkillSelectionDecision::Fallback { .. } => Vec::new(),
        };
        let selected_skills = self.skills.select_by_ids(&selected_skill_ids);

        let mut decision_ms = skill_selector_ms;
        let mut executed_tool_calls = Vec::new();
        let mut tool_outputs = Vec::new();
        let mut citations = Vec::new();
        let mut tool_timings = Vec::new();

        let native_system_prompt = build_native_agent_system_prompt(
            &memory_context,
            system_prompt_override.as_deref(),
            &selected_skills,
        );
        let tool_definitions = build_native_tool_definitions();
        let mut conversation_messages = vec![ModelMessage {
            role: ModelMessageRole::User,
            content: ctx.content.clone(),
            name: None,
            tool_call_id: None,
            tool_calls: Vec::new(),
        }];

        let mut native_reply_text: Option<String> = None;

        for round in 1..=MAX_NATIVE_TOOL_ROUNDS {
            let decision_started_at = Instant::now();
            let turn_result = self
                .model
                .complete_turn(ModelTurnRequest {
                    system_prompt: native_system_prompt.clone(),
                    messages: conversation_messages.clone(),
                    tools: tool_definitions.clone(),
                })
                .await;
            let round_decision_ms = elapsed_ms(decision_started_at);
            decision_ms = decision_ms.saturating_add(round_decision_ms);

            let turn_result = match turn_result {
                Ok(result) => result,
                Err(error) => {
                    self.record_native_turn_decision(
                        &ctx,
                        round,
                        &NativeTurnDecision::Fallback {
                            reason: "model_error",
                            error: Some(error.to_string()),
                        },
                        round_decision_ms,
                    )
                    .await;
                    warn!(
                        ?error,
                        round, "native tool decision call failed; stopping tool loop"
                    );
                    break;
                }
            };

            let assistant_text = turn_result.assistant_text.trim().to_owned();
            let raw_tool_calls = turn_result.tool_calls.clone();
            let planned_tool_calls = sanitize_native_tool_calls(turn_result.tool_calls);
            conversation_messages.push(ModelMessage {
                role: ModelMessageRole::Assistant,
                content: assistant_text.clone(),
                name: None,
                tool_call_id: None,
                tool_calls: raw_tool_calls,
            });

            if planned_tool_calls.is_empty() {
                self.record_native_turn_decision(
                    &ctx,
                    round,
                    &NativeTurnDecision::FinalAnswer {
                        response_text: assistant_text.clone(),
                        payload: json!({
                            "assistant_text": assistant_text,
                            "tool_calls": []
                        }),
                    },
                    round_decision_ms,
                )
                .await;

                if !assistant_text.is_empty() {
                    native_reply_text = Some(assistant_text);
                }
                break;
            }

            self.record_native_turn_decision(
                &ctx,
                round,
                &NativeTurnDecision::ToolRequest {
                    tool_count: planned_tool_calls.len(),
                    payload: json!({
                        "assistant_text_preview": assistant_text,
                        "tool_calls": planned_tool_calls
                            .iter()
                            .map(|call| json!({
                                "id": call.call_id,
                                "tool_name": call.call.tool_name,
                                "args": call.call.args,
                            }))
                            .collect::<Vec<_>>()
                    }),
                },
                round_decision_ms,
            )
            .await;

            let source = format!("native_round_{round}");
            let tool_messages = self
                .execute_native_tool_round(
                    &ctx,
                    planned_tool_calls,
                    &source,
                    &mut executed_tool_calls,
                    &mut tool_outputs,
                    &mut citations,
                    &mut tool_timings,
                )
                .await;
            conversation_messages.extend(tool_messages);
        }

        let tool_execution_ms = tool_timings.iter().fold(0u64, |total, timing| {
            total.saturating_add(timing.duration_ms)
        });

        let (reply_text, final_model_ms) = if let Some(answer) = native_reply_text {
            (answer, 0)
        } else if !tool_outputs.is_empty() {
            let final_model_started_at = Instant::now();
            let fallback_synthesis_prompt = format!(
                "{}\n\nFallback synthesis mode: produce the best final answer from available tool outputs. Do not request tools in this step.",
                native_system_prompt
            );
            let synthesized = self
                .model
                .complete(ModelRequest {
                    system_prompt: fallback_synthesis_prompt,
                    user_prompt: format!(
                        "User request:\n{}\n\nTool outputs:\n{}",
                        ctx.content,
                        tool_exec::format_tool_outputs(&tool_outputs)
                    ),
                })
                .await
                .unwrap_or_else(|error| {
                    warn!(
                        ?error,
                        "native tool loop did not produce final answer; using tool fallback output"
                    );
                    fallback_tool_output_text(&tool_outputs)
                });
            (synthesized, elapsed_ms(final_model_started_at))
        } else {
            (
                "I could not complete this request due to an internal orchestration error."
                    .to_owned(),
                0,
            )
        };

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
            decision_ms,
            tool_execution_ms,
            final_model_ms,
            record_assistant_message_ms,
            tool_calls: tool_timings,
        };
        self.record_message_latency(MessageLatencyRecord {
            user_id: ctx.user_id.clone(),
            guild_id: ctx.guild_id.clone(),
            channel_id: ctx.channel_id.clone(),
            message_id: ctx.message_id.clone(),
            stt_ms: None,
            tts_ms: None,
            final_response_ms: timings.total_ms,
            decision_ms: timings.decision_ms,
            tool_call_ms: timings.tool_execution_ms,
            timestamp: Utc::now(),
        })
        .await;

        if timings.total_ms >= SLOW_REPLY_THRESHOLD_MS {
            warn!(
                user_id = %ctx.user_id,
                guild_id = %ctx.guild_id,
                channel_id = %ctx.channel_id,
                message_id = %ctx.message_id,
                total_ms = timings.total_ms,
                decision_ms = timings.decision_ms,
                tool_execution_ms = timings.tool_execution_ms,
                final_model_ms = timings.final_model_ms,
                "slow reply detected"
            );
        } else {
            info!(
                user_id = %ctx.user_id,
                guild_id = %ctx.guild_id,
                channel_id = %ctx.channel_id,
                message_id = %ctx.message_id,
                total_ms = timings.total_ms,
                decision_ms = timings.decision_ms,
                tool_execution_ms = timings.tool_execution_ms,
                final_model_ms = timings.final_model_ms,
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
}

#[async_trait]
impl VoiceReplyOrchestrator for DefaultChatOrchestrator {
    async fn handle_voice_transcript(&self, message: MessageCtx) -> anyhow::Result<String> {
        let reply = self.handle_message(message).await?;
        Ok(reply.text)
    }
}
