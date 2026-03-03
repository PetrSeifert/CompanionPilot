use std::{sync::Arc, time::Instant};

use async_trait::async_trait;
use chrono::Utc;
use tracing::{debug, info, warn};

use crate::{
    memory::MemoryStore,
    model::{ModelProvider, ModelRequest},
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
mod util;

#[cfg(test)]
mod tests;

use contracts::{
    MemoryDecision, SkillSelectionDecision, ToolFollowupDecision, UnifiedPlanDecision,
};
use prompts::{build_support_context_block, build_system_prompt};
use tool_exec::{dedupe_citations, fallback_tool_output_text, format_tool_outputs};
use util::elapsed_ms;

#[cfg(test)]
use contracts::PlannedToolCall;
#[cfg(test)]
use parse::parse_unified_plan;
#[cfg(test)]
use sanitize::{
    clean_memory_value, enforce_datetime_planning_boundary, sanitize_memory_key,
    sanitize_planned_tool_calls,
};
#[cfg(test)]
use telemetry::truncate_for_log;

pub const SLOW_REPLY_THRESHOLD_MS: u64 = 30_000;

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

        let planner_started_at = Instant::now();
        let planner_decision = self
            .decide_unified_plan(&ctx.content, &memory_context, &selected_skills)
            .await;
        let unified_planner_ms = elapsed_ms(planner_started_at);
        let mut planner_ms = skill_selector_ms.saturating_add(unified_planner_ms);
        self.record_unified_planner_decision(&ctx, &planner_decision, unified_planner_ms)
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

            let followup_started_at = Instant::now();
            let followup_decision = self
                .decide_tool_followup(
                    &ctx.content,
                    &memory_context,
                    &selected_skills,
                    &tool_outputs,
                )
                .await;
            let followup_ms = elapsed_ms(followup_started_at);
            planner_ms = planner_ms.saturating_add(followup_ms);
            self.record_tool_followup_decision(&ctx, tool_round, &followup_decision, followup_ms)
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
                            &selected_skills,
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
                            build_support_context_block(&memory_context, &selected_skills)
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
        self.record_message_latency(MessageLatencyRecord {
            user_id: ctx.user_id.clone(),
            guild_id: ctx.guild_id.clone(),
            channel_id: ctx.channel_id.clone(),
            message_id: ctx.message_id.clone(),
            stt_ms: None,
            tts_ms: None,
            final_response_ms: timings.total_ms,
            decision_ms: timings.planner_ms,
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
}

#[async_trait]
impl VoiceReplyOrchestrator for DefaultChatOrchestrator {
    async fn handle_voice_transcript(&self, message: MessageCtx) -> anyhow::Result<String> {
        let reply = self.handle_message(message).await?;
        Ok(reply.text)
    }
}
