use std::time::Instant;

use chrono::Utc;
use futures::future::join_all;
use tracing::{info, warn};

use crate::{
    model::{ModelMessage, ModelMessageRole},
    types::{MessageCtx, ToolCall, ToolCallRecord, ToolCallTiming},
};

use super::{
    DefaultChatOrchestrator,
    contracts::ExecutedToolOutput,
    sanitize::{SanitizedToolCall, memory_fact_from_store_memory_args},
    telemetry::truncate_for_log,
    util::elapsed_ms,
};

pub(super) struct ToolExecutionResult {
    pub(super) output: ExecutedToolOutput,
    pub(super) timing: ToolCallTiming,
    pub(super) citations: Vec<String>,
}

impl DefaultChatOrchestrator {
    pub(super) async fn execute_native_tool_round(
        &self,
        ctx: &MessageCtx,
        planned_tool_calls: Vec<SanitizedToolCall>,
        source: &str,
        executed_tool_calls: &mut Vec<ToolCall>,
        tool_outputs: &mut Vec<ExecutedToolOutput>,
        citations: &mut Vec<String>,
        tool_timings: &mut Vec<ToolCallTiming>,
    ) -> Vec<ModelMessage> {
        let (program_calls, normal_calls): (Vec<_>, Vec<_>) = planned_tool_calls
            .into_iter()
            .partition(|planned_call| planned_call.call.tool_name == "execute_program");

        let futures = normal_calls
            .into_iter()
            .map(|planned_call| {
                let call = planned_call.call.clone();
                executed_tool_calls.push(call.clone());
                self.execute_single_tool_call(ctx, planned_call, source.to_owned())
            })
            .collect::<Vec<_>>();

        let results = join_all(futures).await;
        let mut tool_messages =
            Vec::with_capacity(results.len().saturating_add(program_calls.len()));
        for result in results {
            citations.extend(result.citations.clone());
            tool_timings.push(result.timing.clone());
            tool_messages.push(ModelMessage {
                role: ModelMessageRole::Tool,
                content: result.output.text.clone(),
                name: Some(result.output.tool_name.clone()),
                tool_call_id: Some(result.output.tool_call_id.clone()),
                tool_calls: Vec::new(),
                reasoning: None,
            });
            tool_outputs.push(result.output);
        }

        for planned_program_call in program_calls {
            let call = planned_program_call.call.clone();
            let call_id = planned_program_call.call_id.clone();
            executed_tool_calls.push(call.clone());

            let program_result = self
                .execute_program(ctx, planned_program_call, source.to_owned())
                .await;

            for step in &program_result.steps {
                citations.extend(step.citations.clone());
                tool_timings.push(step.timing.clone());
                executed_tool_calls.push(step.tool_call.clone());
            }

            let output = ExecutedToolOutput {
                tool_call_id: call_id,
                tool_name: call.tool_name,
                args: call.args,
                success: program_result.success,
                text: program_result.combined_text,
            };
            tool_messages.push(ModelMessage {
                role: ModelMessageRole::Tool,
                content: output.text.clone(),
                name: Some(output.tool_name.clone()),
                tool_call_id: Some(output.tool_call_id.clone()),
                tool_calls: Vec::new(),
                reasoning: None,
            });
            tool_outputs.push(output);
        }

        tool_messages
    }

    pub(super) async fn execute_single_tool_call(
        &self,
        ctx: &MessageCtx,
        planned_call: SanitizedToolCall,
        source: String,
    ) -> ToolExecutionResult {
        let call_id = planned_call.call_id;
        let tool_name = planned_call.call.tool_name;
        let args = planned_call.call.args;

        info!(
            user_id = %ctx.user_id,
            guild_id = %ctx.guild_id,
            channel_id = %ctx.channel_id,
            source = %source,
            tool_name = %tool_name,
            args_json = %args,
            "tool call selected by native orchestration"
        );

        let started_at = Instant::now();
        let (success, text, citations, error) = if tool_name == "store_memory" {
            match memory_fact_from_store_memory_args(&args) {
                Some(fact) => match self.memory.upsert_fact(&ctx.user_id, fact.clone()).await {
                    Ok(()) => (
                        true,
                        format!(
                            "Stored memory fact {}={} (confidence {:.2})",
                            fact.key, fact.value, fact.confidence
                        ),
                        Vec::new(),
                        None,
                    ),
                    Err(error) => (
                        false,
                        format!("store_memory failed: {error}"),
                        Vec::new(),
                        Some(error.to_string()),
                    ),
                },
                None => (
                    false,
                    "store_memory failed: invalid key/value payload".to_owned(),
                    Vec::new(),
                    Some("invalid_store_memory_payload".to_owned()),
                ),
            }
        } else {
            match self.tools.execute(&tool_name, args.clone(), ctx).await {
                Ok(result) => (true, result.text, result.citations, None),
                Err(error) => (
                    false,
                    error.to_string(),
                    Vec::new(),
                    Some(error.to_string()),
                ),
            }
        };
        let duration_ms = elapsed_ms(started_at);

        self.record_tool_call(ToolCallRecord {
            user_id: ctx.user_id.clone(),
            guild_id: ctx.guild_id.clone(),
            channel_id: ctx.channel_id.clone(),
            message_id: ctx.message_id.clone(),
            tool_name: tool_name.clone(),
            source: source.clone(),
            args_json: args.to_string(),
            result_text: truncate_for_log(&text, 1_200),
            citations: citations.clone(),
            success,
            error: error.clone(),
            duration_ms,
            timestamp: Utc::now(),
        })
        .await;

        if success {
            info!(
                user_id = %ctx.user_id,
                source = %source,
                tool_name = %tool_name,
                duration_ms,
                result_citations = citations.len(),
                "tool call completed"
            );
        } else {
            warn!(
                user_id = %ctx.user_id,
                guild_id = %ctx.guild_id,
                channel_id = %ctx.channel_id,
                source = %source,
                tool_name = %tool_name,
                duration_ms,
                error = ?error,
                "tool call failed; continuing orchestration"
            );
        }

        ToolExecutionResult {
            output: ExecutedToolOutput {
                tool_call_id: call_id,
                tool_name: tool_name.clone(),
                args,
                success,
                text,
            },
            timing: ToolCallTiming {
                tool_name,
                duration_ms,
                success,
            },
            citations,
        }
    }
}

pub(super) fn format_tool_outputs(outputs: &[ExecutedToolOutput]) -> String {
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

pub(super) fn fallback_tool_output_text(outputs: &[ExecutedToolOutput]) -> String {
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

pub(super) fn dedupe_citations(citations: Vec<String>) -> Vec<String> {
    let mut deduped = Vec::new();
    for citation in citations {
        if deduped.iter().any(|existing| existing == &citation) {
            continue;
        }
        deduped.push(citation);
    }
    deduped
}
