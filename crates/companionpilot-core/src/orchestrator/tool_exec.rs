use std::time::Instant;

use chrono::Utc;
use tracing::{info, warn};

use crate::types::{MessageCtx, ToolCall, ToolCallRecord, ToolCallTiming};

use super::{
    DefaultChatOrchestrator, contracts::ExecutedToolOutput, telemetry::truncate_for_log,
    util::elapsed_ms,
};

impl DefaultChatOrchestrator {
    pub(super) async fn execute_planned_tool_calls(
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
            let planned_args = tool_call.args.clone();
            let args = planned_args.clone();
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
                planned_args_json = %planned_args,
                args_json = %args,
                "tool call selected by unified planner"
            );

            let tool_result = match self.tools.execute(&tool_name, args.clone(), ctx).await {
                Ok(result) => result,
                Err(error) => {
                    let error_text = error.to_string();
                    let duration_ms = elapsed_ms(tool_started_at);
                    self.record_tool_call(ToolCallRecord {
                        user_id: ctx.user_id.clone(),
                        guild_id: ctx.guild_id.clone(),
                        channel_id: ctx.channel_id.clone(),
                        message_id: ctx.message_id.clone(),
                        tool_name: tool_name.clone(),
                        source: source.to_owned(),
                        args_json: args.to_string(),
                        result_text: String::new(),
                        citations: Vec::new(),
                        success: false,
                        error: Some(error_text.clone()),
                        duration_ms,
                        timestamp: Utc::now(),
                    })
                    .await;
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
            let duration_ms = elapsed_ms(tool_started_at);

            self.record_tool_call(ToolCallRecord {
                user_id: ctx.user_id.clone(),
                guild_id: ctx.guild_id.clone(),
                channel_id: ctx.channel_id.clone(),
                message_id: ctx.message_id.clone(),
                tool_name: tool_name.clone(),
                source: source.to_owned(),
                args_json: args.to_string(),
                result_text: truncate_for_log(&tool_result.text, 1200),
                citations: tool_result.citations.clone(),
                success: true,
                error: None,
                duration_ms,
                timestamp: Utc::now(),
            })
            .await;
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
