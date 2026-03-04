use std::collections::HashMap;

use serde_json::Value;

use crate::{
    model::ModelToolCall,
    types::{MessageCtx, ToolCall, ToolCallTiming},
};

use super::{
    DefaultChatOrchestrator,
    contracts::ExecutedToolOutput,
    sanitize::{SanitizedToolCall, sanitize_native_tool_calls},
};

#[derive(Debug, Clone)]
pub(super) struct ProgramStepExecution {
    pub(super) step_id: String,
    pub(super) output: ExecutedToolOutput,
    pub(super) timing: ToolCallTiming,
    pub(super) citations: Vec<String>,
    pub(super) tool_call: ToolCall,
}

#[derive(Debug, Clone)]
pub(super) struct ProgramExecutionResult {
    pub(super) combined_text: String,
    pub(super) steps: Vec<ProgramStepExecution>,
    pub(super) success: bool,
}

impl DefaultChatOrchestrator {
    pub(super) async fn execute_program(
        &self,
        ctx: &MessageCtx,
        planned_program_call: SanitizedToolCall,
        source: String,
    ) -> ProgramExecutionResult {
        let program_call_id = planned_program_call.call_id;
        let steps = planned_program_call
            .call
            .args
            .get("steps")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();

        let mut step_outputs_by_id = HashMap::<String, String>::new();
        let mut step_executions = Vec::with_capacity(steps.len());
        let mut success = true;

        for step in steps {
            let step_id = step
                .get("step_id")
                .and_then(Value::as_str)
                .map(str::trim)
                .unwrap_or_default()
                .to_owned();
            let tool_name = step
                .get("tool_name")
                .and_then(Value::as_str)
                .map(str::trim)
                .unwrap_or_default()
                .to_owned();
            let arguments = step
                .get("arguments")
                .cloned()
                .unwrap_or_else(|| Value::Object(Default::default()));
            let substituted_arguments = substitute_variables(&arguments, &step_outputs_by_id);

            let synthetic_call_id = format!("{program_call_id}::{step_id}");
            let synthetic_raw_call = ModelToolCall {
                id: synthetic_call_id.clone(),
                name: tool_name.clone(),
                arguments: substituted_arguments.clone(),
            };

            let mut sanitized_calls = sanitize_native_tool_calls(vec![synthetic_raw_call]);
            if sanitized_calls.len() != 1 {
                success = false;
                step_executions.push(ProgramStepExecution {
                    step_id: step_id.clone(),
                    output: ExecutedToolOutput {
                        tool_call_id: synthetic_call_id,
                        tool_name: tool_name.clone(),
                        args: substituted_arguments.clone(),
                        success: false,
                        text: format!(
                            "Program step `{step_id}` failed validation for tool `{tool_name}`"
                        ),
                    },
                    timing: ToolCallTiming {
                        tool_name: tool_name.clone(),
                        duration_ms: 0,
                        success: false,
                    },
                    citations: Vec::new(),
                    tool_call: ToolCall {
                        tool_name,
                        args: substituted_arguments,
                    },
                });
                break;
            }

            let sanitized_call = sanitized_calls.remove(0);
            let step_tool_call = sanitized_call.call.clone();
            let step_source = format!("{source}:program_step:{step_id}");
            let step_result = self
                .execute_single_tool_call(ctx, sanitized_call, step_source)
                .await;
            let step_success = step_result.output.success;
            step_outputs_by_id.insert(step_id.clone(), step_result.output.text.clone());
            step_executions.push(ProgramStepExecution {
                step_id,
                output: step_result.output,
                timing: step_result.timing,
                citations: step_result.citations,
                tool_call: step_tool_call,
            });

            if !step_success {
                success = false;
                break;
            }
        }

        ProgramExecutionResult {
            combined_text: format_program_result(&step_executions),
            steps: step_executions,
            success,
        }
    }
}

pub(super) fn substitute_variables(
    value: &Value,
    step_outputs_by_id: &HashMap<String, String>,
) -> Value {
    match value {
        Value::String(text) => Value::String(substitute_string_variables(text, step_outputs_by_id)),
        Value::Array(items) => Value::Array(
            items
                .iter()
                .map(|item| substitute_variables(item, step_outputs_by_id))
                .collect::<Vec<_>>(),
        ),
        Value::Object(object) => Value::Object(
            object
                .iter()
                .map(|(key, value)| (key.clone(), substitute_variables(value, step_outputs_by_id)))
                .collect(),
        ),
        other => other.clone(),
    }
}

fn substitute_string_variables(
    input: &str,
    step_outputs_by_id: &HashMap<String, String>,
) -> String {
    let mut output = String::with_capacity(input.len());
    let mut cursor = 0usize;

    while let Some(relative_start) = input[cursor..].find("${") {
        let start = cursor + relative_start;
        output.push_str(&input[cursor..start]);

        let token_start = start + 2;
        let remainder = &input[token_start..];
        let Some(relative_end) = remainder.find('}') else {
            output.push_str(&input[start..]);
            cursor = input.len();
            break;
        };
        let token_end = token_start + relative_end;
        let step_id = &input[token_start..token_end];
        if let Some(replacement) = step_outputs_by_id.get(step_id) {
            output.push_str(replacement);
        } else {
            output.push_str(&input[start..=token_end]);
        }

        cursor = token_end + 1;
    }

    if cursor < input.len() {
        output.push_str(&input[cursor..]);
    }

    output
}

fn format_program_result(step_executions: &[ProgramStepExecution]) -> String {
    if step_executions.is_empty() {
        return "Program execution produced no steps.".to_owned();
    }

    step_executions
        .iter()
        .enumerate()
        .map(|(index, step)| {
            let status = if step.output.success {
                "success"
            } else {
                "error"
            };
            format!(
                "{}. Step: {}\nTool: {}\nStatus: {}\nOutput:\n{}",
                index + 1,
                step.step_id,
                step.output.tool_name,
                status,
                step.output.text
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}
