use serde_json::json;
use tracing::warn;

use crate::{model::ModelRequest, skills::SkillDefinition};

use super::{
    DefaultChatOrchestrator,
    contracts::{
        ExecutedToolOutput, SkillSelectionDecision, ToolFollowupDecision, UnifiedPlanDecision,
    },
    parse::{parse_skill_selection_plan, parse_tool_followup_plan, parse_unified_plan},
    prompts::{
        build_skill_selector_prompt, build_tool_followup_prompt, build_unified_planner_prompt,
    },
    sanitize::{
        enforce_datetime_planning_boundary, memory_decision_from_plan, memory_payload,
        sanitize_planned_tool_calls,
    },
    telemetry::truncate_for_log,
    tool_exec::format_tool_outputs,
};

impl DefaultChatOrchestrator {
    pub(super) async fn decide_selected_skills(
        &self,
        user_input: &str,
        memory: &crate::types::MemoryContext,
    ) -> SkillSelectionDecision {
        let skill_metadata = self.skills.metadata_inventory();
        let selector_prompt = build_skill_selector_prompt(memory, &skill_metadata);
        let selector_result = self
            .model
            .complete(ModelRequest {
                system_prompt: selector_prompt,
                user_prompt: user_input.to_owned(),
            })
            .await;

        let selector_result = match selector_result {
            Ok(content) => content,
            Err(error) => {
                warn!(?error, "skill selector model call failed");
                return SkillSelectionDecision::Fallback {
                    reason: "skill_selector_model_error",
                    error: Some(error.to_string()),
                };
            }
        };

        match parse_skill_selection_plan(&selector_result) {
            Ok(plan) => {
                let selected_skill_ids = self.skills.sanitize_selected_ids(plan.selected_skills);
                let rationale = if plan.rationale.trim().is_empty() {
                    "model_skill_selector".to_owned()
                } else {
                    plan.rationale.trim().to_owned()
                };

                SkillSelectionDecision::UseSelection {
                    rationale,
                    selected_skill_ids,
                }
            }
            Err(error) => {
                warn!(
                    ?error,
                    selector_output = %truncate_for_log(&selector_result, 220),
                    "failed to parse skill selector output"
                );
                SkillSelectionDecision::Fallback {
                    reason: "skill_selector_parse_error",
                    error: Some(error.to_string()),
                }
            }
        }
    }

    pub(super) async fn decide_unified_plan(
        &self,
        user_input: &str,
        memory: &crate::types::MemoryContext,
        selected_skills: &[SkillDefinition],
    ) -> UnifiedPlanDecision {
        let planner_prompt = build_unified_planner_prompt(memory, selected_skills);
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

    pub(super) async fn decide_tool_followup(
        &self,
        user_input: &str,
        memory: &crate::types::MemoryContext,
        selected_skills: &[SkillDefinition],
        tool_outputs: &[ExecutedToolOutput],
    ) -> ToolFollowupDecision {
        let planner_prompt = build_tool_followup_prompt(memory, selected_skills);
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
}
