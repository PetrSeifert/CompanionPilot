use tracing::warn;

use crate::model::ModelRequest;

use super::{
    DefaultChatOrchestrator, contracts::SkillSelectionDecision, parse::parse_skill_selection_plan,
    prompts::build_skill_selector_prompt, telemetry::truncate_for_log,
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
}
