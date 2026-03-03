use serde::Deserialize;
use serde_json::Value;

#[derive(Debug, Clone)]
pub(super) enum SkillSelectionDecision {
    UseSelection {
        selected_skill_ids: Vec<String>,
        rationale: String,
    },
    Fallback {
        reason: &'static str,
        error: Option<String>,
    },
}

#[derive(Debug, Clone)]
pub(super) enum NativeTurnDecision {
    ToolRequest {
        tool_count: usize,
        payload: Value,
    },
    FinalAnswer {
        response_text: String,
        payload: Value,
    },
    Fallback {
        reason: &'static str,
        error: Option<String>,
    },
}

#[derive(Debug, Default, Deserialize)]
pub(super) struct SkillSelectionPlan {
    #[serde(default)]
    pub(super) selected_skills: Vec<String>,
    #[serde(default)]
    pub(super) rationale: String,
}

#[derive(Debug, Clone)]
pub(super) struct ExecutedToolOutput {
    pub(super) tool_call_id: String,
    pub(super) tool_name: String,
    pub(super) args: Value,
    pub(super) success: bool,
    pub(super) text: String,
}
