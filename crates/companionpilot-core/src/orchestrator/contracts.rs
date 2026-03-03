use serde::Deserialize;
use serde_json::Value;

use crate::types::{MemoryFact, ToolCall};

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

pub(super) enum UnifiedPlanDecision {
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

pub(super) enum MemoryDecision {
    Store {
        fact: MemoryFact,
        rationale: &'static str,
    },
    Skip {
        reason: &'static str,
    },
}

pub(super) enum ToolFollowupDecision {
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
pub(super) struct UnifiedPlan {
    #[serde(default)]
    pub(super) tool_calls: Vec<PlannedToolCall>,
    #[serde(default)]
    pub(super) memory: PlannedMemory,
    #[serde(default)]
    pub(super) rationale: String,
}

#[derive(Debug, Deserialize)]
pub(super) struct PlannedToolCall {
    pub(super) tool_name: String,
    #[serde(default)]
    pub(super) args: Value,
}

#[derive(Debug, Default, Deserialize)]
pub(super) struct PlannedMemory {
    #[serde(default)]
    pub(super) store: bool,
    #[serde(default)]
    pub(super) key: String,
    #[serde(default)]
    pub(super) value: String,
    #[serde(default)]
    pub(super) confidence: f32,
}

#[derive(Debug, Default, Deserialize)]
pub(super) struct ToolFollowupPlan {
    #[serde(default)]
    pub(super) action: String,
    #[serde(default)]
    pub(super) final_answer: String,
    #[serde(default)]
    pub(super) tool_calls: Vec<PlannedToolCall>,
    #[serde(default)]
    pub(super) rationale: String,
}

#[derive(Debug, Default, Deserialize)]
pub(super) struct SkillSelectionPlan {
    #[serde(default)]
    pub(super) selected_skills: Vec<String>,
    #[serde(default)]
    pub(super) rationale: String,
}

pub(super) struct ExecutedToolOutput {
    pub(super) tool_name: String,
    pub(super) args: Value,
    pub(super) success: bool,
    pub(super) text: String,
}
