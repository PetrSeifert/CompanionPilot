use chrono::Utc;
use serde_json::{Value, json};
use tracing::warn;

use crate::types::{MessageCtx, MessageLatencyRecord, PlannerDecisionRecord, ToolCallRecord};

use super::{
    DefaultChatOrchestrator,
    contracts::{SkillSelectionDecision, ToolFollowupDecision, UnifiedPlanDecision},
};

pub(super) fn truncate_for_log(input: &str, max_len: usize) -> String {
    let mut result = input.replace('\n', "\\n");
    if result.len() > max_len {
        let safe_len = if result.is_char_boundary(max_len) {
            max_len
        } else {
            result
                .char_indices()
                .map(|(index, _)| index)
                .take_while(|index| *index < max_len)
                .last()
                .unwrap_or(0)
        };
        result.truncate(safe_len);
        result.push_str("...");
    }
    result
}

impl DefaultChatOrchestrator {
    pub(super) async fn record_tool_call(&self, call: ToolCallRecord) {
        if let Err(error) = self.memory.record_tool_call(call).await {
            warn!(?error, "failed to persist tool call log");
        }
    }

    pub(super) async fn record_unified_planner_decision(
        &self,
        ctx: &MessageCtx,
        decision: &UnifiedPlanDecision,
        duration_ms: u64,
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
            duration_ms,
        )
        .await;
    }

    pub(super) async fn record_skill_selector_decision(
        &self,
        ctx: &MessageCtx,
        decision: &SkillSelectionDecision,
        duration_ms: u64,
    ) {
        let (decision_value, rationale, payload, success, error) = match decision {
            SkillSelectionDecision::UseSelection {
                rationale,
                selected_skill_ids,
            } => (
                "apply_selection",
                rationale.clone(),
                json!({
                    "selected_skills": selected_skill_ids,
                    "rationale": rationale,
                }),
                true,
                None,
            ),
            SkillSelectionDecision::Fallback { reason, error } => (
                "fallback_empty_selection",
                (*reason).to_owned(),
                json!({}),
                false,
                error.clone(),
            ),
        };

        self.record_planner_decision(
            ctx,
            "skill_selector",
            decision_value,
            rationale,
            payload,
            success,
            error,
            duration_ms,
        )
        .await;
    }

    pub(super) async fn record_tool_followup_decision(
        &self,
        ctx: &MessageCtx,
        round: usize,
        decision: &ToolFollowupDecision,
        duration_ms: u64,
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
            duration_ms,
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
        duration_ms: u64,
    ) {
        let record = PlannerDecisionRecord {
            user_id: ctx.user_id.clone(),
            guild_id: ctx.guild_id.clone(),
            channel_id: ctx.channel_id.clone(),
            message_id: ctx.message_id.clone(),
            planner: planner.to_owned(),
            decision: decision.to_owned(),
            rationale,
            payload_json: payload.to_string(),
            success,
            error,
            duration_ms,
            timestamp: Utc::now(),
        };

        if let Err(store_error) = self.memory.record_planner_decision(record).await {
            warn!(
                ?store_error,
                planner, "failed to persist planner decision log"
            );
        }
    }

    pub(super) async fn record_message_latency(&self, latency: MessageLatencyRecord) {
        if let Err(error) = self.memory.record_message_latency(latency).await {
            warn!(?error, "failed to persist message latency log");
        }
    }
}
