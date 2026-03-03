use chrono::Utc;
use serde_json::{Value, json};
use tracing::warn;

use crate::types::{MessageCtx, MessageLatencyRecord, OrchestrationDecisionRecord, ToolCallRecord};

use super::{
    DefaultChatOrchestrator,
    contracts::{NativeTurnDecision, SkillSelectionDecision},
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

        self.record_orchestration_decision(
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

    pub(super) async fn record_native_turn_decision(
        &self,
        ctx: &MessageCtx,
        round: usize,
        decision: &NativeTurnDecision,
        duration_ms: u64,
    ) {
        let (decision_value, rationale, payload, success, error) = match decision {
            NativeTurnDecision::ToolRequest {
                tool_count,
                payload,
            } => (
                "request_tools",
                format!("round_{round}_tool_request"),
                json!({
                    "round": round,
                    "tool_count": tool_count,
                    "decision": payload
                }),
                true,
                None,
            ),
            NativeTurnDecision::FinalAnswer {
                response_text,
                payload,
            } => (
                "final_answer",
                format!("round_{round}_final"),
                json!({
                    "round": round,
                    "response_preview": truncate_for_log(response_text, 180),
                    "decision": payload
                }),
                true,
                None,
            ),
            NativeTurnDecision::Fallback { reason, error } => (
                "fallback",
                format!("round_{round}_{reason}"),
                json!({ "round": round }),
                false,
                error.clone(),
            ),
        };

        self.record_orchestration_decision(
            ctx,
            "native_turn",
            decision_value,
            rationale,
            payload,
            success,
            error,
            duration_ms,
        )
        .await;
    }

    async fn record_orchestration_decision(
        &self,
        ctx: &MessageCtx,
        stage: &str,
        decision: &str,
        rationale: String,
        payload: Value,
        success: bool,
        error: Option<String>,
        duration_ms: u64,
    ) {
        let record = OrchestrationDecisionRecord {
            user_id: ctx.user_id.clone(),
            guild_id: ctx.guild_id.clone(),
            channel_id: ctx.channel_id.clone(),
            message_id: ctx.message_id.clone(),
            stage: stage.to_owned(),
            decision: decision.to_owned(),
            rationale,
            payload_json: payload.to_string(),
            success,
            error,
            duration_ms,
            timestamp: Utc::now(),
        };

        if let Err(store_error) = self.memory.record_orchestration_decision(record).await {
            warn!(
                ?store_error,
                stage, "failed to persist orchestration decision log"
            );
        }
    }

    pub(super) async fn record_message_latency(&self, latency: MessageLatencyRecord) {
        if let Err(error) = self.memory.record_message_latency(latency).await {
            warn!(?error, "failed to persist message latency log");
        }
    }
}
