use std::{
    fs,
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};

use async_trait::async_trait;
use chrono::Utc;
use serde_json::{Value, json};

use crate::{
    memory::{InMemoryMemoryStore, MemoryStore},
    model::{
        MockModelProvider, ModelMessageRole, ModelProvider, ModelRequest, ModelToolCall,
        ModelTurnRequest, ModelTurnResponse,
    },
    safety::SafetyPolicy,
    skills::SkillCatalog,
    tools::{ToolExecutor, ToolRegistry, ToolResult},
    types::MessageCtx,
};

use super::telemetry::truncate_for_log;
use super::{DefaultChatOrchestrator, sanitize::clean_memory_value, sanitize::sanitize_memory_key};

fn empty_skill_catalog() -> Arc<SkillCatalog> {
    Arc::new(SkillCatalog::default())
}

fn test_skill_catalog_with_marker() -> Arc<SkillCatalog> {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock should be after epoch")
        .as_nanos();
    let temp_dir = std::env::temp_dir().join(format!(
        "companionpilot-orchestrator-skills-{}-{nanos}",
        std::process::id()
    ));
    fs::create_dir_all(&temp_dir).expect("temp skill directory should be created");
    fs::write(
        temp_dir.join("focus.md"),
        r#"---
id: focus-skill
title: Focus Skill
description: Keep attention on key points.
tags: [focus, testing]
---
THIS_BODY_MARKER_SHOULD_NOT_APPEAR_IN_SELECTOR
"#,
    )
    .expect("skill file should be written");

    let catalog = SkillCatalog::load_from_dir(&temp_dir).expect("skill catalog should load");
    let _ = fs::remove_dir_all(temp_dir);
    Arc::new(catalog)
}

#[derive(Debug, Default)]
struct FollowupLoopModelProvider;

#[async_trait]
impl ModelProvider for FollowupLoopModelProvider {
    async fn complete(&self, request: ModelRequest) -> anyhow::Result<String> {
        if request
            .system_prompt
            .contains("You are the skill selector for CompanionPilot.")
        {
            return Ok(json!({
                "selected_skills": [],
                "rationale": "mock_selector_default"
            })
            .to_string());
        }
        Ok("fallback final synthesis".to_owned())
    }

    async fn complete_turn(&self, request: ModelTurnRequest) -> anyhow::Result<ModelTurnResponse> {
        let saw_alpha = request.messages.iter().any(|message| {
            message.role == ModelMessageRole::Tool && message.content.contains("result:alpha")
        });
        let saw_beta = request.messages.iter().any(|message| {
            message.role == ModelMessageRole::Tool && message.content.contains("result:beta")
        });

        if !saw_alpha {
            return Ok(ModelTurnResponse {
                assistant_text: String::new(),
                tool_calls: vec![ModelToolCall {
                    id: "alpha-call".to_owned(),
                    name: "web_search".to_owned(),
                    arguments: json!({
                        "query": "alpha",
                        "max_results": 3
                    }),
                }],
            });
        }
        if !saw_beta {
            return Ok(ModelTurnResponse {
                assistant_text: String::new(),
                tool_calls: vec![ModelToolCall {
                    id: "beta-call".to_owned(),
                    name: "web_search".to_owned(),
                    arguments: json!({
                        "query": "beta",
                        "max_results": 2
                    }),
                }],
            });
        }

        Ok(ModelTurnResponse {
            assistant_text: "Final answer from native tool loop.".to_owned(),
            tool_calls: Vec::new(),
        })
    }
}

#[derive(Debug, Default)]
struct SkillSelectionContractModelProvider {
    stages: Mutex<Vec<String>>,
}

#[async_trait]
impl ModelProvider for SkillSelectionContractModelProvider {
    async fn complete(&self, request: ModelRequest) -> anyhow::Result<String> {
        if request
            .system_prompt
            .contains("You are the skill selector for CompanionPilot.")
        {
            assert!(
                !request
                    .system_prompt
                    .contains("THIS_BODY_MARKER_SHOULD_NOT_APPEAR_IN_SELECTOR"),
                "selector prompt must not include markdown body"
            );
            self.stages
                .lock()
                .expect("stages lock should succeed")
                .push("selector".to_owned());
            return Ok(json!({
                "selected_skills": ["focus-skill", "unknown-skill"],
                "rationale": "metadata match"
            })
            .to_string());
        }

        Ok("fallback synthesis".to_owned())
    }

    async fn complete_turn(&self, request: ModelTurnRequest) -> anyhow::Result<ModelTurnResponse> {
        let stages = self.stages.lock().expect("stages lock should succeed");
        assert_eq!(
            stages.first().map(String::as_str),
            Some("selector"),
            "selector stage must run before native loop"
        );
        drop(stages);

        assert!(
            request.system_prompt.contains("Selected skills guidance:"),
            "native loop prompt should include selected skill guidance"
        );
        assert!(
            request
                .system_prompt
                .contains("THIS_BODY_MARKER_SHOULD_NOT_APPEAR_IN_SELECTOR"),
            "selected skill body should be available in native loop prompt"
        );

        Ok(ModelTurnResponse {
            assistant_text: "selector contract ok".to_owned(),
            tool_calls: Vec::new(),
        })
    }
}

#[derive(Debug, Default)]
struct EmptyTurnModelProvider;

#[async_trait]
impl ModelProvider for EmptyTurnModelProvider {
    async fn complete(&self, request: ModelRequest) -> anyhow::Result<String> {
        if request
            .system_prompt
            .contains("You are the skill selector for CompanionPilot.")
        {
            return Ok(json!({
                "selected_skills": [],
                "rationale": "empty_turn_selector"
            })
            .to_string());
        }
        Ok("Direct fallback answer.".to_owned())
    }

    async fn complete_turn(&self, _request: ModelTurnRequest) -> anyhow::Result<ModelTurnResponse> {
        Ok(ModelTurnResponse {
            assistant_text: String::new(),
            tool_calls: Vec::new(),
        })
    }
}

#[derive(Debug, Default)]
struct StubWebSearchToolExecutor;

#[async_trait]
impl ToolExecutor for StubWebSearchToolExecutor {
    async fn execute(
        &self,
        tool_name: &str,
        args: Value,
        _message_ctx: &MessageCtx,
    ) -> anyhow::Result<ToolResult> {
        if tool_name != "web_search" {
            return Err(anyhow::anyhow!("unknown tool: {tool_name}"));
        }

        let query = args
            .get("query")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("missing query arg"))?;

        Ok(ToolResult {
            text: format!("result:{query}"),
            citations: vec![format!("https://example.com/{query}")],
        })
    }
}

#[tokio::test]
async fn persists_simple_name_fact() {
    let memory = Arc::new(InMemoryMemoryStore::default());
    let orchestrator = DefaultChatOrchestrator::new(
        Arc::new(MockModelProvider),
        memory.clone(),
        Arc::new(ToolRegistry::default()),
        empty_skill_catalog(),
        SafetyPolicy::default(),
    );

    let _ = orchestrator
        .handle_message(MessageCtx {
            message_id: "1".into(),
            user_id: "u1".into(),
            guild_id: "g1".into(),
            channel_id: "c1".into(),
            content: "my name is petr".into(),
            timestamp: Utc::now(),
        })
        .await
        .expect("handle message should succeed");

    let facts = memory
        .search_relevant("u1", "name", 10)
        .await
        .expect("search should succeed");
    assert_eq!(facts.len(), 1);
    assert_eq!(facts[0].key, "name");
}

#[tokio::test]
async fn tool_failure_is_included_in_final_synthesis_context() {
    let memory = Arc::new(InMemoryMemoryStore::default());
    let orchestrator = DefaultChatOrchestrator::new(
        Arc::new(MockModelProvider),
        memory.clone(),
        Arc::new(ToolRegistry::default()),
        empty_skill_catalog(),
        SafetyPolicy::default(),
    );

    let result = orchestrator
        .handle_message(MessageCtx {
            message_id: "3".into(),
            user_id: "u3".into(),
            guild_id: "g1".into(),
            channel_id: "c1".into(),
            content: "search the web for rust async traits".into(),
            timestamp: Utc::now(),
        })
        .await
        .expect("tool failure should still synthesize a final answer");

    assert_eq!(result.tool_calls.len(), 1);
    assert_eq!(result.tool_calls[0].tool_name, "web_search");
    assert!(result.text.contains("Tool summary:"));
    assert!(result.text.contains("web_search tool is not configured"));
}

#[tokio::test]
async fn native_loop_can_run_multiple_tool_rounds_before_final_answer() {
    let memory = Arc::new(InMemoryMemoryStore::default());
    let orchestrator = DefaultChatOrchestrator::new(
        Arc::new(FollowupLoopModelProvider),
        memory.clone(),
        Arc::new(StubWebSearchToolExecutor),
        empty_skill_catalog(),
        SafetyPolicy::default(),
    );

    let result = orchestrator
        .handle_message(MessageCtx {
            message_id: "3b".into(),
            user_id: "u3b".into(),
            guild_id: "g1".into(),
            channel_id: "c1".into(),
            content: "find a final answer using tools".into(),
            timestamp: Utc::now(),
        })
        .await
        .expect("native tool loop should complete");

    assert_eq!(result.tool_calls.len(), 2);
    assert_eq!(result.tool_calls[0].tool_name, "web_search");
    assert_eq!(result.tool_calls[0].args["query"], "alpha");
    assert_eq!(result.tool_calls[1].tool_name, "web_search");
    assert_eq!(result.tool_calls[1].args["query"], "beta");
    assert_eq!(result.text, "Final answer from native tool loop.");
    assert_eq!(result.citations.len(), 2);

    let decisions = memory
        .list_orchestration_decisions("u3b", 20)
        .await
        .expect("orchestration decisions should list");
    let tool_request = decisions
        .iter()
        .find(|decision| decision.stage == "native_turn" && decision.decision == "request_tools")
        .expect("native turn tool request should be logged");
    let payload: Value = serde_json::from_str(&tool_request.payload_json)
        .expect("tool request payload should be valid JSON");
    assert!(
        payload["decision"]["model_input_snapshot"]["user_request"]
            .as_str()
            .is_some(),
        "tool request payload should include model input snapshot"
    );
}

#[tokio::test]
async fn skill_selector_uses_metadata_only_and_runs_before_native_loop() {
    let memory = Arc::new(InMemoryMemoryStore::default());
    let orchestrator = DefaultChatOrchestrator::new(
        Arc::new(SkillSelectionContractModelProvider::default()),
        memory.clone(),
        Arc::new(ToolRegistry::default()),
        test_skill_catalog_with_marker(),
        SafetyPolicy::default(),
    );

    let result = orchestrator
        .handle_message(MessageCtx {
            message_id: "3c".into(),
            user_id: "u3c".into(),
            guild_id: "g1".into(),
            channel_id: "c1".into(),
            content: "focus and summarize this".into(),
            timestamp: Utc::now(),
        })
        .await
        .expect("selector-first pipeline should complete");

    assert_eq!(result.text, "selector contract ok");
    assert!(result.tool_calls.is_empty());

    let decisions = memory
        .list_orchestration_decisions("u3c", 20)
        .await
        .expect("orchestration decisions should list");

    let skill_selector_decision = decisions
        .iter()
        .find(|decision| decision.stage == "skill_selector")
        .expect("skill selector decision should be logged");
    assert_eq!(skill_selector_decision.decision, "apply_selection");

    let payload: Value = serde_json::from_str(&skill_selector_decision.payload_json)
        .expect("skill selector payload should be valid JSON");
    let selected = payload["selected_skills"]
        .as_array()
        .expect("selected_skills should be an array")
        .iter()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>();
    assert_eq!(selected, vec!["focus-skill"]);
}

#[tokio::test]
async fn empty_native_turn_uses_direct_answer_fallback_instead_of_internal_error() {
    let memory = Arc::new(InMemoryMemoryStore::default());
    let orchestrator = DefaultChatOrchestrator::new(
        Arc::new(EmptyTurnModelProvider),
        memory,
        Arc::new(ToolRegistry::default()),
        empty_skill_catalog(),
        SafetyPolicy::default(),
    );

    let result = orchestrator
        .handle_message(MessageCtx {
            message_id: "3d".into(),
            user_id: "u3d".into(),
            guild_id: "g1".into(),
            channel_id: "c1".into(),
            content: "please use a tool".into(),
            timestamp: Utc::now(),
        })
        .await
        .expect("fallback should complete");

    assert_eq!(result.text, "Direct fallback answer.");
    assert!(result.tool_calls.is_empty());
}

#[test]
fn sanitize_memory_key_normalizes_words() {
    assert_eq!(sanitize_memory_key("Favorite Game"), "favorite_game");
}

#[test]
fn clean_memory_value_trims_wrappers_and_preserves_punctuation() {
    assert_eq!(clean_memory_value("\"Petr.\""), "Petr.");
}

#[test]
fn truncate_for_log_preserves_utf8_boundaries() {
    let output = truncate_for_log("hello🎵world", 7);
    assert_eq!(output, "hello...");
}
