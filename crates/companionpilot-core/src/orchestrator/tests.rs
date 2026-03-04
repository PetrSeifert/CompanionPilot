use std::{
    collections::HashMap,
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

use super::program_exec::substitute_variables;
use super::telemetry::truncate_for_log;
use super::{
    DefaultChatOrchestrator,
    sanitize::{clean_memory_value, sanitize_memory_key, sanitize_native_tool_calls},
};

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
                reasoning: None,
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
                reasoning: None,
            });
        }

        Ok(ModelTurnResponse {
            assistant_text: "Final answer from native tool loop.".to_owned(),
            tool_calls: Vec::new(),
            reasoning: None,
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
            reasoning: None,
        })
    }
}

#[derive(Debug, Default)]
struct InvalidToolCallThenFinalModelProvider;

#[async_trait]
impl ModelProvider for InvalidToolCallThenFinalModelProvider {
    async fn complete(&self, request: ModelRequest) -> anyhow::Result<String> {
        if request
            .system_prompt
            .contains("You are the skill selector for CompanionPilot.")
        {
            return Ok(json!({
                "selected_skills": [],
                "rationale": "invalid_tool_then_final_selector"
            })
            .to_string());
        }
        Ok("fallback synthesis".to_owned())
    }

    async fn complete_turn(&self, request: ModelTurnRequest) -> anyhow::Result<ModelTurnResponse> {
        let saw_validation_feedback = request.messages.iter().any(|message| {
            message.role == ModelMessageRole::User
                && message
                    .content
                    .contains("Previous tool call arguments were invalid")
        });
        if saw_validation_feedback {
            return Ok(ModelTurnResponse {
                assistant_text: "Recovered after invalid tool call.".to_owned(),
                tool_calls: Vec::new(),
                reasoning: None,
            });
        }

        Ok(ModelTurnResponse {
            assistant_text: String::new(),
            tool_calls: vec![ModelToolCall {
                id: "bad-run-terminal-command-call".to_owned(),
                name: "run_terminal_command".to_owned(),
                arguments: json!({ "command": "spogo" }),
            }],
            reasoning: None,
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
            reasoning: None,
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

#[derive(Debug, Default)]
struct ProgramChainModelProvider;

#[async_trait]
impl ModelProvider for ProgramChainModelProvider {
    async fn complete(&self, request: ModelRequest) -> anyhow::Result<String> {
        if request
            .system_prompt
            .contains("You are the skill selector for CompanionPilot.")
        {
            return Ok(json!({
                "selected_skills": [],
                "rationale": "program_chain_selector"
            })
            .to_string());
        }
        Ok("fallback synthesis".to_owned())
    }

    async fn complete_turn(&self, request: ModelTurnRequest) -> anyhow::Result<ModelTurnResponse> {
        let program_output = request.messages.iter().find(|message| {
            message.role == ModelMessageRole::Tool
                && message.name.as_deref() == Some("execute_program")
        });
        if let Some(program_output) = program_output {
            assert!(
                program_output.content.contains("result:result:alpha"),
                "program output should include substituted second-step output"
            );
            return Ok(ModelTurnResponse {
                assistant_text: "Program chain complete.".to_owned(),
                tool_calls: Vec::new(),
                reasoning: None,
            });
        }

        Ok(ModelTurnResponse {
            assistant_text: String::new(),
            tool_calls: vec![ModelToolCall {
                id: "program-chain-call".to_owned(),
                name: "execute_program".to_owned(),
                arguments: json!({
                    "reasoning": "step output should feed into next step",
                    "steps": [
                        {
                            "step_id": "first",
                            "tool_name": "web_search",
                            "arguments": {
                                "query": "alpha",
                                "max_results": 1
                            }
                        },
                        {
                            "step_id": "second",
                            "tool_name": "web_search",
                            "arguments": {
                                "query": "${first}",
                                "max_results": 1
                            }
                        }
                    ]
                }),
            }],
            reasoning: None,
        })
    }
}

#[derive(Debug, Default)]
struct ProgramAbortModelProvider;

#[async_trait]
impl ModelProvider for ProgramAbortModelProvider {
    async fn complete(&self, request: ModelRequest) -> anyhow::Result<String> {
        if request
            .system_prompt
            .contains("You are the skill selector for CompanionPilot.")
        {
            return Ok(json!({
                "selected_skills": [],
                "rationale": "program_abort_selector"
            })
            .to_string());
        }
        Ok("fallback synthesis".to_owned())
    }

    async fn complete_turn(&self, request: ModelTurnRequest) -> anyhow::Result<ModelTurnResponse> {
        let program_output = request.messages.iter().find(|message| {
            message.role == ModelMessageRole::Tool
                && message.name.as_deref() == Some("execute_program")
        });
        if let Some(program_output) = program_output {
            assert!(
                program_output.content.contains("forced failure"),
                "program output should include failing step error"
            );
            return Ok(ModelTurnResponse {
                assistant_text: "Program failure handled.".to_owned(),
                tool_calls: Vec::new(),
                reasoning: None,
            });
        }

        Ok(ModelTurnResponse {
            assistant_text: String::new(),
            tool_calls: vec![ModelToolCall {
                id: "program-abort-call".to_owned(),
                name: "execute_program".to_owned(),
                arguments: json!({
                    "reasoning": "stop at first failed step",
                    "steps": [
                        {
                            "step_id": "first",
                            "tool_name": "web_search",
                            "arguments": {
                                "query": "alpha",
                                "max_results": 1
                            }
                        },
                        {
                            "step_id": "second",
                            "tool_name": "web_search",
                            "arguments": {
                                "query": "${first}-force-fail",
                                "max_results": 1
                            }
                        },
                        {
                            "step_id": "third",
                            "tool_name": "web_search",
                            "arguments": {
                                "query": "gamma",
                                "max_results": 1
                            }
                        }
                    ]
                }),
            }],
            reasoning: None,
        })
    }
}

#[derive(Debug, Default)]
struct ProgramWebSearchToolExecutor {
    queries: Mutex<Vec<String>>,
}

#[async_trait]
impl ToolExecutor for ProgramWebSearchToolExecutor {
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
            .ok_or_else(|| anyhow::anyhow!("missing query arg"))?
            .to_owned();
        self.queries
            .lock()
            .expect("queries lock should succeed")
            .push(query.clone());

        if query.contains("force-fail") {
            return Err(anyhow::anyhow!("forced failure for query: {query}"));
        }

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

#[tokio::test]
async fn invalid_tool_call_is_recovered_without_internal_error() {
    let memory = Arc::new(InMemoryMemoryStore::default());
    let orchestrator = DefaultChatOrchestrator::new(
        Arc::new(InvalidToolCallThenFinalModelProvider),
        memory.clone(),
        Arc::new(ToolRegistry::default()),
        empty_skill_catalog(),
        SafetyPolicy::default(),
    );

    let result = orchestrator
        .handle_message(MessageCtx {
            message_id: "3e".into(),
            user_id: "u3e".into(),
            guild_id: "g1".into(),
            channel_id: "c1".into(),
            content: "what song is this?".into(),
            timestamp: Utc::now(),
        })
        .await
        .expect("invalid tool call recovery should complete");

    assert_eq!(result.text, "Recovered after invalid tool call.");
    assert!(result.tool_calls.is_empty());

    let decisions = memory
        .list_orchestration_decisions("u3e", 20)
        .await
        .expect("orchestration decisions should list");
    assert!(
        decisions.iter().any(|decision| {
            decision.stage == "native_turn"
                && decision.decision == "fallback"
                && decision.rationale.contains("invalid_tool_calls")
        }),
        "invalid tool call fallback should be logged"
    );
}

#[tokio::test]
async fn program_two_step_chain_substitutes_variables() {
    let memory = Arc::new(InMemoryMemoryStore::default());
    let tools = Arc::new(ProgramWebSearchToolExecutor::default());
    let orchestrator = DefaultChatOrchestrator::new(
        Arc::new(ProgramChainModelProvider),
        memory,
        tools.clone(),
        empty_skill_catalog(),
        SafetyPolicy::default(),
    );

    let result = orchestrator
        .handle_message(MessageCtx {
            message_id: "3f".into(),
            user_id: "u3f".into(),
            guild_id: "g1".into(),
            channel_id: "c1".into(),
            content: "chain dependent calls".into(),
            timestamp: Utc::now(),
        })
        .await
        .expect("program chain should complete");

    assert_eq!(result.text, "Program chain complete.");
    assert!(
        result
            .tool_calls
            .iter()
            .any(|call| call.tool_name == "execute_program")
    );

    let queries = tools
        .queries
        .lock()
        .expect("queries lock should succeed")
        .clone();
    assert_eq!(queries.len(), 2);
    assert_eq!(queries[0], "alpha");
    assert_eq!(queries[1], "result:alpha");
}

#[tokio::test]
async fn program_aborts_on_step_failure() {
    let memory = Arc::new(InMemoryMemoryStore::default());
    let tools = Arc::new(ProgramWebSearchToolExecutor::default());
    let orchestrator = DefaultChatOrchestrator::new(
        Arc::new(ProgramAbortModelProvider),
        memory,
        tools.clone(),
        empty_skill_catalog(),
        SafetyPolicy::default(),
    );

    let result = orchestrator
        .handle_message(MessageCtx {
            message_id: "3g".into(),
            user_id: "u3g".into(),
            guild_id: "g1".into(),
            channel_id: "c1".into(),
            content: "abort on failure".into(),
            timestamp: Utc::now(),
        })
        .await
        .expect("program failure should still produce a final answer");

    assert_eq!(result.text, "Program failure handled.");
    let queries = tools
        .queries
        .lock()
        .expect("queries lock should succeed")
        .clone();
    assert_eq!(queries.len(), 2);
    assert_eq!(queries[0], "alpha");
    assert_eq!(queries[1], "result:alpha-force-fail");
}

#[test]
fn program_rejects_forward_references() {
    let calls = sanitize_native_tool_calls(vec![ModelToolCall {
        id: "program-1".to_owned(),
        name: "execute_program".to_owned(),
        arguments: json!({
            "reasoning": "invalid forward ref",
            "steps": [
                {
                    "step_id": "first",
                    "tool_name": "web_search",
                    "arguments": { "query": "${second}" }
                },
                {
                    "step_id": "second",
                    "tool_name": "web_search",
                    "arguments": { "query": "ok" }
                }
            ]
        }),
    }]);
    assert!(calls.is_empty());
}

#[test]
fn program_rejects_nested_execute_program() {
    let calls = sanitize_native_tool_calls(vec![ModelToolCall {
        id: "program-2".to_owned(),
        name: "execute_program".to_owned(),
        arguments: json!({
            "reasoning": "nested should fail",
            "steps": [
                {
                    "step_id": "first",
                    "tool_name": "execute_program",
                    "arguments": {}
                }
            ]
        }),
    }]);
    assert!(calls.is_empty());
}

#[test]
fn program_rejects_duplicate_step_ids() {
    let calls = sanitize_native_tool_calls(vec![ModelToolCall {
        id: "program-3".to_owned(),
        name: "execute_program".to_owned(),
        arguments: json!({
            "reasoning": "duplicate ids should fail",
            "steps": [
                {
                    "step_id": "dup",
                    "tool_name": "web_search",
                    "arguments": { "query": "alpha" }
                },
                {
                    "step_id": "dup",
                    "tool_name": "web_search",
                    "arguments": { "query": "beta" }
                }
            ]
        }),
    }]);
    assert!(calls.is_empty());
}

#[test]
fn substitute_variables_replaces_in_nested_json() {
    let mut substitutions = HashMap::new();
    substitutions.insert("step_one".to_owned(), "VALUE".to_owned());
    let input = json!({
        "plain": "no refs",
        "list": [
            "prefix ${step_one}",
            {
                "nested": "${step_one} / ${missing}"
            }
        ],
        "number": 42
    });

    let output = substitute_variables(&input, &substitutions);

    assert_eq!(output["plain"], "no refs");
    assert_eq!(output["list"][0], "prefix VALUE");
    assert_eq!(output["list"][1]["nested"], "VALUE / ${missing}");
    assert_eq!(output["number"], 42);
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
