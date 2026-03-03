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
    model::{MockModelProvider, ModelProvider, ModelRequest},
    safety::SafetyPolicy,
    skills::SkillCatalog,
    tools::{ToolExecutor, ToolRegistry, ToolResult},
    types::{MessageCtx, ToolCall},
};

use super::{
    DefaultChatOrchestrator, PlannedToolCall, clean_memory_value,
    enforce_datetime_planning_boundary, parse_unified_plan, sanitize_memory_key,
    sanitize_planned_tool_calls, truncate_for_log,
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
            .contains("You are the unified planner for CompanionPilot.")
        {
            return Ok(json!({
                "tool_calls": [
                    {
                        "tool_name": "web_search",
                        "args": {
                            "query": "alpha",
                            "max_results": 3
                        }
                    }
                ],
                "memory": {
                    "store": false,
                    "key": "",
                    "value": "",
                    "confidence": 0.0
                },
                "rationale": "need first lookup"
            })
            .to_string());
        }

        if request
            .system_prompt
            .contains("You are the tool follow-up planner for CompanionPilot.")
        {
            if request.user_prompt.contains("result:alpha")
                && !request.user_prompt.contains("result:beta")
            {
                return Ok(json!({
                    "action": "tools",
                    "final_answer": "",
                    "tool_calls": [
                        {
                            "tool_name": "web_search",
                            "args": {
                                "query": "beta",
                                "max_results": 2
                            }
                        }
                    ],
                    "rationale": "need second lookup"
                })
                .to_string());
            }

            if request.user_prompt.contains("result:beta") {
                return Ok(json!({
                    "action": "final",
                    "final_answer": "Final answer from follow-up planner.",
                    "tool_calls": [],
                    "rationale": "have enough evidence"
                })
                .to_string());
            }
        }

        Ok("fallback final synthesis".to_owned())
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

        if request
            .system_prompt
            .contains("You are the unified planner for CompanionPilot.")
        {
            let stages = self.stages.lock().expect("stages lock should succeed");
            assert_eq!(
                stages.first().map(String::as_str),
                Some("selector"),
                "selector stage must run before unified planner"
            );
            drop(stages);

            assert!(
                request.system_prompt.contains("Selected skills guidance:"),
                "unified planner should receive selected skill guidance"
            );
            assert!(
                request
                    .system_prompt
                    .contains("THIS_BODY_MARKER_SHOULD_NOT_APPEAR_IN_SELECTOR"),
                "selected skill body should be available in unified planning stages"
            );
            return Ok(json!({
                "tool_calls": [],
                "memory": {
                    "store": false,
                    "key": "",
                    "value": "",
                    "confidence": 0.0
                },
                "rationale": "no tools needed"
            })
            .to_string());
        }

        self.stages
            .lock()
            .expect("stages lock should succeed")
            .push("final".to_owned());
        assert!(
            request.system_prompt.contains("Selected skills guidance:"),
            "final synthesis prompt should include selected skill guidance"
        );
        Ok("selector contract ok".to_owned())
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
async fn search_command_is_not_a_manual_override() {
    let memory = Arc::new(InMemoryMemoryStore::default());
    let orchestrator = DefaultChatOrchestrator::new(
        Arc::new(MockModelProvider),
        memory,
        Arc::new(ToolRegistry::default()),
        empty_skill_catalog(),
        SafetyPolicy::default(),
    );

    let result = orchestrator
        .handle_message(MessageCtx {
            message_id: "2".into(),
            user_id: "u2".into(),
            guild_id: "g1".into(),
            channel_id: "c1".into(),
            content: "/search rust".into(),
            timestamp: Utc::now(),
        })
        .await
        .expect("planner should be allowed to skip tool usage");

    assert!(result.tool_calls.is_empty());
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
    assert!(result.text.contains("Status: error"));
    assert!(result.text.contains("web_search tool is not configured"));
}

#[tokio::test]
async fn followup_planner_can_run_multiple_tool_rounds_before_final_answer() {
    let memory = Arc::new(InMemoryMemoryStore::default());
    let orchestrator = DefaultChatOrchestrator::new(
        Arc::new(FollowupLoopModelProvider),
        memory,
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
        .expect("follow-up planning loop should complete");

    assert_eq!(result.tool_calls.len(), 2);
    assert_eq!(result.tool_calls[0].tool_name, "web_search");
    assert_eq!(result.tool_calls[0].args["query"], "alpha");
    assert_eq!(result.tool_calls[1].tool_name, "web_search");
    assert_eq!(result.tool_calls[1].args["query"], "beta");
    assert_eq!(result.text, "Final answer from follow-up planner.");
    assert_eq!(result.citations.len(), 2);
}

#[tokio::test]
async fn skill_selector_uses_metadata_only_and_runs_before_unified_planner() {
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
        .list_planner_decisions("u3c", 20)
        .await
        .expect("planner decisions should list");

    let skill_selector_decision = decisions
        .iter()
        .find(|decision| decision.planner == "skill_selector")
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
async fn name_correction_overwrites_previous_memory() {
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
            message_id: "4".into(),
            user_id: "u4".into(),
            guild_id: "g1".into(),
            channel_id: "c1".into(),
            content: "my name is Petrr".into(),
            timestamp: Utc::now(),
        })
        .await
        .expect("first message should succeed");

    let _ = orchestrator
        .handle_message(MessageCtx {
            message_id: "5".into(),
            user_id: "u4".into(),
            guild_id: "g1".into(),
            channel_id: "c1".into(),
            content: "I misspelled my name, it's Petr.".into(),
            timestamp: Utc::now(),
        })
        .await
        .expect("correction message should succeed");

    let facts = memory
        .search_relevant("u4", "name", 10)
        .await
        .expect("search should succeed");
    assert_eq!(facts.len(), 1);
    assert_eq!(facts[0].value, "Petr");
}

#[tokio::test]
async fn short_term_memory_includes_recent_non_fact_turns() {
    let memory = Arc::new(InMemoryMemoryStore::default());
    let orchestrator = DefaultChatOrchestrator::new(
        Arc::new(MockModelProvider),
        memory,
        Arc::new(ToolRegistry::default()),
        empty_skill_catalog(),
        SafetyPolicy::default(),
    );

    let _ = orchestrator
        .handle_message(MessageCtx {
            message_id: "6".into(),
            user_id: "u6".into(),
            guild_id: "g1".into(),
            channel_id: "c1".into(),
            content: "I am 24 years old.".into(),
            timestamp: Utc::now(),
        })
        .await
        .expect("first message should succeed");

    let second = orchestrator
        .handle_message(MessageCtx {
            message_id: "7".into(),
            user_id: "u6".into(),
            guild_id: "g1".into(),
            channel_id: "c1".into(),
            content: "What did I just tell you?".into(),
            timestamp: Utc::now(),
        })
        .await
        .expect("second message should succeed");

    assert!(second.text.contains("Recent conversation turns:"));
    assert!(second.text.contains("user: I am 24 years old."));
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

#[test]
fn parse_unified_plan_from_wrapped_json() {
    let raw = "Result:\n{\"tool_calls\":[],\"memory\":{\"store\":false,\"key\":\"\",\"value\":\"\",\"confidence\":0.0},\"rationale\":\"none\"}\nDone.";
    let plan = parse_unified_plan(raw).expect("wrapped JSON should parse");
    assert!(plan.tool_calls.is_empty());
    assert!(!plan.memory.store);
}

#[test]
fn sanitize_planned_tool_calls_drops_unknown_without_limiting_count() {
    let mut planned_calls = Vec::new();
    planned_calls.push(PlannedToolCall {
        tool_name: "unknown_tool".to_owned(),
        args: json!({}),
    });

    for index in 0..8 {
        planned_calls.push(PlannedToolCall {
            tool_name: "web_search".to_owned(),
            args: json!({
                "query": format!("rust query {index}"),
                "max_results": 5
            }),
        });
    }

    let sanitized = sanitize_planned_tool_calls(planned_calls);
    assert_eq!(sanitized.len(), 8);
    assert_eq!(sanitized[0].tool_name, "web_search");
    assert_eq!(sanitized[7].tool_name, "web_search");
}

#[test]
fn sanitize_planned_tool_calls_allows_current_datetime() {
    let planned_calls = vec![PlannedToolCall {
        tool_name: "current_datetime".to_owned(),
        args: json!({"ignored": true}),
    }];

    let sanitized = sanitize_planned_tool_calls(planned_calls);
    assert_eq!(sanitized.len(), 1);
    assert_eq!(sanitized[0].tool_name, "current_datetime");
    assert_eq!(sanitized[0].args, json!({}));
}

#[test]
fn sanitize_planned_tool_calls_allows_cli_command() {
    let planned_calls = vec![PlannedToolCall {
        tool_name: "cli".to_owned(),
        args: json!({"command": "spogo -h"}),
    }];

    let sanitized = sanitize_planned_tool_calls(planned_calls);
    assert_eq!(sanitized.len(), 1);
    assert_eq!(sanitized[0].tool_name, "cli");
    assert_eq!(sanitized[0].args, json!({ "args": ["spogo", "-h"] }));
}

#[test]
fn sanitize_planned_tool_calls_drops_cli_non_spogo_command() {
    let planned_calls = vec![PlannedToolCall {
        tool_name: "cli".to_owned(),
        args: json!({"command": "ls -la"}),
    }];

    let sanitized = sanitize_planned_tool_calls(planned_calls);
    assert!(sanitized.is_empty());
}

#[test]
fn sanitize_planned_tool_calls_drops_cli_without_command_or_args() {
    let planned_calls = vec![PlannedToolCall {
        tool_name: "cli".to_owned(),
        args: json!({}),
    }];

    let sanitized = sanitize_planned_tool_calls(planned_calls);
    assert!(sanitized.is_empty());
}

#[test]
fn sanitize_planned_tool_calls_preserves_datetime_then_search_order() {
    let planned_calls = vec![
        PlannedToolCall {
            tool_name: "current_datetime".to_owned(),
            args: json!({}),
        },
        PlannedToolCall {
            tool_name: "web_search".to_owned(),
            args: json!({
                "query": "current weather in berlin",
                "max_results": 5
            }),
        },
    ];

    let sanitized = sanitize_planned_tool_calls(planned_calls);
    assert_eq!(sanitized.len(), 2);
    assert_eq!(sanitized[0].tool_name, "current_datetime");
    assert_eq!(sanitized[1].tool_name, "web_search");
    let query = sanitized[1].args["query"]
        .as_str()
        .expect("query should be a string");
    assert_eq!(query, "current weather in berlin");
}

#[test]
fn sanitize_planned_tool_calls_allows_discord_voice_tools() {
    let planned_calls = vec![
        PlannedToolCall {
            tool_name: "discord_voice_join".to_owned(),
            args: json!({"channel_id":"123"}),
        },
        PlannedToolCall {
            tool_name: "discord_voice_leave".to_owned(),
            args: json!({}),
        },
    ];

    let sanitized = sanitize_planned_tool_calls(planned_calls);
    assert_eq!(sanitized.len(), 2);
    assert_eq!(sanitized[0].tool_name, "discord_voice_join");
    assert_eq!(sanitized[0].args["channel_id"], "123");
    assert_eq!(sanitized[1].tool_name, "discord_voice_leave");
}

#[test]
fn enforce_datetime_planning_boundary_runs_datetime_in_isolation() {
    let calls = vec![
        ToolCall {
            tool_name: "web_search".to_owned(),
            args: json!({"query": "major video game releases late 2024 early 2025", "max_results": 10}),
        },
        ToolCall {
            tool_name: "current_datetime".to_owned(),
            args: json!({}),
        },
    ];

    let bounded = enforce_datetime_planning_boundary(calls);
    assert_eq!(bounded.len(), 1);
    assert_eq!(bounded[0].tool_name, "current_datetime");
}

#[test]
fn enforce_datetime_planning_boundary_keeps_non_datetime_plans_unchanged() {
    let calls = vec![ToolCall {
        tool_name: "web_search".to_owned(),
        args: json!({"query": "rust async traits", "max_results": 3}),
    }];

    let bounded = enforce_datetime_planning_boundary(calls.clone());
    assert_eq!(bounded.len(), calls.len());
    assert_eq!(bounded[0].tool_name, "web_search");
    assert_eq!(bounded[0].args, calls[0].args);
}
