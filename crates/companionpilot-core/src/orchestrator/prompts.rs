use serde_json::json;

use crate::{
    skills::{SkillDefinition, SkillMetadata},
    types::MemoryContext,
};

use super::util::{indent_block, truncate_for_prompt};

const SELECTED_SKILL_LIMIT: usize = 5;
const SELECTED_SKILL_BODY_MAX_CHARS: usize = 1_200;

pub(super) fn build_skill_selector_prompt(
    memory: &MemoryContext,
    skill_metadata: &[SkillMetadata],
) -> String {
    let context_block = build_planner_context_block(memory);
    let skill_inventory = build_skill_inventory_for_selector(skill_metadata);

    format!(
        "You are the skill selector for CompanionPilot.
Select zero or more skills relevant to the current user request.
Return strict JSON only (no markdown, no prose) with this exact schema:
{{
  \"selected_skills\": [\"skill-id\", \"...\"],
  \"rationale\": \"short reason\"
}}
Rules:
- Choose only skill IDs from the provided inventory.
- Use only inventory metadata (id/title/description/tags) for selection.
- Do not invent IDs.
- It is valid to return an empty selected_skills array.
Available skills:
{}
{}",
        skill_inventory, context_block
    )
}

pub(super) fn build_unified_planner_prompt(
    memory: &MemoryContext,
    selected_skills: &[SkillDefinition],
) -> String {
    let context_block = build_support_context_block(memory, selected_skills);

    format!(
        "You are the unified planner for CompanionPilot.
Decide both tool usage and memory write for one user message.
Return strict JSON only (no markdown, no prose) with this exact schema:
{{
  \"tool_calls\": [{{\"tool_name\":\"...\",\"args\":{{...}}}}],
  \"memory\": {{
    \"store\": true|false,
    \"key\": \"...\",
    \"value\": \"...\",
    \"confidence\": 0.0-1.0
  }},
  \"rationale\": \"short reason\"
}}
Tool calls are executed sequentially in listed order.
There are no manual commands or manual overrides: all tool usage must come from this decision.
If no tool is needed, return an empty tool_calls array.
If memory should not be stored, set store=false and key/value to empty strings.
Store only durable personal facts (identity, preferences, recurring goals, corrections).
Do not store one-off requests or transient states.
Use web search for latest/current/news/prices/weather or unknown factual claims.
For Spotify requests, use cli for local spogo commands.
cli has a strict policy: only commands starting with `spogo` are allowed; any other command will be blocked.
If uncertain about spogo usage, first call cli with command \"spogo -h\" or \"spogo <subcommand> -h\".
For time-sensitive requests, call current_datetime before web_search so queries and answers are anchored to real current time.
If current_datetime is needed, request only current_datetime in this decision and wait for its output before planning web_search.
Tool inventory:
{}
{}",
        build_tool_inventory_for_planner(),
        context_block
    )
}

pub(super) fn build_tool_followup_prompt(
    memory: &MemoryContext,
    selected_skills: &[SkillDefinition],
) -> String {
    let context_block = build_support_context_block(memory, selected_skills);

    format!(
        "You are the tool follow-up planner for CompanionPilot.
Decide whether the current evidence is enough for a final user-facing answer, or whether more tool calls are needed.
Return strict JSON only (no markdown, no prose) with this exact schema:
{{
  \"action\": \"final\"|\"tools\",
  \"final_answer\": \"non-empty only when action=final\",
  \"tool_calls\": [{{\"tool_name\":\"...\",\"args\":{{...}}}}],
  \"rationale\": \"short reason\"
}}
If action=final, provide the complete final answer and return an empty tool_calls array.
If action=tools, final_answer must be empty and tool_calls must contain at least one valid call.
Only request tools when the current outputs are insufficient or conflicting.
For time-sensitive requests, prefer calling current_datetime before additional web_search calls.
For Spotify requests, use cli for local spogo commands.
cli has a strict policy: only commands starting with `spogo` are allowed; any other command will be blocked.
If uncertain about spogo usage, call cli with command \"spogo -h\" or \"spogo <subcommand> -h\" before guessing.
If current_datetime is needed, call it alone first, then plan web_search in a later tool round.
Tool inventory:
{}
{}",
        build_tool_inventory_for_planner(),
        context_block
    )
}

pub(super) fn build_support_context_block(
    memory: &MemoryContext,
    selected_skills: &[SkillDefinition],
) -> String {
    let mut sections = Vec::new();
    let planner_context = build_planner_context_block(memory);
    if !planner_context.is_empty() {
        sections.push(planner_context.trim().to_owned());
    }
    let selected_skills_block = build_selected_skill_context_block(selected_skills);
    if !selected_skills_block.is_empty() {
        sections.push(selected_skills_block);
    }

    if sections.is_empty() {
        String::new()
    } else {
        format!("{}\n", sections.join("\n\n"))
    }
}

fn build_skill_inventory_for_selector(skill_metadata: &[SkillMetadata]) -> String {
    let inventory = skill_metadata
        .iter()
        .map(|skill| {
            json!({
                "id": &skill.id,
                "title": &skill.title,
                "description": &skill.description,
                "tags": &skill.tags,
            })
        })
        .collect::<Vec<_>>();

    serde_json::to_string_pretty(&inventory).unwrap_or_else(|_| "[]".to_owned())
}

pub(super) fn build_selected_skill_context_block(selected_skills: &[SkillDefinition]) -> String {
    if selected_skills.is_empty() {
        return "Selected skills: none.".to_owned();
    }

    let lines = selected_skills
        .iter()
        .take(SELECTED_SKILL_LIMIT)
        .enumerate()
        .map(|(index, skill)| {
            let tags = if skill.metadata.tags.is_empty() {
                "none".to_owned()
            } else {
                skill.metadata.tags.join(", ")
            };
            format!(
                "{}. id={}\n   title={}\n   description={}\n   tags={}\n   markdown:\n{}",
                index + 1,
                skill.metadata.id,
                skill.metadata.title,
                skill.metadata.description,
                tags,
                indent_block(&truncate_for_prompt(
                    &skill.body_markdown,
                    SELECTED_SKILL_BODY_MAX_CHARS
                ))
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    format!("Selected skills guidance:\n{lines}")
}

fn build_planner_context_block(memory: &MemoryContext) -> String {
    let mut context_lines = Vec::new();
    if let Some(summary) = &memory.summary {
        context_lines.push(format!("Conversation summary: {summary}"));
    }

    if !memory.facts.is_empty() {
        let facts = memory
            .facts
            .iter()
            .map(|fact| format!("{}={}", fact.key, fact.value))
            .collect::<Vec<_>>()
            .join("; ");
        context_lines.push(format!("Known user facts: {facts}"));
    }

    if !memory.recent_messages.is_empty() {
        context_lines.push(build_recent_context_block(&memory.recent_messages));
    }

    if context_lines.is_empty() {
        String::new()
    } else {
        format!("Context:\n{}\n", context_lines.join("\n"))
    }
}

fn build_tool_inventory_for_planner() -> &'static str {
    r#"[
  {
    "tool_name": "current_datetime",
    "args_schema": {},
    "when_to_use": "Need the exact current date/time before time-sensitive lookups or answers.",
    "when_not_to_use": "Question is timeless or explicitly historical."
  },
  {
    "tool_name": "cli",
    "args_schema": {
      "command": "string (recommended, raw shell-like command; must start with `spogo`)",
      "args": "array<string> or string (optional alternative to command; if used must start with `spogo` token)"
    },
    "when_to_use": "Need Spotify operations via local spogo CLI, or need to inspect spogo help/usage via -h.",
    "when_not_to_use": "Request is unrelated to spogo/Spotify."
  },
  {
    "tool_name": "web_search",
    "args_schema": {
      "query": "string (required, non-empty)",
      "max_results": "integer 1-10 (optional, default 5)"
    },
    "when_to_use": "Need external factual information, latest/current info, or web-sourced recommendations.",
    "when_not_to_use": "Casual chat, personal memory recall, or when the answer can be provided from context."
  },
  {
    "tool_name": "discord_voice_join",
    "args_schema": {
      "channel_id": "string Discord channel id (optional; defaults to requester's current voice channel)"
    },
    "when_to_use": "User explicitly asks the assistant to join voice.",
    "when_not_to_use": "User did not request voice channel participation."
  },
  {
    "tool_name": "discord_voice_leave",
    "args_schema": {},
    "when_to_use": "User explicitly asks assistant to leave voice or stop voice interaction.",
    "when_not_to_use": "Bot is not connected to voice."
  }
]"#
}

const DEFAULT_SYSTEM_PROMPT_BASE: &str = "You are CompanionPilot, a helpful Discord AI companion.\nKeep replies concise and practical.\nNever emit XML/JSON/pseudo tool-call markup in normal replies.";

pub(super) fn default_system_prompt_base() -> &'static str {
    DEFAULT_SYSTEM_PROMPT_BASE
}

pub(super) fn build_system_prompt(
    memory: &MemoryContext,
    override_prompt: Option<&str>,
    selected_skills: &[SkillDefinition],
) -> String {
    let mut sections = if let Some(prompt) = override_prompt {
        vec![prompt.to_owned()]
    } else {
        vec![DEFAULT_SYSTEM_PROMPT_BASE.to_owned()]
    };

    if let Some(summary) = &memory.summary {
        sections.push(format!("Conversation summary: {summary}"));
    }

    if !memory.recent_messages.is_empty() {
        sections.push(build_recent_context_block(&memory.recent_messages));
    }

    if !memory.facts.is_empty() {
        let lines = memory
            .facts
            .iter()
            .map(|fact| format!("{} = {}", fact.key, fact.value))
            .collect::<Vec<_>>()
            .join("; ");
        sections.push(format!("Known user facts: {lines}"));
    }

    sections.push(build_selected_skill_context_block(selected_skills));

    sections.join("\n")
}

fn build_recent_context_block(recent_messages: &[String]) -> String {
    if recent_messages.is_empty() {
        return String::new();
    }

    let turns = recent_messages
        .iter()
        .rev()
        .take(8)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .map(|line| format!("- {line}"))
        .collect::<Vec<_>>()
        .join("\n");
    format!("Recent conversation turns:\n{turns}")
}
