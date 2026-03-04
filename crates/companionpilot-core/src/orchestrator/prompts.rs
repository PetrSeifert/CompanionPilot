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
    let context_block = build_context_block(memory);
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

pub(super) fn build_native_agent_system_prompt(
    memory: &MemoryContext,
    override_prompt: Option<&str>,
    selected_skills: &[SkillDefinition],
) -> String {
    let mut sections = if let Some(prompt) = override_prompt {
        vec![prompt.to_owned()]
    } else {
        vec![default_system_prompt_base().to_owned()]
    };

    sections.push(
        "Use tools natively when needed. Prefer direct answers when tools are unnecessary.
When you need current/news/price/weather/factual verification, call `web_search`.
When you need to execute a local command, call `run_terminal_command`.
For `run_terminal_command`, provide either `command` (string) or `args` (array<string> or string).
Do not include shell operators like `|`, `&&`, `||`, `;`, or redirection.
Use `current_datetime` when the request depends on current date/time context.
If a durable user fact is stated or corrected, call `store_memory` with key/value/confidence.
Do not emit pseudo tool-call markup in normal text responses."
            .to_owned(),
    );

    let context_block = build_support_context_block(memory, selected_skills);
    if !context_block.is_empty() {
        sections.push(context_block.trim().to_owned());
    }

    sections.join("\n\n")
}

pub(super) fn build_support_context_block(
    memory: &MemoryContext,
    selected_skills: &[SkillDefinition],
) -> String {
    let mut sections = Vec::new();
    let context = build_context_block(memory);
    if !context.is_empty() {
        sections.push(context.trim().to_owned());
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

pub(super) fn default_system_prompt_base() -> &'static str {
    "You are CompanionPilot, a helpful Discord AI companion.
Keep replies concise and practical."
}

fn build_context_block(memory: &MemoryContext) -> String {
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
