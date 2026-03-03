#[derive(Debug, Clone, Copy)]
pub struct ToolSkill {
    pub markdown: &'static str,
}

pub fn get_tool_skill(tool_name: &str) -> Option<ToolSkill> {
    match tool_name {
        "current_datetime" => Some(ToolSkill {
            markdown: include_str!("skills/current_datetime.md"),
        }),
        "cli" => Some(ToolSkill {
            markdown: include_str!("skills/cli.md"),
        }),
        "web_search" => Some(ToolSkill {
            markdown: include_str!("skills/web_search.md"),
        }),
        "discord_voice_join" => Some(ToolSkill {
            markdown: include_str!("skills/discord_voice_join.md"),
        }),
        "discord_voice_leave" => Some(ToolSkill {
            markdown: include_str!("skills/discord_voice_leave.md"),
        }),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::get_tool_skill;

    #[test]
    fn returns_skill_for_known_tool() {
        let skill = get_tool_skill("web_search").expect("web_search skill should exist");
        assert!(!skill.markdown.trim().is_empty());
    }

    #[test]
    fn returns_none_for_unknown_tool() {
        assert!(get_tool_skill("unknown_tool").is_none());
    }
}
