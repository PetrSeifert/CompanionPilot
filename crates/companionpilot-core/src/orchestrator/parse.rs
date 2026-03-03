use serde::de::DeserializeOwned;

use super::contracts::SkillSelectionPlan;

pub(super) fn parse_skill_selection_plan(
    raw: &str,
) -> Result<SkillSelectionPlan, serde_json::Error> {
    parse_json_plan(raw)
}

fn parse_json_plan<T: DeserializeOwned>(raw: &str) -> Result<T, serde_json::Error> {
    let candidate = raw
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();

    match serde_json::from_str::<T>(candidate) {
        Ok(plan) => Ok(plan),
        Err(original_error) => {
            if let Some(object_candidate) = extract_first_json_object(candidate) {
                serde_json::from_str::<T>(object_candidate).map_err(|_| original_error)
            } else {
                Err(original_error)
            }
        }
    }
}

fn extract_first_json_object(raw: &str) -> Option<&str> {
    let mut start_index: Option<usize> = None;
    let mut depth = 0usize;
    let mut in_string = false;
    let mut is_escaped = false;

    for (index, character) in raw.char_indices() {
        if start_index.is_none() {
            if character == '{' {
                start_index = Some(index);
                depth = 1;
            }
            continue;
        }

        if in_string {
            if is_escaped {
                is_escaped = false;
                continue;
            }
            match character {
                '\\' => is_escaped = true,
                '"' => in_string = false,
                _ => {}
            }
            continue;
        }

        match character {
            '"' => in_string = true,
            '{' => depth += 1,
            '}' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    let start = start_index?;
                    return Some(&raw[start..=index]);
                }
            }
            _ => {}
        }
    }

    None
}
