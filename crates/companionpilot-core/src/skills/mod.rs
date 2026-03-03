use std::{
    collections::{HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
};

use anyhow::Context;
use tracing::warn;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkillMetadata {
    pub id: String,
    pub title: String,
    pub description: String,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct SkillDefinition {
    pub metadata: SkillMetadata,
    pub body_markdown: String,
    pub source_path: PathBuf,
}

#[derive(Debug, Clone, Default)]
pub struct SkillCatalog {
    skills: Vec<SkillDefinition>,
    id_index: HashMap<String, usize>,
}

impl SkillCatalog {
    pub fn load_from_dir(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let path = path.as_ref();
        if !path.exists() {
            warn!(
                path = %path.display(),
                "skills directory is missing; continuing with empty catalog"
            );
            return Ok(Self::default());
        }
        if !path.is_dir() {
            warn!(
                path = %path.display(),
                "skills path is not a directory; continuing with empty catalog"
            );
            return Ok(Self::default());
        }

        let mut skill_paths = fs::read_dir(path)
            .with_context(|| format!("failed to read skills directory {}", path.display()))?
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|entry_path| {
                entry_path.is_file()
                    && entry_path
                        .extension()
                        .and_then(|ext| ext.to_str())
                        .map(|ext| ext.eq_ignore_ascii_case("md"))
                        .unwrap_or(false)
            })
            .collect::<Vec<_>>();
        skill_paths.sort();

        let mut skills = Vec::new();
        let mut id_index = HashMap::new();
        for skill_path in skill_paths {
            let skill = match parse_skill_file(&skill_path) {
                Ok(skill) => skill,
                Err(error) => {
                    warn!(
                        path = %skill_path.display(),
                        ?error,
                        "failed to parse skill file; skipping"
                    );
                    continue;
                }
            };

            if id_index.contains_key(&skill.metadata.id) {
                warn!(
                    skill_id = %skill.metadata.id,
                    path = %skill.source_path.display(),
                    "duplicate skill id detected; keeping first definition and skipping duplicate"
                );
                continue;
            }

            let index = skills.len();
            id_index.insert(skill.metadata.id.clone(), index);
            skills.push(skill);
        }

        Ok(Self { skills, id_index })
    }

    pub fn len(&self) -> usize {
        self.skills.len()
    }

    pub fn is_empty(&self) -> bool {
        self.skills.is_empty()
    }

    pub fn metadata_inventory(&self) -> Vec<SkillMetadata> {
        self.skills
            .iter()
            .map(|skill| skill.metadata.clone())
            .collect()
    }

    pub fn sanitize_selected_ids(&self, selected_ids: Vec<String>) -> Vec<String> {
        let mut deduped = Vec::new();
        let mut seen = HashSet::new();
        for selected_id in selected_ids {
            if !self.id_index.contains_key(&selected_id) {
                continue;
            }
            if !seen.insert(selected_id.clone()) {
                continue;
            }
            deduped.push(selected_id);
        }
        deduped
    }

    pub fn select_by_ids(&self, selected_ids: &[String]) -> Vec<SkillDefinition> {
        selected_ids
            .iter()
            .filter_map(|skill_id| {
                let index = self.id_index.get(skill_id)?;
                self.skills.get(*index).cloned()
            })
            .collect()
    }
}

fn parse_skill_file(path: &Path) -> anyhow::Result<SkillDefinition> {
    let raw = fs::read_to_string(path)
        .with_context(|| format!("failed to read skill file {}", path.display()))?;
    let (frontmatter, body) = split_frontmatter(&raw)?;
    let fields = parse_frontmatter_fields(frontmatter);

    let id = required_field(&fields, "id", path)?;
    let description = required_field(&fields, "description", path)?;
    let title = optional_field(&fields, "title").unwrap_or_else(|| id.clone());
    let tags = parse_tags(optional_field(&fields, "tags"));

    Ok(SkillDefinition {
        metadata: SkillMetadata {
            id,
            title,
            description,
            tags,
        },
        body_markdown: body.trim().to_owned(),
        source_path: path.to_path_buf(),
    })
}

fn split_frontmatter(raw: &str) -> anyhow::Result<(String, String)> {
    let normalized = raw.strip_prefix('\u{feff}').unwrap_or(raw);
    let mut lines = normalized.lines();
    let Some(first_line) = lines.next() else {
        anyhow::bail!("skill file is empty");
    };
    if first_line.trim() != "---" {
        anyhow::bail!("skill file must start with YAML frontmatter");
    }

    let mut frontmatter_lines = Vec::new();
    let mut frontmatter_closed = false;
    for line in lines.by_ref() {
        if line.trim() == "---" {
            frontmatter_closed = true;
            break;
        }
        frontmatter_lines.push(line);
    }
    if !frontmatter_closed {
        anyhow::bail!("frontmatter is not terminated with closing ---");
    }

    let body = lines.collect::<Vec<_>>().join("\n");
    Ok((frontmatter_lines.join("\n"), body))
}

fn parse_frontmatter_fields(frontmatter: String) -> HashMap<String, String> {
    let mut fields = HashMap::new();
    for line in frontmatter.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let Some((key, raw_value)) = trimmed.split_once(':') else {
            continue;
        };
        let key = key.trim().to_ascii_lowercase();
        let value = raw_value
            .trim()
            .trim_matches('"')
            .trim_matches('\'')
            .trim()
            .to_owned();
        fields.insert(key, value);
    }
    fields
}

fn required_field(
    fields: &HashMap<String, String>,
    key: &str,
    path: &Path,
) -> anyhow::Result<String> {
    let value = optional_field(fields, key)
        .with_context(|| format!("missing required `{key}` field in {}", path.display()))?;
    Ok(value)
}

fn optional_field(fields: &HashMap<String, String>, key: &str) -> Option<String> {
    fields
        .get(&key.to_ascii_lowercase())
        .map(String::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

fn parse_tags(raw: Option<String>) -> Vec<String> {
    let Some(raw) = raw else {
        return Vec::new();
    };
    let inner = raw
        .trim()
        .trim_start_matches('[')
        .trim_end_matches(']')
        .trim();

    let mut tags = Vec::new();
    let mut seen = HashSet::new();
    for part in inner.split(',') {
        let tag = part
            .trim()
            .trim_matches('"')
            .trim_matches('\'')
            .trim()
            .to_ascii_lowercase();
        if tag.is_empty() || !seen.insert(tag.clone()) {
            continue;
        }
        tags.push(tag);
    }
    tags
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::PathBuf,
        time::{SystemTime, UNIX_EPOCH},
    };

    use super::SkillCatalog;

    fn temp_test_dir(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "companionpilot-skills-{name}-{}-{nanos}",
            std::process::id()
        ));
        fs::create_dir_all(&path).expect("temp dir should be created");
        path
    }

    #[test]
    fn loads_valid_skills_and_metadata() {
        let dir = temp_test_dir("valid");
        fs::write(
            dir.join("a.md"),
            r#"---
id: productivity-notes
title: Productivity Notes
description: Keep actionable notes concise.
tags: [notes, productivity]
---
# Notes
Keep summaries short.
"#,
        )
        .expect("skill file should be written");

        let catalog = SkillCatalog::load_from_dir(&dir).expect("catalog should load");
        assert_eq!(catalog.len(), 1);
        let metadata = catalog.metadata_inventory();
        assert_eq!(metadata.len(), 1);
        assert_eq!(metadata[0].id, "productivity-notes");
        assert_eq!(metadata[0].title, "Productivity Notes");
        assert_eq!(metadata[0].description, "Keep actionable notes concise.");
        assert_eq!(metadata[0].tags, vec!["notes", "productivity"]);

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn skips_invalid_frontmatter() {
        let dir = temp_test_dir("invalid");
        fs::write(
            dir.join("invalid.md"),
            r#"---
title: Missing Id
description: Missing required id
---
body
"#,
        )
        .expect("invalid skill file should be written");

        let catalog = SkillCatalog::load_from_dir(&dir).expect("catalog load should not fail");
        assert_eq!(catalog.len(), 0);

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn duplicate_ids_keep_first_file() {
        let dir = temp_test_dir("dupe");
        fs::write(
            dir.join("a.md"),
            r#"---
id: dupe-skill
description: first
---
first body
"#,
        )
        .expect("first skill file should be written");
        fs::write(
            dir.join("b.md"),
            r#"---
id: dupe-skill
description: second
---
second body
"#,
        )
        .expect("second skill file should be written");

        let catalog = SkillCatalog::load_from_dir(&dir).expect("catalog should load");
        assert_eq!(catalog.len(), 1);
        let selected = catalog.select_by_ids(&["dupe-skill".to_owned()]);
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].metadata.description, "first");
        assert_eq!(selected[0].body_markdown, "first body");

        let _ = fs::remove_dir_all(dir);
    }
}
