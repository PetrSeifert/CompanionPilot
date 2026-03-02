use anyhow::{Context, bail};
use serde_json::Value;
use tracing::{debug, info};

use super::{SpogoCli, ToolResult};

#[derive(Debug, Clone)]
pub struct SpogoSearchTool {
    cli: SpogoCli,
}

impl Default for SpogoSearchTool {
    fn default() -> Self {
        Self::new(SpogoCli::default())
    }
}

impl SpogoSearchTool {
    pub fn new(cli: SpogoCli) -> Self {
        Self { cli }
    }

    pub async fn search(&self, args: Value) -> anyhow::Result<ToolResult> {
        let query = args
            .get("q")
            .or_else(|| args.get("query"))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| anyhow::anyhow!("spogo_search requires string arg `q`"))?;
        let search_type = normalize_search_type(args.get("type"))
            .ok_or_else(|| anyhow::anyhow!("spogo_search requires arg `type`"))?;

        let limit = args
            .get("limit")
            .and_then(Value::as_u64)
            .map(|value| value.clamp(1, 50));
        let offset = args
            .get("offset")
            .and_then(Value::as_u64)
            .map(|value| value.clamp(0, 1000));

        info!(search_type = %search_type, "spogo search request start");
        debug!(query = %query, "spogo search query");

        let mut command_args = vec![
            "--json".to_owned(),
            "search".to_owned(),
            search_type.to_owned(),
            query.to_owned(),
        ];
        if let Some(limit) = limit {
            command_args.push("--limit".to_owned());
            command_args.push(limit.to_string());
        }
        if let Some(offset) = offset {
            command_args.push("--offset".to_owned());
            command_args.push(offset.to_string());
        }

        let output = self.cli.execute(command_args).await?;
        if output.stdout.trim().is_empty() {
            bail!("spogo_search returned empty response");
        }
        let payload: Value = serde_json::from_str(output.stdout.trim())
            .context("spogo_search returned invalid JSON")?;
        let pretty_payload =
            serde_json::to_string_pretty(&payload).context("failed to format spogo_search JSON")?;

        Ok(ToolResult {
            text: format!("Spogo search response:\n{pretty_payload}"),
            citations: vec![output.command_line],
        })
    }
}

fn normalize_search_type(value: Option<&Value>) -> Option<&'static str> {
    let candidate = match value? {
        Value::String(value) => Some(value.trim().to_ascii_lowercase()),
        Value::Array(values) => values
            .iter()
            .filter_map(Value::as_str)
            .map(str::trim)
            .find(|item| !item.is_empty())
            .map(|item| item.to_ascii_lowercase()),
        _ => None,
    }?;
    if candidate.is_empty() || !is_supported_type(&candidate) {
        return None;
    }
    Some(match candidate.as_str() {
        "track" => "track",
        "artist" => "artist",
        "album" => "album",
        "playlist" => "playlist",
        "show" => "show",
        "episode" => "episode",
        "audiobook" => "audiobook",
        _ => return None,
    })
}

fn is_supported_type(value: &str) -> bool {
    matches!(
        value,
        "album" | "artist" | "playlist" | "track" | "show" | "episode" | "audiobook"
    )
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::normalize_search_type;

    #[test]
    fn normalize_search_type_from_string() {
        let value = json!("TRACK");
        assert_eq!(normalize_search_type(Some(&value)), Some("track"));
    }

    #[test]
    fn normalize_search_type_from_array() {
        let value = json!(["", "artist", "track"]);
        assert_eq!(normalize_search_type(Some(&value)), Some("artist"));
    }

    #[test]
    fn normalize_search_type_rejects_invalid() {
        let value = json!("invalid");
        assert!(normalize_search_type(Some(&value)).is_none());
    }
}
