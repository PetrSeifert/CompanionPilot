use anyhow::Context;
use reqwest::{Client, Url};
use serde_json::Value;
use tracing::{debug, info, warn};

use super::ToolResult;

pub const DEFAULT_SPOTIFY_SEARCH_API_URL: &str = "https://api.peterrock.dev/api/spotify/search";

#[derive(Debug, Clone)]
pub struct SpotifySearchTool {
    client: Client,
    endpoint_url: String,
}

impl Default for SpotifySearchTool {
    fn default() -> Self {
        Self::new(DEFAULT_SPOTIFY_SEARCH_API_URL)
    }
}

impl SpotifySearchTool {
    pub fn new(endpoint_url: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            endpoint_url: endpoint_url.into(),
        }
    }

    pub async fn search(&self, args: Value) -> anyhow::Result<ToolResult> {
        let query = args
            .get("q")
            .or_else(|| args.get("query"))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| anyhow::anyhow!("spotify_search requires string arg `q`"))?;

        let type_csv = normalize_search_types(args.get("type"))
            .ok_or_else(|| anyhow::anyhow!("spotify_search requires arg `type`"))?;

        let limit = args
            .get("limit")
            .and_then(Value::as_u64)
            .map(|value| value.clamp(1, 50));
        let offset = args
            .get("offset")
            .and_then(Value::as_u64)
            .map(|value| value.clamp(0, 1000));
        let market = args
            .get("market")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| is_valid_market_code(value))
            .map(|value| value.to_ascii_uppercase());
        let include_external = args
            .get("include_external")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty());

        info!(types = %type_csv, "spotify search request start");
        debug!(query = %query, "spotify search query");

        let mut url = Url::parse(&self.endpoint_url).context("invalid spotify search base URL")?;
        {
            let mut pairs = url.query_pairs_mut();
            pairs.append_pair("q", query);
            pairs.append_pair("type", &type_csv);
            if let Some(limit) = limit {
                pairs.append_pair("limit", &limit.to_string());
            }
            if let Some(offset) = offset {
                pairs.append_pair("offset", &offset.to_string());
            }
            if let Some(market) = market.as_deref() {
                pairs.append_pair("market", market);
            }
            if let Some(include_external) = include_external {
                pairs.append_pair("include_external", include_external);
            }
        }

        let payload = self
            .client
            .get(url.clone())
            .send()
            .await
            .map_err(|error| {
                warn!(?error, "spotify search request failed");
                error
            })?
            .error_for_status()
            .map_err(|error| {
                warn!(?error, "spotify search returned error status");
                error
            })?
            .json::<Value>()
            .await
            .map_err(|error| {
                warn!(?error, "failed to deserialize spotify search response");
                error
            })?;
        let category_count = payload.as_object().map(|object| object.len()).unwrap_or(0);
        info!(category_count, "spotify search request success");

        let pretty_payload = serde_json::to_string_pretty(&payload)
            .context("failed to format spotify search response as JSON")?;

        Ok(ToolResult {
            text: format!("Spotify search response:\n{pretty_payload}"),
            citations: vec![url.to_string()],
        })
    }
}

fn normalize_search_types(value: Option<&Value>) -> Option<String> {
    let mut normalized = Vec::new();

    let raw_values = match value? {
        Value::String(value) => value.split(',').map(str::to_owned).collect::<Vec<_>>(),
        Value::Array(values) => values
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect::<Vec<_>>(),
        _ => return None,
    };

    for raw in raw_values {
        let value = raw.trim().to_ascii_lowercase();
        if value.is_empty() || !is_supported_type(&value) || normalized.contains(&value) {
            continue;
        }
        normalized.push(value);
    }

    if normalized.is_empty() {
        None
    } else {
        Some(normalized.join(","))
    }
}

fn is_supported_type(value: &str) -> bool {
    matches!(
        value,
        "album" | "artist" | "playlist" | "track" | "show" | "episode" | "audiobook"
    )
}

fn is_valid_market_code(value: &str) -> bool {
    value.len() == 2
        && value
            .chars()
            .all(|character| character.is_ascii_alphabetic())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::normalize_search_types;

    #[test]
    fn normalize_search_types_from_csv() {
        let value = json!("track, artist, invalid,track,album");
        let normalized = normalize_search_types(Some(&value)).expect("types should parse");
        assert_eq!(normalized, "track,artist,album");
    }

    #[test]
    fn normalize_search_types_from_array() {
        let value = json!(["artist", "playlist", ""]);
        let normalized = normalize_search_types(Some(&value)).expect("types should parse");
        assert_eq!(normalized, "artist,playlist");
    }

    #[test]
    fn normalize_search_types_rejects_invalid_payload() {
        let value = json!(["invalid"]);
        assert!(normalize_search_types(Some(&value)).is_none());
    }
}
