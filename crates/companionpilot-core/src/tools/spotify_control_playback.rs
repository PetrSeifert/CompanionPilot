use anyhow::{Context, bail};
use reqwest::{Client, StatusCode, Url};
use serde::Deserialize;
use serde_json::{Map, Value, json};
use tracing::{info, warn};

use super::ToolResult;

pub const DEFAULT_SPOTIFY_CONTROL_BASE_URL: &str = "https://api.peterrock.dev/api/spotify/control";

#[derive(Debug, Clone)]
pub struct SpotifyControlPlaybackTool {
    client: Client,
    base_url: String,
    admin_token: String,
}

#[derive(Debug)]
struct PreparedControlRequest {
    user_id: String,
    action: String,
    payload: Value,
}

#[derive(Debug, Deserialize)]
struct ControlArgs {
    user_id: String,
    action: String,
    #[serde(default)]
    device_id: Option<String>,
    #[serde(default)]
    state: Option<Value>,
    #[serde(default)]
    position_ms: Option<u64>,
    #[serde(default)]
    volume: Option<u64>,
    #[serde(default)]
    uris: Vec<String>,
    #[serde(default)]
    context_uri: Option<String>,
    #[serde(default)]
    uri: Option<String>,
    #[serde(default)]
    device_ids: Vec<String>,
    #[serde(default)]
    play: Option<bool>,
}

impl SpotifyControlPlaybackTool {
    pub fn new(base_url: impl Into<String>, admin_token: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.into(),
            admin_token: admin_token.into(),
        }
    }

    pub async fn control_playback(&self, args: Value) -> anyhow::Result<ToolResult> {
        let prepared = prepare_control_request(args)?;
        info!(
            user_id = %prepared.user_id,
            action = %prepared.action,
            "spotify control playback request start"
        );

        let endpoint_url = build_action_url(&self.base_url, &prepared.user_id, &prepared.action)?;
        let response = self
            .client
            .post(endpoint_url.clone())
            .bearer_auth(&self.admin_token)
            .json(&prepared.payload)
            .send()
            .await
            .map_err(|error| {
                warn!(
                    ?error,
                    user_id = %prepared.user_id,
                    action = %prepared.action,
                    "spotify control playback request failed"
                );
                error
            })?;

        let status = response.status();
        let response_text = response.text().await.unwrap_or_default();
        if !status.is_success() {
            let response_summary = truncate_for_output(response_text.trim());
            warn!(
                status = %status,
                user_id = %prepared.user_id,
                action = %prepared.action,
                body = %response_summary,
                "spotify control playback returned error status"
            );
            bail!(
                "spotify control request failed with status {}: {}. {}",
                status.as_u16(),
                status.canonical_reason().unwrap_or("unknown error"),
                response_summary
            );
        }

        let mut lines = vec![
            format!("Spotify user id: {}", prepared.user_id),
            format!("Action: {}", prepared.action),
            format!("Result: success ({})", status),
        ];
        if !response_text.trim().is_empty() && status != StatusCode::NO_CONTENT {
            lines.push(format!(
                "Response: {}",
                truncate_for_output(response_text.trim())
            ));
        }

        Ok(ToolResult {
            text: lines.join("\n"),
            citations: vec![endpoint_url.to_string()],
        })
    }
}

fn build_action_url(base_url: &str, user_id: &str, action: &str) -> anyhow::Result<Url> {
    let mut url = Url::parse(base_url).context("invalid spotify control base URL")?;
    let mut segments = url
        .path_segments_mut()
        .map_err(|_| anyhow::anyhow!("spotify control base URL cannot be a base"))?;
    segments.push(user_id);
    segments.push(action);
    drop(segments);
    Ok(url)
}

fn prepare_control_request(args: Value) -> anyhow::Result<PreparedControlRequest> {
    let parsed: ControlArgs = serde_json::from_value(args).context(
        "spotify_control_playback expects JSON args with user_id, action, and optional action-specific fields",
    )?;

    let user_id = parsed.user_id.trim().to_owned();
    if user_id.is_empty() {
        bail!("spotify_control_playback requires non-empty user_id");
    }

    let action = parsed.action.trim().to_ascii_lowercase();
    if !is_supported_action(&action) {
        bail!(
            "unsupported spotify_control_playback action: {}",
            parsed.action
        );
    }

    let payload = build_payload(&action, &parsed)?;

    Ok(PreparedControlRequest {
        user_id,
        action,
        payload,
    })
}

fn build_payload(action: &str, args: &ControlArgs) -> anyhow::Result<Value> {
    let mut payload = Map::new();

    if let Some(device_id) = normalize_opt_string(args.device_id.as_deref()) {
        payload.insert("device_id".to_owned(), Value::String(device_id));
    }

    match action {
        "play" | "pause" | "toggle" | "next" | "previous" => {}
        "shuffle" => {
            let state = args
                .state
                .as_ref()
                .and_then(Value::as_bool)
                .context("shuffle action requires boolean state")?;
            payload.insert("state".to_owned(), Value::Bool(state));
        }
        "repeat" => {
            let state = args
                .state
                .as_ref()
                .and_then(Value::as_str)
                .map(str::trim)
                .context("repeat action requires string state (off|track|context)")?;
            if !matches!(state, "off" | "track" | "context") {
                bail!("repeat state must be one of: off, track, context");
            }
            payload.insert("state".to_owned(), Value::String(state.to_owned()));
        }
        "seek" => {
            let position_ms = args
                .position_ms
                .context("seek action requires position_ms in milliseconds")?;
            payload.insert("position_ms".to_owned(), json!(position_ms));
        }
        "volume" => {
            let volume = args
                .volume
                .context("volume action requires volume (0..100)")?;
            if volume > 100 {
                bail!("volume must be in range 0..100");
            }
            payload.insert("volume".to_owned(), json!(volume));
        }
        "playback" => {
            let uris = normalize_string_vec(&args.uris);
            let context_uri = normalize_opt_string(args.context_uri.as_deref());
            if uris.is_empty() && context_uri.is_none() {
                bail!("playback action requires at least one uri in uris or context_uri");
            }
            if !uris.is_empty() {
                payload.insert("uris".to_owned(), json!(uris));
            }
            if let Some(context_uri) = context_uri {
                payload.insert("context_uri".to_owned(), Value::String(context_uri));
            }
            if let Some(position_ms) = args.position_ms {
                payload.insert("position_ms".to_owned(), json!(position_ms));
            }
        }
        "queue" => {
            let uri =
                normalize_opt_string(args.uri.as_deref()).context("queue action requires uri")?;
            payload.insert("uri".to_owned(), Value::String(uri));
        }
        "transfer" => {
            let device_ids = normalize_string_vec(&args.device_ids);
            if device_ids.is_empty() {
                bail!("transfer action requires non-empty device_ids");
            }
            payload.insert("device_ids".to_owned(), json!(device_ids));
            if let Some(play) = args.play {
                payload.insert("play".to_owned(), Value::Bool(play));
            }
        }
        _ => {
            bail!("unsupported action");
        }
    }

    Ok(Value::Object(payload))
}

fn is_supported_action(action: &str) -> bool {
    matches!(
        action,
        "play"
            | "pause"
            | "toggle"
            | "next"
            | "previous"
            | "shuffle"
            | "repeat"
            | "seek"
            | "volume"
            | "playback"
            | "queue"
            | "transfer"
    )
}

fn normalize_opt_string(raw: Option<&str>) -> Option<String> {
    raw.map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn normalize_string_vec(values: &[String]) -> Vec<String> {
    values
        .iter()
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn truncate_for_output(raw: &str) -> String {
    const LIMIT: usize = 220;
    if raw.chars().count() <= LIMIT {
        return raw.to_owned();
    }

    let truncated = raw.chars().take(LIMIT).collect::<String>();
    format!("{truncated}...")
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{build_payload, is_supported_action, prepare_control_request};

    #[test]
    fn supports_documented_actions() {
        assert!(is_supported_action("play"));
        assert!(is_supported_action("transfer"));
        assert!(!is_supported_action("like"));
    }

    #[test]
    fn prepares_simple_next_request() {
        let prepared = prepare_control_request(json!({
            "user_id": "alice123",
            "action": "NEXT",
            "device_id": "dev-1"
        }))
        .expect("request should parse");

        assert_eq!(prepared.user_id, "alice123");
        assert_eq!(prepared.action, "next");
        assert_eq!(prepared.payload["device_id"], "dev-1");
    }

    #[test]
    fn repeat_requires_supported_state() {
        let result = prepare_control_request(json!({
            "user_id": "alice123",
            "action": "repeat",
            "state": "invalid"
        }));
        assert!(result.is_err());
    }

    #[test]
    fn playback_requires_target() {
        let result = prepare_control_request(json!({
            "user_id": "alice123",
            "action": "playback"
        }));
        assert!(result.is_err());
    }

    #[test]
    fn transfer_requires_device_ids() {
        let result = prepare_control_request(json!({
            "user_id": "alice123",
            "action": "transfer",
            "device_ids": []
        }));
        assert!(result.is_err());
    }

    #[test]
    fn volume_is_limited_to_percentage_range() {
        let parsed = serde_json::from_value(json!({
            "user_id": "alice123",
            "action": "volume",
            "volume": 120
        }))
        .expect("args should parse");
        let result = build_payload("volume", &parsed);
        assert!(result.is_err());
    }
}
