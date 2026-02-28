use anyhow::Context;
use reqwest::Client;
use serde_json::Value;
use tracing::{info, warn};

use super::ToolResult;

const DEFAULT_PLAYING_STATUS_URL: &str = "https://api.peterrock.dev/api/spotify/playing-status";

#[derive(Debug, Clone)]
pub struct SpotifyPlayingStatusTool {
    client: Client,
    endpoint_url: String,
}

impl Default for SpotifyPlayingStatusTool {
    fn default() -> Self {
        Self::new(DEFAULT_PLAYING_STATUS_URL)
    }
}

impl SpotifyPlayingStatusTool {
    pub fn new(endpoint_url: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            endpoint_url: endpoint_url.into(),
        }
    }

    pub async fn get_playing_status(&self, _args: Value) -> anyhow::Result<ToolResult> {
        info!("spotify playing status request start");

        let payload = self
            .client
            .get(&self.endpoint_url)
            .send()
            .await
            .map_err(|error| {
                warn!(?error, "spotify playing status request failed");
                error
            })?
            .error_for_status()
            .map_err(|error| {
                warn!(?error, "spotify playing status returned error status");
                error
            })?
            .json::<Value>()
            .await
            .map_err(|error| {
                warn!(
                    ?error,
                    "failed to deserialize spotify playing status response"
                );
                error
            })?;

        let text = format_playing_status(&payload)
            .context("spotify_playing_status response format was not recognized")?;

        Ok(ToolResult {
            text,
            citations: vec![self.endpoint_url.clone()],
        })
    }
}

fn format_playing_status(payload: &Value) -> Option<String> {
    let (user, status) = extract_user_and_status(payload)?;

    let display_name = user
        .get("display_name")
        .and_then(Value::as_str)
        .unwrap_or("Unknown");
    let is_playing = status
        .get("is_playing")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    let mut lines = vec![format!("Spotify user: {display_name}")];

    if !is_playing {
        lines.push("Playback status: not currently playing".to_owned());
        return Some(lines.join("\n"));
    }

    lines.push("Playback status: currently playing".to_owned());

    if let Some(track) = status.get("track").and_then(Value::as_object) {
        let track_name = track
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("Unknown track");
        let artist = track
            .get("artist")
            .and_then(Value::as_str)
            .unwrap_or("Unknown artist");
        let album = track
            .get("album")
            .and_then(Value::as_str)
            .unwrap_or("Unknown album");

        lines.push(format!("Track: {track_name}"));
        lines.push(format!("Artist: {artist}"));
        lines.push(format!("Album: {album}"));

        let progress_ms = status
            .get("progress_ms")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let duration_ms = track
            .get("duration_ms")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        if duration_ms > 0 {
            lines.push(format!(
                "Progress: {} / {}",
                format_millis(progress_ms),
                format_millis(duration_ms)
            ));
        }

        if let Some(uri) = track.get("uri").and_then(Value::as_str) {
            lines.push(format!("URI: {uri}"));
        }
    }

    Some(lines.join("\n"))
}

fn extract_user_and_status(payload: &Value) -> Option<(&Value, &Value)> {
    let array = payload.as_array()?;
    if array.len() == 2 && array[0].is_object() && array[1].is_object() {
        return Some((&array[0], &array[1]));
    }

    for item in array {
        if let Some(found) = extract_user_and_status(item) {
            return Some(found);
        }
    }

    None
}

fn format_millis(ms: u64) -> String {
    let total_seconds = ms / 1000;
    let minutes = total_seconds / 60;
    let seconds = total_seconds % 60;
    format!("{minutes:02}:{seconds:02}")
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::format_playing_status;

    #[test]
    fn formats_current_track_from_nested_array_payload() {
        let payload = json!([
            [
                {
                    "display_name": "Petr Seifert",
                    "id": "3137pjdifc6dxkr7hny6se6av3my"
                },
                {
                    "is_playing": true,
                    "track": {
                        "id": "45JYEmfoWSpCA3Paut7YXE",
                        "name": "MIDDLE OF THE NIGHT",
                        "artist": "Elley Duhe",
                        "album": "PHOENIX",
                        "album_art_url": "https://i.scdn.co/image/ab67616d0000b273e2d712966a13667c0ebdf469",
                        "duration_ms": 184453,
                        "uri": "spotify:track:45JYEmfoWSpCA3Paut7YXE"
                    },
                    "progress_ms": 53338
                }
            ]
        ]);

        let text = format_playing_status(&payload).expect("payload should parse");
        assert!(text.contains("Spotify user: Petr Seifert"));
        assert!(text.contains("Track: MIDDLE OF THE NIGHT"));
        assert!(text.contains("Artist: Elley Duhe"));
        assert!(text.contains("Progress: 00:53 / 03:04"));
    }

    #[test]
    fn formats_not_playing_state() {
        let payload = json!([
            {
                "display_name": "Petr Seifert",
                "id": "3137pjdifc6dxkr7hny6se6av3my"
            },
            {
                "is_playing": false,
                "track": null,
                "progress_ms": 0
            }
        ]);

        let text = format_playing_status(&payload).expect("payload should parse");
        assert!(text.contains("Playback status: not currently playing"));
        assert!(!text.contains("Track:"));
    }
}
