use anyhow::Context;
use reqwest::Client;
use serde::Deserialize;
use serde_json::Value;
use tracing::{info, warn};

use super::ToolResult;

pub const DEFAULT_SPOTIFY_USERS_API_URL: &str = "https://api.peterrock.dev/api/spotify/users";

#[derive(Debug, Clone)]
pub struct SpotifyUsersListTool {
    client: Client,
    endpoint_url: String,
    admin_token: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SpotifyUserSummary {
    id: String,
    #[serde(default)]
    display_name: Option<String>,
}

impl Default for SpotifyUsersListTool {
    fn default() -> Self {
        Self::new(DEFAULT_SPOTIFY_USERS_API_URL, None)
    }
}

impl SpotifyUsersListTool {
    pub fn new(endpoint_url: impl Into<String>, admin_token: Option<String>) -> Self {
        Self {
            client: Client::new(),
            endpoint_url: endpoint_url.into(),
            admin_token,
        }
    }

    pub async fn list_users(&self, _args: Value) -> anyhow::Result<ToolResult> {
        info!("spotify users list request start");

        let request = self.client.get(&self.endpoint_url);
        let request = if let Some(admin_token) = &self.admin_token {
            request.bearer_auth(admin_token)
        } else {
            request
        };

        let payload = request
            .send()
            .await
            .map_err(|error| {
                warn!(?error, "spotify users list request failed");
                error
            })?
            .error_for_status()
            .map_err(|error| {
                warn!(?error, "spotify users list returned error status");
                error
            })?
            .json::<Vec<SpotifyUserSummary>>()
            .await
            .map_err(|error| {
                warn!(?error, "failed to deserialize spotify users list response");
                error
            })?;

        let text = format_users_list(&payload)
            .context("spotify_users_list response format was not recognized")?;

        Ok(ToolResult {
            text,
            citations: vec![self.endpoint_url.clone()],
        })
    }
}

fn format_users_list(users: &[SpotifyUserSummary]) -> Option<String> {
    if users.is_empty() {
        return Some("Tracked Spotify users: none".to_owned());
    }

    let mut lines = vec![format!("Tracked Spotify users: {}", users.len())];
    for user in users {
        if user.id.trim().is_empty() {
            return None;
        }

        let display_name = user
            .display_name
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or("Unknown");
        lines.push(format!("- {} ({})", display_name, user.id.trim()));
    }

    Some(lines.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::{SpotifyUserSummary, format_users_list};

    #[test]
    fn formats_non_empty_users_list() {
        let users = vec![
            SpotifyUserSummary {
                id: "alice123".to_owned(),
                display_name: Some("Alice".to_owned()),
            },
            SpotifyUserSummary {
                id: "bob456".to_owned(),
                display_name: Some("Bob".to_owned()),
            },
        ];

        let text = format_users_list(&users).expect("users should format");
        assert!(text.contains("Tracked Spotify users: 2"));
        assert!(text.contains("- Alice (alice123)"));
        assert!(text.contains("- Bob (bob456)"));
    }

    #[test]
    fn formats_empty_users_list() {
        let text = format_users_list(&[]).expect("empty users should format");
        assert_eq!(text, "Tracked Spotify users: none");
    }
}
