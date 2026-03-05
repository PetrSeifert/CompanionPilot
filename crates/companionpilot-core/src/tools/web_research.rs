use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::{debug, info, warn};

use super::ToolResult;

const POLL_INTERVAL_SECS: u64 = 10;
const MAX_POLL_DURATION_SECS: u64 = 180;

#[derive(Debug, Clone)]
pub struct TavilyResearchTool {
    client: Client,
    api_key: String,
}

impl TavilyResearchTool {
    pub fn new(api_key: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
        }
    }

    pub async fn research(&self, args: Value) -> anyhow::Result<ToolResult> {
        let input = args
            .get("input")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("web_research requires string arg `input`"))?;
        if input.trim().is_empty() {
            anyhow::bail!("web_research requires a non-empty `input`");
        }

        info!("tavily research start");
        debug!(input = %input, "tavily research input");

        let mut payload = ResearchRequest {
            api_key: &self.api_key,
            input,
            model: None,
        };

        if let Some(model) = args.get("model").and_then(Value::as_str) {
            if matches!(model, "mini" | "pro" | "auto") {
                payload.model = Some(model.to_owned());
            }
        }

        // Start the research task
        let start_response = self
            .client
            .post("https://api.tavily.com/research")
            .json(&payload)
            .send()
            .await
            .map_err(|error| {
                warn!(?error, "tavily research request failed");
                error
            })?
            .error_for_status()
            .map_err(|error| {
                warn!(?error, "tavily research returned error status");
                error
            })?
            .json::<ResearchStartResponse>()
            .await
            .map_err(|error| {
                warn!(?error, "failed to deserialize tavily research start response");
                error
            })?;

        let request_id = start_response.request_id;
        info!(request_id = %request_id, "tavily research task submitted, polling");

        // Poll for results
        let poll_url = format!("https://api.tavily.com/research/{request_id}");
        let max_attempts = MAX_POLL_DURATION_SECS / POLL_INTERVAL_SECS;

        for attempt in 1..=max_attempts {
            if attempt > 1 {
                tokio::time::sleep(std::time::Duration::from_secs(POLL_INTERVAL_SECS)).await;
            }

            let poll_response = self
                .client
                .get(&poll_url)
                .bearer_auth(&self.api_key)
                .send()
                .await
                .map_err(|error| {
                    warn!(?error, attempt, "tavily research poll failed");
                    error
                })?
                .error_for_status()
                .map_err(|error| {
                    warn!(?error, attempt, "tavily research poll returned error status");
                    error
                })?
                .json::<ResearchPollResponse>()
                .await
                .map_err(|error| {
                    warn!(?error, attempt, "failed to deserialize tavily research poll");
                    error
                })?;

            debug!(
                attempt,
                status = %poll_response.status,
                "tavily research poll"
            );

            match poll_response.status.as_str() {
                "completed" => {
                    info!(attempt, "tavily research completed");
                    let content = poll_response
                        .content
                        .unwrap_or_else(|| "Research completed but no content returned.".to_owned());
                    let citations: Vec<String> = poll_response
                        .sources
                        .unwrap_or_default()
                        .into_iter()
                        .map(|source| source.url)
                        .collect();
                    return Ok(ToolResult {
                        text: content,
                        citations,
                    });
                }
                "failed" => {
                    let error_msg = poll_response
                        .content
                        .unwrap_or_else(|| "Research task failed.".to_owned());
                    warn!(attempt, error = %error_msg, "tavily research failed");
                    anyhow::bail!("Tavily research failed: {error_msg}");
                }
                _ => {
                    // Still processing — continue polling
                }
            }
        }

        anyhow::bail!(
            "Tavily research timed out after {}s (request_id: {request_id})",
            MAX_POLL_DURATION_SECS
        )
    }
}

#[derive(Debug, Serialize)]
struct ResearchRequest<'a> {
    api_key: &'a str,
    input: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ResearchStartResponse {
    request_id: String,
}

#[derive(Debug, Deserialize)]
struct ResearchPollResponse {
    status: String,
    content: Option<String>,
    sources: Option<Vec<ResearchSource>>,
}

#[derive(Debug, Deserialize)]
struct ResearchSource {
    url: String,
}
