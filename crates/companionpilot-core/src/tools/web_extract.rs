use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::{info, warn};

use super::ToolResult;

#[derive(Debug, Clone)]
pub struct TavilyExtractTool {
    client: Client,
    api_key: String,
}

impl TavilyExtractTool {
    pub fn new(api_key: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
        }
    }

    pub async fn extract(&self, args: Value) -> anyhow::Result<ToolResult> {
        let urls: Vec<String> = args
            .get("urls")
            .and_then(Value::as_array)
            .ok_or_else(|| anyhow::anyhow!("web_extract requires array arg `urls`"))?
            .iter()
            .filter_map(Value::as_str)
            .map(ToOwned::to_owned)
            .collect();
        if urls.is_empty() {
            anyhow::bail!("web_extract requires at least one URL in `urls`");
        }

        info!(url_count = urls.len(), "tavily extract start");

        let mut payload = ExtractRequest {
            api_key: &self.api_key,
            urls,
            query: None,
            chunks_per_source: None,
            extract_depth: None,
        };

        if let Some(query) = args.get("query").and_then(Value::as_str) {
            if !query.trim().is_empty() {
                payload.query = Some(query.to_owned());
            }
        }
        if let Some(chunks) = args.get("chunks_per_source").and_then(Value::as_u64) {
            payload.chunks_per_source = Some(chunks.clamp(1, 5) as usize);
        }
        if let Some(depth) = args.get("extract_depth").and_then(Value::as_str) {
            if matches!(depth, "basic" | "advanced") {
                payload.extract_depth = Some(depth.to_owned());
            }
        }

        let response = self
            .client
            .post("https://api.tavily.com/extract")
            .json(&payload)
            .send()
            .await
            .map_err(|error| {
                warn!(?error, "tavily extract request failed");
                error
            })?
            .error_for_status()
            .map_err(|error| {
                warn!(?error, "tavily extract returned error status");
                error
            })?
            .json::<ExtractResponse>()
            .await
            .map_err(|error| {
                warn!(?error, "failed to deserialize tavily extract response");
                error
            })?;

        info!(
            result_count = response.results.len(),
            failed_count = response.failed_results.len(),
            "tavily extract success"
        );

        // Extract exists to return full page content, so per-page limits are generous.
        // Aggregate cap prevents extreme output when many URLs are requested.
        const PER_PAGE_CHARS: usize = 16_000;
        const AGGREGATE_CHARS: usize = 80_000;

        let mut citations = Vec::new();
        let mut lines = Vec::new();
        let mut total_chars: usize = 0;

        for item in response.results {
            citations.push(item.url.clone());
            let content = item.raw_content.unwrap_or_default();
            let remaining = AGGREGATE_CHARS.saturating_sub(total_chars);
            if remaining == 0 {
                lines.push(format!("URL: {}\n(content omitted — aggregate limit reached)", item.url));
                continue;
            }
            let limit = PER_PAGE_CHARS.min(remaining);
            let truncated = truncate_chars(&content, limit);
            total_chars += truncated.len();
            lines.push(format!("URL: {}\n{truncated}", item.url));
        }

        if !response.failed_results.is_empty() {
            lines.push("Failed URLs:".to_owned());
            for fail in response.failed_results {
                let error = fail.error.unwrap_or_else(|| "unknown error".to_owned());
                lines.push(format!("- {} ({})", fail.url, error));
            }
        }

        if lines.is_empty() {
            lines.push("No content extracted.".to_owned());
        }

        Ok(ToolResult {
            text: lines.join("\n\n"),
            citations,
        })
    }
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    if text.len() <= max_chars {
        return text.to_owned();
    }
    let boundary = text
        .char_indices()
        .take_while(|(i, _)| *i < max_chars)
        .last()
        .map(|(i, ch)| i + ch.len_utf8())
        .unwrap_or(max_chars);
    format!("{}...", &text[..boundary])
}

#[derive(Debug, Serialize)]
struct ExtractRequest<'a> {
    api_key: &'a str,
    urls: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    query: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    chunks_per_source: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    extract_depth: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ExtractResponse {
    #[serde(default)]
    results: Vec<ExtractResult>,
    #[serde(default)]
    failed_results: Vec<ExtractFailedResult>,
}

#[derive(Debug, Deserialize)]
struct ExtractResult {
    url: String,
    raw_content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ExtractFailedResult {
    url: String,
    error: Option<String>,
}
