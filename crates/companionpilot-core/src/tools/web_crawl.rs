use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::{info, warn};

use super::ToolResult;

#[derive(Debug, Clone)]
pub struct TavilyCrawlTool {
    client: Client,
    api_key: String,
}

impl TavilyCrawlTool {
    pub fn new(api_key: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
        }
    }

    pub async fn crawl(&self, args: Value) -> anyhow::Result<ToolResult> {
        let url = args
            .get("url")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("web_crawl requires string arg `url`"))?;
        if url.trim().is_empty() {
            anyhow::bail!("web_crawl requires a non-empty `url`");
        }

        info!(url = %url, "tavily crawl start");

        let mut payload = CrawlRequest {
            api_key: &self.api_key,
            url,
            max_depth: None,
            max_breadth: None,
            limit: None,
            instructions: None,
            chunks_per_source: None,
            extract_depth: None,
            select_paths: None,
            exclude_paths: None,
        };

        if let Some(depth) = args.get("max_depth").and_then(Value::as_u64) {
            payload.max_depth = Some(depth.clamp(1, 5) as usize);
        }
        if let Some(breadth) = args.get("max_breadth").and_then(Value::as_u64) {
            payload.max_breadth = Some(breadth.clamp(1, 500) as usize);
        }
        if let Some(limit) = args.get("limit").and_then(Value::as_u64) {
            payload.limit = Some(limit.clamp(1, 500) as usize);
        }
        if let Some(instructions) = args.get("instructions").and_then(Value::as_str) {
            if !instructions.trim().is_empty() {
                payload.instructions = Some(instructions.to_owned());
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

        let extract_paths = |key: &str| -> Option<Vec<String>> {
            let arr = args.get(key)?.as_array()?;
            let paths: Vec<String> = arr
                .iter()
                .filter_map(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .take(20)
                .map(ToOwned::to_owned)
                .collect();
            if paths.is_empty() { None } else { Some(paths) }
        };
        payload.select_paths = extract_paths("select_paths");
        payload.exclude_paths = extract_paths("exclude_paths");

        let response = self
            .client
            .post("https://api.tavily.com/crawl")
            .json(&payload)
            .send()
            .await
            .map_err(|error| {
                warn!(?error, "tavily crawl request failed");
                error
            })?
            .error_for_status()
            .map_err(|error| {
                warn!(?error, "tavily crawl returned error status");
                error
            })?
            .json::<CrawlResponse>()
            .await
            .map_err(|error| {
                warn!(?error, "failed to deserialize tavily crawl response");
                error
            })?;

        info!(
            base_url = %response.base_url,
            result_count = response.results.len(),
            "tavily crawl success"
        );

        const MAX_INLINE_PAGES: usize = 10;

        let mut citations = Vec::new();
        let total_pages = response.results.len();
        let mut lines = vec![format!(
            "Crawled {} pages from {}",
            total_pages, response.base_url
        )];

        for item in &response.results {
            citations.push(item.url.clone());
        }

        for item in response.results.iter().take(MAX_INLINE_PAGES) {
            let content = item.raw_content.as_deref().unwrap_or_default();
            let truncated = truncate_chars(content, 1000);
            lines.push(format!("URL: {}\n{truncated}", item.url));
        }

        if total_pages > MAX_INLINE_PAGES {
            let omitted = total_pages - MAX_INLINE_PAGES;
            lines.push(format!(
                "(+{omitted} more pages omitted from text; all {total_pages} URLs included in citations)"
            ));
        }

        if total_pages == 0 {
            lines.push("No pages crawled.".to_owned());
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
struct CrawlRequest<'a> {
    api_key: &'a str,
    url: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_depth: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_breadth: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    limit: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    instructions: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    chunks_per_source: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    extract_depth: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    select_paths: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    exclude_paths: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
struct CrawlResponse {
    #[serde(default)]
    base_url: String,
    #[serde(default)]
    results: Vec<CrawlResult>,
}

#[derive(Debug, Deserialize)]
struct CrawlResult {
    url: String,
    raw_content: Option<String>,
}
