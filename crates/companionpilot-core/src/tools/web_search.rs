use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::{debug, info, warn};

use super::ToolResult;

#[derive(Debug, Clone)]
pub struct TavilyWebSearchTool {
    client: Client,
    api_key: String,
}

impl TavilyWebSearchTool {
    pub fn new(api_key: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
        }
    }

    pub async fn search(&self, args: Value) -> anyhow::Result<ToolResult> {
        let query = args
            .get("query")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("web_search requires string arg `query`"))?;
        let max_results = args
            .get("max_results")
            .and_then(Value::as_u64)
            .unwrap_or(5)
            .clamp(1, 20);

        let search_depth = args
            .get("search_depth")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);
        let topic = args
            .get("topic")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);
        let time_range = args
            .get("time_range")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);

        let extract_domains = |key: &str| -> Option<Vec<String>> {
            let arr = args.get(key)?.as_array()?;
            let domains: Vec<String> = arr
                .iter()
                .filter_map(Value::as_str)
                .map(ToOwned::to_owned)
                .collect();
            if domains.is_empty() { None } else { Some(domains) }
        };
        let include_domains = extract_domains("include_domains");
        let exclude_domains = extract_domains("exclude_domains");

        info!(max_results, "tavily web search start");
        debug!(query = %query, "tavily query");

        let payload = TavilyRequest {
            api_key: &self.api_key,
            query,
            max_results: max_results as usize,
            include_answer: true,
            search_depth,
            topic,
            time_range,
            include_domains,
            exclude_domains,
        };

        let response = self
            .client
            .post("https://api.tavily.com/search")
            .json(&payload)
            .send()
            .await
            .map_err(|error| {
                warn!(?error, "tavily request failed");
                error
            })?
            .error_for_status()
            .map_err(|error| {
                warn!(?error, "tavily returned error status");
                error
            })?
            .json::<TavilyResponse>()
            .await
            .map_err(|error| {
                warn!(?error, "failed to deserialize tavily response");
                error
            })?;

        info!(
            result_count = response.results.len(),
            has_answer = response.answer.is_some(),
            "tavily web search success"
        );

        let mut citations = Vec::new();
        let mut lines = Vec::new();
        if let Some(answer) = response.answer {
            lines.push(format!("Summary: {answer}"));
        }

        for item in response.results {
            citations.push(item.url.clone());
            let mut line = format!("- {} ({})", item.title, item.url);
            if let Some(content) = item.content {
                let snippet = if content.len() > 300 {
                    let boundary = content
                        .char_indices()
                        .take_while(|(i, _)| *i < 300)
                        .last()
                        .map(|(i, ch)| i + ch.len_utf8())
                        .unwrap_or(300);
                    format!("{}...", &content[..boundary])
                } else {
                    content
                };
                line.push_str(&format!("\n  {snippet}"));
            }
            lines.push(line);
        }

        if lines.is_empty() {
            lines.push("No search results returned.".to_owned());
        }

        Ok(ToolResult {
            text: lines.join("\n"),
            citations,
        })
    }
}

#[derive(Debug, Serialize)]
struct TavilyRequest<'a> {
    api_key: &'a str,
    query: &'a str,
    max_results: usize,
    include_answer: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    search_depth: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    topic: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    time_range: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    include_domains: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    exclude_domains: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
struct TavilyResponse {
    answer: Option<String>,
    results: Vec<TavilyResult>,
}

#[derive(Debug, Deserialize)]
struct TavilyResult {
    title: String,
    url: String,
    content: Option<String>,
}
