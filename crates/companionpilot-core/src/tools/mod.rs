mod current_datetime;
mod web_search;

use async_trait::async_trait;
use serde_json::Value;

pub use current_datetime::CurrentDateTimeTool;
pub use web_search::TavilyWebSearchTool;

#[derive(Debug, Clone)]
pub struct ToolResult {
    pub text: String,
    pub citations: Vec<String>,
}

#[async_trait]
pub trait ToolExecutor: Send + Sync {
    async fn execute(&self, tool_name: &str, args: Value) -> anyhow::Result<ToolResult>;
}

#[derive(Debug, Default)]
pub struct ToolRegistry {
    pub current_datetime: CurrentDateTimeTool,
    pub web_search: Option<TavilyWebSearchTool>,
}

#[async_trait]
impl ToolExecutor for ToolRegistry {
    async fn execute(&self, tool_name: &str, args: Value) -> anyhow::Result<ToolResult> {
        match tool_name {
            "current_datetime" => self.current_datetime.get_now(args).await,
            "web_search" => {
                let tool = self
                    .web_search
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("web_search tool is not configured"))?;
                tool.search(args).await
            }
            _ => Err(anyhow::anyhow!("unknown tool: {tool_name}")),
        }
    }
}
