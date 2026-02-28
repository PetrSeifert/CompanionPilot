use chrono::Utc;
use serde_json::Value;

use super::ToolResult;

#[derive(Debug, Clone, Default)]
pub struct CurrentDateTimeTool;

impl CurrentDateTimeTool {
    pub async fn get_now(&self, _args: Value) -> anyhow::Result<ToolResult> {
        let now = Utc::now();
        let text = format!(
            "Current UTC datetime: {}\nCurrent UTC date: {}\nCurrent UTC year: {}",
            now.to_rfc3339(),
            now.format("%Y-%m-%d"),
            now.format("%Y")
        );

        Ok(ToolResult {
            text,
            citations: Vec::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::CurrentDateTimeTool;

    #[tokio::test]
    async fn returns_utc_datetime_fields() {
        let tool = CurrentDateTimeTool;
        let result = tool
            .get_now(json!({}))
            .await
            .expect("current_datetime should succeed");

        assert!(result.text.contains("Current UTC datetime:"));
        assert!(result.text.contains("Current UTC date:"));
        assert!(result.text.contains("Current UTC year:"));
        assert!(result.citations.is_empty());
    }
}
