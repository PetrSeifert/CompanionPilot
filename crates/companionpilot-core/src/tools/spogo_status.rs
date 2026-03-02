use anyhow::Context;
use serde_json::Value;

use super::{SpogoCli, ToolResult};

#[derive(Debug, Clone)]
pub struct SpogoStatusTool {
    cli: SpogoCli,
    account_label: Option<String>,
}

impl Default for SpogoStatusTool {
    fn default() -> Self {
        Self::new(SpogoCli::default(), None)
    }
}

impl SpogoStatusTool {
    pub fn new(cli: SpogoCli, account_label: Option<String>) -> Self {
        Self { cli, account_label }
    }

    pub async fn get_status(&self, _args: Value) -> anyhow::Result<ToolResult> {
        let output = self
            .cli
            .execute(vec!["--json".to_owned(), "status".to_owned()])
            .await?;
        let payload: Value = serde_json::from_str(output.stdout.trim())
            .context("spogo_status returned invalid JSON")?;
        let pretty_payload =
            serde_json::to_string_pretty(&payload).context("failed to format spogo_status JSON")?;

        let mut lines = Vec::new();
        if let Some(label) = self
            .account_label
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            lines.push(format!("Spotify account: {label}"));
        }
        lines.push("Spogo status response:".to_owned());
        lines.push(pretty_payload);

        Ok(ToolResult {
            text: lines.join("\n"),
            citations: vec![output.command_line],
        })
    }
}
