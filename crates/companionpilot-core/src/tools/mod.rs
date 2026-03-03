mod current_datetime;
mod spogo_cli;
mod spogo_cli_tool;
mod spogo_control;
mod spogo_search;
mod spogo_status;
mod web_search;

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use crate::{types::MessageCtx, voice::VoiceManager};

pub use current_datetime::CurrentDateTimeTool;
pub use spogo_cli::{DEFAULT_SPOGO_BIN_PATH, DEFAULT_SPOGO_TIMEOUT_MS, SpogoCli};
pub use spogo_cli_tool::CliTool;
pub(crate) use spogo_cli_tool::sanitize_cli_invocation_args;
pub use spogo_control::SpogoControlTool;
pub use spogo_search::SpogoSearchTool;
pub use spogo_status::SpogoStatusTool;
pub use web_search::TavilyWebSearchTool;

#[derive(Debug, Clone)]
pub struct ToolResult {
    pub text: String,
    pub citations: Vec<String>,
}

#[async_trait]
pub trait ToolExecutor: Send + Sync {
    async fn execute(
        &self,
        tool_name: &str,
        args: Value,
        message_ctx: &MessageCtx,
    ) -> anyhow::Result<ToolResult>;
}

#[derive(Debug, Default)]
pub struct ToolRegistry {
    pub current_datetime: CurrentDateTimeTool,
    pub cli: CliTool,
    pub spogo_control: SpogoControlTool,
    pub spogo_status: SpogoStatusTool,
    pub spogo_search: SpogoSearchTool,
    pub web_search: Option<TavilyWebSearchTool>,
    pub voice: Option<Arc<VoiceManager>>,
}

#[async_trait]
impl ToolExecutor for ToolRegistry {
    async fn execute(
        &self,
        tool_name: &str,
        args: Value,
        message_ctx: &MessageCtx,
    ) -> anyhow::Result<ToolResult> {
        match tool_name {
            "current_datetime" => self.current_datetime.get_now(args).await,
            "cli" => self.cli.execute_command(args).await,
            "spogo_control" => self.spogo_control.control(args).await,
            "spogo_status" => self.spogo_status.get_status(args).await,
            "spogo_search" => self.spogo_search.search(args).await,
            "web_search" => {
                let tool = self
                    .web_search
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("web_search tool is not configured"))?;
                tool.search(args).await
            }
            "discord_voice_join" => {
                let manager = self
                    .voice
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("voice tools are not configured"))?;
                let text = manager
                    .join_for_requester(&message_ctx.guild_id, &message_ctx.user_id, &args)
                    .await?;
                Ok(ToolResult {
                    text,
                    citations: Vec::new(),
                })
            }
            "discord_voice_listen_turn" => {
                let manager = self
                    .voice
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("voice tools are not configured"))?;
                let text = manager
                    .listen_and_respond_for_requester(
                        &message_ctx.guild_id,
                        &message_ctx.user_id,
                        &args,
                    )
                    .await?;
                Ok(ToolResult {
                    text,
                    citations: Vec::new(),
                })
            }
            "discord_voice_leave" => {
                let manager = self
                    .voice
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("voice tools are not configured"))?;
                let text = manager
                    .leave_for_requester(&message_ctx.guild_id, &message_ctx.user_id)
                    .await?;
                Ok(ToolResult {
                    text,
                    citations: Vec::new(),
                })
            }
            _ => Err(anyhow::anyhow!("unknown tool: {tool_name}")),
        }
    }
}
