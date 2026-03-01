mod current_datetime;
mod spotify_control_playback;
mod spotify_playing_status;
mod spotify_users_list;
mod web_search;

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use crate::{types::MessageCtx, voice::VoiceManager};

pub use current_datetime::CurrentDateTimeTool;
pub use spotify_control_playback::{DEFAULT_SPOTIFY_CONTROL_BASE_URL, SpotifyControlPlaybackTool};
pub use spotify_playing_status::SpotifyPlayingStatusTool;
pub use spotify_users_list::{DEFAULT_SPOTIFY_USERS_API_URL, SpotifyUsersListTool};
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
    pub spotify_control_playback: Option<SpotifyControlPlaybackTool>,
    pub spotify_playing_status: SpotifyPlayingStatusTool,
    pub spotify_users_list: SpotifyUsersListTool,
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
            "spotify_control_playback" => {
                let tool = self.spotify_control_playback.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("spotify_control_playback tool is not configured")
                })?;
                tool.control_playback(args).await
            }
            "spotify_playing_status" => self.spotify_playing_status.get_playing_status(args).await,
            "spotify_users_list" => self.spotify_users_list.list_users(args).await,
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
