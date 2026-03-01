use std::sync::Arc;

use chrono::Utc;
use serenity::{
    async_trait,
    model::{channel::Message, gateway::GatewayIntents, prelude::VoiceState},
    prelude::*,
};
use songbird::{SerenityInit, Songbird};
use tracing::{error, info, warn};

use crate::{orchestrator::DefaultChatOrchestrator, types::MessageCtx, voice::VoiceManager};

struct Handler {
    orchestrator: Arc<DefaultChatOrchestrator>,
    voice: Option<Arc<VoiceManager>>,
}

#[async_trait]
impl EventHandler for Handler {
    async fn message(&self, ctx: Context, msg: Message) {
        if msg.author.bot {
            return;
        }

        let guild_id = msg
            .guild_id
            .map(|id| id.to_string())
            .unwrap_or_else(|| "dm".to_owned());

        let request = MessageCtx {
            message_id: msg.id.to_string(),
            user_id: msg.author.id.to_string(),
            guild_id,
            channel_id: msg.channel_id.to_string(),
            content: msg.content.clone(),
            timestamp: Utc::now(),
        };

        match self.orchestrator.handle_message(request).await {
            Ok(reply) => {
                if reply.timings.total_ms >= 30_000 {
                    warn!(
                        user_id = %msg.author.id,
                        channel_id = %msg.channel_id,
                        message_id = %msg.id,
                        total_ms = reply.timings.total_ms,
                        planner_ms = reply.timings.planner_ms,
                        tool_execution_ms = reply.timings.tool_execution_ms,
                        final_model_ms = reply.timings.final_model_ms,
                        "slow Discord reply detected"
                    );
                } else {
                    info!(
                        user_id = %msg.author.id,
                        channel_id = %msg.channel_id,
                        message_id = %msg.id,
                        total_ms = reply.timings.total_ms,
                        planner_ms = reply.timings.planner_ms,
                        tool_execution_ms = reply.timings.tool_execution_ms,
                        final_model_ms = reply.timings.final_model_ms,
                        "Discord reply ready"
                    );
                }
                let suppress_text_reply = reply
                    .tool_calls
                    .iter()
                    .any(|call| call.tool_name == "discord_voice_listen_turn");

                if suppress_text_reply || reply.text.trim().is_empty() {
                    return;
                }

                if let Err(error) = msg.channel_id.say(&ctx.http, reply.text).await {
                    error!(?error, "failed to send Discord message");
                }
            }
            Err(error) => {
                error!(?error, "failed to process Discord message");
            }
        }
    }

    async fn voice_state_update(&self, _ctx: Context, old: Option<VoiceState>, new: VoiceState) {
        let Some(voice) = &self.voice else {
            return;
        };

        let guild_id = new
            .guild_id
            .or_else(|| old.as_ref().and_then(|state| state.guild_id));
        let Some(guild_id) = guild_id else {
            return;
        };

        let channel_id = new
            .channel_id
            .or_else(|| old.as_ref().and_then(|state| state.channel_id));
        voice
            .update_user_voice_state(
                guild_id.get(),
                new.user_id.get(),
                channel_id.map(|id| id.get()),
            )
            .await;
    }
}

pub async fn start_discord_bot(
    token: String,
    orchestrator: Arc<DefaultChatOrchestrator>,
    voice: Option<Arc<VoiceManager>>,
) -> anyhow::Result<()> {
    let intents = GatewayIntents::GUILD_MESSAGES
        | GatewayIntents::GUILDS
        | GatewayIntents::GUILD_VOICE_STATES
        | GatewayIntents::DIRECT_MESSAGES
        | GatewayIntents::MESSAGE_CONTENT;

    let handler = Handler {
        orchestrator,
        voice: voice.clone(),
    };

    let mut builder = Client::builder(token, intents).event_handler(handler);

    if let Some(voice_manager) = &voice {
        let songbird = Songbird::serenity_from_config(VoiceManager::songbird_config());
        voice_manager.set_songbird(songbird.clone()).await;
        builder = builder.register_songbird_with(songbird);
    }

    let mut client = builder.await?;

    info!("starting Discord gateway client");
    client.start().await?;
    Ok(())
}
