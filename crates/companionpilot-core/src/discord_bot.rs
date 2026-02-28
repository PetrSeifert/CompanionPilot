use std::sync::Arc;

use chrono::Utc;
use serenity::{
    async_trait,
    model::{channel::Message, gateway::GatewayIntents},
    prelude::*,
};
use tracing::{error, info, warn};

use crate::{orchestrator::DefaultChatOrchestrator, types::MessageCtx};

struct Handler {
    orchestrator: Arc<DefaultChatOrchestrator>,
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
                if let Err(error) = msg.channel_id.say(&ctx.http, reply.text).await {
                    error!(?error, "failed to send Discord message");
                }
            }
            Err(error) => {
                error!(?error, "failed to process Discord message");
            }
        }
    }
}

pub async fn start_discord_bot(
    token: String,
    orchestrator: Arc<DefaultChatOrchestrator>,
) -> anyhow::Result<()> {
    let intents = GatewayIntents::GUILD_MESSAGES
        | GatewayIntents::DIRECT_MESSAGES
        | GatewayIntents::MESSAGE_CONTENT;

    let handler = Handler { orchestrator };
    let mut client = Client::builder(token, intents)
        .event_handler(handler)
        .await?;

    info!("starting Discord gateway client");
    client.start().await?;
    Ok(())
}
