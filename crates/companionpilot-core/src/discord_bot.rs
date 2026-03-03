use std::{collections::HashSet, sync::Arc};

use chrono::Utc;
use serenity::{
    async_trait,
    model::{
        channel::Message,
        gateway::{GatewayIntents, Ready},
        id::GuildId,
        prelude::VoiceState,
    },
    prelude::*,
};
use songbird::{SerenityInit, Songbird};
use tracing::{error, info, warn};

use crate::{
    orchestrator::DefaultChatOrchestrator, runtime_settings::RuntimeSettingsManager,
    types::MessageCtx, voice::VoiceManager,
};

const DISCORD_MESSAGE_MAX_CHARS: usize = 2_000;

struct Handler {
    orchestrator: Arc<DefaultChatOrchestrator>,
    voice: Option<Arc<VoiceManager>>,
    runtime_settings: Arc<RuntimeSettingsManager>,
}

impl Handler {
    async fn seed_voice_states_from_cache(&self, ctx: &Context, guilds: &[GuildId]) {
        let Some(voice) = &self.voice else {
            return;
        };

        let mut cached_states = Vec::new();
        for guild_id in guilds {
            let Some(guild) = ctx.cache.guild(*guild_id) else {
                continue;
            };

            for (user_id, state) in &guild.voice_states {
                cached_states.push((
                    guild_id.get(),
                    user_id.get(),
                    state.channel_id.map(|id| id.get()),
                ));
            }
        }

        for (guild_id, user_id, channel_id) in cached_states {
            voice
                .update_user_voice_state(guild_id, user_id, channel_id)
                .await;
        }
    }
}

#[async_trait]
impl EventHandler for Handler {
    async fn ready(&self, _ctx: Context, ready: Ready) {
        info!(user = %ready.user.name, "Discord gateway ready");
    }

    async fn cache_ready(&self, ctx: Context, guilds: Vec<GuildId>) {
        self.seed_voice_states_from_cache(&ctx, &guilds).await;
        info!(
            guild_count = guilds.len(),
            "seeded cached voice states during startup"
        );
    }

    async fn message(&self, ctx: Context, msg: Message) {
        if msg.author.bot {
            return;
        }
        let bot_user_id = ctx.cache.current_user().id;
        let is_direct_message = msg.guild_id.is_none();
        let mentions_bot = msg.mentions.iter().any(|user| user.id == bot_user_id);
        let runtime_settings = self.runtime_settings.get().await;
        let allowed_channel_ids =
            parse_allowed_channel_ids(&runtime_settings.discord_allowed_channel_ids);
        let allowlist_enabled = !allowed_channel_ids.is_empty();
        let channel_allowed =
            !allowlist_enabled || allowed_channel_ids.contains(&msg.channel_id.get());
        if !should_process_message(
            is_direct_message,
            mentions_bot,
            channel_allowed,
            allowlist_enabled,
        ) {
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
                if reply.text.trim().is_empty() {
                    return;
                }

                for (chunk_index, chunk) in
                    split_message_for_discord(&reply.text, DISCORD_MESSAGE_MAX_CHARS)
                        .into_iter()
                        .enumerate()
                {
                    if let Err(error) = msg.channel_id.say(&ctx.http, chunk).await {
                        error!(?error, chunk_index, "failed to send Discord message chunk");
                        break;
                    }
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

        let channel_id = new.channel_id;
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
    runtime_settings: Arc<RuntimeSettingsManager>,
) -> anyhow::Result<()> {
    let intents = GatewayIntents::GUILD_MESSAGES
        | GatewayIntents::GUILDS
        | GatewayIntents::GUILD_VOICE_STATES
        | GatewayIntents::DIRECT_MESSAGES
        | GatewayIntents::MESSAGE_CONTENT;

    let handler = Handler {
        orchestrator,
        voice: voice.clone(),
        runtime_settings,
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

pub fn parse_allowed_channel_ids(raw: &str) -> HashSet<u64> {
    raw.split([',', ';', ' ', '\n', '\t'])
        .filter_map(|value| {
            let trimmed = value.trim();
            if trimmed.is_empty() {
                return None;
            }
            trimmed.parse::<u64>().ok()
        })
        .collect()
}

fn should_process_message(
    is_direct_message: bool,
    mentions_bot: bool,
    channel_allowed: bool,
    allowlist_enabled: bool,
) -> bool {
    if is_direct_message {
        return true;
    }

    if !channel_allowed {
        return false;
    }

    mentions_bot || allowlist_enabled
}

fn split_message_for_discord(text: &str, max_chars: usize) -> Vec<&str> {
    if text.is_empty() || max_chars == 0 {
        return Vec::new();
    }

    let mut chunks = Vec::new();
    let mut start = 0usize;
    let mut chars_in_chunk = 0usize;

    for (byte_index, _) in text.char_indices() {
        if chars_in_chunk == max_chars {
            chunks.push(&text[start..byte_index]);
            start = byte_index;
            chars_in_chunk = 0;
        }
        chars_in_chunk += 1;
    }

    if start < text.len() {
        chunks.push(&text[start..]);
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::{
        DISCORD_MESSAGE_MAX_CHARS, parse_allowed_channel_ids, should_process_message,
        split_message_for_discord,
    };

    #[test]
    fn split_message_for_discord_enforces_limit() {
        let text = "a".repeat(DISCORD_MESSAGE_MAX_CHARS + 25);
        let chunks = split_message_for_discord(&text, DISCORD_MESSAGE_MAX_CHARS);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].chars().count(), DISCORD_MESSAGE_MAX_CHARS);
        assert_eq!(chunks[1].chars().count(), 25);
    }

    #[test]
    fn split_message_for_discord_handles_multibyte_text() {
        let text = "😄".repeat(DISCORD_MESSAGE_MAX_CHARS + 1);
        let chunks = split_message_for_discord(&text, DISCORD_MESSAGE_MAX_CHARS);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].chars().count(), DISCORD_MESSAGE_MAX_CHARS);
        assert_eq!(chunks[1].chars().count(), 1);
    }

    #[test]
    fn parse_allowed_channel_ids_parses_delimited_values() {
        let ids = parse_allowed_channel_ids("123, 456;789\n101112");
        assert_eq!(ids.len(), 4);
        assert!(ids.contains(&123));
        assert!(ids.contains(&456));
        assert!(ids.contains(&789));
        assert!(ids.contains(&101112));
    }

    #[test]
    fn parse_allowed_channel_ids_ignores_invalid_entries() {
        let ids = parse_allowed_channel_ids("123,abc,, -5");
        assert_eq!(ids.len(), 1);
        assert!(ids.contains(&123));
    }

    #[test]
    fn should_process_message_allows_direct_messages() {
        assert!(should_process_message(true, false, false, false));
    }

    #[test]
    fn should_process_message_requires_mention_without_allowlist() {
        assert!(!should_process_message(false, false, true, false));
        assert!(should_process_message(false, true, true, false));
    }

    #[test]
    fn should_process_message_allows_allowlisted_channels_without_mention() {
        assert!(should_process_message(false, false, true, true));
        assert!(!should_process_message(false, false, false, true));
    }
}
