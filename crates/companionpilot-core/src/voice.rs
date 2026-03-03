use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::Arc,
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

use anyhow::Context;
use async_trait::async_trait;
use chrono::Utc;
use reqwest::{
    Client,
    multipart::{Form, Part},
};
use serde::Deserialize;
use serde_json::Value;
use serenity::all::{ChannelId, GuildId};
use songbird::{
    Config as SongbirdConfig, Songbird,
    driver::DecodeMode,
    events::{CoreEvent, Event, EventContext, EventHandler as VoiceEventHandler, TrackEvent},
    tracks::PlayMode,
};
use tokio::sync::{Mutex, Notify, RwLock};
use tracing::{info, warn};

use crate::{
    memory::MemoryStore,
    runtime_settings::RuntimeSettingsManager,
    types::{MessageCtx, MessageLatencyRecord},
};

const DEFAULT_LISTEN_WINDOW_MS: u64 = 12_000;
const DEFAULT_CHUNK_GAP_MS: u64 = 700;
const MIN_LISTEN_WINDOW_MS: u64 = 1_000;
const MAX_LISTEN_WINDOW_MS: u64 = 60_000;
const MIN_CHUNK_GAP_MS: u64 = 100;
const MAX_CHUNK_GAP_MS: u64 = 3_000;
const MAX_TTS_INPUT_CHARS: usize = 4_000;

#[derive(Debug, Clone)]
pub struct VoiceRuntimeConfig {
    pub openai_api_key: String,
    pub stt_model: String,
    pub tts_model: String,
    pub tts_voice: String,
    pub allowlist: HashSet<(u64, u64)>,
    pub idle_timeout: Duration,
    pub default_chunk_gap: Duration,
    pub default_listen_window: Duration,
    pub default_max_turn: Duration,
}

impl VoiceRuntimeConfig {
    pub fn parse_allowlist(raw: &str) -> HashSet<(u64, u64)> {
        let mut entries = HashSet::new();
        for pair in raw.split(',') {
            let trimmed = pair.trim();
            if trimmed.is_empty() {
                continue;
            }

            let mut parts = trimmed.split(':');
            let Some(guild_raw) = parts.next() else {
                continue;
            };
            let Some(channel_raw) = parts.next() else {
                continue;
            };
            if parts.next().is_some() {
                continue;
            }

            let Ok(guild_id) = guild_raw.trim().parse::<u64>() else {
                continue;
            };
            let Ok(channel_id) = channel_raw.trim().parse::<u64>() else {
                continue;
            };

            entries.insert((guild_id, channel_id));
        }
        entries
    }
}

#[derive(Debug, Clone)]
struct AudioChunk {
    speaker_label: String,
    pcm_samples: Vec<i16>,
}

#[derive(Debug)]
struct CapturedTurn {
    speakers: Vec<String>,
    pcm_samples: Vec<i16>,
}

#[derive(Debug)]
struct VoiceSession {
    channel_id: u64,
    requester_user_id: u64,
    chunk_queue: Mutex<VecDeque<AudioChunk>>,
    queue_notify: Notify,
    listen_lock: Mutex<()>,
    last_activity: Mutex<Instant>,
    closed: AtomicBool,
    tts_playing: AtomicBool,
}

impl VoiceSession {
    fn new(channel_id: u64, requester_user_id: u64) -> Self {
        Self {
            channel_id,
            requester_user_id,
            chunk_queue: Mutex::new(VecDeque::new()),
            queue_notify: Notify::new(),
            listen_lock: Mutex::new(()),
            last_activity: Mutex::new(Instant::now()),
            closed: AtomicBool::new(false),
            tts_playing: AtomicBool::new(false),
        }
    }

    async fn touch(&self) {
        *self.last_activity.lock().await = Instant::now();
    }

    async fn elapsed_since_last_activity(&self) -> Duration {
        self.last_activity.lock().await.elapsed()
    }

    fn close(&self) {
        self.closed.store(true, Ordering::Relaxed);
        self.queue_notify.notify_waiters();
    }

    fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Relaxed)
    }

    fn set_tts_playing(&self, value: bool) {
        self.tts_playing.store(value, Ordering::Relaxed);
    }

    async fn clear_chunks(&self) {
        self.chunk_queue.lock().await.clear();
    }

    async fn push_chunk(&self, chunk: AudioChunk) {
        if self.is_closed() || self.tts_playing.load(Ordering::Relaxed) {
            return;
        }
        self.chunk_queue.lock().await.push_back(chunk);
        self.touch().await;
        self.queue_notify.notify_waiters();
    }

    async fn next_chunk(&self) -> anyhow::Result<AudioChunk> {
        loop {
            if self.is_closed() {
                anyhow::bail!("voice session closed");
            }
            if let Some(chunk) = self.chunk_queue.lock().await.pop_front() {
                return Ok(chunk);
            }
            self.queue_notify.notified().await;
        }
    }

    async fn capture_turn(
        &self,
        listen_window: Duration,
        chunk_gap: Duration,
        max_turn: Duration,
    ) -> anyhow::Result<CapturedTurn> {
        let first_chunk = tokio::time::timeout(listen_window, self.next_chunk())
            .await
            .context("timed out waiting for next speaking event")??;

        let turn_started_at = Instant::now();
        let mut speakers = HashSet::new();
        let mut pcm_samples = Vec::new();

        speakers.insert(first_chunk.speaker_label);
        pcm_samples.extend(first_chunk.pcm_samples);

        loop {
            let elapsed = turn_started_at.elapsed();
            if elapsed >= max_turn {
                break;
            }

            let max_wait = (max_turn - elapsed).min(chunk_gap);
            let next_result = tokio::time::timeout(max_wait, self.next_chunk()).await;
            let Ok(next_chunk) = next_result else {
                break;
            };
            let next_chunk = next_chunk?;

            speakers.insert(next_chunk.speaker_label);
            pcm_samples.extend(next_chunk.pcm_samples);
        }

        if pcm_samples.is_empty() {
            anyhow::bail!("captured speaking turn had no PCM audio");
        }

        let mut speaker_labels = speakers.into_iter().collect::<Vec<_>>();
        speaker_labels.sort();
        Ok(CapturedTurn {
            speakers: speaker_labels,
            pcm_samples,
        })
    }
}

#[derive(Clone)]
struct VoiceReceiveHandler {
    session: Arc<VoiceSession>,
}

#[async_trait]
impl VoiceEventHandler for VoiceReceiveHandler {
    async fn act(&self, ctx: &EventContext<'_>) -> Option<Event> {
        if let EventContext::VoiceTick(tick) = ctx {
            for (ssrc, voice_data) in &tick.speaking {
                let Some(decoded) = &voice_data.decoded_voice else {
                    continue;
                };
                if decoded.is_empty() {
                    continue;
                }

                self.session
                    .push_chunk(AudioChunk {
                        speaker_label: format!("ssrc:{ssrc}"),
                        pcm_samples: decoded.clone(),
                    })
                    .await;
            }
        }

        None
    }
}

#[derive(Clone)]
struct PlaybackFinishedHandler {
    session: Arc<VoiceSession>,
}

#[async_trait]
impl VoiceEventHandler for PlaybackFinishedHandler {
    async fn act(&self, _ctx: &EventContext<'_>) -> Option<Event> {
        self.session.set_tts_playing(false);
        None
    }
}

#[async_trait]
pub trait VoiceReplyOrchestrator: Send + Sync {
    async fn handle_voice_transcript(&self, message: MessageCtx) -> anyhow::Result<String>;
}

pub struct VoiceManager {
    config: VoiceRuntimeConfig,
    runtime_settings: Arc<RuntimeSettingsManager>,
    memory: Arc<dyn MemoryStore>,
    sessions: RwLock<HashMap<u64, Arc<VoiceSession>>>,
    user_voice_channels: RwLock<HashMap<(u64, u64), u64>>,
    songbird: RwLock<Option<Arc<Songbird>>>,
    orchestrator: RwLock<Option<Arc<dyn VoiceReplyOrchestrator>>>,
    openai: OpenAiAudioClient,
}

impl std::fmt::Debug for VoiceManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VoiceManager")
            .field("allowlist_len", &self.config.allowlist.len())
            .field("idle_timeout", &self.config.idle_timeout)
            .finish_non_exhaustive()
    }
}

impl VoiceManager {
    pub fn new(
        config: VoiceRuntimeConfig,
        runtime_settings: Arc<RuntimeSettingsManager>,
        memory: Arc<dyn MemoryStore>,
    ) -> Arc<Self> {
        Arc::new(Self {
            openai: OpenAiAudioClient::new(config.openai_api_key.clone()),
            config,
            runtime_settings,
            memory,
            sessions: RwLock::new(HashMap::new()),
            user_voice_channels: RwLock::new(HashMap::new()),
            songbird: RwLock::new(None),
            orchestrator: RwLock::new(None),
        })
    }

    pub fn songbird_config() -> SongbirdConfig {
        SongbirdConfig::default().decode_mode(DecodeMode::Decode)
    }

    pub async fn set_songbird(&self, manager: Arc<Songbird>) {
        *self.songbird.write().await = Some(manager);
    }

    pub async fn set_orchestrator(&self, orchestrator: Arc<dyn VoiceReplyOrchestrator>) {
        *self.orchestrator.write().await = Some(orchestrator);
    }

    pub fn start_idle_reaper(self: &Arc<Self>) {
        let manager = Arc::clone(self);
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(30)).await;
                if let Err(error) = manager.cleanup_idle_sessions().await {
                    warn!(?error, "voice idle cleanup failed");
                }
            }
        });
    }

    pub async fn update_user_voice_state(
        &self,
        guild_id: u64,
        user_id: u64,
        channel_id: Option<u64>,
    ) {
        let mut states = self.user_voice_channels.write().await;
        match channel_id {
            Some(channel_id) => {
                states.insert((guild_id, user_id), channel_id);
            }
            None => {
                states.remove(&(guild_id, user_id));
            }
        }
    }

    pub async fn join_for_requester(
        self: &Arc<Self>,
        guild_id_raw: &str,
        requester_user_id_raw: &str,
        args: &Value,
    ) -> anyhow::Result<String> {
        let guild_id = parse_discord_id(guild_id_raw, "guild_id")?;
        let requester_user_id = parse_discord_id(requester_user_id_raw, "requester_user_id")?;

        let channel_id =
            if let Some(raw_channel_id) = args.get("channel_id").and_then(Value::as_str) {
                parse_discord_id(raw_channel_id, "channel_id")?
            } else {
                self.user_voice_channels
                    .read()
                    .await
                    .get(&(guild_id, requester_user_id))
                    .copied()
                    .context("requesting user is not currently in a voice channel")?
            };

        self.ensure_allowlisted(guild_id, channel_id).await?;

        let songbird = self.songbird().await?;
        let guild_id_key = GuildId::new(guild_id);
        let channel_id_key = ChannelId::new(channel_id);

        let call_lock = songbird
            .join(guild_id_key, channel_id_key)
            .await
            .with_context(|| {
                format!("failed to join voice channel {channel_id} in guild {guild_id}")
            })?;

        let session = Arc::new(VoiceSession::new(channel_id, requester_user_id));
        {
            let mut call = call_lock.lock().await;
            call.remove_all_global_events();
            call.add_global_event(
                Event::Core(CoreEvent::VoiceTick),
                VoiceReceiveHandler {
                    session: Arc::clone(&session),
                },
            );
        }

        if let Some(previous) = self
            .sessions
            .write()
            .await
            .insert(guild_id, Arc::clone(&session))
        {
            previous.close();
        }
        session.touch().await;
        self.spawn_auto_listen_loop(guild_id, Arc::clone(&session));

        info!(guild_id, channel_id, "voice join succeeded");
        Ok(format!(
            "Joined voice channel {channel_id}. Natural voice listening is active."
        ))
    }

    pub async fn leave_for_requester(
        &self,
        guild_id_raw: &str,
        requester_user_id_raw: &str,
    ) -> anyhow::Result<String> {
        let guild_id = parse_discord_id(guild_id_raw, "guild_id")?;
        let requester_user_id = parse_discord_id(requester_user_id_raw, "requester_user_id")?;

        let session = self
            .sessions
            .read()
            .await
            .get(&guild_id)
            .cloned()
            .context("no active voice session for this guild")?;
        self.ensure_requester_in_channel(guild_id, requester_user_id, session.channel_id)
            .await?;

        let songbird = self.songbird().await?;
        songbird
            .remove(GuildId::new(guild_id))
            .await
            .with_context(|| format!("failed to leave voice session in guild {guild_id}"))?;

        session.close();
        self.sessions.write().await.remove(&guild_id);
        info!(guild_id, "voice session removed");
        Ok("Left the voice channel.".to_owned())
    }

    pub async fn listen_and_respond_for_requester(
        &self,
        guild_id_raw: &str,
        requester_user_id_raw: &str,
        args: &Value,
    ) -> anyhow::Result<String> {
        let guild_id = parse_discord_id(guild_id_raw, "guild_id")?;
        let requester_user_id = parse_discord_id(requester_user_id_raw, "requester_user_id")?;
        let session = self
            .sessions
            .read()
            .await
            .get(&guild_id)
            .cloned()
            .context("bot is not connected to voice in this guild")?;

        self.ensure_requester_in_channel(guild_id, requester_user_id, session.channel_id)
            .await?;
        self.ensure_allowlisted(guild_id, session.channel_id)
            .await?;

        let runtime_settings = self.runtime_settings.get().await;
        let listen_window_ms = args
            .get("listen_window_ms")
            .and_then(Value::as_u64)
            .unwrap_or(runtime_settings.voice_listen_window_ms)
            .clamp(MIN_LISTEN_WINDOW_MS, MAX_LISTEN_WINDOW_MS);
        let chunk_gap_ms = args
            .get("chunk_gap_ms")
            .and_then(Value::as_u64)
            .unwrap_or(runtime_settings.voice_chunk_gap_ms)
            .clamp(MIN_CHUNK_GAP_MS, MAX_CHUNK_GAP_MS);
        let max_turn_ms = args
            .get("max_turn_ms")
            .and_then(Value::as_u64)
            .unwrap_or(runtime_settings.voice_max_turn_ms)
            .max(chunk_gap_ms);

        let listen_window = Duration::from_millis(listen_window_ms);
        let chunk_gap = Duration::from_millis(chunk_gap_ms);
        let max_turn = Duration::from_millis(max_turn_ms);

        let transcript = self
            .process_voice_turn(
                guild_id,
                Arc::clone(&session),
                listen_window,
                chunk_gap,
                max_turn,
            )
            .await?;

        let truncated_transcript = truncate_for_tool_result(&transcript, 220);
        Ok(format!(
            "Processed voice turn and replied in voice. Transcript: {truncated_transcript}"
        ))
    }

    async fn process_voice_turn(
        &self,
        guild_id: u64,
        session: Arc<VoiceSession>,
        listen_window: Duration,
        chunk_gap: Duration,
        max_turn: Duration,
    ) -> anyhow::Result<String> {
        let captured_turn = {
            let _listen_guard = session.listen_lock.lock().await;
            session.clear_chunks().await;
            session
                .capture_turn(listen_window, chunk_gap, max_turn)
                .await
                .context("failed to capture a voice turn")?
        };
        session.touch().await;

        let turn_started_at = Instant::now();
        let stt_started_at = Instant::now();
        let runtime_settings = self.runtime_settings.get().await;
        let wav_payload = pcm_i16_to_wav_bytes(&captured_turn.pcm_samples, 2, 48_000);
        let transcript = self
            .openai
            .transcribe_wav(&runtime_settings.openai_stt_model, &wav_payload)
            .await
            .context("STT transcription failed")?;
        let stt_ms = elapsed_ms(stt_started_at);
        let transcript = transcript.trim();
        if transcript.is_empty() {
            anyhow::bail!("transcription returned empty text");
        }

        let speaker_prefix = if captured_turn.speakers.is_empty() {
            String::new()
        } else {
            format!("[speakers:{}] ", captured_turn.speakers.join(","))
        };
        let transcript_for_orchestrator = format!("{speaker_prefix}{transcript}");

        let memory_user_id = session.requester_user_id.to_string();
        let orchestrator = self
            .orchestrator
            .read()
            .await
            .clone()
            .context("voice orchestrator is not configured")?;
        let message_ctx = MessageCtx {
            message_id: format!("voice-turn-{}", Utc::now().timestamp_millis()),
            user_id: memory_user_id,
            guild_id: guild_id.to_string(),
            channel_id: session.channel_id.to_string(),
            content: transcript_for_orchestrator,
            timestamp: Utc::now(),
        };
        let reply_text = orchestrator
            .handle_voice_transcript(message_ctx.clone())
            .await
            .context("failed to generate assistant reply for voice turn")?;

        let tts_started_at = Instant::now();
        let reply_for_tts = clamp_tts_input(&reply_text);
        let tts_audio = self
            .openai
            .synthesize_mp3(
                &runtime_settings.openai_tts_model,
                &runtime_settings.openai_tts_voice,
                &reply_for_tts,
            )
            .await
            .context("TTS synthesis failed")?;
        self.play_tts_audio(guild_id, Arc::clone(&session), tts_audio)
            .await?;
        let tts_ms = elapsed_ms(tts_started_at);
        let final_response_ms = elapsed_ms(turn_started_at);
        if let Err(error) = self
            .memory
            .record_message_latency(MessageLatencyRecord {
                user_id: message_ctx.user_id.clone(),
                guild_id: message_ctx.guild_id.clone(),
                channel_id: message_ctx.channel_id.clone(),
                message_id: message_ctx.message_id.clone(),
                stt_ms: Some(stt_ms),
                tts_ms: Some(tts_ms),
                final_response_ms,
                decision_ms: 0,
                tool_call_ms: 0,
                timestamp: Utc::now(),
            })
            .await
        {
            warn!(?error, "failed to persist voice latency metrics");
        }
        session.touch().await;

        Ok(transcript.to_owned())
    }

    fn spawn_auto_listen_loop(self: &Arc<Self>, guild_id: u64, session: Arc<VoiceSession>) {
        let manager = Arc::clone(self);
        tokio::spawn(async move {
            info!(guild_id, "automatic voice listening loop started");
            loop {
                if session.is_closed() {
                    break;
                }
                let runtime_settings = manager.runtime_settings.get().await;

                let result = manager
                    .process_voice_turn(
                        guild_id,
                        Arc::clone(&session),
                        Duration::from_millis(runtime_settings.voice_listen_window_ms),
                        Duration::from_millis(runtime_settings.voice_chunk_gap_ms),
                        Duration::from_millis(runtime_settings.voice_max_turn_ms),
                    )
                    .await;

                match result {
                    Ok(transcript) => {
                        info!(
                            guild_id,
                            transcript = %truncate_for_tool_result(&transcript, 180),
                            "automatic voice turn completed"
                        );
                    }
                    Err(error) => {
                        let message = error.to_string();
                        if message.contains("timed out waiting for next speaking event")
                            || message.contains("voice session closed")
                        {
                            continue;
                        }
                        warn!(guild_id, ?error, "automatic voice turn failed");
                        tokio::time::sleep(Duration::from_millis(400)).await;
                    }
                }
            }
            info!(guild_id, "automatic voice listening loop stopped");
        });
    }

    async fn play_tts_audio(
        &self,
        guild_id: u64,
        session: Arc<VoiceSession>,
        audio_bytes: Vec<u8>,
    ) -> anyhow::Result<()> {
        if audio_bytes.len() < 64 {
            anyhow::bail!(
                "TTS response was unexpectedly small ({} bytes)",
                audio_bytes.len()
            );
        }

        let audio_magic = audio_magic(&audio_bytes);
        let songbird = self.songbird().await?;
        let handler_lock = songbird
            .get(GuildId::new(guild_id))
            .context("bot is no longer connected to voice")?;
        let mut handler = handler_lock.lock().await;
        session.set_tts_playing(true);
        let track = handler.play_input(audio_bytes.into());
        track
            .add_event(
                Event::Track(TrackEvent::End),
                PlaybackFinishedHandler {
                    session: Arc::clone(&session),
                },
            )
            .context("failed to attach tts track end event")?;
        track
            .add_event(
                Event::Track(TrackEvent::Error),
                PlaybackFinishedHandler {
                    session: Arc::clone(&session),
                },
            )
            .context("failed to attach tts track error event")?;

        // Give Songbird a short window to parse/decode and report any immediate track error.
        tokio::time::sleep(Duration::from_millis(250)).await;
        let state = track
            .get_info()
            .await
            .context("unable to query Songbird track state after enqueue")?;
        match state.playing {
            PlayMode::Errored(error) => {
                session.set_tts_playing(false);
                anyhow::bail!("Songbird playback error: {error:?} (audio_magic={audio_magic})")
            }
            PlayMode::End => {
                session.set_tts_playing(false);
                warn!(
                    guild_id,
                    audio_magic = %audio_magic,
                    "tts track ended very quickly; check audio format and Discord permissions"
                );
            }
            _ => {}
        }

        info!(
            guild_id,
            audio_magic = %audio_magic,
            "tts track enqueued for voice playback"
        );
        Ok(())
    }

    async fn cleanup_idle_sessions(&self) -> anyhow::Result<()> {
        let runtime_settings = self.runtime_settings.get().await;
        let idle_timeout = Duration::from_secs(runtime_settings.voice_idle_timeout_sec);
        if idle_timeout.is_zero() {
            return Ok(());
        }

        let mut stale_guilds = Vec::new();
        {
            let sessions = self.sessions.read().await;
            for (guild_id, session) in sessions.iter() {
                if session.elapsed_since_last_activity().await >= idle_timeout {
                    stale_guilds.push(*guild_id);
                }
            }
        }

        if stale_guilds.is_empty() {
            return Ok(());
        }

        let songbird = self.songbird().await?;
        for guild_id in &stale_guilds {
            if let Err(error) = songbird.remove(GuildId::new(*guild_id)).await {
                warn!(
                    guild_id = *guild_id,
                    ?error,
                    "failed leaving idle voice session"
                );
            }
        }

        let mut sessions = self.sessions.write().await;
        for guild_id in stale_guilds {
            if let Some(session) = sessions.remove(&guild_id) {
                session.close();
            }
            info!(guild_id, "idle voice session removed");
        }

        Ok(())
    }

    async fn songbird(&self) -> anyhow::Result<Arc<Songbird>> {
        self.songbird
            .read()
            .await
            .clone()
            .context("Songbird voice manager is not registered")
    }

    async fn ensure_requester_in_channel(
        &self,
        guild_id: u64,
        requester_user_id: u64,
        expected_channel_id: u64,
    ) -> anyhow::Result<()> {
        let current_channel = self
            .user_voice_channels
            .read()
            .await
            .get(&(guild_id, requester_user_id))
            .copied()
            .context("requesting user is not currently in voice")?;
        if current_channel != expected_channel_id {
            anyhow::bail!("requesting user must be in the same voice channel as the bot");
        }
        Ok(())
    }

    async fn ensure_allowlisted(&self, guild_id: u64, channel_id: u64) -> anyhow::Result<()> {
        let runtime_settings = self.runtime_settings.get().await;
        let allowlist = VoiceRuntimeConfig::parse_allowlist(&runtime_settings.voice_allowlist);
        if allowlist.is_empty() {
            anyhow::bail!("voice allowlist is empty; no channel is currently permitted");
        }
        if !allowlist.contains(&(guild_id, channel_id)) {
            anyhow::bail!("voice channel is not in configured allowlist");
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct OpenAiAudioClient {
    client: Client,
    api_key: String,
}

impl OpenAiAudioClient {
    fn new(api_key: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
        }
    }

    async fn transcribe_wav(&self, stt_model: &str, wav_audio: &[u8]) -> anyhow::Result<String> {
        #[derive(Debug, Deserialize)]
        struct TranscriptionResponse {
            text: String,
        }

        let audio_part = Part::bytes(wav_audio.to_vec())
            .file_name("voice-turn.wav")
            .mime_str("audio/wav")?;
        let form = Form::new()
            .part("file", audio_part)
            .text("model", stt_model.to_owned());

        let response = self
            .client
            .post("https://api.openai.com/v1/audio/transcriptions")
            .bearer_auth(&self.api_key)
            .multipart(form)
            .send()
            .await?
            .error_for_status()?
            .json::<TranscriptionResponse>()
            .await?;

        Ok(response.text)
    }

    async fn synthesize_mp3(
        &self,
        tts_model: &str,
        tts_voice: &str,
        text: &str,
    ) -> anyhow::Result<Vec<u8>> {
        let payload = serde_json::json!({
            "model": tts_model,
            "voice": tts_voice,
            "input": text,
            "response_format": "mp3"
        });

        let response = self
            .client
            .post("https://api.openai.com/v1/audio/speech")
            .bearer_auth(&self.api_key)
            .json(&payload)
            .send()
            .await?
            .error_for_status()?;

        Ok(response.bytes().await?.to_vec())
    }
}

fn clamp_tts_input(input: &str) -> String {
    let trimmed = input.trim();
    if trimmed.chars().count() <= MAX_TTS_INPUT_CHARS {
        return trimmed.to_owned();
    }
    trimmed.chars().take(MAX_TTS_INPUT_CHARS).collect()
}

fn truncate_for_tool_result(input: &str, max_chars: usize) -> String {
    let compact = input.replace('\n', " ");
    if compact.chars().count() <= max_chars {
        return compact;
    }
    compact.chars().take(max_chars).collect::<String>() + "..."
}

fn audio_magic(bytes: &[u8]) -> String {
    if bytes.len() >= 4 && &bytes[0..4] == b"RIFF" {
        return "wav".to_owned();
    }
    if bytes.len() >= 3 && &bytes[0..3] == b"ID3" {
        return "mp3-id3".to_owned();
    }
    if bytes.len() >= 2 && bytes[0] == 0xFF && (bytes[1] & 0xE0) == 0xE0 {
        return "mp3-frame".to_owned();
    }
    let preview = bytes
        .iter()
        .take(8)
        .map(|byte| format!("{byte:02X}"))
        .collect::<Vec<_>>()
        .join("");
    format!("unknown:{preview}")
}

fn parse_discord_id(raw: &str, field_name: &str) -> anyhow::Result<u64> {
    raw.parse::<u64>()
        .with_context(|| format!("invalid {field_name} `{raw}`"))
}

fn elapsed_ms(started_at: Instant) -> u64 {
    let ms = started_at.elapsed().as_millis();
    if ms > u128::from(u64::MAX) {
        u64::MAX
    } else {
        ms as u64
    }
}

fn pcm_i16_to_wav_bytes(samples: &[i16], channels: u16, sample_rate: u32) -> Vec<u8> {
    let bits_per_sample = 16u16;
    let bytes_per_sample = (bits_per_sample / 8) as u32;
    let data_size = (samples.len() as u32) * bytes_per_sample;
    let byte_rate = sample_rate * channels as u32 * bytes_per_sample;
    let block_align = channels * (bits_per_sample / 8);
    let chunk_size = 36 + data_size;

    let mut wav = Vec::with_capacity((44 + data_size) as usize);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&chunk_size.to_le_bytes());
    wav.extend_from_slice(b"WAVE");
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&channels.to_le_bytes());
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&byte_rate.to_le_bytes());
    wav.extend_from_slice(&block_align.to_le_bytes());
    wav.extend_from_slice(&bits_per_sample.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_size.to_le_bytes());

    for sample in samples {
        wav.extend_from_slice(&sample.to_le_bytes());
    }

    wav
}

pub fn default_listen_window() -> Duration {
    Duration::from_millis(DEFAULT_LISTEN_WINDOW_MS)
}

pub fn default_chunk_gap() -> Duration {
    Duration::from_millis(DEFAULT_CHUNK_GAP_MS)
}

#[cfg(test)]
mod tests {
    use super::{VoiceRuntimeConfig, pcm_i16_to_wav_bytes};

    #[test]
    fn allowlist_parser_reads_pairs() {
        let parsed = VoiceRuntimeConfig::parse_allowlist("1:2,3:4,invalid");
        assert_eq!(parsed.len(), 2);
        assert!(parsed.contains(&(1, 2)));
        assert!(parsed.contains(&(3, 4)));
    }

    #[test]
    fn wav_header_size_matches_payload() {
        let samples = vec![0_i16; 480];
        let wav = pcm_i16_to_wav_bytes(&samples, 2, 48_000);
        assert!(wav.len() > 44);
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
    }
}
