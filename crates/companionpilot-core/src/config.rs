use std::{env, net::SocketAddr};

#[derive(Debug, Clone)]
pub struct AppConfig {
    pub http_bind: SocketAddr,
    pub discord_token: Option<String>,
    pub model_provider: String,
    pub openrouter_api_key: Option<String>,
    pub openrouter_model: String,
    pub openrouter_referer: Option<String>,
    pub openrouter_title: Option<String>,
    pub openai_api_key: Option<String>,
    pub openai_stt_model: String,
    pub openai_tts_model: String,
    pub openai_tts_voice: String,
    pub tavily_api_key: Option<String>,
    pub database_url: Option<String>,
    pub redis_url: Option<String>,
    pub voice_enabled: bool,
    pub voice_allowlist: String,
    pub voice_idle_timeout_sec: u64,
    pub voice_chunk_gap_ms: u64,
    pub voice_max_turn_ms: u64,
    pub voice_listen_window_ms: u64,
}

impl AppConfig {
    pub fn from_env() -> anyhow::Result<Self> {
        let port = env::var("PORT").unwrap_or_else(|_| "8080".to_owned());
        let http_bind = env::var("HTTP_BIND").unwrap_or_else(|_| format!("0.0.0.0:{port}"));
        let http_bind = http_bind.parse()?;

        Ok(Self {
            http_bind,
            discord_token: env::var("DISCORD_TOKEN").ok(),
            model_provider: env::var("MODEL_PROVIDER").unwrap_or_else(|_| "auto".to_owned()),
            openrouter_api_key: env::var("OPENROUTER_API_KEY").ok(),
            openrouter_model: env::var("OPENROUTER_MODEL")
                .unwrap_or_else(|_| "anthropic/claude-3.5-sonnet".to_owned()),
            openrouter_referer: env::var("OPENROUTER_REFERER").ok(),
            openrouter_title: env::var("OPENROUTER_TITLE").ok(),
            openai_api_key: env::var("OPENAI_API_KEY").ok(),
            openai_stt_model: env::var("OPENAI_STT_MODEL")
                .unwrap_or_else(|_| "gpt-4o-mini-transcribe".to_owned()),
            openai_tts_model: env::var("OPENAI_TTS_MODEL")
                .unwrap_or_else(|_| "gpt-4o-mini-tts".to_owned()),
            openai_tts_voice: env::var("OPENAI_TTS_VOICE").unwrap_or_else(|_| "alloy".to_owned()),
            tavily_api_key: env::var("TAVILY_API_KEY").ok(),
            database_url: env::var("DATABASE_URL").ok(),
            redis_url: env::var("REDIS_URL").ok(),
            voice_enabled: env_bool("VOICE_ENABLED", false),
            voice_allowlist: env::var("VOICE_ALLOWLIST").unwrap_or_default(),
            voice_idle_timeout_sec: env_u64("VOICE_IDLE_TIMEOUT_SEC", 300),
            voice_chunk_gap_ms: env_u64("VOICE_CHUNK_GAP_MS", 700),
            voice_max_turn_ms: env_u64("VOICE_MAX_TURN_MS", 12_000),
            voice_listen_window_ms: env_u64("VOICE_LISTEN_WINDOW_MS", 12_000),
        })
    }
}

fn env_bool(name: &str, default: bool) -> bool {
    env::var(name)
        .ok()
        .map(|raw| {
            matches!(
                raw.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(default)
}

fn env_u64(name: &str, default: u64) -> u64 {
    env::var(name)
        .ok()
        .and_then(|raw| raw.trim().parse::<u64>().ok())
        .unwrap_or(default)
}
