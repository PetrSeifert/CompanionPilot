use std::{io::ErrorKind, path::PathBuf, sync::Arc};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use sqlx::{PgPool, postgres::PgPoolOptions};
use tokio::{fs, sync::RwLock};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RuntimeSettings {
    pub model_provider: String,
    pub openrouter_model: String,
    pub openai_stt_model: String,
    pub openai_tts_model: String,
    pub openai_tts_voice: String,
    pub voice_allowlist: String,
    pub voice_idle_timeout_sec: u64,
    pub voice_listen_window_ms: u64,
    pub voice_chunk_gap_ms: u64,
    pub voice_max_turn_ms: u64,
}

impl RuntimeSettings {
    pub fn normalized(&self) -> anyhow::Result<Self> {
        let normalized = Self {
            model_provider: normalize_model_provider(&self.model_provider)?,
            openrouter_model: normalize_non_empty("openrouter_model", &self.openrouter_model)?,
            openai_stt_model: normalize_non_empty("openai_stt_model", &self.openai_stt_model)?,
            openai_tts_model: normalize_non_empty("openai_tts_model", &self.openai_tts_model)?,
            openai_tts_voice: normalize_non_empty("openai_tts_voice", &self.openai_tts_voice)?,
            voice_allowlist: self.voice_allowlist.trim().to_owned(),
            voice_idle_timeout_sec: self.voice_idle_timeout_sec,
            voice_listen_window_ms: self.voice_listen_window_ms,
            voice_chunk_gap_ms: self.voice_chunk_gap_ms,
            voice_max_turn_ms: self.voice_max_turn_ms,
        };
        normalized.validate()?;
        Ok(normalized)
    }

    fn validate(&self) -> anyhow::Result<()> {
        if self.voice_listen_window_ms == 0 {
            anyhow::bail!("voice_listen_window_ms must be greater than 0");
        }
        if self.voice_chunk_gap_ms == 0 {
            anyhow::bail!("voice_chunk_gap_ms must be greater than 0");
        }
        if self.voice_max_turn_ms == 0 {
            anyhow::bail!("voice_max_turn_ms must be greater than 0");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RuntimeSettingsUpdate {
    pub model_provider: Option<String>,
    pub openrouter_model: Option<String>,
    pub openai_stt_model: Option<String>,
    pub openai_tts_model: Option<String>,
    pub openai_tts_voice: Option<String>,
    pub voice_allowlist: Option<String>,
    pub voice_idle_timeout_sec: Option<u64>,
    pub voice_listen_window_ms: Option<u64>,
    pub voice_chunk_gap_ms: Option<u64>,
    pub voice_max_turn_ms: Option<u64>,
}

impl RuntimeSettingsUpdate {
    pub fn apply_to(&self, existing: &RuntimeSettings) -> anyhow::Result<RuntimeSettings> {
        let candidate = RuntimeSettings {
            model_provider: self
                .model_provider
                .clone()
                .unwrap_or_else(|| existing.model_provider.clone()),
            openrouter_model: self
                .openrouter_model
                .clone()
                .unwrap_or_else(|| existing.openrouter_model.clone()),
            openai_stt_model: self
                .openai_stt_model
                .clone()
                .unwrap_or_else(|| existing.openai_stt_model.clone()),
            openai_tts_model: self
                .openai_tts_model
                .clone()
                .unwrap_or_else(|| existing.openai_tts_model.clone()),
            openai_tts_voice: self
                .openai_tts_voice
                .clone()
                .unwrap_or_else(|| existing.openai_tts_voice.clone()),
            voice_allowlist: self
                .voice_allowlist
                .clone()
                .unwrap_or_else(|| existing.voice_allowlist.clone()),
            voice_idle_timeout_sec: self
                .voice_idle_timeout_sec
                .unwrap_or(existing.voice_idle_timeout_sec),
            voice_listen_window_ms: self
                .voice_listen_window_ms
                .unwrap_or(existing.voice_listen_window_ms),
            voice_chunk_gap_ms: self
                .voice_chunk_gap_ms
                .unwrap_or(existing.voice_chunk_gap_ms),
            voice_max_turn_ms: self.voice_max_turn_ms.unwrap_or(existing.voice_max_turn_ms),
        };
        candidate.normalized()
    }
}

#[async_trait]
pub trait RuntimeSettingsStore: Send + Sync {
    async fn load(&self) -> anyhow::Result<Option<RuntimeSettings>>;
    async fn save(&self, settings: &RuntimeSettings) -> anyhow::Result<()>;
}

#[derive(Debug)]
pub struct FileRuntimeSettingsStore {
    path: PathBuf,
}

impl FileRuntimeSettingsStore {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }
}

#[async_trait]
impl RuntimeSettingsStore for FileRuntimeSettingsStore {
    async fn load(&self) -> anyhow::Result<Option<RuntimeSettings>> {
        let raw = match fs::read_to_string(&self.path).await {
            Ok(raw) => raw,
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        let settings: RuntimeSettings = serde_json::from_str(&raw)?;
        Ok(Some(settings.normalized()?))
    }

    async fn save(&self, settings: &RuntimeSettings) -> anyhow::Result<()> {
        let normalized = settings.normalized()?;
        if let Some(parent) = self.path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).await?;
            }
        }
        let raw = serde_json::to_string_pretty(&normalized)?;
        fs::write(&self.path, raw).await?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct PostgresRuntimeSettingsStore {
    pool: PgPool,
}

impl PostgresRuntimeSettingsStore {
    pub async fn connect(database_url: &str) -> anyhow::Result<Self> {
        let pool = PgPoolOptions::new()
            .max_connections(4)
            .connect(database_url)
            .await?;
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS runtime_settings (
                 slot SMALLINT PRIMARY KEY,
                 payload_json TEXT NOT NULL,
                 updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
             )",
        )
        .execute(&pool)
        .await?;
        Ok(Self { pool })
    }
}

#[async_trait]
impl RuntimeSettingsStore for PostgresRuntimeSettingsStore {
    async fn load(&self) -> anyhow::Result<Option<RuntimeSettings>> {
        let payload = sqlx::query_scalar::<_, String>(
            "SELECT payload_json
             FROM runtime_settings
             WHERE slot = 1",
        )
        .fetch_optional(&self.pool)
        .await?;
        let Some(payload) = payload else {
            return Ok(None);
        };
        let settings: RuntimeSettings = serde_json::from_str(&payload)?;
        Ok(Some(settings.normalized()?))
    }

    async fn save(&self, settings: &RuntimeSettings) -> anyhow::Result<()> {
        let normalized = settings.normalized()?;
        let payload = serde_json::to_string(&normalized)?;
        sqlx::query(
            "INSERT INTO runtime_settings (slot, payload_json, updated_at)
             VALUES (1, $1, NOW())
             ON CONFLICT (slot)
             DO UPDATE SET payload_json = EXCLUDED.payload_json, updated_at = NOW()",
        )
        .bind(payload)
        .execute(&self.pool)
        .await?;
        Ok(())
    }
}

#[derive(Clone)]
pub struct RuntimeSettingsManager {
    settings: Arc<RwLock<RuntimeSettings>>,
    store: Arc<dyn RuntimeSettingsStore>,
}

impl RuntimeSettingsManager {
    pub async fn new(
        defaults: RuntimeSettings,
        store: Arc<dyn RuntimeSettingsStore>,
    ) -> anyhow::Result<Arc<Self>> {
        let defaults = defaults.normalized()?;
        let stored = store.load().await?;
        let current = match stored {
            Some(value) => value.normalized()?,
            None => {
                store.save(&defaults).await?;
                defaults
            }
        };
        Ok(Arc::new(Self {
            settings: Arc::new(RwLock::new(current)),
            store,
        }))
    }

    pub async fn get(&self) -> RuntimeSettings {
        self.settings.read().await.clone()
    }

    pub async fn replace(&self, new_settings: RuntimeSettings) -> anyhow::Result<RuntimeSettings> {
        let normalized = new_settings.normalized()?;
        self.store.save(&normalized).await?;
        *self.settings.write().await = normalized.clone();
        Ok(normalized)
    }

    pub async fn update(&self, update: RuntimeSettingsUpdate) -> anyhow::Result<RuntimeSettings> {
        let current = self.get().await;
        let merged = update.apply_to(&current)?;
        self.replace(merged).await
    }
}

fn normalize_model_provider(value: &str) -> anyhow::Result<String> {
    let normalized = value.trim().to_ascii_lowercase();
    if matches!(normalized.as_str(), "auto" | "openrouter" | "mock") {
        Ok(normalized)
    } else {
        anyhow::bail!("model_provider must be one of: auto, openrouter, mock");
    }
}

fn normalize_non_empty(field: &str, value: &str) -> anyhow::Result<String> {
    let normalized = value.trim().to_owned();
    if normalized.is_empty() {
        anyhow::bail!("{field} must not be empty");
    }
    Ok(normalized)
}

#[cfg(test)]
mod tests {
    use super::{RuntimeSettings, RuntimeSettingsUpdate};

    fn baseline() -> RuntimeSettings {
        RuntimeSettings {
            model_provider: "auto".to_owned(),
            openrouter_model: "anthropic/claude-3.5-sonnet".to_owned(),
            openai_stt_model: "gpt-4o-mini-transcribe".to_owned(),
            openai_tts_model: "gpt-4o-mini-tts".to_owned(),
            openai_tts_voice: "alloy".to_owned(),
            voice_allowlist: "1:2".to_owned(),
            voice_idle_timeout_sec: 300,
            voice_listen_window_ms: 12_000,
            voice_chunk_gap_ms: 700,
            voice_max_turn_ms: 12_000,
        }
    }

    #[test]
    fn update_rejects_invalid_provider() {
        let existing = baseline();
        let update = RuntimeSettingsUpdate {
            model_provider: Some("invalid".to_owned()),
            ..RuntimeSettingsUpdate::default()
        };
        assert!(update.apply_to(&existing).is_err());
    }

    #[test]
    fn update_merges_partial_values() {
        let existing = baseline();
        let update = RuntimeSettingsUpdate {
            openai_tts_voice: Some("nova".to_owned()),
            voice_chunk_gap_ms: Some(900),
            ..RuntimeSettingsUpdate::default()
        };
        let merged = update.apply_to(&existing).expect("settings should merge");
        assert_eq!(merged.openai_tts_voice, "nova");
        assert_eq!(merged.voice_chunk_gap_ms, 900);
        assert_eq!(merged.model_provider, "auto");
    }
}
