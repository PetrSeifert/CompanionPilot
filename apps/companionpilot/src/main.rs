use std::sync::Arc;

use companionpilot_core::{
    config::AppConfig,
    discord_bot,
    http::{self, AppState},
    memory::{InMemoryMemoryStore, MemoryStore, PostgresMemoryStore},
    model::{ModelProvider, RuntimeModelProvider},
    orchestrator::DefaultChatOrchestrator,
    runtime_settings::{
        FileRuntimeSettingsStore, PostgresRuntimeSettingsStore, RuntimeSettings,
        RuntimeSettingsManager, RuntimeSettingsStore,
    },
    safety::SafetyPolicy,
    skills::SkillCatalog,
    tools::{
        CliTool, CurrentDateTimeTool, SpogoCli, TavilyCrawlTool, TavilyExtractTool,
        TavilyResearchTool, TavilyWebSearchTool, ToolExecutor, ToolRegistry,
    },
    voice::{VoiceManager, VoiceRuntimeConfig},
};
use tokio::net::TcpListener;
use tracing::{info, warn};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();
    init_tracing();

    let config = AppConfig::from_env()?;
    let runtime_settings = build_runtime_settings_manager(&config).await?;
    let runtime_snapshot = runtime_settings.get().await;

    let model = build_model_provider(&config, runtime_settings.clone());
    let memory = build_memory_store(&config).await?;
    let voice = build_voice_manager(
        &config,
        runtime_settings.clone(),
        &runtime_snapshot,
        memory.clone(),
    );
    let tools = build_tools(&config, voice.clone());
    let skills = Arc::new(SkillCatalog::load_from_dir("skills")?);
    if skills.is_empty() {
        warn!("No runtime skills loaded from ./skills; skill selector will run with empty catalog");
    } else {
        info!(
            skill_count = skills.len(),
            "Loaded runtime skills from ./skills"
        );
    }

    let memory_for_dashboard = memory.clone();
    let orchestrator = Arc::new(DefaultChatOrchestrator::new(
        model,
        memory,
        tools,
        skills,
        SafetyPolicy::default(),
    ));
    if let Some(voice_manager) = &voice {
        voice_manager.set_orchestrator(orchestrator.clone()).await;
        voice_manager.start_idle_reaper();
    }

    if let Some(discord_token) = config.discord_token.clone() {
        let allowed_channel_ids =
            discord_bot::parse_allowed_channel_ids(&runtime_snapshot.discord_allowed_channel_ids);
        if !runtime_snapshot
            .discord_allowed_channel_ids
            .trim()
            .is_empty()
            && allowed_channel_ids.is_empty()
        {
            warn!(
                "DISCORD_ALLOWED_CHANNEL_IDS is set but contains no valid channel IDs; allowlist is disabled"
            );
        }
        let discord_orchestrator = orchestrator.clone();
        let discord_voice = voice.clone();
        let discord_runtime_settings = runtime_settings.clone();
        tokio::spawn(async move {
            if let Err(error) = discord_bot::start_discord_bot(
                discord_token,
                discord_orchestrator,
                discord_voice,
                discord_runtime_settings,
            )
            .await
            {
                warn!(?error, "Discord bot stopped with error");
            }
        });
    } else {
        warn!("DISCORD_TOKEN is not set; Discord bot is disabled");
    }

    if config.api_auth_token.is_none() {
        warn!(
            "API_AUTH_TOKEN is not configured; protected HTTP endpoints will return 503 until configured"
        );
    }

    let app = http::router(AppState {
        orchestrator,
        memory: memory_for_dashboard,
        runtime_settings,
        api_auth_token: config.api_auth_token.clone(),
    });
    let listener = TcpListener::bind(config.http_bind).await?;
    info!("CompanionPilot HTTP API listening on {}", config.http_bind);

    axum::serve(listener, app).await?;
    Ok(())
}

fn init_tracing() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .compact()
        .init();
}

fn build_model_provider(
    config: &AppConfig,
    runtime_settings: Arc<RuntimeSettingsManager>,
) -> Arc<dyn ModelProvider> {
    if config.openrouter_api_key.is_none() {
        warn!(
            "OPENROUTER_API_KEY is missing; runtime MODEL_PROVIDER=openrouter will fall back to mock"
        );
    }
    Arc::new(RuntimeModelProvider::new(
        runtime_settings,
        config.openrouter_api_key.clone(),
        config.openrouter_referer.clone(),
        config.openrouter_title.clone(),
    ))
}

async fn build_memory_store(config: &AppConfig) -> anyhow::Result<Arc<dyn MemoryStore>> {
    if let Some(database_url) = &config.database_url {
        let store = PostgresMemoryStore::connect(database_url).await?;
        info!("Connected to Postgres memory store");
        Ok(Arc::new(store))
    } else {
        warn!("DATABASE_URL not set; using in-memory store");
        Ok(Arc::new(InMemoryMemoryStore::default()))
    }
}

fn build_tools(config: &AppConfig, voice: Option<Arc<VoiceManager>>) -> Arc<dyn ToolExecutor> {
    let web_search = config
        .tavily_api_key
        .as_ref()
        .map(|key| TavilyWebSearchTool::new(key.clone()));
    let web_extract = config
        .tavily_api_key
        .as_ref()
        .map(|key| TavilyExtractTool::new(key.clone()));
    let web_crawl = config
        .tavily_api_key
        .as_ref()
        .map(|key| TavilyCrawlTool::new(key.clone()));
    let web_research = config
        .tavily_api_key
        .as_ref()
        .map(|key| TavilyResearchTool::new(key.clone()));
    let spogo_cli = SpogoCli::new(
        config.spogo_bin_path.clone(),
        config.spogo_config_dir.clone(),
        config.spogo_timeout_ms,
    );
    let cli_tool = CliTool::new(spogo_cli);

    if web_search.is_none() {
        warn!("TAVILY_API_KEY not set; model-selected web_search calls will fail");
    }
    if config.spogo_config_dir.trim().is_empty() {
        warn!("SPOGO_CONFIG_DIR is empty; model-selected cli spogo calls will fail");
    }

    Arc::new(ToolRegistry {
        current_datetime: CurrentDateTimeTool,
        cli: cli_tool,
        web_search,
        web_extract,
        web_crawl,
        web_research,
        voice,
    })
}

fn build_voice_manager(
    config: &AppConfig,
    runtime_settings: Arc<RuntimeSettingsManager>,
    initial_settings: &RuntimeSettings,
    memory: Arc<dyn MemoryStore>,
) -> Option<Arc<VoiceManager>> {
    if !config.voice_enabled {
        return None;
    }

    let Some(openai_api_key) = config.openai_api_key.clone() else {
        warn!("VOICE_ENABLED is true but OPENAI_API_KEY is missing; voice is disabled");
        return None;
    };

    let allowlist = VoiceRuntimeConfig::parse_allowlist(&initial_settings.voice_allowlist);
    if allowlist.is_empty() {
        warn!(
            "VOICE_ENABLED is true but VOICE_ALLOWLIST has no valid guild:channel entries; voice tools will fail until configured"
        );
    }

    Some(VoiceManager::new(
        VoiceRuntimeConfig {
            openai_api_key,
            stt_model: initial_settings.openai_stt_model.clone(),
            tts_model: initial_settings.openai_tts_model.clone(),
            tts_voice: initial_settings.openai_tts_voice.clone(),
            allowlist,
            idle_timeout: std::time::Duration::from_secs(initial_settings.voice_idle_timeout_sec),
            default_chunk_gap: std::time::Duration::from_millis(
                initial_settings.voice_chunk_gap_ms,
            ),
            default_listen_window: std::time::Duration::from_millis(
                initial_settings.voice_listen_window_ms,
            ),
            default_max_turn: std::time::Duration::from_millis(initial_settings.voice_max_turn_ms),
        },
        runtime_settings,
        memory,
    ))
}

async fn build_runtime_settings_manager(
    config: &AppConfig,
) -> anyhow::Result<Arc<RuntimeSettingsManager>> {
    let defaults = RuntimeSettings {
        model_provider: config.model_provider.clone(),
        openrouter_model: config.openrouter_model.clone(),
        openai_stt_model: config.openai_stt_model.clone(),
        openai_tts_model: config.openai_tts_model.clone(),
        openai_tts_voice: config.openai_tts_voice.clone(),
        discord_allowed_channel_ids: config.discord_allowed_channel_ids.clone(),
        voice_allowlist: config.voice_allowlist.clone(),
        voice_idle_timeout_sec: config.voice_idle_timeout_sec,
        voice_listen_window_ms: config.voice_listen_window_ms,
        voice_chunk_gap_ms: config.voice_chunk_gap_ms,
        voice_max_turn_ms: config.voice_max_turn_ms,
    };

    let store: Arc<dyn RuntimeSettingsStore> = if let Some(database_url) = &config.database_url {
        info!("using Postgres runtime settings store");
        Arc::new(PostgresRuntimeSettingsStore::connect(database_url).await?)
    } else {
        info!(
            path = %config.runtime_settings_path,
            "using file runtime settings store"
        );
        Arc::new(FileRuntimeSettingsStore::new(
            config.runtime_settings_path.clone(),
        ))
    };

    RuntimeSettingsManager::new(defaults, store).await
}
