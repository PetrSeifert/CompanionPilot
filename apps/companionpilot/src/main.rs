use std::sync::Arc;

use companionpilot_core::{
    config::AppConfig,
    discord_bot,
    http::{self, AppState},
    memory::{InMemoryMemoryStore, MemoryStore, PostgresMemoryStore},
    model::{MockModelProvider, ModelProvider, OpenRouterProvider},
    orchestrator::DefaultChatOrchestrator,
    safety::SafetyPolicy,
    tools::{
        CliTool, CurrentDateTimeTool, SpogoCli, SpogoControlTool, SpogoSearchTool, SpogoStatusTool,
        TavilyWebSearchTool, ToolExecutor, ToolRegistry,
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

    let model = build_model_provider(&config);
    let memory = build_memory_store(&config).await?;
    let voice = build_voice_manager(&config, memory.clone());
    let tools = build_tools(&config, voice.clone());

    let memory_for_dashboard = memory.clone();
    let orchestrator = Arc::new(DefaultChatOrchestrator::new(
        model,
        memory,
        tools,
        SafetyPolicy::default(),
    ));
    if let Some(voice_manager) = &voice {
        voice_manager.set_orchestrator(orchestrator.clone()).await;
        voice_manager.start_idle_reaper();
    }

    if let Some(discord_token) = config.discord_token.clone() {
        let allowed_channel_ids =
            discord_bot::parse_allowed_channel_ids(&config.discord_allowed_channel_ids);
        if !config.discord_allowed_channel_ids.trim().is_empty() && allowed_channel_ids.is_empty() {
            warn!(
                "DISCORD_ALLOWED_CHANNEL_IDS is set but contains no valid channel IDs; allowlist is disabled"
            );
        }
        let discord_orchestrator = orchestrator.clone();
        let discord_voice = voice.clone();
        tokio::spawn(async move {
            if let Err(error) =
                discord_bot::start_discord_bot(
                    discord_token,
                    discord_orchestrator,
                    discord_voice,
                    allowed_channel_ids,
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

fn build_model_provider(config: &AppConfig) -> Arc<dyn ModelProvider> {
    let provider = config.model_provider.to_lowercase();
    match provider.as_str() {
        "openrouter" => {
            if let Some(api_key) = config.openrouter_api_key.clone() {
                info!(model = %config.openrouter_model, "using OpenRouter model provider");
                Arc::new(OpenRouterProvider::new(
                    api_key,
                    config.openrouter_model.clone(),
                    config.openrouter_referer.clone(),
                    config.openrouter_title.clone(),
                ))
            } else {
                warn!("MODEL_PROVIDER=openrouter but OPENROUTER_API_KEY is missing; using mock");
                Arc::new(MockModelProvider)
            }
        }
        "mock" => {
            warn!("MODEL_PROVIDER=mock; using mock model provider");
            Arc::new(MockModelProvider)
        }
        "auto" => {
            if let Some(api_key) = config.openrouter_api_key.clone() {
                info!(
                    model = %config.openrouter_model,
                    "using OpenRouter model provider (auto mode)"
                );
                Arc::new(OpenRouterProvider::new(
                    api_key,
                    config.openrouter_model.clone(),
                    config.openrouter_referer.clone(),
                    config.openrouter_title.clone(),
                ))
            } else {
                warn!("No OPENROUTER_API_KEY configured; using mock model provider");
                Arc::new(MockModelProvider)
            }
        }
        other => {
            warn!(
                provider = %other,
                "unknown MODEL_PROVIDER value; valid values are auto|openrouter|mock; falling back to auto"
            );
            if let Some(api_key) = config.openrouter_api_key.clone() {
                Arc::new(OpenRouterProvider::new(
                    api_key,
                    config.openrouter_model.clone(),
                    config.openrouter_referer.clone(),
                    config.openrouter_title.clone(),
                ))
            } else {
                Arc::new(MockModelProvider)
            }
        }
    }
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
    let spogo_cli = SpogoCli::new(
        config.spogo_bin_path.clone(),
        config.spogo_config_dir.clone(),
        config.spogo_timeout_ms,
    );
    let spogo_control =
        SpogoControlTool::new(spogo_cli.clone(), config.spogo_account_label.clone());
    let spogo_status = SpogoStatusTool::new(spogo_cli.clone(), config.spogo_account_label.clone());
    let cli_tool = CliTool::new(spogo_cli.clone());
    let spogo_search = SpogoSearchTool::new(spogo_cli);

    if web_search.is_none() {
        warn!("TAVILY_API_KEY not set; planner-selected web_search calls will fail");
    }
    if config.spogo_config_dir.trim().is_empty() {
        warn!("SPOGO_CONFIG_DIR is empty; planner-selected spogo tool calls will fail");
    }

    Arc::new(ToolRegistry {
        current_datetime: CurrentDateTimeTool,
        cli: cli_tool,
        spogo_control,
        spogo_status,
        spogo_search,
        web_search,
        voice,
    })
}

fn build_voice_manager(
    config: &AppConfig,
    memory: Arc<dyn MemoryStore>,
) -> Option<Arc<VoiceManager>> {
    if !config.voice_enabled {
        return None;
    }

    let Some(openai_api_key) = config.openai_api_key.clone() else {
        warn!("VOICE_ENABLED is true but OPENAI_API_KEY is missing; voice is disabled");
        return None;
    };

    let allowlist = VoiceRuntimeConfig::parse_allowlist(&config.voice_allowlist);
    if allowlist.is_empty() {
        warn!(
            "VOICE_ENABLED is true but VOICE_ALLOWLIST has no valid guild:channel entries; voice tools will fail until configured"
        );
    }

    Some(VoiceManager::new(
        VoiceRuntimeConfig {
            openai_api_key,
            stt_model: config.openai_stt_model.clone(),
            tts_model: config.openai_tts_model.clone(),
            tts_voice: config.openai_tts_voice.clone(),
            allowlist,
            idle_timeout: std::time::Duration::from_secs(config.voice_idle_timeout_sec),
            default_chunk_gap: std::time::Duration::from_millis(config.voice_chunk_gap_ms),
            default_listen_window: std::time::Duration::from_millis(config.voice_listen_window_ms),
            default_max_turn: std::time::Duration::from_millis(config.voice_max_turn_ms),
        },
        memory,
    ))
}
