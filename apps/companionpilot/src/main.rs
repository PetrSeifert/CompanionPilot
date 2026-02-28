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
        CurrentDateTimeTool, SpotifyPlayingStatusTool, TavilyWebSearchTool, ToolExecutor,
        ToolRegistry,
    },
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
    let tools = build_tools(&config);

    let memory_for_dashboard = memory.clone();
    let orchestrator = Arc::new(DefaultChatOrchestrator::new(
        model,
        memory,
        tools,
        SafetyPolicy::default(),
    ));

    if let Some(discord_token) = config.discord_token.clone() {
        let discord_orchestrator = orchestrator.clone();
        tokio::spawn(async move {
            if let Err(error) =
                discord_bot::start_discord_bot(discord_token, discord_orchestrator).await
            {
                warn!(?error, "Discord bot stopped with error");
            }
        });
    } else {
        warn!("DISCORD_TOKEN is not set; Discord bot is disabled");
    }

    if config.redis_url.is_none() {
        warn!("REDIS_URL is not configured; using stateless in-process cache only");
    }

    let app = http::router(AppState {
        orchestrator,
        memory: memory_for_dashboard,
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

fn build_tools(config: &AppConfig) -> Arc<dyn ToolExecutor> {
    let web_search = config
        .tavily_api_key
        .as_ref()
        .map(|key| TavilyWebSearchTool::new(key.clone()));

    if web_search.is_none() {
        warn!("TAVILY_API_KEY not set; planner-selected web_search calls will fail");
    }

    Arc::new(ToolRegistry {
        current_datetime: CurrentDateTimeTool,
        spotify_playing_status: SpotifyPlayingStatusTool::default(),
        web_search,
    })
}
