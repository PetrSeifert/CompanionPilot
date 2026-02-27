use std::sync::Arc;

use companionpilot::{
    config::AppConfig,
    discord_bot,
    http::{self, AppState},
    memory::{InMemoryMemoryStore, MemoryStore, PostgresMemoryStore},
    model::{MockModelProvider, ModelProvider, OpenAiProvider},
    orchestrator::DefaultChatOrchestrator,
    safety::SafetyPolicy,
    tools::{TavilyWebSearchTool, ToolExecutor, ToolRegistry},
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

    let app = http::router(AppState { orchestrator });
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
    if let Some(api_key) = config.openai_api_key.clone() {
        Arc::new(OpenAiProvider::new(api_key, config.openai_model.clone()))
    } else {
        warn!("OPENAI_API_KEY not set; using mock model provider");
        Arc::new(MockModelProvider)
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
        warn!("TAVILY_API_KEY not set; /search command will fail");
    }

    Arc::new(ToolRegistry { web_search })
}
