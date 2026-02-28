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
    pub tavily_api_key: Option<String>,
    pub database_url: Option<String>,
    pub redis_url: Option<String>,
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
            tavily_api_key: env::var("TAVILY_API_KEY").ok(),
            database_url: env::var("DATABASE_URL").ok(),
            redis_url: env::var("REDIS_URL").ok(),
        })
    }
}
