use std::{env, net::SocketAddr};

#[derive(Debug, Clone)]
pub struct AppConfig {
    pub http_bind: SocketAddr,
    pub discord_token: Option<String>,
    pub openai_api_key: Option<String>,
    pub openai_model: String,
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
            openai_api_key: env::var("OPENAI_API_KEY").ok(),
            openai_model: env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_owned()),
            tavily_api_key: env::var("TAVILY_API_KEY").ok(),
            database_url: env::var("DATABASE_URL").ok(),
            redis_url: env::var("REDIS_URL").ok(),
        })
    }
}
