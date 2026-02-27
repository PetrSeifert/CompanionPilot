mod mock;
mod openrouter;

use async_trait::async_trait;

pub use mock::MockModelProvider;
pub use openrouter::OpenRouterProvider;

#[derive(Debug, Clone)]
pub struct ModelRequest {
    pub system_prompt: String,
    pub user_prompt: String,
}

#[async_trait]
pub trait ModelProvider: Send + Sync {
    async fn complete(&self, request: ModelRequest) -> anyhow::Result<String>;
}
