use async_trait::async_trait;

use super::{ModelProvider, ModelRequest};

#[derive(Debug, Default)]
pub struct MockModelProvider;

#[async_trait]
impl ModelProvider for MockModelProvider {
    async fn complete(&self, request: ModelRequest) -> anyhow::Result<String> {
        Ok(format!(
            "CompanionPilot mock reply.\n\nSystem: {}\n\nUser: {}",
            request.system_prompt, request.user_prompt
        ))
    }
}
