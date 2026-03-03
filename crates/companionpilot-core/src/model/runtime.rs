use std::sync::Arc;

use async_trait::async_trait;

use crate::runtime_settings::RuntimeSettingsManager;

use super::{
    MockModelProvider, ModelProvider, ModelRequest, ModelTurnRequest, ModelTurnResponse,
    OpenRouterProvider,
};

pub struct RuntimeModelProvider {
    runtime_settings: Arc<RuntimeSettingsManager>,
    openrouter: Option<OpenRouterProvider>,
    mock: MockModelProvider,
}

impl RuntimeModelProvider {
    pub fn new(
        runtime_settings: Arc<RuntimeSettingsManager>,
        openrouter_api_key: Option<String>,
        openrouter_referer: Option<String>,
        openrouter_title: Option<String>,
    ) -> Self {
        let openrouter = openrouter_api_key.map(|api_key| {
            OpenRouterProvider::new(
                api_key,
                "anthropic/claude-3.5-sonnet".to_owned(),
                openrouter_referer,
                openrouter_title,
            )
        });
        Self {
            runtime_settings,
            openrouter,
            mock: MockModelProvider,
        }
    }
}

#[async_trait]
impl ModelProvider for RuntimeModelProvider {
    async fn complete(&self, request: ModelRequest) -> anyhow::Result<String> {
        let settings = self.runtime_settings.get().await;
        match settings.model_provider.as_str() {
            "mock" => self.mock.complete(request).await,
            "openrouter" => {
                let Some(openrouter) = &self.openrouter else {
                    return self.mock.complete(request).await;
                };
                openrouter
                    .complete_with_model(&settings.openrouter_model, request)
                    .await
            }
            "auto" => {
                if let Some(openrouter) = &self.openrouter {
                    openrouter
                        .complete_with_model(&settings.openrouter_model, request)
                        .await
                } else {
                    self.mock.complete(request).await
                }
            }
            _ => self.mock.complete(request).await,
        }
    }

    async fn complete_turn(&self, request: ModelTurnRequest) -> anyhow::Result<ModelTurnResponse> {
        let settings = self.runtime_settings.get().await;
        match settings.model_provider.as_str() {
            "mock" => self.mock.complete_turn(request).await,
            "openrouter" => {
                let Some(openrouter) = &self.openrouter else {
                    return self.mock.complete_turn(request).await;
                };
                openrouter
                    .complete_turn_with_model(&settings.openrouter_model, request)
                    .await
            }
            "auto" => {
                if let Some(openrouter) = &self.openrouter {
                    openrouter
                        .complete_turn_with_model(&settings.openrouter_model, request)
                        .await
                } else {
                    self.mock.complete_turn(request).await
                }
            }
            _ => self.mock.complete_turn(request).await,
        }
    }
}
