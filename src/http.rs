use std::sync::Arc;

use axum::{
    Json, Router,
    extract::State,
    routing::{get, post},
};
use chrono::Utc;
use serde::Deserialize;
use tower_http::trace::TraceLayer;

use crate::{orchestrator::DefaultChatOrchestrator, types::MessageCtx};

#[derive(Clone)]
pub struct AppState {
    pub orchestrator: Arc<DefaultChatOrchestrator>,
}

#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    pub user_id: String,
    #[serde(default = "default_guild")]
    pub guild_id: String,
    #[serde(default = "default_channel")]
    pub channel_id: String,
    pub content: String,
}

fn default_guild() -> String {
    "local".to_owned()
}

fn default_channel() -> String {
    "local".to_owned()
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/chat", post(chat))
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

async fn health() -> &'static str {
    "ok"
}

async fn chat(
    State(state): State<AppState>,
    Json(request): Json<ChatRequest>,
) -> Result<Json<crate::types::OrchestratorReply>, (axum::http::StatusCode, String)> {
    let message = MessageCtx {
        message_id: format!("http-{}", Utc::now().timestamp_millis()),
        user_id: request.user_id,
        guild_id: request.guild_id,
        channel_id: request.channel_id,
        content: request.content,
        timestamp: Utc::now(),
    };

    let reply = state
        .orchestrator
        .handle_message(message)
        .await
        .map_err(internal_error)?;

    Ok(Json(reply))
}

fn internal_error(error: anyhow::Error) -> (axum::http::StatusCode, String) {
    (
        axum::http::StatusCode::INTERNAL_SERVER_ERROR,
        error.to_string(),
    )
}
