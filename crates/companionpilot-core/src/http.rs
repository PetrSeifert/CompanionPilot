use std::sync::Arc;

use axum::{
    Json, Router,
    extract::{Path, Query, State},
    http::header,
    response::IntoResponse,
    routing::{delete, get, post},
};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use tower_http::trace::TraceLayer;

use crate::{
    memory::MemoryStore,
    orchestrator::DefaultChatOrchestrator,
    types::{MessageCtx, OrchestratorReply},
};

static DASHBOARD_HTML: &str = include_str!("dashboard.html");

#[derive(Clone)]
pub struct AppState {
    pub orchestrator: Arc<DefaultChatOrchestrator>,
    pub memory: Arc<dyn MemoryStore>,
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

#[derive(Debug, Deserialize)]
pub struct LimitQuery {
    #[serde(default = "default_limit")]
    pub limit: usize,
}

fn default_guild() -> String {
    "local".to_owned()
}

fn default_channel() -> String {
    "local".to_owned()
}

fn default_limit() -> usize {
    200
}

#[derive(Serialize)]
struct DeletedResponse {
    deleted: u64,
}

#[derive(Serialize)]
struct DeletedBoolResponse {
    deleted: bool,
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/", get(index))
        .route("/health", get(health))
        .route("/chat", post(chat))
        .route("/dashboard", get(dashboard))
        .route("/api/users", get(api_list_users))
        .route(
            "/api/users/{user_id}/messages",
            get(api_list_messages).delete(api_clear_messages),
        )
        .route(
            "/api/users/{user_id}/facts",
            get(api_list_facts).delete(api_clear_facts),
        )
        .route("/api/users/{user_id}/facts/{key}", delete(api_delete_fact))
        .route(
            "/api/users/{user_id}/tool-calls",
            get(api_list_tool_calls).delete(api_clear_tool_calls),
        )
        .route(
            "/api/users/{user_id}/decisions",
            get(api_list_decisions).delete(api_clear_decisions),
        )
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

async fn index() -> &'static str {
    "CompanionPilot API"
}

async fn health() -> &'static str {
    "ok"
}

async fn dashboard() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
        DASHBOARD_HTML,
    )
}

async fn chat(
    State(state): State<AppState>,
    Json(request): Json<ChatRequest>,
) -> Result<Json<OrchestratorReply>, (axum::http::StatusCode, String)> {
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

// --- Dashboard API handlers ---

async fn api_list_users(
    State(state): State<AppState>,
    Query(query): Query<LimitQuery>,
) -> Result<impl IntoResponse, (axum::http::StatusCode, String)> {
    let users = state
        .memory
        .list_users(query.limit)
        .await
        .map_err(internal_error)?;
    Ok(Json(users))
}

async fn api_list_messages(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
    Query(query): Query<LimitQuery>,
) -> Result<impl IntoResponse, (axum::http::StatusCode, String)> {
    let messages = state
        .memory
        .list_chat_messages(&user_id, query.limit)
        .await
        .map_err(internal_error)?;
    Ok(Json(messages))
}

async fn api_clear_messages(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
) -> Result<Json<DeletedResponse>, (axum::http::StatusCode, String)> {
    let deleted = state
        .memory
        .clear_chat_messages(&user_id)
        .await
        .map_err(internal_error)?;
    Ok(Json(DeletedResponse { deleted }))
}

async fn api_list_facts(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
    Query(query): Query<LimitQuery>,
) -> Result<impl IntoResponse, (axum::http::StatusCode, String)> {
    let facts = state
        .memory
        .list_facts(&user_id, query.limit)
        .await
        .map_err(internal_error)?;
    Ok(Json(facts))
}

async fn api_clear_facts(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
) -> Result<Json<DeletedResponse>, (axum::http::StatusCode, String)> {
    let deleted = state
        .memory
        .clear_facts(&user_id)
        .await
        .map_err(internal_error)?;
    Ok(Json(DeletedResponse { deleted }))
}

async fn api_delete_fact(
    State(state): State<AppState>,
    Path((user_id, key)): Path<(String, String)>,
) -> Result<Json<DeletedBoolResponse>, (axum::http::StatusCode, String)> {
    let deleted = state
        .memory
        .delete_fact(&user_id, &key)
        .await
        .map_err(internal_error)?;
    Ok(Json(DeletedBoolResponse { deleted }))
}

async fn api_list_tool_calls(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
    Query(query): Query<LimitQuery>,
) -> Result<impl IntoResponse, (axum::http::StatusCode, String)> {
    let calls = state
        .memory
        .list_tool_calls(&user_id, query.limit)
        .await
        .map_err(internal_error)?;
    Ok(Json(calls))
}

async fn api_clear_tool_calls(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
) -> Result<Json<DeletedResponse>, (axum::http::StatusCode, String)> {
    let deleted = state
        .memory
        .clear_tool_calls(&user_id)
        .await
        .map_err(internal_error)?;
    Ok(Json(DeletedResponse { deleted }))
}

async fn api_list_decisions(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
    Query(query): Query<LimitQuery>,
) -> Result<impl IntoResponse, (axum::http::StatusCode, String)> {
    let decisions = state
        .memory
        .list_planner_decisions(&user_id, query.limit)
        .await
        .map_err(internal_error)?;
    Ok(Json(decisions))
}

async fn api_clear_decisions(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
) -> Result<Json<DeletedResponse>, (axum::http::StatusCode, String)> {
    let deleted = state
        .memory
        .clear_planner_decisions(&user_id)
        .await
        .map_err(internal_error)?;
    Ok(Json(DeletedResponse { deleted }))
}

fn internal_error(error: anyhow::Error) -> (axum::http::StatusCode, String) {
    (
        axum::http::StatusCode::INTERNAL_SERVER_ERROR,
        format!("internal error: {error}"),
    )
}
