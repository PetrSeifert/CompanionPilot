use std::sync::Arc;

use axum::{
    Json, Router,
    extract::{Path, Query, State},
    http::{Request, StatusCode, header},
    middleware::{self, Next},
    response::IntoResponse,
    response::Response,
    routing::{delete, get, post},
};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use tower_http::trace::TraceLayer;

use crate::{
    memory::MemoryStore,
    orchestrator::DefaultChatOrchestrator,
    runtime_settings::{RuntimeSettings, RuntimeSettingsManager, RuntimeSettingsUpdate},
    types::{MessageCtx, OrchestratorReply},
};

static DASHBOARD_HTML: &str = include_str!("dashboard.html");

#[derive(Clone)]
pub struct AppState {
    pub orchestrator: Arc<DefaultChatOrchestrator>,
    pub memory: Arc<dyn MemoryStore>,
    pub runtime_settings: Arc<RuntimeSettingsManager>,
    pub api_auth_token: Option<String>,
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
    let protected = Router::new()
        .route("/chat", post(chat))
        .route("/dashboard", get(dashboard))
        .route(
            "/api/runtime-settings",
            get(api_get_runtime_settings).put(api_update_runtime_settings),
        )
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
        .route(
            "/api/users/{user_id}/latencies",
            get(api_list_message_latencies).delete(api_clear_message_latencies),
        )
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            require_bearer_auth,
        ));

    Router::new()
        .route("/", get(index))
        .route("/health", get(health))
        .merge(protected)
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

async fn require_bearer_auth(
    State(state): State<AppState>,
    request: Request<axum::body::Body>,
    next: Next,
) -> Result<Response, (StatusCode, String)> {
    let expected_token = state
        .api_auth_token
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or((
            StatusCode::SERVICE_UNAVAILABLE,
            "API auth token is not configured".to_owned(),
        ))?;

    let provided_token = request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|raw| raw.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "));

    if provided_token != Some(expected_token) {
        return Err((
            StatusCode::UNAUTHORIZED,
            "missing or invalid bearer token".to_owned(),
        ));
    }

    Ok(next.run(request).await)
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

async fn api_get_runtime_settings(
    State(state): State<AppState>,
) -> Result<Json<RuntimeSettings>, (axum::http::StatusCode, String)> {
    Ok(Json(state.runtime_settings.get().await))
}

async fn api_update_runtime_settings(
    State(state): State<AppState>,
    Json(update): Json<RuntimeSettingsUpdate>,
) -> Result<Json<RuntimeSettings>, (axum::http::StatusCode, String)> {
    let updated = state
        .runtime_settings
        .update(update)
        .await
        .map_err(|error| {
            (
                axum::http::StatusCode::BAD_REQUEST,
                format!("invalid runtime settings: {error}"),
            )
        })?;
    Ok(Json(updated))
}

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

async fn api_list_message_latencies(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
    Query(query): Query<LimitQuery>,
) -> Result<impl IntoResponse, (axum::http::StatusCode, String)> {
    let latencies = state
        .memory
        .list_message_latencies(&user_id, query.limit)
        .await
        .map_err(internal_error)?;
    Ok(Json(latencies))
}

async fn api_clear_message_latencies(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
) -> Result<Json<DeletedResponse>, (axum::http::StatusCode, String)> {
    let deleted = state
        .memory
        .clear_message_latencies(&user_id)
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
