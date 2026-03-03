use std::sync::Arc;

use axum::{
    Json, Router,
    extract::{Path, Query, State},
    http::{HeaderValue, Request, StatusCode, header},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{delete, get, post},
};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use subtle::ConstantTimeEq;
use tower_http::trace::TraceLayer;

use crate::{
    memory::MemoryStore,
    orchestrator::DefaultChatOrchestrator,
    runtime_settings::{RuntimeSettings, RuntimeSettingsManager, RuntimeSettingsUpdate},
    types::{MessageCtx, OrchestratorReply},
};

const BASIC_AUTH_CHALLENGE: &str = r#"Basic realm="CompanionPilot", charset="UTF-8""#;

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
            require_api_auth,
        ));

    Router::new()
        .route("/", get(index))
        .route("/health", get(health))
        .merge(protected)
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

async fn require_api_auth(
    State(state): State<AppState>,
    request: Request<axum::body::Body>,
    next: Next,
) -> Result<Response, Response> {
    let expected_token = state
        .api_auth_token
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(api_auth_unavailable_response)?;

    let provided_auth = request.headers().get(header::AUTHORIZATION);
    if !authorization_matches_expected_token(provided_auth, expected_token) {
        return Err(unauthorized_response());
    }

    Ok(next.run(request).await)
}

fn authorization_matches_expected_token(
    authorization_header: Option<&HeaderValue>,
    expected_token: &str,
) -> bool {
    let Some(raw_header) = authorization_header else {
        return false;
    };

    let Ok(value) = raw_header.to_str() else {
        return false;
    };

    if let Some(token) = value.strip_prefix("Bearer ") {
        return constant_time_str_eq(token, expected_token);
    }

    let Some(encoded_credentials) = value.strip_prefix("Basic ") else {
        return false;
    };
    let Ok(decoded_bytes) = STANDARD.decode(encoded_credentials) else {
        return false;
    };
    let Ok(decoded_credentials) = std::str::from_utf8(&decoded_bytes) else {
        return false;
    };

    let Some((username, password)) = decoded_credentials.split_once(':') else {
        return false;
    };

    // Accept token as either username or password to work with common browser credential prompts.
    constant_time_str_eq(username, expected_token) || constant_time_str_eq(password, expected_token)
}

fn constant_time_str_eq(candidate: &str, expected: &str) -> bool {
    let candidate_bytes = candidate.as_bytes();
    let expected_bytes = expected.as_bytes();
    if candidate_bytes.len() != expected_bytes.len() {
        return false;
    }
    candidate_bytes.ct_eq(expected_bytes).into()
}

fn unauthorized_response() -> Response {
    (
        StatusCode::UNAUTHORIZED,
        [(header::WWW_AUTHENTICATE, BASIC_AUTH_CHALLENGE)],
        "missing or invalid credentials",
    )
        .into_response()
}

fn api_auth_unavailable_response() -> Response {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        "API auth token is not configured",
    )
        .into_response()
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
        .list_orchestration_decisions(&user_id, query.limit)
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
        .clear_orchestration_decisions(&user_id)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn make_authorization_value(value: &str) -> HeaderValue {
        HeaderValue::from_str(value).expect("valid authorization header")
    }

    #[test]
    fn accepts_bearer_token() {
        let expected = "dashboard-secret";
        let header = make_authorization_value("Bearer dashboard-secret");
        assert!(authorization_matches_expected_token(
            Some(&header),
            expected
        ));
    }

    #[test]
    fn accepts_basic_token_as_password() {
        let expected = "dashboard-secret";
        let encoded = STANDARD.encode(format!("operator:{expected}"));
        let header = make_authorization_value(&format!("Basic {encoded}"));
        assert!(authorization_matches_expected_token(
            Some(&header),
            expected
        ));
    }

    #[test]
    fn accepts_basic_token_as_username() {
        let expected = "dashboard-secret";
        let encoded = STANDARD.encode(format!("{expected}:ignored"));
        let header = make_authorization_value(&format!("Basic {encoded}"));
        assert!(authorization_matches_expected_token(
            Some(&header),
            expected
        ));
    }

    #[test]
    fn rejects_invalid_authorization() {
        let expected = "dashboard-secret";
        let invalid_bearer = make_authorization_value("Bearer wrong");
        assert!(!authorization_matches_expected_token(
            Some(&invalid_bearer),
            expected
        ));

        let malformed_basic = make_authorization_value("Basic not-base64!");
        assert!(!authorization_matches_expected_token(
            Some(&malformed_basic),
            expected
        ));

        assert!(!authorization_matches_expected_token(None, expected));
    }
}
