use std::sync::Arc;

use axum::{
    Json, Router,
    extract::{Path, Query, State},
    response::Html,
    routing::{get, post},
};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use tower_http::trace::TraceLayer;

use crate::{
    memory::MemoryStore,
    orchestrator::DefaultChatOrchestrator,
    types::{MessageCtx, OrchestratorReply},
};

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
struct LimitQuery {
    limit: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct DashboardChatRequest {
    content: String,
    #[serde(default = "default_dashboard_guild")]
    guild_id: String,
    #[serde(default = "default_dashboard_channel")]
    channel_id: String,
    #[serde(default)]
    system_prompt: Option<String>,
}

#[derive(Debug, Deserialize)]
struct UpsertFactRequest {
    key: String,
    value: String,
    confidence: Option<f32>,
}

#[derive(Debug, Serialize)]
struct DefaultSystemPromptResponse {
    system_prompt: String,
}

fn default_guild() -> String {
    "local".to_owned()
}

fn default_channel() -> String {
    "local".to_owned()
}

fn default_dashboard_guild() -> String {
    "dashboard".to_owned()
}

fn default_dashboard_channel() -> String {
    "web".to_owned()
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/", get(index))
        .route("/app", get(dashboard_app))
        .route("/health", get(health))
        .route("/chat", post(chat))
        .route(
            "/api/dashboard/default-system-prompt",
            get(get_dashboard_default_system_prompt),
        )
        .route("/api/dashboard/users", get(list_users))
        .route("/api/dashboard/users/{user_id}/facts", get(list_user_facts))
        .route(
            "/api/dashboard/users/{user_id}/facts",
            post(upsert_user_fact),
        )
        .route(
            "/api/dashboard/users/{user_id}/facts/{key}",
            axum::routing::delete(delete_user_fact),
        )
        .route("/api/dashboard/users/{user_id}/chats", get(list_user_chats))
        .route(
            "/api/dashboard/users/{user_id}/chats",
            axum::routing::delete(clear_user_chats),
        )
        .route(
            "/api/dashboard/users/{user_id}/chats/{message_id}",
            axum::routing::delete(delete_user_chat_message),
        )
        .route(
            "/api/dashboard/users/{user_id}/toolcalls",
            get(list_user_tool_calls),
        )
        .route(
            "/api/dashboard/users/{user_id}/toolcalls",
            axum::routing::delete(clear_user_tool_calls),
        )
        .route(
            "/api/dashboard/users/{user_id}/planners",
            get(list_user_planner_decisions),
        )
        .route(
            "/api/dashboard/users/{user_id}/planners",
            axum::routing::delete(clear_user_planner_decisions),
        )
        .route(
            "/api/dashboard/users/{user_id}/chat",
            post(send_dashboard_message),
        )
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

async fn index() -> Html<&'static str> {
    Html("<a href=\"/app\">CompanionPilot Dashboard</a>")
}

async fn dashboard_app() -> Html<&'static str> {
    Html(DASHBOARD_HTML)
}

async fn health() -> &'static str {
    "ok"
}

async fn get_dashboard_default_system_prompt() -> Json<DefaultSystemPromptResponse> {
    Json(DefaultSystemPromptResponse {
        system_prompt: crate::orchestrator::default_system_prompt_base().to_owned(),
    })
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

async fn list_users(
    State(state): State<AppState>,
    Query(query): Query<LimitQuery>,
) -> Result<Json<Vec<crate::types::UserDashboardSummary>>, (axum::http::StatusCode, String)> {
    let limit = query.limit.unwrap_or(50).clamp(1, 500);
    let users = state
        .memory
        .list_users(limit)
        .await
        .map_err(internal_error)?;
    Ok(Json(users))
}

async fn list_user_facts(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
    Query(query): Query<LimitQuery>,
) -> Result<Json<Vec<crate::types::MemoryFact>>, (axum::http::StatusCode, String)> {
    let limit = query.limit.unwrap_or(100).clamp(1, 500);
    let facts = state
        .memory
        .list_facts(&user_id, limit)
        .await
        .map_err(internal_error)?;
    Ok(Json(facts))
}

async fn upsert_user_fact(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
    Json(request): Json<UpsertFactRequest>,
) -> Result<Json<crate::types::MemoryFact>, (axum::http::StatusCode, String)> {
    let key = request.key.trim().to_lowercase();
    let value = request.value.trim().to_owned();
    if key.is_empty() || value.is_empty() {
        return Err((
            axum::http::StatusCode::BAD_REQUEST,
            "key and value must be non-empty".to_owned(),
        ));
    }

    let fact = crate::types::MemoryFact {
        key,
        value,
        confidence: request.confidence.unwrap_or(0.95).clamp(0.0, 1.0),
        source: "dashboard_manual".to_owned(),
        updated_at: Utc::now(),
    };

    state
        .memory
        .upsert_fact(&user_id, fact.clone())
        .await
        .map_err(internal_error)?;
    Ok(Json(fact))
}

async fn delete_user_fact(
    State(state): State<AppState>,
    Path((user_id, key)): Path<(String, String)>,
) -> Result<axum::http::StatusCode, (axum::http::StatusCode, String)> {
    let deleted = state
        .memory
        .delete_fact(&user_id, &key)
        .await
        .map_err(internal_error)?;
    Ok(if deleted {
        axum::http::StatusCode::NO_CONTENT
    } else {
        axum::http::StatusCode::NOT_FOUND
    })
}

async fn list_user_chats(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
    Query(query): Query<LimitQuery>,
) -> Result<Json<Vec<crate::types::ChatMessageRecord>>, (axum::http::StatusCode, String)> {
    let limit = query.limit.unwrap_or(200).clamp(1, 1000);
    let messages = state
        .memory
        .list_chat_messages(&user_id, limit)
        .await
        .map_err(internal_error)?;
    Ok(Json(messages))
}

async fn delete_user_chat_message(
    State(state): State<AppState>,
    Path((user_id, message_id)): Path<(String, String)>,
) -> Result<axum::http::StatusCode, (axum::http::StatusCode, String)> {
    let deleted = state
        .memory
        .delete_chat_message(&user_id, &message_id)
        .await
        .map_err(internal_error)?;
    Ok(if deleted {
        axum::http::StatusCode::NO_CONTENT
    } else {
        axum::http::StatusCode::NOT_FOUND
    })
}

async fn clear_user_chats(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    let deleted = state
        .memory
        .clear_chat_messages(&user_id)
        .await
        .map_err(internal_error)?;
    Ok(Json(serde_json::json!({ "deleted": deleted })))
}

async fn clear_user_tool_calls(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    let deleted = state
        .memory
        .clear_tool_calls(&user_id)
        .await
        .map_err(internal_error)?;
    Ok(Json(serde_json::json!({ "deleted": deleted })))
}

async fn clear_user_planner_decisions(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    let deleted = state
        .memory
        .clear_planner_decisions(&user_id)
        .await
        .map_err(internal_error)?;
    Ok(Json(serde_json::json!({ "deleted": deleted })))
}

async fn list_user_tool_calls(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
    Query(query): Query<LimitQuery>,
) -> Result<Json<Vec<crate::types::ToolCallRecord>>, (axum::http::StatusCode, String)> {
    let limit = query.limit.unwrap_or(150).clamp(1, 1000);
    let calls = state
        .memory
        .list_tool_calls(&user_id, limit)
        .await
        .map_err(internal_error)?;
    Ok(Json(calls))
}

async fn list_user_planner_decisions(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
    Query(query): Query<LimitQuery>,
) -> Result<Json<Vec<crate::types::PlannerDecisionRecord>>, (axum::http::StatusCode, String)> {
    let limit = query.limit.unwrap_or(200).clamp(1, 1000);
    let decisions = state
        .memory
        .list_planner_decisions(&user_id, limit)
        .await
        .map_err(internal_error)?;
    Ok(Json(decisions))
}

async fn send_dashboard_message(
    State(state): State<AppState>,
    Path(user_id): Path<String>,
    Json(request): Json<DashboardChatRequest>,
) -> Result<Json<OrchestratorReply>, (axum::http::StatusCode, String)> {
    let DashboardChatRequest {
        content,
        guild_id,
        channel_id,
        system_prompt,
    } = request;
    let message = MessageCtx {
        message_id: format!("dashboard-{}", Utc::now().timestamp_millis()),
        user_id,
        guild_id,
        channel_id,
        content,
        timestamp: Utc::now(),
    };

    let reply = state
        .orchestrator
        .handle_message_with_system_prompt_override(message, system_prompt)
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

const DASHBOARD_HTML: &str = r#"<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>CompanionPilot Control Deck</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #081620;
      --bg-2: #102837;
      --panel: rgba(15, 30, 45, 0.92);
      --panel-bright: rgba(27, 47, 66, 0.95);
      --text: #e8f3ff;
      --muted: #95aec4;
      --accent: #5de2a2;
      --accent-2: #3ea5ff;
      --danger: #ff6f7d;
      --border: rgba(140, 180, 210, 0.3);
    }

    * { box-sizing: border-box; }

    html, body {
      height: 100%;
      overflow: hidden;
      margin: 0;
    }

    body {
      font-family: "Space Grotesk", "Trebuchet MS", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at 8% 12%, rgba(62, 165, 255, 0.28), transparent 30%),
        radial-gradient(circle at 95% 4%, rgba(93, 226, 162, 0.2), transparent 34%),
        linear-gradient(145deg, var(--bg) 0%, var(--bg-2) 55%, #071018 100%);
      animation: bgShift 18s ease-in-out infinite;
    }

    @keyframes bgShift {
      0%   { background-position: 0% 0%, 100% 0%, center; }
      50%  { background-position: 5% 8%, 95% 12%, center; }
      100% { background-position: 0% 0%, 100% 0%, center; }
    }

    .shell {
      height: 100%;
      max-width: 1300px;
      margin: 0 auto;
      padding: 14px 22px;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }

    .header {
      padding: 14px 20px;
      border: 1px solid var(--border);
      border-radius: 18px;
      background: linear-gradient(120deg, rgba(20,38,56,.94), rgba(10,23,34,.9));
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      flex-shrink: 0;
    }

    .title {
      margin: 0;
      font-size: clamp(1.1rem, 2.5vw, 1.7rem);
      letter-spacing: 0.02em;
    }

    .subtitle {
      margin: 3px 0 0;
      color: var(--muted);
      font-size: .85rem;
    }

    .header-right {
      display: flex;
      align-items: center;
      gap: 10px;
    }

    .status-badge {
      font-family: "IBM Plex Mono", monospace;
      font-size: .84rem;
      border: 1px solid rgba(93,226,162,.4);
      color: var(--accent);
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(19, 39, 33, .65);
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .status-dot {
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: var(--accent);
      animation: pulse 2s ease-in-out infinite;
      flex-shrink: 0;
    }

    .status-dot.unhealthy {
      background: var(--danger);
      animation: none;
    }

    @keyframes pulse {
      0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(93,226,162,.5); }
      50%       { opacity: 0.75; box-shadow: 0 0 0 5px rgba(93,226,162,0); }
    }

    .refresh-btn {
      width: 34px;
      height: 34px;
      padding: 0;
      border-radius: 50%;
      border: 1px solid var(--border);
      background: rgba(20, 42, 59, .85);
      color: var(--accent-2);
      font-size: 1.15rem;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: filter .15s, transform .25s;
    }
    .refresh-btn:hover { filter: brightness(1.25); transform: rotate(30deg); }

    .grid {
      flex: 1;
      min-height: 0;
      display: grid;
      grid-template-columns: 300px 1fr 360px;
      gap: 14px;
    }

    .panel {
      border: 1px solid var(--border);
      background: var(--panel);
      border-radius: 16px;
      overflow: hidden;
      box-shadow: 0 14px 36px rgba(0,0,0,.2);
      display: flex;
      flex-direction: column;
      min-height: 0;
    }

    .panel-head {
      margin: 0;
      padding: 14px 16px;
      font-size: 1rem;
      font-weight: 700;
      border-bottom: 1px solid var(--border);
      background: rgba(20, 39, 56, .9);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      flex-shrink: 0;
    }

    .badge {
      font-family: "IBM Plex Mono", monospace;
      font-size: .72rem;
      padding: 2px 7px;
      border-radius: 999px;
      background: rgba(62,165,255,.15);
      border: 1px solid rgba(62,165,255,.32);
      color: var(--accent-2);
      font-weight: 600;
    }

    #users {
      flex: 1;
      overflow-y: auto;
    }

    .user-item {
      padding: 12px 14px;
      border-bottom: 1px solid rgba(140,180,210,.12);
      border-left: 3px solid transparent;
      cursor: pointer;
      transition: background .18s ease, border-color .18s ease;
    }
    .user-item:hover { background: rgba(41, 66, 88, 0.35); }
    .user-item.active {
      background: rgba(62, 165, 255, .22);
      border-left-color: var(--accent-2);
    }
    .user-id   { font-size: .9rem; font-family: "IBM Plex Mono", monospace; }
    .user-meta { color: var(--muted); font-size: .78rem; margin-top: 5px; }

    #chat {
      flex: 1;
      overflow-y: auto;
      padding: 14px;
      min-height: 0;
    }

    .memory-list, .tool-list, .planner-list {
      padding: 10px 12px 14px;
      overflow-y: auto;
      flex: 1;
      min-height: 0;
    }

    .stack {
      display: flex;
      flex-direction: column;
      flex: 1;
      min-height: 0;
      overflow: hidden;
    }

    .stack-section {
      display: flex;
      flex-direction: column;
      min-height: 0;
      overflow: hidden;
      border-bottom: 1px solid var(--border);
    }
    .stack-section:last-child { border-bottom: none; }

    .sec-facts    { flex: 1.3; }
    .sec-tools    { flex: 1.7; }
    .sec-planners { flex: 1.0; }

    .subpanel-title {
      margin: 0;
      padding: 8px 12px;
      font-size: .88rem;
      color: var(--accent-2);
      border-bottom: 1px solid rgba(140,180,210,.18);
      background: rgba(18, 35, 50, .8);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      flex-shrink: 0;
    }

    .fact {
      background: var(--panel-bright);
      border: 1px solid rgba(140,180,210,.25);
      border-radius: 12px;
      padding: 10px 12px;
      margin-bottom: 10px;
    }
    .fact-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
      margin-bottom: 4px;
    }
    .fact-key  { color: var(--accent); font-weight: 700; font-size: .9rem; }
    .fact-val  { font-size: .9rem; }
    .fact-meta { color: var(--muted); font-size: .78rem; margin-top: 6px; }

    .fact-form {
      display: grid;
      grid-template-columns: 1fr 1fr auto;
      gap: 8px;
      padding: 10px 12px;
      border-bottom: 1px solid rgba(140,180,210,.18);
      background: rgba(14, 32, 46, .65);
      flex-shrink: 0;
    }
    .fact-form input {
      background: rgba(8, 19, 29, .9);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 8px 9px;
      min-width: 0;
      font-size: .82rem;
    }

    .fact-error {
      color: var(--danger);
      font-size: .82rem;
      padding: 4px 12px;
      flex-shrink: 0;
    }

    #chat-error:empty, .fact-error:empty { display: none; }

    .tool-call {
      background: var(--panel-bright);
      border: 1px solid rgba(140,180,210,.25);
      border-radius: 12px;
      padding: 10px 12px;
      margin-bottom: 10px;
    }
    .tool-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
      margin-bottom: 6px;
      font-size: .84rem;
      font-family: "IBM Plex Mono", monospace;
    }
    .tool-name   { color: var(--accent); font-weight: 600; }
    .tool-source { color: var(--muted); }
    .tool-status {
      font-size: .75rem;
      padding: 2px 8px;
      border-radius: 999px;
      border: 1px solid rgba(140,180,210,.3);
    }
    .tool-status.ok  { color: var(--accent); border-color: rgba(93,226,162,.45); background: rgba(18,58,41,.45); }
    .tool-status.err { color: var(--danger); border-color: rgba(255,111,125,.45); background: rgba(70,24,31,.45); }
    .tool-query {
      font-size: .82rem;
      margin-bottom: 6px;
      color: #d2e9ff;
      line-height: 1.35;
      overflow-wrap: anywhere;
      word-break: break-word;
    }
    .tool-result {
      white-space: pre-wrap;
      font-size: .79rem;
      color: var(--muted);
      line-height: 1.35;
      margin-top: 4px;
      overflow-wrap: anywhere;
      word-break: break-word;
    }
    .tool-result-preview { overflow-wrap: anywhere; word-break: break-word; }
    details.tool-expand           { margin-top: 6px; font-size: .78rem; }
    details.tool-expand > summary { cursor: pointer; color: var(--accent-2); user-select: none; }

    .planner-item {
      background: var(--panel-bright);
      border: 1px solid rgba(140,180,210,.25);
      border-radius: 12px;
      padding: 10px 12px;
      margin-bottom: 10px;
    }
    .planner-head {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      align-items: center;
      margin-bottom: 6px;
      font-family: "IBM Plex Mono", monospace;
      font-size: .8rem;
    }
    .planner-name     { color: var(--accent-2); }
    .planner-decision { color: var(--text); }

    .chat-row { margin-bottom: 10px; display: flex; }
    .chat-row.user      { justify-content: flex-end; }
    .chat-row.assistant { justify-content: flex-start; }
    .chat-content { max-width: 86%; display: grid; gap: 4px; }

    .bubble {
      border-radius: 14px;
      padding: 10px 12px;
      font-size: .9rem;
      line-height: 1.35;
      border: 1px solid rgba(140,180,210,.2);
      background: rgba(32, 57, 78, .72);
      white-space: pre-wrap;
      word-break: break-word;
    }
    .chat-row.user .bubble { background: rgba(62,165,255,.2); border-color: rgba(62,165,255,.35); }

    .bubble-meta {
      color: var(--muted);
      font-size: .72rem;
      font-family: "IBM Plex Mono", monospace;
      padding: 0 2px;
    }

    .bubble.pending {
      border-color: rgba(93,226,162,.35);
      background: rgba(20, 56, 49, .45);
      color: #dff9ee;
      font-family: "IBM Plex Mono", monospace;
      font-size: .82rem;
      white-space: nowrap;
    }

    .dots { display: inline-block; width: 14px; text-align: left; }

    .del-msg-btn { opacity: 0; transition: opacity .15s; }
    .chat-row:hover .del-msg-btn { opacity: 1; }

    .composer {
      border-top: 1px solid var(--border);
      padding: 12px;
      display: grid;
      gap: 8px;
      background: rgba(12, 27, 38, 0.9);
      flex-shrink: 0;
    }
    .composer-actions {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
    }
    .composer-buttons { display: flex; gap: 8px; align-items: center; }
    .send-status {
      font-size: .82rem;
      color: var(--muted);
      font-family: "IBM Plex Mono", monospace;
      white-space: nowrap;
    }
    .send-status.sending { color: var(--accent); }

    textarea, button {
      font: inherit;
      border-radius: 10px;
      border: 1px solid var(--border);
    }

    textarea {
      width: 100%;
      min-height: 68px;
      max-height: 180px;
      background: rgba(8, 19, 29, .9);
      color: var(--text);
      resize: vertical;
      padding: 10px 11px;
    }
    textarea:focus { outline: none; border-color: rgba(62,165,255,.5); }
    .system-prompt-wrap { margin-top: 10px; }
    .system-prompt-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
      margin-bottom: 6px;
    }
    .system-prompt-label {
      display: block;
      margin: 0;
      font-size: .78rem;
      color: var(--muted);
      letter-spacing: .02em;
      text-transform: uppercase;
      font-family: "IBM Plex Mono", monospace;
    }
    .system-prompt-input {
      min-height: 74px;
      max-height: 220px;
      font-family: "IBM Plex Mono", monospace;
      font-size: .86rem;
    }

    button {
      padding: 10px 12px;
      background: linear-gradient(110deg, rgba(93,226,162,.2), rgba(62,165,255,.25));
      color: var(--text);
      cursor: pointer;
      font-weight: 700;
    }
    button:hover { filter: brightness(1.1); }

    .tiny-btn {
      padding: 4px 8px;
      font-size: .74rem;
      border-radius: 8px;
      border: 1px solid rgba(140,180,210,.35);
      background: rgba(20, 42, 59, .85);
      color: var(--text);
      cursor: pointer;
    }
    .tiny-btn.danger { border-color: rgba(255,111,125,.45); background: rgba(73,26,34,.75); color: #ffd8dc; }
    .tiny-btn.warn   { border-color: rgba(255,202,96,.55);  background: rgba(77,56,18,.76);  color: #ffe4a8; }

    .empty { padding: 16px; color: var(--muted); font-size: .9rem; }
    .error { color: var(--danger); font-size: .84rem; padding: 0 2px; }

    /* Slim scrollbars */
    ::-webkit-scrollbar       { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(140,180,210,.25); border-radius: 99px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(140,180,210,.45); }

    @media (max-width: 1100px) {
      html, body { height: auto; overflow: auto; }
      .shell     { height: auto; }
      .grid      { grid-template-columns: 1fr; }
      #chat      { max-height: 50vh; }
      #users     { max-height: 30vh; }
      .memory-list, .tool-list, .planner-list { max-height: 28vh; }
      .stack     { height: auto; }
      .stack-section { flex: none; }
    }
  </style>
</head>
<body>
  <main class="shell">
    <header class="header">
      <div>
        <h1 class="title">CompanionPilot Control Deck</h1>
        <p class="subtitle">Inspect users, memory facts, and full conversation trails.</p>
      </div>
      <div class="header-right">
        <button id="refresh-btn" class="refresh-btn" title="Refresh data">&#x21BB;</button>
        <div class="status-badge">
          <span id="status-dot" class="status-dot"></span>
          <span id="status-text">Checking...</span>
        </div>
      </div>
    </header>

    <section class="grid">
      <article class="panel">
        <div class="panel-head">
          <span>Users</span>
          <span id="user-count" class="badge">0</span>
        </div>
        <div id="users"></div>
      </article>

      <article class="panel">
        <div class="panel-head">
          <span>Chat Timeline</span>
          <span id="chat-count" class="badge">0</span>
        </div>
        <div id="chat"></div>
        <div class="composer">
          <textarea id="prompt" placeholder="Send a test message&#x2026; (Ctrl+Enter to send)"></textarea>
          <div class="system-prompt-wrap">
            <div class="system-prompt-head">
              <label class="system-prompt-label" for="system-prompt">System Prompt (editable)</label>
              <button id="system-prompt-default" type="button" class="tiny-btn">Use Default</button>
            </div>
            <textarea id="system-prompt" class="system-prompt-input" placeholder="Loading default system prompt..."></textarea>
          </div>
          <div class="composer-actions">
            <div class="composer-buttons">
              <button id="send">Send</button>
              <button id="clear-chat" class="tiny-btn warn">Clear History</button>
            </div>
            <div id="send-status" class="send-status">Ready</div>
          </div>
          <div id="chat-error" class="error"></div>
        </div>
      </article>

      <article class="panel">
        <div class="panel-head">Memory &amp; Decisions</div>
        <div class="stack">
          <div class="stack-section sec-facts">
            <div class="subpanel-title">
              <span>Memory Facts</span>
              <span id="facts-count" class="badge">0</span>
            </div>
            <div class="fact-form">
              <input id="fact-key" placeholder="key (e.g. name)" />
              <input id="fact-value" placeholder="value" />
              <button id="fact-save" class="tiny-btn">Save Fact</button>
            </div>
            <div id="fact-error" class="fact-error"></div>
            <div id="facts" class="memory-list"></div>
          </div>
          <div class="stack-section sec-tools">
            <div class="subpanel-title">
              <span>Tool Calls</span>
              <div style="display:flex;align-items:center;gap:6px">
                <button class="tiny-btn warn" id="clear-tools">Clear</button>
                <span id="tools-count" class="badge">0</span>
              </div>
            </div>
            <div id="toolcalls" class="tool-list"></div>
          </div>
          <div class="stack-section sec-planners">
            <div class="subpanel-title">
              <span>Planner Decisions</span>
              <div style="display:flex;align-items:center;gap:6px">
                <button class="tiny-btn warn" id="clear-planners">Clear</button>
                <span id="planners-count" class="badge">0</span>
              </div>
            </div>
            <div id="planners" class="planner-list"></div>
          </div>
        </div>
      </article>
    </section>
  </main>

  <script>
    const usersEl        = document.getElementById("users");
    const factsEl        = document.getElementById("facts");
    const toolCallsEl    = document.getElementById("toolcalls");
    const plannersEl     = document.getElementById("planners");
    const chatEl         = document.getElementById("chat");
    const statusDotEl    = document.getElementById("status-dot");
    const statusTextEl   = document.getElementById("status-text");
    const sendBtn        = document.getElementById("send");
    const clearChatBtn   = document.getElementById("clear-chat");
    const promptEl       = document.getElementById("prompt");
    const systemPromptEl = document.getElementById("system-prompt");
    const systemPromptDefaultBtn = document.getElementById("system-prompt-default");
    const sendStatusEl   = document.getElementById("send-status");
    const chatErrorEl    = document.getElementById("chat-error");
    const factErrorEl    = document.getElementById("fact-error");
    const factKeyEl      = document.getElementById("fact-key");
    const factValueEl    = document.getElementById("fact-value");
    const factSaveBtn    = document.getElementById("fact-save");
    const clearToolsBtn  = document.getElementById("clear-tools");
    const clearPlannersBtn = document.getElementById("clear-planners");
    const refreshBtn     = document.getElementById("refresh-btn");
    const userCountEl    = document.getElementById("user-count");
    const chatCountEl    = document.getElementById("chat-count");
    const factsCountEl   = document.getElementById("facts-count");
    const toolsCountEl   = document.getElementById("tools-count");
    const plannersCountEl = document.getElementById("planners-count");

    let selectedUser   = null;
    let isSending      = false;
    let pendingBubbleId = null;
    const SYSTEM_PROMPT_STORAGE_KEY = "companionpilot.dashboard.systemPrompt";
    let defaultSystemPrompt = [
      "You are CompanionPilot, a helpful Discord AI companion.",
      "Keep replies concise and practical.",
      "Never emit XML/JSON/pseudo tool-call markup in normal replies."
    ].join("\n");

    const fmtDate = (iso) => {
      const date = new Date(iso);
      if (Number.isNaN(date.getTime())) return iso;
      return date.toLocaleString();
    };

    const escapeHtml = (value) => (value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;");

    const queryFromArgs = (argsJson) => {
      try {
        const parsed = JSON.parse(argsJson);
        return parsed.query || argsJson;
      } catch {
        return argsJson;
      }
    };

    const renderExpandableText = (value, maxChars = 320) => {
      const safe = escapeHtml(value || "");
      if (safe.length <= maxChars) {
        return `<div class="tool-result">${safe || "n/a"}</div>`;
      }
      const preview = safe.slice(0, maxChars) + "...";
      return `
        <div class="tool-result tool-result-preview">${preview}</div>
        <details class="tool-expand">
          <summary>Show full result</summary>
          <div class="tool-result">${safe}</div>
        </details>
      `;
    };

    async function expectOk(resp) {
      if (resp.ok) return;
      const text = await resp.text();
      throw new Error(text || `request failed (${resp.status})`);
    }

    let lastSendStatus = "Ready";

    function fmtMs(ms) {
      if (typeof ms !== "number" || !Number.isFinite(ms)) return null;
      if (ms >= 1000) return `${(ms / 1000).toFixed(2)}s`;
      return `${Math.round(ms)}ms`;
    }

    function buildSendTimingStatus(clientMs, timings) {
      const parts = [];
      const clientLabel = fmtMs(clientMs);
      if (clientLabel) parts.push(`Client ${clientLabel}`);

      if (timings && typeof timings.total_ms === "number") {
        parts.push(`Server ${fmtMs(timings.total_ms)}`);
      }
      if (timings && typeof timings.final_model_ms === "number") {
        parts.push(`Model ${fmtMs(timings.final_model_ms)}`);
      }
      if (
        timings &&
        typeof timings.tool_execution_ms === "number" &&
        timings.tool_execution_ms > 0
      ) {
        parts.push(`Tools ${fmtMs(timings.tool_execution_ms)}`);
      }

      return parts.length ? parts.join(" | ") : "Ready";
    }

    function setSendingState(sending) {
      isSending = sending;
      sendBtn.disabled = sending;
      promptEl.disabled = sending;
      systemPromptEl.disabled = sending;
      systemPromptDefaultBtn.disabled = sending;
      sendBtn.textContent = sending ? "Sending..." : "Send";
      sendStatusEl.textContent = sending ? "Waiting for response..." : lastSendStatus;
      sendStatusEl.classList.toggle("sending", sending);
    }

    function normalizedPrompt(value) {
      return (value || "").trim();
    }

    function shouldSendSystemPromptOverride(value) {
      const candidate = normalizedPrompt(value);
      if (!candidate) return false;
      return candidate !== normalizedPrompt(defaultSystemPrompt);
    }

    function persistSystemPrompt() {
      try {
        if (!shouldSendSystemPromptOverride(systemPromptEl.value)) {
          localStorage.removeItem(SYSTEM_PROMPT_STORAGE_KEY);
          return;
        }
        localStorage.setItem(SYSTEM_PROMPT_STORAGE_KEY, systemPromptEl.value);
      } catch {}
    }

    async function initSystemPromptEditor() {
      let savedPrompt = null;
      try {
        savedPrompt = localStorage.getItem(SYSTEM_PROMPT_STORAGE_KEY);
      } catch {}

      try {
        const resp = await fetch("/api/dashboard/default-system-prompt");
        if (resp.ok) {
          const payload = await resp.json();
          if (typeof payload.system_prompt === "string" && payload.system_prompt.trim()) {
            defaultSystemPrompt = payload.system_prompt;
          }
        }
      } catch {}

      if (savedPrompt && normalizedPrompt(savedPrompt)) {
        systemPromptEl.value = savedPrompt;
      } else {
        systemPromptEl.value = defaultSystemPrompt;
      }
      persistSystemPrompt();
    }

    function addOptimisticUserBubble(text) {
      const markup = `
        <div class="chat-row user" id="optimistic-msg">
          <div class="chat-content">
            <div class="bubble">${escapeHtml(text)}</div>
          </div>
        </div>
      `;
      chatEl.insertAdjacentHTML("beforeend", markup);
      chatEl.scrollTop = chatEl.scrollHeight;
    }

    function removeOptimisticUserBubble() {
      const node = document.getElementById("optimistic-msg");
      if (node) node.remove();
    }

    function addPendingAssistantBubble() {
      pendingBubbleId = `pending-${Date.now()}`;
      const markup = `
        <div class="chat-row assistant" id="${pendingBubbleId}">
          <div class="bubble pending">Thinking<span class="dots">...</span></div>
        </div>
      `;
      chatEl.insertAdjacentHTML("beforeend", markup);
      chatEl.scrollTop = chatEl.scrollHeight;
    }

    function removePendingAssistantBubble() {
      if (!pendingBubbleId) return;
      const node = document.getElementById(pendingBubbleId);
      if (node) node.remove();
      pendingBubbleId = null;
    }

    async function checkHealth() {
      try {
        const resp = await fetch("/health");
        const text = await resp.text();
        const healthy = text === "ok";
        statusDotEl.className = "status-dot" + (healthy ? "" : " unhealthy");
        statusTextEl.textContent = healthy ? "System healthy" : "Unhealthy";
      } catch {
        statusDotEl.className = "status-dot unhealthy";
        statusTextEl.textContent = "Health check failed";
      }
    }

    async function loadUsers() {
      const resp  = await fetch("/api/dashboard/users");
      const users = await resp.json();
      usersEl.innerHTML = "";

      if (!users.length) {
        usersEl.innerHTML    = '<div class="empty">No users yet. Send a message via Discord or /chat first.</div>';
        factsEl.innerHTML    = '<div class="empty">No memory facts loaded.</div>';
        toolCallsEl.innerHTML = '<div class="empty">No tool calls recorded.</div>';
        plannersEl.innerHTML  = '<div class="empty">No planner decisions recorded.</div>';
        chatEl.innerHTML     = '<div class="empty">No chat history loaded.</div>';
        selectedUser = null;
        lastSendStatus = "No users";
        sendStatusEl.textContent = "No users";
        sendStatusEl.classList.remove("sending");
        userCountEl.textContent    = "0";
        chatCountEl.textContent    = "0";
        factsCountEl.textContent   = "0";
        toolsCountEl.textContent   = "0";
        plannersCountEl.textContent = "0";
        return;
      }

      userCountEl.textContent = users.length;

      if (!selectedUser) {
        selectedUser = users[0].user_id;
      }

      for (const user of users) {
        const node = document.createElement("div");
        node.className = "user-item" + (selectedUser === user.user_id ? " active" : "");
        node.innerHTML = `
          <div class="user-id">${escapeHtml(user.user_id)}</div>
          <div class="user-meta">${user.message_count} msgs &bull; ${user.fact_count} facts &bull; ${fmtDate(user.last_activity)}</div>
        `;
        node.onclick = async () => {
          selectedUser = user.user_id;
          usersEl.querySelectorAll(".user-item").forEach(el => {
            el.classList.toggle("active", el === node);
          });
          await renderSelectedUser();
          promptEl.focus();
        };
        usersEl.appendChild(node);
      }

      if (!isSending) {
        if (lastSendStatus === "No users") {
          lastSendStatus = "Ready";
        }
        sendStatusEl.textContent = lastSendStatus;
        sendStatusEl.classList.remove("sending");
      }
    }

    async function renderSelectedUser() {
      if (!selectedUser) return;

      const [factsResp, chatsResp, toolCallsResp, plannersResp] = await Promise.all([
        fetch(`/api/dashboard/users/${encodeURIComponent(selectedUser)}/facts?limit=200`),
        fetch(`/api/dashboard/users/${encodeURIComponent(selectedUser)}/chats?limit=300`),
        fetch(`/api/dashboard/users/${encodeURIComponent(selectedUser)}/toolcalls?limit=200`),
        fetch(`/api/dashboard/users/${encodeURIComponent(selectedUser)}/planners?limit=250`)
      ]);

      const facts            = await factsResp.json();
      const chats            = await chatsResp.json();
      const toolCalls        = await toolCallsResp.json();
      const plannerDecisions = await plannersResp.json();

      factsCountEl.textContent    = facts.length;
      chatCountEl.textContent     = chats.length;
      toolsCountEl.textContent    = toolCalls.length;
      plannersCountEl.textContent = plannerDecisions.length;

      factsEl.innerHTML = facts.length
        ? facts.map((f) => `
            <div class="fact">
              <div class="fact-head">
                <div class="fact-key">${escapeHtml(f.key)}</div>
                <button class="tiny-btn danger" onclick="deleteFact('${encodeURIComponent(f.key)}')">Delete</button>
              </div>
              <div class="fact-val">${escapeHtml(f.value)}</div>
              <div class="fact-meta">confidence=${f.confidence.toFixed(2)} &bull; ${fmtDate(f.updated_at)}</div>
            </div>
          `).join("")
        : '<div class="empty">No memory facts for this user.</div>';

      chatEl.innerHTML = chats.length
        ? chats.map((m) => `
            <div class="chat-row ${m.role}">
              <div class="chat-content">
                <div class="bubble">${escapeHtml(m.content)}</div>
                <div class="bubble-meta">${m.role} &bull; ${m.guild_id}/${m.channel_id} &bull; ${fmtDate(m.timestamp)}</div>
                <div>
                  <button class="tiny-btn danger del-msg-btn" onclick="deleteChatMessage('${encodeURIComponent(m.id)}')">Delete</button>
                </div>
              </div>
            </div>
          `).join("")
        : '<div class="empty">No chat messages for this user.</div>';

      toolCallsEl.innerHTML = toolCalls.length
        ? toolCalls.map((call) => `
            <div class="tool-call">
              <div class="tool-head">
                <span class="tool-name">${escapeHtml(call.tool_name)}</span>
                <span class="tool-source">${escapeHtml(call.source)}</span>
                <span class="tool-status ${call.success ? "ok" : "err"}">${call.success ? "success" : "error"}</span>
              </div>
              <div class="tool-query">query: ${escapeHtml(queryFromArgs(call.args_json))}</div>
              ${renderExpandableText(call.success ? call.result_text : (call.error || "unknown tool error"))}
              <div class="fact-meta">${fmtDate(call.timestamp)} &bull; ${call.citations.length} citations</div>
            </div>
          `).join("")
        : '<div class="empty">No tool calls for this user.</div>';

      plannersEl.innerHTML = plannerDecisions.length
        ? plannerDecisions.map((entry) => `
            <div class="planner-item">
              <div class="planner-head">
                <span class="planner-name">${escapeHtml(entry.planner)}</span>
                <span class="planner-decision">${escapeHtml(entry.decision)}</span>
                <span class="tool-status ${entry.success ? "ok" : "err"}">${entry.success ? "ok" : "error"}</span>
              </div>
              <div class="tool-query">reason: ${escapeHtml(entry.rationale)}</div>
              ${renderExpandableText(entry.payload_json, 220)}
              ${entry.error ? `<div class="error">${escapeHtml(entry.error)}</div>` : ""}
              <div class="fact-meta">${fmtDate(entry.timestamp)}</div>
            </div>
          `).join("")
        : '<div class="empty">No planner decisions for this user.</div>';

      chatEl.scrollTop = chatEl.scrollHeight;
    }

    sendBtn.onclick = async () => {
      if (isSending) return;
      chatErrorEl.textContent = "";
      const content = promptEl.value.trim();
      if (!selectedUser) {
        chatErrorEl.textContent = "Select a user first.";
        return;
      }
      if (!content) {
        chatErrorEl.textContent = "Message cannot be empty.";
        return;
      }

      setSendingState(true);
      promptEl.value = "";
      addOptimisticUserBubble(content);
      addPendingAssistantBubble();
      const requestStartedAt = performance.now();
      try {
        const systemPrompt = systemPromptEl.value.trim();
        const payload = { content };
        if (shouldSendSystemPromptOverride(systemPrompt)) {
          payload.system_prompt = systemPrompt;
        }
        const resp = await fetch(`/api/dashboard/users/${encodeURIComponent(selectedUser)}/chat`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify(payload)
        });
        await expectOk(resp);
        const reply = await resp.json();
        const clientMs = performance.now() - requestStartedAt;
        lastSendStatus = buildSendTimingStatus(clientMs, reply.timings);
        removeOptimisticUserBubble();
        removePendingAssistantBubble();
        await renderSelectedUser();
        await loadUsers();
      } catch (error) {
        removeOptimisticUserBubble();
        removePendingAssistantBubble();
        chatErrorEl.textContent = `Send failed: ${error.message || error}`;
        lastSendStatus = "Ready";
        promptEl.value = content;
      } finally {
        setSendingState(false);
      }
    };

    async function saveFact() {
      factErrorEl.textContent = "";
      if (!selectedUser) {
        factErrorEl.textContent = "Select a user first.";
        return;
      }
      const key   = factKeyEl.value.trim();
      const value = factValueEl.value.trim();
      if (!key || !value) {
        factErrorEl.textContent = "Fact key and value are required.";
        return;
      }

      factSaveBtn.disabled = true;
      try {
        const resp = await fetch(`/api/dashboard/users/${encodeURIComponent(selectedUser)}/facts`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ key, value, confidence: 0.98 })
        });
        await expectOk(resp);
        factKeyEl.value   = "";
        factValueEl.value = "";
        await renderSelectedUser();
        factKeyEl.focus();
      } catch (error) {
        factErrorEl.textContent = `Save fact failed: ${error.message || error}`;
      } finally {
        factSaveBtn.disabled = false;
      }
    }

    async function deleteFact(encodedKey) {
      if (!selectedUser) return;
      const key = decodeURIComponent(encodedKey);
      if (!confirm(`Delete fact '${key}'?`)) return;

      try {
        const resp = await fetch(`/api/dashboard/users/${encodeURIComponent(selectedUser)}/facts/${encodeURIComponent(key)}`, {
          method: "DELETE"
        });
        await expectOk(resp);
        await renderSelectedUser();
      } catch (error) {
        factErrorEl.textContent = `Delete fact failed: ${error.message || error}`;
      }
    }

    async function deleteChatMessage(encodedMessageId) {
      if (!selectedUser) return;
      const messageId = decodeURIComponent(encodedMessageId);
      if (!confirm("Delete this conversation message?")) return;

      try {
        const resp = await fetch(`/api/dashboard/users/${encodeURIComponent(selectedUser)}/chats/${encodeURIComponent(messageId)}`, {
          method: "DELETE"
        });
        await expectOk(resp);
        await renderSelectedUser();
        await loadUsers();
      } catch (error) {
        chatErrorEl.textContent = `Delete message failed: ${error.message || error}`;
      }
    }

    async function clearConversation() {
      if (!selectedUser) return;
      if (!confirm("Clear full conversation history for this user?")) return;

      clearChatBtn.disabled = true;
      try {
        const resp = await fetch(`/api/dashboard/users/${encodeURIComponent(selectedUser)}/chats`, {
          method: "DELETE"
        });
        await expectOk(resp);
        await renderSelectedUser();
        await loadUsers();
      } catch (error) {
        chatErrorEl.textContent = `Clear conversation failed: ${error.message || error}`;
      } finally {
        clearChatBtn.disabled = false;
      }
    }

    async function clearToolCalls() {
      if (!selectedUser) return;
      if (!confirm("Clear all tool call logs for this user?")) return;

      clearToolsBtn.disabled = true;
      try {
        const resp = await fetch(`/api/dashboard/users/${encodeURIComponent(selectedUser)}/toolcalls`, {
          method: "DELETE"
        });
        await expectOk(resp);
        await renderSelectedUser();
        await loadUsers();
      } catch (error) {
        chatErrorEl.textContent = `Clear tool calls failed: ${error.message || error}`;
      } finally {
        clearToolsBtn.disabled = false;
      }
    }

    async function clearPlannerDecisions() {
      if (!selectedUser) return;
      if (!confirm("Clear all planner decision logs for this user?")) return;

      clearPlannersBtn.disabled = true;
      try {
        const resp = await fetch(`/api/dashboard/users/${encodeURIComponent(selectedUser)}/planners`, {
          method: "DELETE"
        });
        await expectOk(resp);
        await renderSelectedUser();
        await loadUsers();
      } catch (error) {
        chatErrorEl.textContent = `Clear planner decisions failed: ${error.message || error}`;
      } finally {
        clearPlannersBtn.disabled = false;
      }
    }

    window.deleteFact        = deleteFact;
    window.deleteChatMessage = deleteChatMessage;

    refreshBtn.onclick = async () => {
      await loadUsers();
      await renderSelectedUser();
    };

    promptEl.addEventListener("keydown", (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
        event.preventDefault();
        sendBtn.click();
      }
    });

    factValueEl.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        saveFact();
      }
    });

    factSaveBtn.addEventListener("click", saveFact);
    clearChatBtn.addEventListener("click", clearConversation);
    clearToolsBtn.addEventListener("click", clearToolCalls);
    clearPlannersBtn.addEventListener("click", clearPlannerDecisions);
    systemPromptDefaultBtn.addEventListener("click", () => {
      systemPromptEl.value = defaultSystemPrompt;
      persistSystemPrompt();
    });
    systemPromptEl.addEventListener("input", persistSystemPrompt);

    (async function bootstrap() {
      await checkHealth();
      await initSystemPromptEditor();
      await loadUsers();
      await renderSelectedUser();
      setInterval(checkHealth, 30000);
    })();
  </script>
</body>
</html>"#;
