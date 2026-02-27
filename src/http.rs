use std::sync::Arc;

use axum::{
    Json, Router,
    extract::{Path, Query, State},
    response::Html,
    routing::{get, post},
};
use chrono::Utc;
use serde::Deserialize;
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
}

#[derive(Debug, Deserialize)]
struct UpsertFactRequest {
    key: String,
    value: String,
    confidence: Option<f32>,
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
            "/api/dashboard/users/{user_id}/planners",
            get(list_user_planner_decisions),
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
    let message = MessageCtx {
        message_id: format!("dashboard-{}", Utc::now().timestamp_millis()),
        user_id,
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

    body {
      margin: 0;
      min-height: 100vh;
      font-family: "Space Grotesk", "Trebuchet MS", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at 8% 12%, rgba(62, 165, 255, 0.28), transparent 30%),
        radial-gradient(circle at 95% 4%, rgba(93, 226, 162, 0.2), transparent 34%),
        linear-gradient(145deg, var(--bg) 0%, var(--bg-2) 55%, #071018 100%);
      animation: bgShift 18s ease-in-out infinite;
    }

    @keyframes bgShift {
      0% { background-position: 0% 0%, 100% 0%, center; }
      50% { background-position: 5% 8%, 95% 12%, center; }
      100% { background-position: 0% 0%, 100% 0%, center; }
    }

    .shell {
      max-width: 1300px;
      margin: 0 auto;
      padding: 28px 22px 30px;
      display: grid;
      gap: 14px;
    }

    .header {
      padding: 18px 20px;
      border: 1px solid var(--border);
      border-radius: 18px;
      background: linear-gradient(120deg, rgba(20,38,56,.94), rgba(10,23,34,.9));
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
    }

    .title {
      margin: 0;
      font-size: clamp(1.4rem, 3vw, 2.1rem);
      letter-spacing: 0.02em;
    }

    .subtitle {
      margin: 3px 0 0;
      color: var(--muted);
      font-size: .95rem;
    }

    .status {
      font-family: "IBM Plex Mono", monospace;
      font-size: .84rem;
      border: 1px solid rgba(93,226,162,.4);
      color: var(--accent);
      padding: 8px 10px;
      border-radius: 999px;
      background: rgba(19, 39, 33, .65);
    }

    .grid {
      display: grid;
      grid-template-columns: 300px 1fr 360px;
      gap: 14px;
    }

    .panel {
      border: 1px solid var(--border);
      background: var(--panel);
      border-radius: 16px;
      overflow: hidden;
      min-height: 300px;
      box-shadow: 0 14px 36px rgba(0,0,0,.2);
    }

    .panel h2 {
      margin: 0;
      padding: 14px 16px;
      font-size: 1rem;
      border-bottom: 1px solid var(--border);
      background: rgba(20, 39, 56, .9);
    }

    .users {
      max-height: 72vh;
      overflow: auto;
    }

    .user-item {
      padding: 12px 14px;
      border-bottom: 1px solid rgba(140,180,210,.12);
      cursor: pointer;
      transition: .18s background ease;
    }

    .user-item:hover { background: rgba(41, 66, 88, 0.35); }
    .user-item.active { background: rgba(62, 165, 255, .26); }
    .user-id { font-size: .9rem; font-family: "IBM Plex Mono", monospace; }
    .user-meta { color: var(--muted); font-size: .78rem; margin-top: 5px; }

    .memory-list, .chat-list, .tool-list, .planner-list {
      padding: 10px 12px 14px;
      max-height: none;
      min-height: 0;
      overflow: auto;
    }

    .stack {
      display: grid;
      grid-template-rows: minmax(0, 1fr) minmax(0, 1fr) minmax(0, 1fr);
      gap: 0;
      height: 100%;
    }
    .stack > div {
      min-height: 0;
      display: flex;
      flex-direction: column;
    }

    .subpanel-title {
      margin: 0;
      padding: 10px 12px;
      font-size: .9rem;
      color: var(--accent-2);
      border-top: 1px solid var(--border);
      border-bottom: 1px solid rgba(140,180,210,.18);
      background: rgba(18, 35, 50, .8);
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
    .fact-key {
      color: var(--accent);
      font-weight: 700;
      font-size: .9rem;
    }
    .fact-val { font-size: .9rem; }
    .fact-meta { color: var(--muted); font-size: .78rem; margin-top: 6px; }
    .fact-form {
      display: grid;
      grid-template-columns: 1fr 1fr auto;
      gap: 8px;
      padding: 10px 12px;
      border-bottom: 1px solid rgba(140,180,210,.18);
      background: rgba(14, 32, 46, .65);
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
    .tool-name { color: var(--accent); font-weight: 600; }
    .tool-source { color: var(--muted); }
    .tool-status {
      font-size: .75rem;
      padding: 2px 8px;
      border-radius: 999px;
      border: 1px solid rgba(140,180,210,.3);
    }
    .tool-status.ok {
      color: var(--accent);
      border-color: rgba(93,226,162,.45);
      background: rgba(18, 58, 41, .45);
    }
    .tool-status.err {
      color: var(--danger);
      border-color: rgba(255,111,125,.45);
      background: rgba(70, 24, 31, .45);
    }
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
    .tool-result-preview {
      overflow-wrap: anywhere;
      word-break: break-word;
    }
    details.tool-expand {
      margin-top: 6px;
      font-size: .78rem;
    }
    details.tool-expand > summary {
      cursor: pointer;
      color: var(--accent-2);
      user-select: none;
    }

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
    .planner-name { color: var(--accent-2); }
    .planner-decision { color: var(--text); }

    .chat-row {
      margin-bottom: 10px;
      display: flex;
    }
    .chat-row.user { justify-content: flex-end; }
    .chat-row.assistant { justify-content: flex-start; }
    .chat-content {
      max-width: 86%;
      display: grid;
      gap: 4px;
    }
    .bubble {
      border-radius: 14px;
      padding: 10px 12px;
      font-size: .9rem;
      line-height: 1.35;
      border: 1px solid rgba(140,180,210,.2);
      background: rgba(32, 57, 78, .72);
    }
    .chat-row.user .bubble {
      background: rgba(62,165,255,.2);
      border-color: rgba(62,165,255,.35);
    }
    .bubble small {
      display: block;
      margin-top: 6px;
      color: var(--muted);
      font-size: .72rem;
      font-family: "IBM Plex Mono", monospace;
    }
    .bubble.pending {
      border-color: rgba(93,226,162,.35);
      background: rgba(20, 56, 49, .45);
      color: #dff9ee;
      font-family: "IBM Plex Mono", monospace;
      font-size: .82rem;
    }
    .dots {
      display: inline-block;
      width: 14px;
      text-align: left;
    }

    .composer {
      border-top: 1px solid var(--border);
      padding: 12px;
      display: grid;
      gap: 8px;
      background: rgba(12, 27, 38, 0.9);
    }
    .composer-actions {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
    }
    .composer-buttons {
      display: flex;
      gap: 8px;
      align-items: center;
    }
    .send-status {
      font-size: .82rem;
      color: var(--muted);
      font-family: "IBM Plex Mono", monospace;
      white-space: nowrap;
    }
    .send-status.sending {
      color: var(--accent);
    }

    textarea, button {
      font: inherit;
      border-radius: 10px;
      border: 1px solid var(--border);
    }

    textarea {
      width: 100%;
      min-height: 86px;
      background: rgba(8, 19, 29, .9);
      color: var(--text);
      resize: vertical;
      padding: 10px 11px;
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
    .tiny-btn.danger {
      border-color: rgba(255,111,125,.45);
      background: rgba(73, 26, 34, .75);
      color: #ffd8dc;
    }
    .tiny-btn.warn {
      border-color: rgba(255,202,96,.55);
      background: rgba(77, 56, 18, .76);
      color: #ffe4a8;
    }

    .empty {
      padding: 16px;
      color: var(--muted);
      font-size: .9rem;
    }

    .error { color: var(--danger); font-size: .84rem; padding: 0 2px; }

    @media (max-width: 1100px) {
      .grid { grid-template-columns: 1fr; }
      .users { max-height: 30vh; }
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
      <div id="status" class="status">Checking health...</div>
    </header>

    <section class="grid">
      <article class="panel">
        <h2>Users</h2>
        <div id="users" class="users"></div>
      </article>

      <article class="panel">
        <h2>Chat Timeline</h2>
        <div id="chat" class="chat-list"></div>
        <div class="composer">
          <textarea id="prompt" placeholder="Send a test message as selected user..."></textarea>
          <div class="composer-actions">
            <div class="composer-buttons">
              <button id="send">Send to CompanionPilot</button>
              <button id="clear-chat" class="tiny-btn warn">Clear Conversation</button>
            </div>
            <div id="send-status" class="send-status">Ready</div>
          </div>
          <div id="chat-error" class="error"></div>
        </div>
      </article>

      <article class="panel">
        <h2>Memory + Tool Calls</h2>
        <div class="stack">
          <div>
            <h3 class="subpanel-title">Memory Facts</h3>
            <div class="fact-form">
              <input id="fact-key" placeholder="key (e.g. name)" />
              <input id="fact-value" placeholder="value" />
              <button id="fact-save" class="tiny-btn">Save Fact</button>
            </div>
            <div id="facts" class="memory-list"></div>
          </div>
          <div>
            <h3 class="subpanel-title">Tool Calls</h3>
            <div id="toolcalls" class="tool-list"></div>
          </div>
          <div>
            <h3 class="subpanel-title">Planner Decisions</h3>
            <div id="planners" class="planner-list"></div>
          </div>
        </div>
      </article>
    </section>
  </main>

  <script>
    const usersEl = document.getElementById("users");
    const factsEl = document.getElementById("facts");
    const toolCallsEl = document.getElementById("toolcalls");
    const plannersEl = document.getElementById("planners");
    const chatEl = document.getElementById("chat");
    const statusEl = document.getElementById("status");
    const sendBtn = document.getElementById("send");
    const clearChatBtn = document.getElementById("clear-chat");
    const promptEl = document.getElementById("prompt");
    const sendStatusEl = document.getElementById("send-status");
    const chatErrorEl = document.getElementById("chat-error");
    const factKeyEl = document.getElementById("fact-key");
    const factValueEl = document.getElementById("fact-value");
    const factSaveBtn = document.getElementById("fact-save");
    let selectedUser = null;
    let isSending = false;
    let pendingBubbleId = null;

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

    function setSendingState(sending) {
      isSending = sending;
      sendBtn.disabled = sending;
      promptEl.disabled = sending;
      sendBtn.textContent = sending ? "Sending..." : "Send to CompanionPilot";
      sendStatusEl.textContent = sending ? "Waiting for response..." : "Ready";
      sendStatusEl.classList.toggle("sending", sending);
    }

    function addPendingAssistantBubble() {
      pendingBubbleId = `pending-${Date.now()}`;
      const markup = `
        <div class="chat-row assistant" id="${pendingBubbleId}">
          <div class="bubble pending">CompanionPilot is thinking<span class="dots">...</span></div>
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
        statusEl.textContent = text === "ok" ? "System healthy" : "Unhealthy";
      } catch {
        statusEl.textContent = "Health check failed";
      }
    }

    async function loadUsers() {
      const resp = await fetch("/api/dashboard/users");
      const users = await resp.json();
      usersEl.innerHTML = "";

      if (!users.length) {
        usersEl.innerHTML = '<div class="empty">No users yet. Send a message via Discord or /chat first.</div>';
        factsEl.innerHTML = '<div class="empty">No memory facts loaded.</div>';
        toolCallsEl.innerHTML = '<div class="empty">No tool calls recorded.</div>';
        plannersEl.innerHTML = '<div class="empty">No planner decisions recorded.</div>';
        chatEl.innerHTML = '<div class="empty">No chat history loaded.</div>';
        selectedUser = null;
        sendStatusEl.textContent = "No users";
        sendStatusEl.classList.remove("sending");
        return;
      }

      for (const user of users) {
        const node = document.createElement("div");
        node.className = "user-item" + (selectedUser === user.user_id ? " active" : "");
        node.innerHTML = `
          <div class="user-id">${escapeHtml(user.user_id)}</div>
          <div class="user-meta">${user.message_count} msgs • ${user.fact_count} facts • ${fmtDate(user.last_activity)}</div>
        `;
        node.onclick = async () => {
          selectedUser = user.user_id;
          await renderSelectedUser();
          await loadUsers();
        };
        usersEl.appendChild(node);
      }

      if (!selectedUser) {
        selectedUser = users[0].user_id;
      }
      if (!isSending) {
        sendStatusEl.textContent = "Ready";
        sendStatusEl.classList.remove("sending");
      }
      await renderSelectedUser();
    }

    async function renderSelectedUser() {
      if (!selectedUser) return;

      const [factsResp, chatsResp, toolCallsResp, plannersResp] = await Promise.all([
        fetch(`/api/dashboard/users/${encodeURIComponent(selectedUser)}/facts?limit=200`),
        fetch(`/api/dashboard/users/${encodeURIComponent(selectedUser)}/chats?limit=300`),
        fetch(`/api/dashboard/users/${encodeURIComponent(selectedUser)}/toolcalls?limit=200`),
        fetch(`/api/dashboard/users/${encodeURIComponent(selectedUser)}/planners?limit=250`)
      ]);

      const facts = await factsResp.json();
      const chats = await chatsResp.json();
      const toolCalls = await toolCallsResp.json();
      const plannerDecisions = await plannersResp.json();

      factsEl.innerHTML = facts.length
        ? facts.map((f) => `
            <div class="fact">
              <div class="fact-head">
                <div class="fact-key">${escapeHtml(f.key)}</div>
                <button class="tiny-btn danger" onclick="deleteFact('${encodeURIComponent(f.key)}')">Delete</button>
              </div>
              <div class="fact-val">${escapeHtml(f.value)}</div>
              <div class="fact-meta">confidence=${f.confidence.toFixed(2)} • ${fmtDate(f.updated_at)}</div>
            </div>
          `).join("")
        : '<div class="empty">No memory facts for this user.</div>';

      chatEl.innerHTML = chats.length
        ? chats.map((m) => `
            <div class="chat-row ${m.role}">
              <div class="chat-content">
                <div class="bubble">
                  ${escapeHtml(m.content)}
                  <small>${m.role} • ${m.guild_id}/${m.channel_id} • ${fmtDate(m.timestamp)}</small>
                </div>
                <div>
                  <button class="tiny-btn danger" onclick="deleteChatMessage('${encodeURIComponent(m.id)}')">Delete</button>
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
              <div class="fact-meta">${fmtDate(call.timestamp)} • ${call.citations.length} citations</div>
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
      if (isSending) {
        return;
      }
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
      addPendingAssistantBubble();
      try {
        const resp = await fetch(`/api/dashboard/users/${encodeURIComponent(selectedUser)}/chat`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ content })
        });
        await expectOk(resp);
        promptEl.value = "";
        removePendingAssistantBubble();
        await renderSelectedUser();
        await loadUsers();
      } catch (error) {
        removePendingAssistantBubble();
        chatErrorEl.textContent = `Send failed: ${error.message || error}`;
      } finally {
        setSendingState(false);
      }
    };

    async function saveFact() {
      chatErrorEl.textContent = "";
      if (!selectedUser) {
        chatErrorEl.textContent = "Select a user first.";
        return;
      }
      const key = factKeyEl.value.trim();
      const value = factValueEl.value.trim();
      if (!key || !value) {
        chatErrorEl.textContent = "Fact key and value are required.";
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
        factKeyEl.value = "";
        factValueEl.value = "";
        await renderSelectedUser();
      } catch (error) {
        chatErrorEl.textContent = `Save fact failed: ${error.message || error}`;
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
        chatErrorEl.textContent = `Delete fact failed: ${error.message || error}`;
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

    window.deleteFact = deleteFact;
    window.deleteChatMessage = deleteChatMessage;

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

    (async function bootstrap() {
      await checkHealth();
      await loadUsers();
      setInterval(checkHealth, 30000);
    })();
  </script>
</body>
</html>"#;
