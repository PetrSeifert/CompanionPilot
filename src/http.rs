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
        .route("/api/dashboard/users/{user_id}/chats", get(list_user_chats))
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

    .memory-list, .chat-list {
      padding: 10px 12px 14px;
      max-height: 42vh;
      overflow: auto;
    }

    .fact {
      background: var(--panel-bright);
      border: 1px solid rgba(140,180,210,.25);
      border-radius: 12px;
      padding: 10px 12px;
      margin-bottom: 10px;
    }
    .fact-key {
      color: var(--accent);
      font-weight: 700;
      font-size: .9rem;
      margin-bottom: 3px;
    }
    .fact-val { font-size: .9rem; }
    .fact-meta { color: var(--muted); font-size: .78rem; margin-top: 6px; }

    .chat-row {
      margin-bottom: 10px;
      display: flex;
    }
    .chat-row.user { justify-content: flex-end; }
    .bubble {
      max-width: 86%;
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

    .composer {
      border-top: 1px solid var(--border);
      padding: 12px;
      display: grid;
      gap: 8px;
      background: rgba(12, 27, 38, 0.9);
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
          <button id="send">Send to CompanionPilot</button>
          <div id="chat-error" class="error"></div>
        </div>
      </article>

      <article class="panel">
        <h2>Memory Facts</h2>
        <div id="facts" class="memory-list"></div>
      </article>
    </section>
  </main>

  <script>
    const usersEl = document.getElementById("users");
    const factsEl = document.getElementById("facts");
    const chatEl = document.getElementById("chat");
    const statusEl = document.getElementById("status");
    const sendBtn = document.getElementById("send");
    const promptEl = document.getElementById("prompt");
    const chatErrorEl = document.getElementById("chat-error");
    let selectedUser = null;

    const fmtDate = (iso) => {
      const date = new Date(iso);
      if (Number.isNaN(date.getTime())) return iso;
      return date.toLocaleString();
    };

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
        chatEl.innerHTML = '<div class="empty">No chat history loaded.</div>';
        selectedUser = null;
        return;
      }

      for (const user of users) {
        const node = document.createElement("div");
        node.className = "user-item" + (selectedUser === user.user_id ? " active" : "");
        node.innerHTML = `
          <div class="user-id">${user.user_id}</div>
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
      await renderSelectedUser();
    }

    async function renderSelectedUser() {
      if (!selectedUser) return;

      const [factsResp, chatsResp] = await Promise.all([
        fetch(`/api/dashboard/users/${encodeURIComponent(selectedUser)}/facts?limit=200`),
        fetch(`/api/dashboard/users/${encodeURIComponent(selectedUser)}/chats?limit=300`)
      ]);

      const facts = await factsResp.json();
      const chats = await chatsResp.json();

      factsEl.innerHTML = facts.length
        ? facts.map((f) => `
            <div class="fact">
              <div class="fact-key">${f.key}</div>
              <div class="fact-val">${f.value}</div>
              <div class="fact-meta">confidence=${f.confidence.toFixed(2)} • ${fmtDate(f.updated_at)}</div>
            </div>
          `).join("")
        : '<div class="empty">No memory facts for this user.</div>';

      chatEl.innerHTML = chats.length
        ? chats.map((m) => `
            <div class="chat-row ${m.role}">
              <div class="bubble">
                ${m.content}
                <small>${m.role} • ${m.guild_id}/${m.channel_id} • ${fmtDate(m.timestamp)}</small>
              </div>
            </div>
          `).join("")
        : '<div class="empty">No chat messages for this user.</div>';

      chatEl.scrollTop = chatEl.scrollHeight;
    }

    sendBtn.onclick = async () => {
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

      sendBtn.disabled = true;
      try {
        const resp = await fetch(`/api/dashboard/users/${encodeURIComponent(selectedUser)}/chat`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ content })
        });
        if (!resp.ok) {
          const text = await resp.text();
          throw new Error(text || "request failed");
        }
        promptEl.value = "";
        await renderSelectedUser();
        await loadUsers();
      } catch (error) {
        chatErrorEl.textContent = `Send failed: ${error.message || error}`;
      } finally {
        sendBtn.disabled = false;
      }
    };

    (async function bootstrap() {
      await checkHealth();
      await loadUsers();
      setInterval(checkHealth, 30000);
    })();
  </script>
</body>
</html>"#;
