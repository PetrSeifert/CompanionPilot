CREATE TABLE IF NOT EXISTS tool_call_logs (
    id BIGSERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    guild_id TEXT NOT NULL,
    channel_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    source TEXT NOT NULL,
    args_json TEXT NOT NULL,
    result_text TEXT NOT NULL,
    citations_text TEXT NOT NULL DEFAULT '',
    success BOOLEAN NOT NULL,
    error TEXT,
    timestamp TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tool_call_logs_user_time
    ON tool_call_logs (user_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_tool_call_logs_tool_time
    ON tool_call_logs (tool_name, timestamp DESC);
