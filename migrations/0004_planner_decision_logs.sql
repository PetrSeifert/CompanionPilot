CREATE TABLE IF NOT EXISTS planner_decision_logs (
    id BIGSERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    guild_id TEXT NOT NULL,
    channel_id TEXT NOT NULL,
    planner TEXT NOT NULL,
    decision TEXT NOT NULL,
    rationale TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    success BOOLEAN NOT NULL,
    error TEXT,
    timestamp TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_planner_decision_logs_user_time
    ON planner_decision_logs (user_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_planner_decision_logs_planner_time
    ON planner_decision_logs (planner, timestamp DESC);
