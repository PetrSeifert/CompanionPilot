ALTER TABLE chat_messages
    ADD COLUMN IF NOT EXISTS message_id TEXT;

UPDATE chat_messages
SET message_id = id::text
WHERE message_id IS NULL OR message_id = '';

CREATE INDEX IF NOT EXISTS idx_chat_messages_user_message
    ON chat_messages (user_id, message_id);

ALTER TABLE tool_call_logs
    ADD COLUMN IF NOT EXISTS message_id TEXT;

ALTER TABLE tool_call_logs
    ADD COLUMN IF NOT EXISTS duration_ms BIGINT NOT NULL DEFAULT 0;

UPDATE tool_call_logs
SET message_id = ''
WHERE message_id IS NULL;

CREATE INDEX IF NOT EXISTS idx_tool_call_logs_user_message_time
    ON tool_call_logs (user_id, message_id, timestamp DESC);

ALTER TABLE planner_decision_logs
    ADD COLUMN IF NOT EXISTS message_id TEXT;

ALTER TABLE planner_decision_logs
    ADD COLUMN IF NOT EXISTS duration_ms BIGINT NOT NULL DEFAULT 0;

UPDATE planner_decision_logs
SET message_id = ''
WHERE message_id IS NULL;

CREATE INDEX IF NOT EXISTS idx_planner_decision_logs_user_message_time
    ON planner_decision_logs (user_id, message_id, timestamp DESC);

CREATE TABLE IF NOT EXISTS message_latency_logs (
    id BIGSERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    guild_id TEXT NOT NULL,
    channel_id TEXT NOT NULL,
    message_id TEXT NOT NULL,
    stt_ms BIGINT,
    tts_ms BIGINT,
    final_response_ms BIGINT NOT NULL DEFAULT 0,
    decision_ms BIGINT NOT NULL DEFAULT 0,
    tool_call_ms BIGINT NOT NULL DEFAULT 0,
    timestamp TIMESTAMPTZ NOT NULL,
    UNIQUE (user_id, guild_id, channel_id, message_id)
);

CREATE INDEX IF NOT EXISTS idx_message_latency_logs_user_time
    ON message_latency_logs (user_id, timestamp DESC);
