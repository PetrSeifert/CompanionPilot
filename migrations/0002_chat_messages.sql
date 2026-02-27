CREATE TABLE IF NOT EXISTS chat_messages (
    id BIGSERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    guild_id TEXT NOT NULL,
    channel_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_user_time
    ON chat_messages (user_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_chat_messages_guild_channel_time
    ON chat_messages (guild_id, channel_id, timestamp DESC);
