# discord_voice_join tool skill

Purpose:
- Connect the bot to a voice channel for interactive voice usage.

Args schema:
- `{ "channel_id": "string" }` optional.
- If omitted/empty, runtime should default to requester's current voice channel.

Usage guidance:
- Use only when user explicitly asks the assistant to join/connect to voice.
- If a specific channel is requested and available, pass its ID.
- Do not invent channel IDs; omit `channel_id` when unknown.

Failure handling:
- If voice tools are not configured or requester is not in voice, return failure clearly.
