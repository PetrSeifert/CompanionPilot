# discord_voice_leave tool skill

Purpose:
- Disconnect the bot from voice for the requester's guild/session.

Args schema:
- `{}` only.

Usage guidance:
- Use when user asks to leave/disconnect/stop voice participation.
- Do not use for normal text-only requests.

Failure handling:
- If bot is not connected or voice tools are unavailable, surface clear failure state.
