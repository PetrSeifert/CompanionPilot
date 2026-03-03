# current_datetime tool skill

Purpose:
- Return the exact current UTC datetime/date/year so downstream reasoning can anchor time-sensitive answers.

Args schema:
- `{}` only. Ignore any incoming args.

Usage guidance:
- Use this before web lookups when user asks about "today", "latest", "current", deadlines, schedules, or anything time-sensitive.
- This tool is informational and side-effect free.

Failure handling:
- If the tool fails, do not fabricate current time; surface failure and rely on fallback logic.
