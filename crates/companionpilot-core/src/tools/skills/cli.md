# cli tool skill

Purpose:
- Execute local Spotify commands through `spogo`.

Strict safety policy:
- Allowed commands must start with token `spogo`.
- Non-`spogo` commands are blocked.
- Shell syntax is disallowed (`|`, `&&`, `;`, backticks, redirection, `$(`, etc.).

Args schema:
- Preferred: `{ "command": "spogo <subcommand> ..." }`
- Alternative: `{ "args": ["spogo", "<subcommand>", "..."] }`
- Empty args are invalid.

Operational guidance:
- If uncertain about subcommand syntax, call help first: `spogo -h` or `spogo <subcommand> -h`.
- Keep arguments plain and minimal. Do not include shell wrappers.
- Do not attempt non-Spotify tasks with this tool.

Validation expectations:
- Commands with first token not equal to `spogo` must be rejected.
- Parsed argument list max length is 64 tokens.
