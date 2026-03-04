---
id: spotify-cli
title: Spotify CLI Operations
description: Handle Spotify tasks via local spogo command execution.
tags: [spotify, cli, spogo]
---

# spogo

Use `run_terminal_command` for Spotify playback/search by invoking `spogo`.
Pass either:
- `command` as a single string (example: `spogo status`)
- `args` as argv tokens (example: `["spogo","status"]`)

Common CLI commands
- Search: `spogo search track "query"`
- Playback: `spogo play|pause|next|prev`
- Devices: `spogo device list`, `spogo device set "<name|id>"`
- Status: `spogo status`
