---
id: spotify-cli
title: spogo
description: Control Spotify from the terminal using the spogo CLI. Use this skill when the user wants to search for music, control playback, manage their library/playlists, or inspect Spotify state via the command line.
tags: [spotify, cli, spogo]
---

# spogo - Spotify Power CLI

`spogo` is a terminal CLI for Spotify that uses browser cookies for auth. It talks to Spotify's internal web endpoints (no rate limits, no app registration required).

**Always pass `--json`** to get structured output. Never parse human-readable output.

## Core Rule: Always Use --json

```bash
spogo --json <command>
```

The `--json` flag goes **before** the subcommand (it's a global flag).

JSON output is written to stdout; errors/warnings go to stderr.

## Playback Control

```bash
spogo --json status             # current track, device, shuffle, repeat, progress
spogo --json play <uri|url>     # play track/album/playlist/show/episode/artist
spogo --json pause
spogo --json next
spogo --json prev
spogo --json seek 1:30          # seek to mm:ss or ms
spogo --json volume 50          # 0-100
spogo --json shuffle on|off
spogo --json repeat off|track|context
```

## Search

```bash
spogo --json search track "query" [--limit N] [--offset N]
spogo --json search album "query"
spogo --json search artist "query"
spogo --json search playlist "query"
spogo --json search episode "query"
spogo --json search show "query"
```

Search JSON output:
```json
{
  "type": "track",
  "total": 100,
  "limit": 20,
  "offset": 0,
  "items": [
    {
      "id": "7hQJA50XrCWABAu5v6QZ4i",
      "uri": "spotify:track:7hQJA50XrCWABAu5v6QZ4i",
      "name": "Buddy Holly",
      "type": "track",
      "artists": ["Weezer"],
      "album": "Weezer (Blue Album)",
      "duration_ms": 147960,
      "explicit": false,
      "is_playable": true
    }
  ]
}
```

## Info (by ID or URL)

```bash
spogo --json track info <id|url>
spogo --json album info <id|url>
spogo --json artist info <id|url>
spogo --json playlist info <id|url>
spogo --json show info <id|url>
spogo --json episode info <id|url>
```

## Queue

```bash
spogo --json queue show         # currently playing + queued tracks
spogo --json queue add <uri>    # add to queue
```

Queue JSON:
```json
{
  "currently_playing": { "id": "...", "name": "...", "artists": ["..."], ... },
  "queue": [
    { "id": "...", "name": "...", "artists": ["..."], ... }
  ]
}
```

## Devices

```bash
spogo --json device list        # list available devices
spogo --json device set "My Phone"   # switch active device
```

Device list JSON: array of `{ "id", "name", "type", "volume_percent", "is_active", "is_restricted" }`.

## Library

```bash
spogo --json library tracks list [--limit N]
spogo --json library tracks add <uri...>
spogo --json library tracks remove <uri...>
spogo --json library albums list [--limit N]
spogo --json library albums add <uri...>
spogo --json library albums remove <uri...>
spogo --json library artists list [--limit N]
spogo --json library artists follow <uri...>
spogo --json library artists unfollow <uri...>
spogo --json library playlists list [--limit N]
```

Library list JSON: `{ "total": N, "items": [...] }`

## Playlists

```bash
spogo --json playlist create "Name" [--public] [--collab]
spogo --json playlist add <playlist-id> <track-uri...>
spogo --json playlist remove <playlist-id> <track-uri...>
spogo --json playlist tracks <playlist-id> [--limit N]
```

## Status JSON Schema

```json
{
  "is_playing": true,
  "progress_ms": 45000,
  "item": {
    "id": "...",
    "uri": "spotify:track:...",
    "name": "Track Name",
    "type": "track",
    "artists": ["Artist"],
    "album": "Album Name",
    "duration_ms": 200000,
    "explicit": false,
    "is_playable": true
  },
  "device": {
    "id": "...",
    "name": "Device Name",
    "type": "speaker",
    "volume_percent": 70,
    "is_active": true,
    "is_restricted": false
  },
  "shuffle": false,
  "repeat": "off"
}
```

## URIs and IDs

Spotify URIs have the form `spotify:track:7hQJA50XrCWABAu5v6QZ4i`. You can pass:
- Full URI: `spotify:track:7hQJA50XrCWABAu5v6QZ4i`
- Short ID: `7hQJA50XrCWABAu5v6QZ4i` (add `--type track` when needed)
- Web URL: `https://open.spotify.com/track/7hQJA50XrCWABAu5v6QZ4i`

## Engine Selection

Default engine is `connect` (Spotify's internal API, no rate limits). Rarely need to change, but:

```bash
spogo --engine auto ...    # try connect, fall back to web
spogo --engine web ...     # force Spotify Web API
```

## Exit Codes

- `0` - success
- `1` - generic error
- `2` - invalid usage
- `3` - auth/cookies missing or invalid -> tell user that he needs to authenticate using cookies
- `4` - network error / timeout

## Typical AI Workflow

1. Check status: `spogo --json status`
2. Search for content: `spogo --json search track "..."  --limit 5`
3. Get URI from `items[0].uri`
4. Play: `spogo --json play <uri>`
5. Verify: `spogo --json status`
