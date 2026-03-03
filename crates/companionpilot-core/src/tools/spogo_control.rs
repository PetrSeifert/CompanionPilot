use anyhow::{Context, bail};
use serde::Deserialize;
use serde_json::Value;
use tracing::info;

use super::{SpogoCli, ToolResult};

#[derive(Debug, Clone)]
pub struct SpogoControlTool {
    cli: SpogoCli,
    account_label: Option<String>,
}

impl Default for SpogoControlTool {
    fn default() -> Self {
        Self::new(SpogoCli::default(), None)
    }
}

#[derive(Debug, Deserialize)]
struct ControlArgs {
    action: String,
    #[serde(default)]
    device_id: Option<String>,
    #[serde(default)]
    state: Option<Value>,
    #[serde(default)]
    position_ms: Option<u64>,
    #[serde(default)]
    volume: Option<u64>,
    #[serde(default)]
    uris: Vec<String>,
    #[serde(default)]
    context_uri: Option<String>,
    #[serde(default)]
    uri: Option<String>,
    #[serde(default)]
    device_ids: Vec<String>,
    #[serde(default)]
    play: Option<bool>,
}

#[derive(Debug)]
struct PreparedAction {
    action: String,
    command_sequences: Vec<Vec<String>>,
}

impl SpogoControlTool {
    pub fn new(cli: SpogoCli, account_label: Option<String>) -> Self {
        Self { cli, account_label }
    }

    pub async fn control(&self, args: Value) -> anyhow::Result<ToolResult> {
        let parsed: ControlArgs = serde_json::from_value(args).context(
            "spogo_control expects JSON args with action and optional action-specific fields",
        )?;
        let prepared = prepare_control_action(parsed)?;
        info!(action = %prepared.action, "spogo control request start");

        let mut citations = Vec::new();
        let mut responses = Vec::new();
        for (index, command) in prepared.command_sequences.iter().enumerate() {
            let output = self.cli.execute(command.clone()).await?;
            citations.push(output.command_line);
            let response = output.stdout.trim();
            if !response.is_empty() {
                responses.push(truncate_for_output(response));
            }

            if prepared.action == "toggle" && index == 0 {
                let payload: Value = serde_json::from_str(response)
                    .context("toggle action expected JSON status output")?;
                let currently_playing = extract_playing_flag(&payload)
                    .context("toggle action could not infer playback state from status output")?;
                let toggle_cmd = if currently_playing { "pause" } else { "play" };
                let mut next_command = command
                    .iter()
                    .take_while(|arg| arg.as_str() != "status")
                    .cloned()
                    .collect::<Vec<_>>();
                next_command.push(toggle_cmd.to_owned());
                let toggle_output = self.cli.execute(next_command).await?;
                citations.push(toggle_output.command_line);
                let toggle_response = toggle_output.stdout.trim();
                if !toggle_response.is_empty() {
                    responses.push(truncate_for_output(toggle_response));
                }
            }
        }

        let mut lines = Vec::new();
        if let Some(label) = self
            .account_label
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            lines.push(format!("Spotify account: {label}"));
        }
        lines.push(format!("Action: {}", prepared.action));
        lines.push(format!(
            "Result: success ({} command{})",
            citations.len(),
            if citations.len() == 1 { "" } else { "s" }
        ));
        if !responses.is_empty() {
            lines.push(format!("Response: {}", responses.join(" | ")));
        }

        Ok(ToolResult {
            text: lines.join("\n"),
            citations,
        })
    }
}

fn prepare_control_action(args: ControlArgs) -> anyhow::Result<PreparedAction> {
    let action = args.action.trim().to_ascii_lowercase();
    if !is_supported_action(&action) {
        bail!("unsupported spogo_control action: {}", args.action);
    }

    let mut command_prefix = vec!["--json".to_owned()];
    if let Some(device_id) = normalize_opt_string(args.device_id.as_deref()) {
        command_prefix.push("--device".to_owned());
        command_prefix.push(device_id);
    }

    let command_sequences = match action.as_str() {
        "play" => vec![with_prefix(&command_prefix, &["play"])],
        "pause" => vec![with_prefix(&command_prefix, &["pause"])],
        "toggle" => build_toggle_commands(&command_prefix),
        "next" => vec![with_prefix(&command_prefix, &["next"])],
        "previous" => vec![with_prefix(&command_prefix, &["prev"])],
        "shuffle" => {
            let state = args
                .state
                .as_ref()
                .and_then(Value::as_bool)
                .context("shuffle action requires boolean state")?;
            let value = if state { "on" } else { "off" };
            vec![with_prefix(&command_prefix, &["shuffle", value])]
        }
        "repeat" => {
            let state = args
                .state
                .as_ref()
                .and_then(Value::as_str)
                .map(str::trim)
                .context("repeat action requires string state (off|track|context)")?;
            if !matches!(state, "off" | "track" | "context") {
                bail!("repeat state must be one of: off, track, context");
            }
            vec![with_prefix(&command_prefix, &["repeat", state])]
        }
        "seek" => {
            let position_ms = args
                .position_ms
                .context("seek action requires position_ms in milliseconds")?;
            let seconds = format_seek_seconds(position_ms);
            vec![with_prefix_owned(
                &command_prefix,
                vec!["seek".to_owned(), seconds],
            )]
        }
        "volume" => {
            let volume = args
                .volume
                .context("volume action requires volume (0..100)")?;
            if volume > 100 {
                bail!("volume must be in range 0..100");
            }
            vec![with_prefix_owned(
                &command_prefix,
                vec!["volume".to_owned(), volume.to_string()],
            )]
        }
        "playback" => {
            let uris = normalize_string_vec(&args.uris);
            let target = if let Some(uri) = uris.first() {
                uri.to_owned()
            } else {
                normalize_opt_string(args.context_uri.as_deref())
                    .context("playback action requires at least one uri in uris or context_uri")?
            };
            vec![with_prefix_owned(
                &command_prefix,
                vec!["play".to_owned(), target],
            )]
        }
        "queue" => {
            let uri =
                normalize_opt_string(args.uri.as_deref()).context("queue action requires uri")?;
            vec![with_prefix_owned(
                &command_prefix,
                vec!["queue".to_owned(), "add".to_owned(), uri],
            )]
        }
        "transfer" => {
            let device_id = normalize_string_vec(&args.device_ids)
                .into_iter()
                .next()
                .context("transfer action requires non-empty device_ids")?;
            let mut sequence = vec![with_prefix_owned(
                &command_prefix,
                vec!["device".to_owned(), "set".to_owned(), device_id],
            )];
            if args.play == Some(true) {
                sequence.push(with_prefix(&command_prefix, &["play"]));
            }
            sequence
        }
        _ => bail!("unsupported action"),
    };

    Ok(PreparedAction {
        action,
        command_sequences,
    })
}

fn build_toggle_commands(command_prefix: &[String]) -> Vec<Vec<String>> {
    vec![with_prefix(command_prefix, &["status"])]
}

fn with_prefix(prefix: &[String], args: &[&str]) -> Vec<String> {
    with_prefix_owned(
        prefix,
        args.iter().map(|value| (*value).to_owned()).collect(),
    )
}

fn with_prefix_owned(prefix: &[String], args: Vec<String>) -> Vec<String> {
    let mut output = prefix.to_vec();
    output.extend(args);
    output
}

fn normalize_opt_string(value: Option<&str>) -> Option<String> {
    value
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(ToOwned::to_owned)
}

fn normalize_string_vec(value: &[String]) -> Vec<String> {
    value
        .iter()
        .map(String::as_str)
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn format_seek_seconds(position_ms: u64) -> String {
    let seconds = position_ms / 1000;
    let millis = position_ms % 1000;
    if millis == 0 {
        seconds.to_string()
    } else {
        format!("{seconds}.{millis:03}")
    }
}

fn truncate_for_output(value: &str) -> String {
    let max = 240usize;
    if value.len() <= max {
        value.to_owned()
    } else {
        let mut end = max;
        while end > 0 && !value.is_char_boundary(end) {
            end -= 1;
        }
        format!("{}...", &value[..end])
    }
}

fn extract_playing_flag(value: &Value) -> Option<bool> {
    if let Some(state) = value.get("is_playing").and_then(Value::as_bool) {
        return Some(state);
    }
    if let Some(state) = value.get("isPlaying").and_then(Value::as_bool) {
        return Some(state);
    }
    if let Some(state) = value.get("playing").and_then(Value::as_bool) {
        return Some(state);
    }
    if let Some(object) = value.as_object() {
        for nested in object.values() {
            if let Some(state) = extract_playing_flag(nested) {
                return Some(state);
            }
        }
    }
    if let Some(array) = value.as_array() {
        for nested in array {
            if let Some(state) = extract_playing_flag(nested) {
                return Some(state);
            }
        }
    }
    None
}

fn is_supported_action(action: &str) -> bool {
    matches!(
        action,
        "play"
            | "pause"
            | "toggle"
            | "next"
            | "previous"
            | "shuffle"
            | "repeat"
            | "seek"
            | "volume"
            | "playback"
            | "queue"
            | "transfer"
    )
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{
        ControlArgs, format_seek_seconds, prepare_control_action, truncate_for_output,
    };

    fn parse_args(value: serde_json::Value) -> ControlArgs {
        serde_json::from_value(value).expect("control args should deserialize")
    }

    #[test]
    fn prepare_playback_action_prefers_first_uri() {
        let args = parse_args(json!({
            "action": "playback",
            "uris": ["spotify:track:abc", "spotify:track:def"]
        }));
        let prepared = prepare_control_action(args).expect("playback action should prepare");
        assert_eq!(prepared.command_sequences.len(), 1);
        assert_eq!(prepared.command_sequences[0][1], "play");
        assert_eq!(prepared.command_sequences[0][2], "spotify:track:abc");
    }

    #[test]
    fn prepare_transfer_action_appends_play_when_requested() {
        let args = parse_args(json!({
            "action": "transfer",
            "device_ids": ["device-1"],
            "play": true
        }));
        let prepared = prepare_control_action(args).expect("transfer action should prepare");
        assert_eq!(prepared.command_sequences.len(), 2);
        assert_eq!(prepared.command_sequences[0][1], "device");
        assert_eq!(prepared.command_sequences[0][2], "set");
        assert_eq!(prepared.command_sequences[1][1], "play");
    }

    #[test]
    fn prepare_shuffle_action_requires_boolean_state() {
        let args = parse_args(json!({
            "action": "shuffle",
            "state": "on"
        }));
        assert!(prepare_control_action(args).is_err());
    }

    #[test]
    fn prepare_seek_action_preserves_subsecond_precision() {
        let args = parse_args(json!({
            "action": "seek",
            "position_ms": 12_345
        }));
        let prepared = prepare_control_action(args).expect("seek action should prepare");
        assert_eq!(prepared.command_sequences.len(), 1);
        assert_eq!(prepared.command_sequences[0][1], "seek");
        assert_eq!(prepared.command_sequences[0][2], "12.345");
    }

    #[test]
    fn format_seek_seconds_omits_fraction_for_whole_seconds() {
        assert_eq!(format_seek_seconds(15_000), "15");
    }

    #[test]
    fn truncate_for_output_handles_multibyte_utf8() {
        let text = "a".repeat(239) + "😄tail";
        let truncated = truncate_for_output(&text);
        assert!(truncated.ends_with("..."));
        assert!(!truncated.contains("tail"));
    }
}
