use anyhow::{Context, bail};
use serde_json::{Value, json};

use super::{SpogoCli, ToolResult};

#[derive(Debug, Clone)]
pub struct CliTool {
    cli: SpogoCli,
}

impl Default for CliTool {
    fn default() -> Self {
        Self::new(SpogoCli::default())
    }
}

impl CliTool {
    pub fn new(cli: SpogoCli) -> Self {
        Self { cli }
    }

    pub async fn execute_command(&self, args: Value) -> anyhow::Result<ToolResult> {
        let command_args = validate_and_extract_spogo_args(parse_cli_args(&args)?)?;
        let output = self.cli.execute(command_args).await?;

        let stdout = output.stdout.trim();
        let stderr = output.stderr.trim();

        let mut lines = if output.exit_code == 0 {
            vec!["CLI command completed with exit_code=0.".to_owned()]
        } else {
            vec![format!(
                "CLI command failed with exit_code={}.",
                output.exit_code
            )]
        };
        if !stdout.is_empty() {
            lines.push(format!("stdout:\n{stdout}"));
        }
        if !stderr.is_empty() {
            lines.push(format!("stderr:\n{stderr}"));
        }
        if stdout.is_empty() && stderr.is_empty() {
            lines.push("No output returned.".to_owned());
        }

        Ok(ToolResult {
            text: lines.join("\n"),
            citations: vec![output.command_line],
        })
    }
}

pub(crate) fn sanitize_cli_invocation_args(raw_args: &Value) -> Option<Value> {
    let args = parse_cli_args(raw_args).ok()?;
    validate_and_extract_spogo_args(args.clone()).ok()?;
    Some(json!({ "args": args }))
}

fn validate_and_extract_spogo_args(command_args: Vec<String>) -> anyhow::Result<Vec<String>> {
    let Some((command, subcommand_args)) = command_args.split_first() else {
        bail!("cli requires at least one token in args");
    };
    if command != "spogo" {
        bail!(
            "CLI command `{}` is not allowed. Only `spogo` commands are allowed.",
            command
        );
    }
    if command_args
        .iter()
        .any(|token| has_disallowed_shell_syntax(token))
    {
        bail!(
            "CLI command contains disallowed shell syntax. Allowed usage is plain spogo args only."
        );
    }
    if subcommand_args.is_empty() {
        bail!("CLI command must include a spogo subcommand or flag. Use `spogo -h` for help.");
    }
    Ok(subcommand_args.to_vec())
}

fn parse_cli_args(args: &Value) -> anyhow::Result<Vec<String>> {
    let parsed = if let Some(command) = args
        .get("command")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        command
            .split_whitespace()
            .map(ToOwned::to_owned)
            .collect::<Vec<_>>()
    } else {
        let raw = args
            .get("args")
            .context("cli requires `command` (string) or `args` (array<string> or string)")?;
        match raw {
            Value::String(raw) => raw
                .split_whitespace()
                .map(ToOwned::to_owned)
                .collect::<Vec<_>>(),
            Value::Array(values) => values
                .iter()
                .filter_map(Value::as_str)
                .map(str::trim)
                .filter(|item| !item.is_empty())
                .map(ToOwned::to_owned)
                .collect::<Vec<_>>(),
            _ => bail!("cli `args` must be array<string> or string"),
        }
    };

    if parsed.is_empty() {
        bail!("cli requires at least one CLI argument");
    }
    if parsed.len() > 64 {
        bail!("cli supports up to 64 arguments per command");
    }

    Ok(parsed)
}

fn has_disallowed_shell_syntax(token: &str) -> bool {
    matches!(token, "|" | "||" | "&" | "&&")
        || token.contains(';')
        || token.contains('`')
        || token.contains("$(")
        || token.contains('>')
        || token.contains('<')
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{parse_cli_args, sanitize_cli_invocation_args, validate_and_extract_spogo_args};

    #[test]
    fn parses_command_string() {
        let args = json!({ "command": "spogo search track muse" });
        let parsed = parse_cli_args(&args).expect("command should parse");
        assert_eq!(parsed, vec!["spogo", "search", "track", "muse"]);
    }

    #[test]
    fn parses_string_args() {
        let args = json!({ "args": "spogo search track muse" });
        let parsed = parse_cli_args(&args).expect("args should parse");
        assert_eq!(parsed, vec!["spogo", "search", "track", "muse"]);
    }

    #[test]
    fn parses_array_args() {
        let args = json!({ "args": ["spogo", "-h"] });
        let parsed = parse_cli_args(&args).expect("args should parse");
        assert_eq!(parsed, vec!["spogo", "-h"]);
    }

    #[test]
    fn rejects_empty_args() {
        let args = json!({ "args": [] });
        assert!(parse_cli_args(&args).is_err());
    }

    #[test]
    fn rejects_missing_command_and_args() {
        let args = json!({});
        assert!(parse_cli_args(&args).is_err());
    }

    #[test]
    fn rejects_non_spogo_commands() {
        let args = vec!["ls".to_owned(), "-la".to_owned()];
        assert!(validate_and_extract_spogo_args(args).is_err());
    }

    #[test]
    fn rejects_bare_spogo_command() {
        let args = vec!["spogo".to_owned()];
        assert!(validate_and_extract_spogo_args(args).is_err());
    }

    #[test]
    fn rejects_shell_metacharacters() {
        let args = vec!["spogo".to_owned(), "search;".to_owned(), "track".to_owned()];
        assert!(validate_and_extract_spogo_args(args).is_err());
    }

    #[test]
    fn sanitize_cli_args_canonicalizes_command_to_args() {
        let raw = json!({ "command": "spogo -h" });
        let sanitized = sanitize_cli_invocation_args(&raw).expect("sanitization should pass");
        assert_eq!(sanitized, json!({ "args": ["spogo", "-h"] }));
    }

    #[test]
    fn sanitize_cli_args_rejects_non_spogo_command() {
        let raw = json!({ "command": "ls -la" });
        assert!(sanitize_cli_invocation_args(&raw).is_none());
    }
}
