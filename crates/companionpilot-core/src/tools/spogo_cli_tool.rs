use anyhow::bail;
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
    let command_tokens = args
        .get("command")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|command| shell_split(command))
        .transpose()?
        .unwrap_or_default();

    let args_tokens = if let Some(raw) = args.get("args") {
        match raw {
            Value::String(raw) => shell_split(raw)?,
            Value::Array(values) => {
                let elements: Vec<String> = values
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::trim)
                    .filter(|item| !item.is_empty())
                    .map(ToOwned::to_owned)
                    .collect();
                fix_array_tokens(elements)?
            }
            _ => bail!("cli `args` must be array<string> or string"),
        }
    } else {
        Vec::new()
    };

    let parsed = if !command_tokens.is_empty() && !args_tokens.is_empty() {
        // Some providers emit split payloads like {"command":"spogo search","args":"track muse"}.
        // Normalize that into canonical argv form and preserve all provided args.
        if args_tokens.first().map(String::as_str) == Some("spogo") {
            args_tokens
        } else {
            let mut combined = command_tokens;
            combined.extend(args_tokens);
            combined
        }
    } else if !command_tokens.is_empty() {
        command_tokens
    } else if !args_tokens.is_empty() {
        args_tokens
    } else {
        bail!("cli requires `command` (string) or `args` (array<string> or string)");
    };

    if parsed.is_empty() {
        bail!("cli requires at least one CLI argument");
    }
    if parsed.len() > 64 {
        bail!("cli supports up to 64 arguments per command");
    }

    Ok(parsed)
}

/// Parse a shell-like string respecting double and single quoted regions.
/// Quotes are stripped from the output; content inside quotes is preserved as one token.
fn shell_split(s: &str) -> anyhow::Result<Vec<String>> {
    let mut tokens: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut in_double = false;
    let mut in_single = false;

    for c in s.chars() {
        match c {
            '"' if !in_single => in_double = !in_double,
            '\'' if !in_double => in_single = !in_single,
            ' ' | '\t' | '\n' if !in_double && !in_single => {
                if !current.is_empty() {
                    tokens.push(std::mem::take(&mut current));
                }
            }
            _ => current.push(c),
        }
    }

    if in_double || in_single {
        bail!("cli args contain an unclosed quote");
    }
    if !current.is_empty() {
        tokens.push(current);
    }

    Ok(tokens)
}

/// Fix array tokens where the AI mistakenly embedded shell quote characters into individual
/// elements and split a quoted string across multiple elements.
///
/// For example: `["\"NEVER", "Neurosama\""]` → `["NEVER Neurosama"]`
///
/// When any element carries a leading or trailing quote character it means the AI was treating
/// the array like a shell command string. In that case we rejoin all elements with spaces and
/// re-parse using `shell_split`, which correctly reassembles the intended arguments.
///
/// When no element has embedded quotes the elements are returned verbatim, preserving any
/// legitimate spaces inside an individual element (e.g. a device name like `"My Phone"`).
fn fix_array_tokens(elements: Vec<String>) -> anyhow::Result<Vec<String>> {
    let has_embedded_quotes = elements.iter().any(|e| {
        e.starts_with('"') || e.starts_with('\'') || e.ends_with('"') || e.ends_with('\'')
    });

    if has_embedded_quotes {
        // Rejoin and let shell_split reassemble quoted spans correctly.
        shell_split(&elements.join(" "))
    } else {
        Ok(elements)
    }
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
    fn parses_split_command_and_args_payload() {
        let args = json!({ "command": "spogo", "args": "status" });
        let parsed = parse_cli_args(&args).expect("split payload should normalize");
        assert_eq!(parsed, vec!["spogo", "status"]);
    }

    #[test]
    fn preserves_split_command_and_args_tokens() {
        let args = json!({ "command": "spogo search", "args": "track muse" });
        let parsed = parse_cli_args(&args).expect("split payload should preserve args");
        assert_eq!(parsed, vec!["spogo", "search", "track", "muse"]);
    }

    #[test]
    fn prefers_args_when_it_contains_full_spogo_command() {
        let args = json!({ "command": "spogo search", "args": "spogo status" });
        let parsed = parse_cli_args(&args).expect("args should parse");
        assert_eq!(parsed, vec!["spogo", "status"]);
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
    fn sanitize_cli_args_accepts_split_command_and_args_payload() {
        let raw = json!({ "command": "spogo", "args": "status" });
        let sanitized = sanitize_cli_invocation_args(&raw).expect("sanitization should pass");
        assert_eq!(sanitized, json!({ "args": ["spogo", "status"] }));
    }

    #[test]
    fn sanitize_cli_args_rejects_non_spogo_command() {
        let raw = json!({ "command": "ls -la" });
        assert!(sanitize_cli_invocation_args(&raw).is_none());
    }

    // --- shell_split and fix_array_tokens ---

    #[test]
    fn shell_split_handles_double_quoted_multi_word() {
        // "NEVER Neurosama" should be a single token, not two
        let result = super::shell_split(
            r#"spogo --json search track "NEVER Neurosama" --limit 1"#,
        )
        .unwrap();
        assert_eq!(
            result,
            vec!["spogo", "--json", "search", "track", "NEVER Neurosama", "--limit", "1"]
        );
    }

    #[test]
    fn shell_split_handles_single_quoted_multi_word() {
        let result = super::shell_split("spogo --json search track 'NEVER Neurosama' --limit 1")
            .unwrap();
        assert_eq!(
            result,
            vec!["spogo", "--json", "search", "track", "NEVER Neurosama", "--limit", "1"]
        );
    }

    #[test]
    fn shell_split_errors_on_unclosed_quote() {
        assert!(super::shell_split(r#"spogo search "unclosed"#).is_err());
    }

    #[test]
    fn fix_array_tokens_rejoins_split_quoted_query() {
        // Simulates: ["spogo", "--json", "search", "track", "\"NEVER", "Neurosama\"", "--limit", "1"]
        let elements = vec![
            "spogo".to_owned(),
            "--json".to_owned(),
            "search".to_owned(),
            "track".to_owned(),
            "\"NEVER".to_owned(),
            "Neurosama\"".to_owned(),
            "--limit".to_owned(),
            "1".to_owned(),
        ];
        let result = super::fix_array_tokens(elements).unwrap();
        assert_eq!(
            result,
            vec!["spogo", "--json", "search", "track", "NEVER Neurosama", "--limit", "1"]
        );
    }

    #[test]
    fn fix_array_tokens_preserves_clean_elements_with_spaces() {
        // When no embedded quotes, elements with spaces are passed through verbatim
        let elements = vec![
            "spogo".to_owned(),
            "--json".to_owned(),
            "device".to_owned(),
            "set".to_owned(),
            "My Phone".to_owned(),
        ];
        let result = super::fix_array_tokens(elements.clone()).unwrap();
        assert_eq!(result, elements);
    }

    #[test]
    fn parse_cli_args_handles_quoted_string_form() {
        let args = json!({
            "args": r#"spogo --json search track "NEVER Neurosama" --limit 1"#
        });
        let parsed = parse_cli_args(&args).unwrap();
        assert_eq!(
            parsed,
            vec!["spogo", "--json", "search", "track", "NEVER Neurosama", "--limit", "1"]
        );
    }

    #[test]
    fn parse_cli_args_handles_split_quoted_array_form() {
        let args = json!({
            "args": ["spogo", "--json", "search", "track", "\"NEVER", "Neurosama\"", "--limit", "1"]
        });
        let parsed = parse_cli_args(&args).unwrap();
        assert_eq!(
            parsed,
            vec!["spogo", "--json", "search", "track", "NEVER Neurosama", "--limit", "1"]
        );
    }
}
