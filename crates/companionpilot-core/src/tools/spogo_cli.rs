use std::process::Stdio;
use std::time::{Duration, Instant};

use anyhow::Context;
use tokio::process::Command;
use tokio::time::timeout;
use tracing::{info, warn};

pub const DEFAULT_SPOGO_BIN_PATH: &str = "/usr/local/bin/spogo";
pub const DEFAULT_SPOGO_TIMEOUT_MS: u64 = 8_000;

#[derive(Debug, Clone)]
pub struct SpogoCli {
    bin_path: String,
    config_dir: String,
    timeout: Duration,
}

impl Default for SpogoCli {
    fn default() -> Self {
        Self::new(
            DEFAULT_SPOGO_BIN_PATH,
            "/data/spogo",
            DEFAULT_SPOGO_TIMEOUT_MS,
        )
    }
}

#[derive(Debug, Clone)]
pub struct SpogoCommandOutput {
    pub command_line: String,
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
}

impl SpogoCli {
    pub fn new(
        bin_path: impl Into<String>,
        config_dir: impl Into<String>,
        timeout_ms: u64,
    ) -> Self {
        Self {
            bin_path: bin_path.into(),
            config_dir: config_dir.into(),
            timeout: Duration::from_millis(timeout_ms.max(1_000)),
        }
    }

    pub async fn execute(&self, command_args: Vec<String>) -> anyhow::Result<SpogoCommandOutput> {
        let mut full_args = vec![
            "--config".to_owned(),
            self.config_file_path(),
            "--timeout".to_owned(),
            format!("{}ms", self.timeout.as_millis()),
        ];
        full_args.extend(command_args);

        let command_line = format_command_line(&self.bin_path, &full_args);
        info!(command = %command_line, "spogo command start");

        // Safety: this invokes the binary directly without a shell.
        // Tokens are passed as argv via Command::args, so shell metacharacters are not interpreted.
        let mut command = Command::new(&self.bin_path);
        command
            .args(&full_args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let started = Instant::now();
        let output = timeout(self.timeout, command.output())
            .await
            .with_context(|| {
                format!(
                    "spogo command timed out after {}ms",
                    self.timeout.as_millis()
                )
            })?
            .with_context(|| format!("failed to execute spogo command: {command_line}"))?;
        let duration_ms = started.elapsed().as_millis() as u64;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let exit_code = output.status.code().unwrap_or(-1);
        if !output.status.success() {
            let details = summarize_output(&stderr, &stdout);
            warn!(
                command = %command_line,
                exit_code,
                duration_ms,
                details = %details,
                "spogo command failed"
            );
            anyhow::bail!(
                "spogo command failed (exit_code={exit_code}) after {duration_ms}ms: {details}"
            );
        }

        Ok(SpogoCommandOutput {
            command_line,
            stdout,
            stderr,
            exit_code,
        })
    }

    fn config_file_path(&self) -> String {
        let trimmed = self.config_dir.trim_end_matches(['/', '\\']);
        format!("{trimmed}/config.toml")
    }
}

fn summarize_output(stderr: &str, stdout: &str) -> String {
    let stderr = stderr.trim();
    if !stderr.is_empty() {
        return truncate_for_output(stderr);
    }
    let stdout = stdout.trim();
    if !stdout.is_empty() {
        return truncate_for_output(stdout);
    }
    "no output".to_owned()
}

fn truncate_for_output(value: &str) -> String {
    let max = 320usize;
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

fn format_command_line(bin_path: &str, args: &[String]) -> String {
    let mut rendered = vec![shell_quote(bin_path)];
    rendered.extend(args.iter().map(|arg| shell_quote(arg)));
    rendered.join(" ")
}

fn shell_quote(value: &str) -> String {
    if value.is_empty() {
        return "''".to_owned();
    }
    if value
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '/' | '\\' | '-' | '_' | '.' | ':' | '='))
    {
        value.to_owned()
    } else {
        format!("'{}'", value.replace('\'', "'\"'\"'"))
    }
}

#[cfg(test)]
mod tests {
    use super::truncate_for_output;

    #[test]
    fn truncate_for_output_handles_multibyte_utf8() {
        let text = "a".repeat(319) + "😄tail";
        let truncated = truncate_for_output(&text);
        assert!(truncated.ends_with("..."));
        assert!(!truncated.contains("tail"));
    }
}
