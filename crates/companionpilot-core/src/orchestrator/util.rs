use std::time::Instant;

pub(super) fn truncate_for_prompt(input: &str, max_len: usize) -> String {
    if input.len() <= max_len {
        return input.to_owned();
    }

    let safe_len = input.floor_char_boundary(max_len);
    let mut truncated = input[..safe_len].to_owned();
    truncated.push_str("...");
    truncated
}

pub(super) fn indent_block(input: &str) -> String {
    input
        .lines()
        .map(|line| format!("      {line}"))
        .collect::<Vec<_>>()
        .join("\n")
}

pub(super) fn elapsed_ms(started_at: Instant) -> u64 {
    started_at
        .elapsed()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}
