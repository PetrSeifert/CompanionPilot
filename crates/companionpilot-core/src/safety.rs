#[derive(Debug, Clone)]
pub struct SafetyPolicy {
    blocked_terms: Vec<String>,
}

impl Default for SafetyPolicy {
    fn default() -> Self {
        Self {
            blocked_terms: vec!["rm -rf".to_owned(), "token leak".to_owned()],
        }
    }
}

impl SafetyPolicy {
    pub fn validate_user_message(&self, input: &str) -> Vec<String> {
        let lowercase = input.to_lowercase();
        self.blocked_terms
            .iter()
            .filter(|term| lowercase.contains(term.as_str()))
            .map(|term| format!("blocked-term:{term}"))
            .collect()
    }
}
