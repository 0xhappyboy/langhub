use crate::types::{LangHubError, LangHubResult};
pub struct TextTool;
impl TextTool {
    pub fn extract_json(text: &str) -> LangHubResult<String> {
        let text = text.trim();
        let json_start = text.find('{');
        let json_end = text.rfind('}');
        if let (Some(start), Some(end)) = (json_start, json_end) {
            if start < end {
                let candidate = &text[start..=end];
                if candidate.starts_with('{') && candidate.ends_with('}') {
                    return Ok(candidate.to_string());
                }
            }
        }
        let bracket_start = text.find('[');
        let bracket_end = text.rfind(']');
        if let (Some(start), Some(end)) = (bracket_start, bracket_end) {
            if start < end {
                let candidate = &text[start..=end];
                if candidate.starts_with('[') && candidate.ends_with(']') {
                    return Ok(candidate.to_string());
                }
            }
        }
        Err(LangHubError::ParseError(
            "No valid JSON found in text".to_string(),
        ))
    }
    pub fn chunk_text(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
        let mut chunks = Vec::new();
        if text.is_empty() {
            return chunks;
        }
        let chars: Vec<char> = text.chars().collect();
        let step = chunk_size.saturating_sub(overlap);
        let mut start = 0;
        while start < chars.len() {
            let end = (start + chunk_size).min(chars.len());
            let chunk: String = chars[start..end].iter().collect();
            chunks.push(chunk);
            start += step;
            if start >= chars.len() {
                break;
            }
        }
        chunks
    }
    pub fn truncate(text: &str, max_length: usize, suffix: &str) -> String {
        if text.len() <= max_length {
            return text.to_string();
        }
        let truncated = &text[..max_length];
        if let Some(last_space) = truncated.rfind(' ') {
            format!("{}{}", &text[..last_space], suffix)
        } else if let Some(last_newline) = truncated.rfind('\n') {
            format!("{}{}", &text[..last_newline], suffix)
        } else {
            format!("{}{}", truncated, suffix)
        }
    }
    pub fn normalize_whitespace(text: &str) -> String {
        let trimmed = text.trim();
        let mut result = String::with_capacity(trimmed.len());
        let mut prev_was_space = false;
        for c in trimmed.chars() {
            if c.is_whitespace() {
                if !prev_was_space {
                    result.push(' ');
                    prev_was_space = true;
                }
            } else {
                result.push(c);
                prev_was_space = false;
            }
        }
        result
    }
    pub fn count_tokens_approx(text: &str) -> usize {
        let mut count = 0;
        let mut in_word = false;
        for c in text.chars() {
            if c.is_whitespace() {
                if in_word {
                    count += 1;
                    in_word = false;
                }
            } else {
                in_word = true;
            }
        }
        if in_word {
            count += 1;
        }
        count
    }
}
pub struct PromptTool;
impl PromptTool {
    pub fn few_shot_prompt(instruction: &str, examples: &[(&str, &str)], query: &str) -> String {
        let mut prompt = String::new();
        prompt.push_str(instruction);
        prompt.push_str("\n\n");
        if !examples.is_empty() {
            prompt.push_str("Examples:\n");
            for (input, output) in examples {
                prompt.push_str(&format!("Input: {}\n", input));
                prompt.push_str(&format!("Output: {}\n\n", output));
            }
        }
        prompt.push_str(&format!("Input: {}\n", query));
        prompt.push_str("Output: ");
        prompt
    }
    pub fn chain_of_thought_prompt(question: &str) -> String {
        format!(
            "Let's solve this step by step.\n\nQuestion: {}\n\nStep 1: ",
            question
        )
    }
    pub fn system_prompt(role: &str, capabilities: &[&str], constraints: &[&str]) -> String {
        let mut prompt = String::new();
        prompt.push_str(&format!("You are {}. ", role));
        if !capabilities.is_empty() {
            prompt.push_str("\n\nCapabilities:\n");
            for cap in capabilities {
                prompt.push_str(&format!("- {}\n", cap));
            }
        }
        if !constraints.is_empty() {
            prompt.push_str("\n\nConstraints:\n");
            for constraint in constraints {
                prompt.push_str(&format!("- {}\n", constraint));
            }
        }
        prompt
    }
}
