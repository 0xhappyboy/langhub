use crate::types::*;
use serde::{Deserialize, Serialize};
use std::future::Future;
use std::{collections::HashMap, pin::Pin};
mod anthropic;
mod deepseek;
mod google;
mod openai;

pub use anthropic::{Anthropic, AnthropicModel};
pub use deepseek::{DeepSeek, DeepSeekModel};
pub use google::{GoogleAI, GoogleModel};
pub use openai::{OpenAI, OpenAIModel};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMResult {
    pub text: String,
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Debug, Clone)]
pub struct LLMOptions {
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub stop_sequences: Option<Vec<String>>,
    pub seed: Option<u64>,
    pub response_format: Option<ResponseFormat>,
}

#[derive(Debug, Clone)]
pub enum ResponseFormat {
    Text,
    Json,
    JsonSchema { schema: serde_json::Value },
}

impl Default for LLMOptions {
    fn default() -> Self {
        Self {
            temperature: Some(0.7),
            max_tokens: Some(4096),
            top_p: Some(0.95),
            top_k: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop_sequences: None,
            seed: None,
            response_format: None,
        }
    }
}

pub trait LLM: Send + Sync {
    fn generate(&self, prompt: &str) -> Pin<Box<dyn Future<Output = Result<LLMResult>> + Send + '_>>;

    fn generate_with_options(
        &self,
        prompt: &str,
        options: LLMOptions,
    ) -> Pin<Box<dyn Future<Output = Result<LLMResult>> + Send + '_>>;

    fn generate_batch(
        &self,
        prompts: Vec<String>,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<LLMResult>>> + Send + '_>> {
        Box::pin(async move {
            let mut results = Vec::new();
            for prompt in prompts {
                results.push(self.generate(&prompt).await?);
            }
            Ok(results)
        })
    }

    fn chat(
        &self,
        messages: Vec<ChatMessage>,
    ) -> Pin<Box<dyn Future<Output = Result<LLMResult>> + Send + '_>> {
        Box::pin(async move {
            let prompt = messages
                .iter()
                .map(|m| format!("{}: {}", m.role, m.content))
                .collect::<Vec<_>>()
                .join("\n");
            self.generate(&prompt).await
        })
    }

    fn get_model_name(&self) -> &str;
    fn get_provider_name(&self) -> &str;
    fn supports_function_calling(&self) -> bool {
        false
    }
    fn supports_json_mode(&self) -> bool {
        false
    }
    fn max_context_length(&self) -> Option<usize> {
        None
    }

    fn get_provider_enum(&self) -> ModelProvider;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    pub name: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub r#type: String,
    pub function: FunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

impl ChatMessage {
    pub fn user(content: &str) -> Self {
        Self {
            role: "user".to_string(),
            content: content.to_string(),
            name: None,
            tool_calls: None,
        }
    }

    pub fn assistant(content: &str) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.to_string(),
            name: None,
            tool_calls: None,
        }
    }

    pub fn system(content: &str) -> Self {
        Self {
            role: "system".to_string(),
            content: content.to_string(),
            name: None,
            tool_calls: None,
        }
    }
}