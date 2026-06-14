//! LLM providers module

mod alibaba;
mod anthropic;
mod azure;
mod baichuan;
mod baidu;
mod cohere;
mod custom;
mod deepseek;
mod fireworks;
mod google;
mod groq;
mod huggingface;
mod minimax;
mod mistral;
mod moonshot;
mod openai;
mod perplexity;
mod replicate;
mod tencent;
mod together;
mod yi;
mod zhipu;

pub use alibaba::{AlibabaModel, AlibabaTongyi};
pub use anthropic::{Anthropic, AnthropicModel};
pub use azure::{AzureModel, AzureOpenAI};
pub use baichuan::{Baichuan, BaichuanModel};
pub use baidu::{BaiduModel, BaiduWenxin};
pub use cohere::{Cohere, CohereModel};
pub use custom::*;
pub use deepseek::{DeepSeek, DeepSeekModel};
pub use fireworks::{Fireworks, FireworksModel};
pub use google::{GoogleAI, GoogleModel};
pub use groq::{Groq, GroqModel};
pub use huggingface::{HuggingFace, HuggingFaceModel};
pub use minimax::{MiniMax, MiniMaxModel};
pub use mistral::{Mistral, MistralModel};
pub use moonshot::{Moonshot, MoonshotModel};
pub use openai::{OpenAI, OpenAIModel};
pub use perplexity::{Perplexity, PerplexityModel};
pub use replicate::{Replicate, ReplicateModel};
pub use tencent::{TencentHunyuan, TencentModel};
pub use together::{Together, TogetherModel};
pub use yi::{Yi, YiModel};
pub use zhipu::{ZhipuAI, ZhipuModel};

use crate::types::{ChatMessage, LangHubError, ModelProvider, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

/// Token usage information from LLM API response
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// Complete LLM API response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMResult {
    /// The generated text content
    pub text: String,
    /// Complete raw response from API (includes usage, finish_reason, logprobs, etc.)
    pub raw_response: serde_json::Value,
}

impl LLMResult {
    /// Extract usage from raw response (works for OpenAI, Anthropic, Google, etc.)
    pub fn extract_usage(&self) -> Option<Usage> {
        extract_usage_from_raw(&self.raw_response)
    }
}

/// Extract usage from various LLM API response formats
pub fn extract_usage_from_raw(raw: &serde_json::Value) -> Option<Usage> {
    // OpenAI / OpenAI-compatible format (DeepSeek, Mistral, Groq, Together, etc.)
    if let Some(u) = raw.get("usage") {
        if let (Some(prompt), Some(completion), Some(total)) = (
            u.get("prompt_tokens").and_then(|v| v.as_u64()),
            u.get("completion_tokens").and_then(|v| v.as_u64()),
            u.get("total_tokens").and_then(|v| v.as_u64()),
        ) {
            return Some(Usage {
                prompt_tokens: prompt as u32,
                completion_tokens: completion as u32,
                total_tokens: total as u32,
            });
        }
    }

    // Anthropic format
    if let Some(u) = raw.get("usage") {
        if let (Some(prompt), Some(completion)) = (
            u.get("input_tokens").and_then(|v| v.as_u64()),
            u.get("output_tokens").and_then(|v| v.as_u64()),
        ) {
            return Some(Usage {
                prompt_tokens: prompt as u32,
                completion_tokens: completion as u32,
                total_tokens: (prompt + completion) as u32,
            });
        }
    }

    // Google Gemini format
    if let Some(u) = raw.get("usageMetadata") {
        if let (Some(prompt), Some(completion), Some(total)) = (
            u.get("promptTokenCount").and_then(|v| v.as_u64()),
            u.get("candidatesTokenCount").and_then(|v| v.as_u64()),
            u.get("totalTokenCount").and_then(|v| v.as_u64()),
        ) {
            return Some(Usage {
                prompt_tokens: prompt as u32,
                completion_tokens: completion as u32,
                total_tokens: total as u32,
            });
        }
    }

    // Cohere format
    if let Some(u) = raw.get("meta").and_then(|m| m.get("billed_units")) {
        if let Some(prompt) = u.get("input_tokens").and_then(|v| v.as_u64()) {
            if let Some(completion) = u.get("output_tokens").and_then(|v| v.as_u64()) {
                return Some(Usage {
                    prompt_tokens: prompt as u32,
                    completion_tokens: completion as u32,
                    total_tokens: (prompt + completion) as u32,
                });
            }
        }
    }

    // HuggingFace format (some endpoints return usage)
    if let Some(u) = raw.get("usage") {
        if let (Some(prompt), Some(completion)) = (
            u.get("prompt_tokens").and_then(|v| v.as_u64()),
            u.get("completion_tokens").and_then(|v| v.as_u64()),
        ) {
            return Some(Usage {
                prompt_tokens: prompt as u32,
                completion_tokens: completion as u32,
                total_tokens: (prompt + completion) as u32,
            });
        }
    }

    None
}

#[derive(Debug, Clone)]
pub struct LLMOptions {
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub repetition_penalty: Option<f32>,
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
            repetition_penalty: None,
            stop_sequences: None,
            seed: None,
            response_format: None,
        }
    }
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

/// LLM trait - unified interface for all providers
pub trait LLM: Send + Sync {
    /// Generate text from a prompt
    fn generate(
        &self,
        prompt: &str,
    ) -> Pin<Box<dyn Future<Output = Result<LLMResult>> + Send + '_>>;

    /// Generate text with options
    fn generate_with_options(
        &self,
        prompt: &str,
        options: LLMOptions,
    ) -> Pin<Box<dyn Future<Output = Result<LLMResult>> + Send + '_>>;

    /// Chat with message history
    fn chat(
        &self,
        messages: Vec<ChatMessage>,
    ) -> Pin<Box<dyn Future<Output = Result<LLMResult>> + Send + '_>>;

    /// Chat with options
    fn chat_with_options(
        &self,
        messages: Vec<ChatMessage>,
        options: LLMOptions,
    ) -> Pin<Box<dyn Future<Output = Result<LLMResult>> + Send + '_>> {
        Box::pin(async move { self.chat(messages).await })
    }

    /// Get model name
    fn get_model_name(&self) -> String;

    /// Get provider name
    fn get_provider_name(&self) -> String;

    /// Get provider enum
    fn get_provider_enum(&self) -> ModelProvider {
        ModelProvider::Custom
    }

    /// Check if provider supports function calling
    fn supports_function_calling(&self) -> bool {
        false
    }

    /// Check if provider supports JSON mode
    fn supports_json_mode(&self) -> bool {
        false
    }

    /// Get max context length
    fn max_context_length(&self) -> Option<usize> {
        None
    }
}
