mod alibaba;
mod anthropic;
mod azure;
mod baichuan;
mod baidu;
mod cohere;
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

pub trait LLM: Send + Sync {
    fn generate(
        &self,
        prompt: &str,
    ) -> Pin<Box<dyn Future<Output = Result<LLMResult>> + Send + '_>>;

    fn generate_with_options(
        &self,
        prompt: &str,
        options: LLMOptions,
    ) -> Pin<Box<dyn Future<Output = Result<LLMResult>> + Send + '_>>;

    fn chat(
        &self,
        messages: Vec<ChatMessage>,
    ) -> Pin<Box<dyn Future<Output = Result<LLMResult>> + Send + '_>>;

    fn get_model_name(&self) -> String;

    fn get_provider_name(&self) -> String;

    fn get_provider_enum(&self) -> ModelProvider {
        ModelProvider::Local
    }

    fn supports_function_calling(&self) -> bool {
        false
    }

    fn supports_json_mode(&self) -> bool {
        false
    }

    fn max_context_length(&self) -> Option<usize> {
        None
    }
}
