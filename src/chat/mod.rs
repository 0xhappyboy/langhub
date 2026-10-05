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
use crate::types::{ChatMessage, LLMVendor, LangHubError, LangHubResult};
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
use serde::{Deserialize, Serialize};
use std::future::Future;
use std::pin::Pin;
use std::{collections::HashMap, fmt};
pub use tencent::{TencentHunyuan, TencentModel};
pub use together::{Together, TogetherModel};
pub use yi::{Yi, YiModel};
pub use zhipu::{ZhipuAI, ZhipuModel};
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
    ) -> Pin<Box<dyn Future<Output = LangHubResult<LLMResult>> + Send + '_>>;
    /// Generate text with options
    fn generate_with_options(
        &self,
        prompt: &str,
        options: LLMOptions,
    ) -> Pin<Box<dyn Future<Output = LangHubResult<LLMResult>> + Send + '_>>;
    /// Chat with message history
    fn chat(
        &self,
        messages: Vec<ChatMessage>,
        model: Option<&str>,
    ) -> Pin<Box<dyn Future<Output = LangHubResult<LLMResult>> + Send + '_>>;
    /// Chat with options
    fn chat_with_options<'a>(
        &'a self,
        messages: Vec<ChatMessage>,
        options: LLMOptions,
        model: Option<&'a str>,
    ) -> Pin<Box<dyn Future<Output = LangHubResult<LLMResult>> + Send + 'a>> {
        Box::pin(async move { self.chat(messages, model).await })
    }
    /// Get model name
    fn get_model_name(&self) -> String;
    /// Get provider name
    fn get_provider_name(&self) -> String;
    /// Get provider enum
    fn get_provider_enum(&self) -> ChatModelProvider {
        ChatModelProvider::Custom
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChatModelProvider {
    OpenAI,
    Anthropic,
    Google,
    DeepSeek,
    Cohere,
    HuggingFace,
    Azure,
    Mistral,
    Groq,
    Together,
    Replicate,
    Fireworks,
    Perplexity,
    Baidu,
    Alibaba,
    Tencent,
    Zhipu,
    MiniMax,
    Moonshot,
    Baichuan,
    Yi,
    Custom,
}
impl fmt::Display for ChatModelProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ChatModelProvider::OpenAI => write!(f, "OpenAI"),
            ChatModelProvider::Anthropic => write!(f, "Anthropic"),
            ChatModelProvider::Google => write!(f, "Google"),
            ChatModelProvider::DeepSeek => write!(f, "DeepSeek"),
            ChatModelProvider::Cohere => write!(f, "Cohere"),
            ChatModelProvider::HuggingFace => write!(f, "HuggingFace"),
            ChatModelProvider::Azure => write!(f, "Azure"),
            ChatModelProvider::Mistral => write!(f, "Mistral"),
            ChatModelProvider::Groq => write!(f, "Groq"),
            ChatModelProvider::Together => write!(f, "Together"),
            ChatModelProvider::Replicate => write!(f, "Replicate"),
            ChatModelProvider::Fireworks => write!(f, "Fireworks"),
            ChatModelProvider::Perplexity => write!(f, "Perplexity"),
            ChatModelProvider::Baidu => write!(f, "Baidu"),
            ChatModelProvider::Alibaba => write!(f, "Alibaba"),
            ChatModelProvider::Tencent => write!(f, "Tencent"),
            ChatModelProvider::Zhipu => write!(f, "Zhipu"),
            ChatModelProvider::MiniMax => write!(f, "MiniMax"),
            ChatModelProvider::Moonshot => write!(f, "Moonshot"),
            ChatModelProvider::Baichuan => write!(f, "Baichuan"),
            ChatModelProvider::Yi => write!(f, "Yi"),
            ChatModelProvider::Custom => write!(f, "Custom"),
        }
    }
}
impl ChatModelProvider {
    /// Parse a frontend provider string to `ChatModelProvider`.
    pub fn parse_chat_provider(name: &str) -> LangHubResult<ChatModelProvider> {
        match name.to_lowercase().as_str() {
            "openai" => Ok(ChatModelProvider::OpenAI),
            "anthropic" => Ok(ChatModelProvider::Anthropic),
            "google" => Ok(ChatModelProvider::Google),
            "deepseek" => Ok(ChatModelProvider::DeepSeek),
            "cohere" => Ok(ChatModelProvider::Cohere),
            "huggingface" | "hugging_face" => Ok(ChatModelProvider::HuggingFace),
            "azure" => Ok(ChatModelProvider::Azure),
            "mistral" => Ok(ChatModelProvider::Mistral),
            "groq" => Ok(ChatModelProvider::Groq),
            "together" => Ok(ChatModelProvider::Together),
            "replicate" => Ok(ChatModelProvider::Replicate),
            "fireworks" => Ok(ChatModelProvider::Fireworks),
            "perplexity" => Ok(ChatModelProvider::Perplexity),
            "baidu" => Ok(ChatModelProvider::Baidu),
            "alibaba" => Ok(ChatModelProvider::Alibaba),
            "tencent" => Ok(ChatModelProvider::Tencent),
            "zhipu" => Ok(ChatModelProvider::Zhipu),
            "minimax" => Ok(ChatModelProvider::MiniMax),
            "moonshot" => Ok(ChatModelProvider::Moonshot),
            "baichuan" => Ok(ChatModelProvider::Baichuan),
            "yi" => Ok(ChatModelProvider::Yi),
            "custom" => Ok(ChatModelProvider::Custom),
            other => Err(LangHubError::LLMError(format!(
                "Unknown chat provider: {}",
                other
            ))),
        }
    }
    /// provider id .
    pub fn id(&self) -> &'static str {
        match self {
            ChatModelProvider::OpenAI => "openai",
            ChatModelProvider::Anthropic => "anthropic",
            ChatModelProvider::Google => "google",
            ChatModelProvider::DeepSeek => "deepseek",
            ChatModelProvider::Cohere => "cohere",
            ChatModelProvider::HuggingFace => "huggingface",
            ChatModelProvider::Azure => "azure",
            ChatModelProvider::Mistral => "mistral",
            ChatModelProvider::Groq => "groq",
            ChatModelProvider::Together => "together",
            ChatModelProvider::Replicate => "replicate",
            ChatModelProvider::Fireworks => "fireworks",
            ChatModelProvider::Perplexity => "perplexity",
            ChatModelProvider::Baidu => "baidu",
            ChatModelProvider::Alibaba => "alibaba",
            ChatModelProvider::Tencent => "tencent",
            ChatModelProvider::Zhipu => "zhipu",
            ChatModelProvider::MiniMax => "minimax",
            ChatModelProvider::Moonshot => "moonshot",
            ChatModelProvider::Baichuan => "baichuan",
            ChatModelProvider::Yi => "yi",
            ChatModelProvider::Custom => "custom",
        }
    }
    pub fn all() -> Vec<ChatModelProvider> {
        vec![
            ChatModelProvider::OpenAI,
            ChatModelProvider::Anthropic,
            ChatModelProvider::Google,
            ChatModelProvider::DeepSeek,
            ChatModelProvider::Cohere,
            ChatModelProvider::HuggingFace,
            ChatModelProvider::Azure,
            ChatModelProvider::Mistral,
            ChatModelProvider::Groq,
            ChatModelProvider::Together,
            ChatModelProvider::Replicate,
            ChatModelProvider::Fireworks,
            ChatModelProvider::Perplexity,
            ChatModelProvider::Baidu,
            ChatModelProvider::Alibaba,
            ChatModelProvider::Tencent,
            ChatModelProvider::Zhipu,
            ChatModelProvider::MiniMax,
            ChatModelProvider::Moonshot,
            ChatModelProvider::Baichuan,
            ChatModelProvider::Yi,
            ChatModelProvider::Custom,
        ]
    }
    /// Returns the vendor of this model provider.
    pub fn vendor(&self) -> LLMVendor {
        match self {
            ChatModelProvider::OpenAI => LLMVendor::OpenAI,
            ChatModelProvider::Anthropic => LLMVendor::Anthropic,
            ChatModelProvider::Google => LLMVendor::Google,
            ChatModelProvider::DeepSeek => LLMVendor::DeepSeek,
            ChatModelProvider::Cohere => LLMVendor::Cohere,
            ChatModelProvider::HuggingFace => LLMVendor::HuggingFace,
            ChatModelProvider::Azure => LLMVendor::Microsoft,
            ChatModelProvider::Mistral => LLMVendor::Mistral,
            ChatModelProvider::Groq => LLMVendor::Groq,
            ChatModelProvider::Together => LLMVendor::Together,
            ChatModelProvider::Replicate => LLMVendor::Replicate,
            ChatModelProvider::Fireworks => LLMVendor::Fireworks,
            ChatModelProvider::Perplexity => LLMVendor::Perplexity,
            ChatModelProvider::Baidu => LLMVendor::Baidu,
            ChatModelProvider::Alibaba => LLMVendor::Alibaba,
            ChatModelProvider::Tencent => LLMVendor::Tencent,
            ChatModelProvider::Zhipu => LLMVendor::Zhipu,
            ChatModelProvider::MiniMax => LLMVendor::MiniMax,
            ChatModelProvider::Moonshot => LLMVendor::Moonshot,
            ChatModelProvider::Baichuan => LLMVendor::Baichuan,
            ChatModelProvider::Yi => LLMVendor::Yi,
            ChatModelProvider::Custom => LLMVendor::Custom,
        }
    }
    /// Returns a short human-readable description of this provider (English).
    pub fn description(&self) -> &'static str {
        match self {
            ChatModelProvider::OpenAI => {
                "OpenAI: GPT-4, GPT-4o and o-series reasoning models with strong general capability"
            }
            ChatModelProvider::Anthropic => {
                "Anthropic Claude: long-context, safety-focused models with strong reasoning and tool use"
            }
            ChatModelProvider::Google => {
                "Google Gemini: multimodal models with very long context and native vision/audio grounding"
            }
            ChatModelProvider::DeepSeek => {
                "DeepSeek: cost-effective chat, coder and reasoning models with strong math and code ability"
            }
            ChatModelProvider::Cohere => {
                "Cohere Command: enterprise RAG and tool-use oriented language models"
            }
            ChatModelProvider::HuggingFace => {
                "HuggingFace: hosted open-source models via Inference API for flexible experimentation"
            }
            ChatModelProvider::Azure => {
                "Azure OpenAI: Microsoft-hosted OpenAI models with enterprise compliance and regional deployment"
            }
            ChatModelProvider::Mistral => {
                "Mistral AI: efficient European models from tiny to large, including Codestral for code"
            }
            ChatModelProvider::Groq => {
                "Groq: ultra-low-latency inference for open-source models via custom LPU hardware"
            }
            ChatModelProvider::Together => {
                "Together.ai: hosted open-source models with broad selection and competitive pricing"
            }
            ChatModelProvider::Replicate => {
                "Replicate: run open-source models in the cloud with pay-per-use pricing"
            }
            ChatModelProvider::Fireworks => {
                "Fireworks AI: fast inference for open-source models with function calling support"
            }
            ChatModelProvider::Perplexity => {
                "Perplexity Sonar: online-search-augmented models with real-time web grounding"
            }
            ChatModelProvider::Baidu => {
                "Baidu ERNIE: Chinese-first language models with strong Chinese understanding"
            }
            ChatModelProvider::Alibaba => {
                "Alibaba Qwen: Tongyi Qianwen models with multilingual and multimodal capability"
            }
            ChatModelProvider::Tencent => {
                "Tencent Hunyuan: Chinese language models with enterprise-grade deployment"
            }
            ChatModelProvider::Zhipu => {
                "Zhipu GLM: Chinese-first models with strong bilingual and reasoning capability"
            }
            ChatModelProvider::MiniMax => {
                "MiniMax abab: long-context Chinese models with strong conversational ability"
            }
            ChatModelProvider::Moonshot => {
                "Moonshot Kimi: long-context Chinese models with strong document understanding"
            }
            ChatModelProvider::Baichuan => {
                "Baichuan: Chinese-first models with strong general and medical-domain capability"
            }
            ChatModelProvider::Yi => {
                "01.AI Yi: bilingual open-source models with strong reasoning and long context"
            }
            ChatModelProvider::Custom => {
                "Custom: any OpenAI-compatible endpoint with your own API base URL"
            }
        }
    }
    /// Returns a short human-readable description of this provider (Chinese).
    pub fn description_zh(&self) -> &'static str {
        match self {
            ChatModelProvider::OpenAI => "OpenAI：GPT-4、GPT-4o 与 o 系列推理模型，通用能力最强",
            ChatModelProvider::Anthropic => {
                "Anthropic Claude：长上下文、安全导向，推理与工具调用能力强"
            }
            ChatModelProvider::Google => {
                "Google Gemini：多模态模型，超长上下文，原生视觉与音频理解"
            }
            ChatModelProvider::DeepSeek => {
                "DeepSeek：高性价比对话、代码与推理模型，数学和编程能力突出"
            }
            ChatModelProvider::Cohere => "Cohere Command：面向企业 RAG 与工具调用的语言模型",
            ChatModelProvider::HuggingFace => {
                "HuggingFace：通过 Inference API 托管开源模型，便于灵活实验"
            }
            ChatModelProvider::Azure => {
                "Azure OpenAI：微软托管的 OpenAI 模型，企业合规与区域化部署"
            }
            ChatModelProvider::Mistral => {
                "Mistral AI：高效的欧洲模型，覆盖 tiny 到 large，含 Codestral 代码模型"
            }
            ChatModelProvider::Groq => "Groq：基于自研 LPU 硬件的超低延迟开源模型推理",
            ChatModelProvider::Together => "Together.ai：托管开源模型，选择丰富、价格有竞争力",
            ChatModelProvider::Replicate => "Replicate：云端运行开源模型，按量付费",
            ChatModelProvider::Fireworks => "Fireworks AI：开源模型高速推理，支持函数调用",
            ChatModelProvider::Perplexity => "Perplexity Sonar：联网搜索增强模型，实时 Web 接地",
            ChatModelProvider::Baidu => "百度文心 ERNIE：中文优先语言模型，中文理解强",
            ChatModelProvider::Alibaba => "阿里云通义千问 Qwen：多语言、多模态能力",
            ChatModelProvider::Tencent => "腾讯混元：中文语言模型，面向企业级部署",
            ChatModelProvider::Zhipu => "智谱 GLM：中文优先模型，中英双语与推理能力强",
            ChatModelProvider::MiniMax => "MiniMax abab：长上下文中文模型，对话能力强",
            ChatModelProvider::Moonshot => "月之暗面 Kimi：长上下文中文模型，文档理解强",
            ChatModelProvider::Baichuan => "百川智能：中文优先模型，通用与医疗领域能力强",
            ChatModelProvider::Yi => "零一万物 Yi：中英双语开源模型，推理与长上下文能力强",
            ChatModelProvider::Custom => "自定义：任意 OpenAI 兼容端点，可配置自己的 API Base URL",
        }
    }
    pub fn supports_function_calling(&self) -> bool {
        match self {
            ChatModelProvider::OpenAI => true,
            ChatModelProvider::Anthropic => true,
            ChatModelProvider::Google => true,
            ChatModelProvider::DeepSeek => true,
            ChatModelProvider::Cohere => true,
            ChatModelProvider::Zhipu => true,
            ChatModelProvider::Moonshot => true,
            ChatModelProvider::Mistral => true,
            ChatModelProvider::Groq => true,
            ChatModelProvider::Together => true,
            ChatModelProvider::Fireworks => true,
            ChatModelProvider::Perplexity => true,
            ChatModelProvider::Baidu => true,
            ChatModelProvider::Alibaba => true,
            ChatModelProvider::Tencent => true,
            ChatModelProvider::MiniMax => true,
            ChatModelProvider::Baichuan => true,
            ChatModelProvider::Yi => true,
            ChatModelProvider::Custom => true,
            _ => false,
        }
    }
    pub fn supports_json_mode(&self) -> bool {
        match self {
            ChatModelProvider::OpenAI => true,
            ChatModelProvider::Anthropic => true,
            ChatModelProvider::DeepSeek => true,
            ChatModelProvider::Google => true,
            ChatModelProvider::Mistral => true,
            ChatModelProvider::Groq => true,
            ChatModelProvider::Together => true,
            ChatModelProvider::Fireworks => true,
            ChatModelProvider::Perplexity => true,
            ChatModelProvider::Baidu => true,
            ChatModelProvider::Alibaba => true,
            ChatModelProvider::Tencent => true,
            ChatModelProvider::Zhipu => true,
            ChatModelProvider::MiniMax => true,
            ChatModelProvider::Moonshot => true,
            ChatModelProvider::Baichuan => true,
            ChatModelProvider::Yi => true,
            ChatModelProvider::Custom => true,
            _ => false,
        }
    }
    /// Returns a short emoji icon for this provider.
    pub fn icon(&self) -> &'static str {
        match self {
            ChatModelProvider::OpenAI => "🔵",
            ChatModelProvider::Anthropic => "🟣",
            ChatModelProvider::Google => "🔴",
            ChatModelProvider::DeepSeek => "🟢",
            ChatModelProvider::Cohere => "📐",
            ChatModelProvider::HuggingFace => "🤗",
            ChatModelProvider::Azure => "☁️",
            ChatModelProvider::Mistral => "🪶",
            ChatModelProvider::Groq => "⚡",
            ChatModelProvider::Together => "🤝",
            ChatModelProvider::Replicate => "🔁",
            ChatModelProvider::Fireworks => "🎆",
            ChatModelProvider::Perplexity => "🔎",
            ChatModelProvider::Baidu => "🔍",
            ChatModelProvider::Alibaba => "☁️",
            ChatModelProvider::Tencent => "🐧",
            ChatModelProvider::Zhipu => "🧠",
            ChatModelProvider::MiniMax => "🎯",
            ChatModelProvider::Moonshot => "🌙",
            ChatModelProvider::Baichuan => "🌊",
            ChatModelProvider::Yi => "1️⃣",
            ChatModelProvider::Custom => "🦛",
        }
    }
    /// Whether this provider requires an API key.
    pub fn requires_api_key(&self) -> bool {
        true
    }
    /// Whether this provider needs additional config fields beyond the API key.
    pub fn needs_extra_config(&self) -> bool {
        matches!(
            self,
            ChatModelProvider::Azure
                | ChatModelProvider::Baidu
                | ChatModelProvider::Tencent
                | ChatModelProvider::MiniMax
                | ChatModelProvider::Custom
        )
    }
    /// Returns the extra config fields required by this provider.
    pub fn extra_config_fields(&self) -> Vec<(String, String, String, bool)> {
        match self {
            ChatModelProvider::Azure => vec![
                (
                    "endpoint".to_string(),
                    "Endpoint URL".to_string(),
                    "https://your-resource.openai.azure.com/".to_string(),
                    true,
                ),
                (
                    "deployment_name".to_string(),
                    "Deployment Name".to_string(),
                    "gpt-4".to_string(),
                    true,
                ),
            ],
            ChatModelProvider::Baidu => vec![(
                "secret_key".to_string(),
                "Secret Key".to_string(),
                "your secret key".to_string(),
                true,
            )],
            ChatModelProvider::Tencent => vec![
                (
                    "secret_id".to_string(),
                    "Secret ID".to_string(),
                    "your secret id".to_string(),
                    true,
                ),
                (
                    "secret_key".to_string(),
                    "Secret Key".to_string(),
                    "your secret key".to_string(),
                    true,
                ),
            ],
            ChatModelProvider::MiniMax => vec![(
                "group_id".to_string(),
                "Group ID".to_string(),
                "your group id".to_string(),
                true,
            )],
            ChatModelProvider::Custom => vec![(
                "api_base".to_string(),
                "API Base URL".to_string(),
                "https://api.example.com/v1".to_string(),
                true,
            )],
            _ => vec![],
        }
    }
    /// Returns all concrete models under this provider, as
    /// `(model_id, display_name, is_recommended)` tuples.
    pub fn models(&self) -> Vec<(String, String, bool)> {
        match self {
            ChatModelProvider::OpenAI => vec![
                ("gpt-4o".to_string(), "GPT-4o".to_string(), true),
                ("gpt-4o-mini".to_string(), "GPT-4o mini".to_string(), true),
                ("gpt-4-turbo".to_string(), "GPT-4 Turbo".to_string(), false),
                ("gpt-4".to_string(), "GPT-4".to_string(), false),
                ("gpt-4-32k".to_string(), "GPT-4 32K".to_string(), false),
                (
                    "gpt-3.5-turbo".to_string(),
                    "GPT-3.5 Turbo".to_string(),
                    false,
                ),
                (
                    "gpt-3.5-turbo-16k".to_string(),
                    "GPT-3.5 Turbo 16K".to_string(),
                    false,
                ),
                ("o1".to_string(), "o1".to_string(), false),
                ("o1-mini".to_string(), "o1-mini".to_string(), false),
                ("o1-preview".to_string(), "o1-preview".to_string(), false),
                ("o3-mini".to_string(), "o3-mini".to_string(), false),
            ],
            ChatModelProvider::Anthropic => vec![
                (
                    "claude-sonnet-4-20250514".to_string(),
                    "Claude Sonnet 4".to_string(),
                    true,
                ),
                (
                    "claude-opus-4-20250514".to_string(),
                    "Claude Opus 4".to_string(),
                    true,
                ),
                (
                    "claude-3-7-sonnet-20250219".to_string(),
                    "Claude 3.7 Sonnet".to_string(),
                    false,
                ),
                (
                    "claude-3-5-sonnet-20241022".to_string(),
                    "Claude 3.5 Sonnet".to_string(),
                    false,
                ),
                (
                    "claude-3-5-haiku-20241022".to_string(),
                    "Claude 3.5 Haiku".to_string(),
                    false,
                ),
                (
                    "claude-3-opus-20240229".to_string(),
                    "Claude 3 Opus".to_string(),
                    false,
                ),
                (
                    "claude-3-sonnet-20240229".to_string(),
                    "Claude 3 Sonnet".to_string(),
                    false,
                ),
                (
                    "claude-3-haiku-20240307".to_string(),
                    "Claude 3 Haiku".to_string(),
                    false,
                ),
                ("claude-2.1".to_string(), "Claude 2.1".to_string(), false),
                ("claude-2.0".to_string(), "Claude 2.0".to_string(), false),
            ],
            ChatModelProvider::Google => vec![
                (
                    "gemini-2.5-pro".to_string(),
                    "Gemini 2.5 Pro".to_string(),
                    true,
                ),
                (
                    "gemini-2.5-flash".to_string(),
                    "Gemini 2.5 Flash".to_string(),
                    true,
                ),
                (
                    "gemini-2.0-flash".to_string(),
                    "Gemini 2.0 Flash".to_string(),
                    false,
                ),
                (
                    "gemini-2.0-flash-lite".to_string(),
                    "Gemini 2.0 Flash Lite".to_string(),
                    false,
                ),
                (
                    "gemini-1.5-pro".to_string(),
                    "Gemini 1.5 Pro".to_string(),
                    false,
                ),
                (
                    "gemini-1.5-flash".to_string(),
                    "Gemini 1.5 Flash".to_string(),
                    false,
                ),
                (
                    "gemini-1.5-pro-vision".to_string(),
                    "Gemini 1.5 Pro Vision".to_string(),
                    false,
                ),
                (
                    "gemini-1.0-pro".to_string(),
                    "Gemini 1.0 Pro".to_string(),
                    false,
                ),
            ],
            ChatModelProvider::DeepSeek => vec![
                (
                    "deepseek-chat".to_string(),
                    "DeepSeek Chat".to_string(),
                    true,
                ),
                (
                    "deepseek-reasoner".to_string(),
                    "DeepSeek Reasoner".to_string(),
                    true,
                ),
                (
                    "deepseek-coder".to_string(),
                    "DeepSeek Coder".to_string(),
                    false,
                ),
            ],
            ChatModelProvider::Cohere => vec![
                (
                    "command-a-03-2025".to_string(),
                    "Command A".to_string(),
                    true,
                ),
                ("command-r-plus".to_string(), "Command R+".to_string(), true),
                ("command-r".to_string(), "Command R".to_string(), false),
                (
                    "command-r7b-12-2024".to_string(),
                    "Command R7B".to_string(),
                    false,
                ),
                (
                    "command-light".to_string(),
                    "Command Light".to_string(),
                    false,
                ),
                (
                    "command-nightly".to_string(),
                    "Command Nightly".to_string(),
                    false,
                ),
            ],
            ChatModelProvider::HuggingFace => vec![
                (
                    "meta-llama/Meta-Llama-3-8B-Instruct".to_string(),
                    "Llama 3 8B Instruct".to_string(),
                    true,
                ),
                (
                    "meta-llama/Meta-Llama-3-70B-Instruct".to_string(),
                    "Llama 3 70B Instruct".to_string(),
                    true,
                ),
                (
                    "meta-llama/Llama-2-7b-chat-hf".to_string(),
                    "Llama 2 7B Chat".to_string(),
                    false,
                ),
                (
                    "meta-llama/Llama-2-13b-chat-hf".to_string(),
                    "Llama 2 13B Chat".to_string(),
                    false,
                ),
                (
                    "meta-llama/Llama-2-70b-chat-hf".to_string(),
                    "Llama 2 70B Chat".to_string(),
                    false,
                ),
                (
                    "mistralai/Mistral-7B-Instruct-v0.2".to_string(),
                    "Mistral 7B Instruct".to_string(),
                    false,
                ),
                (
                    "mistralai/Mixtral-8x7B-Instruct-v0.1".to_string(),
                    "Mixtral 8x7B Instruct".to_string(),
                    false,
                ),
                (
                    "Qwen/Qwen2.5-72B-Instruct".to_string(),
                    "Qwen 2.5 72B Instruct".to_string(),
                    false,
                ),
                (
                    "tiiuae/falcon-7b-instruct".to_string(),
                    "Falcon 7B Instruct".to_string(),
                    false,
                ),
                (
                    "tiiuae/falcon-40b-instruct".to_string(),
                    "Falcon 40B Instruct".to_string(),
                    false,
                ),
            ],
            ChatModelProvider::Azure => vec![
                ("gpt-4o".to_string(), "GPT-4o (Azure)".to_string(), true),
                (
                    "gpt-4o-mini".to_string(),
                    "GPT-4o mini (Azure)".to_string(),
                    true,
                ),
                (
                    "gpt-4-turbo".to_string(),
                    "GPT-4 Turbo (Azure)".to_string(),
                    false,
                ),
                ("gpt-4".to_string(), "GPT-4 (Azure)".to_string(), false),
                (
                    "gpt-35-turbo".to_string(),
                    "GPT-3.5 Turbo (Azure)".to_string(),
                    false,
                ),
            ],
            ChatModelProvider::Mistral => vec![
                (
                    "mistral-large-latest".to_string(),
                    "Mistral Large".to_string(),
                    true,
                ),
                (
                    "mistral-medium-latest".to_string(),
                    "Mistral Medium".to_string(),
                    true,
                ),
                (
                    "mistral-small-latest".to_string(),
                    "Mistral Small".to_string(),
                    false,
                ),
                (
                    "pixtral-large-latest".to_string(),
                    "Pixtral Large".to_string(),
                    false,
                ),
                (
                    "codestral-latest".to_string(),
                    "Codestral".to_string(),
                    false,
                ),
                (
                    "open-mixtral-8x22b".to_string(),
                    "Mixtral 8x22B".to_string(),
                    false,
                ),
                (
                    "open-mistral-7b".to_string(),
                    "Mistral 7B".to_string(),
                    false,
                ),
            ],
            ChatModelProvider::Groq => vec![
                (
                    "llama-3.3-70b-versatile".to_string(),
                    "Llama 3.3 70B Versatile".to_string(),
                    true,
                ),
                (
                    "llama-3.1-8b-instant".to_string(),
                    "Llama 3.1 8B Instant".to_string(),
                    true,
                ),
                (
                    "mixtral-8x7b-32768".to_string(),
                    "Mixtral 8x7B".to_string(),
                    false,
                ),
                ("gemma2-9b-it".to_string(), "Gemma 2 9B".to_string(), false),
                (
                    "deepseek-r1-distill-llama-70b".to_string(),
                    "DeepSeek R1 Distill Llama 70B".to_string(),
                    false,
                ),
                (
                    "llama3-8b-8192".to_string(),
                    "Llama 3 8B".to_string(),
                    false,
                ),
                (
                    "llama3-70b-8192".to_string(),
                    "Llama 3 70B".to_string(),
                    false,
                ),
            ],
            ChatModelProvider::Together => vec![
                (
                    "meta-llama/Llama-3.3-70B-Instruct-Turbo".to_string(),
                    "Llama 3.3 70B Turbo".to_string(),
                    true,
                ),
                (
                    "Qwen/Qwen2.5-72B-Instruct-Turbo".to_string(),
                    "Qwen 2.5 72B Turbo".to_string(),
                    true,
                ),
                (
                    "meta-llama/Llama-3-8b-chat-hf".to_string(),
                    "Llama 3 8B Chat".to_string(),
                    false,
                ),
                (
                    "meta-llama/Llama-3-70b-chat-hf".to_string(),
                    "Llama 3 70B Chat".to_string(),
                    false,
                ),
                (
                    "meta-llama/Llama-2-70b-chat-hf".to_string(),
                    "Llama 2 70B Chat".to_string(),
                    false,
                ),
                (
                    "mistralai/Mixtral-8x7B-Instruct-v0.1".to_string(),
                    "Mixtral 8x7B".to_string(),
                    false,
                ),
                (
                    "mistralai/Mistral-7B-Instruct-v0.2".to_string(),
                    "Mistral 7B".to_string(),
                    false,
                ),
                (
                    "google/gemma-7b-it".to_string(),
                    "Gemma 7B".to_string(),
                    false,
                ),
                (
                    "deepseek-ai/DeepSeek-R1".to_string(),
                    "DeepSeek R1".to_string(),
                    false,
                ),
            ],
            ChatModelProvider::Replicate => vec![
                (
                    "meta/meta-llama-3-8b-instruct".to_string(),
                    "Llama 3 8B Instruct".to_string(),
                    true,
                ),
                (
                    "meta/meta-llama-3-70b-instruct".to_string(),
                    "Llama 3 70B Instruct".to_string(),
                    true,
                ),
                (
                    "mistralai/mixtral-8x7b-instruct-v0.1".to_string(),
                    "Mixtral 8x7B".to_string(),
                    false,
                ),
                (
                    "mistralai/mistral-7b-instruct-v0.2".to_string(),
                    "Mistral 7B".to_string(),
                    false,
                ),
            ],
            ChatModelProvider::Fireworks => vec![
                (
                    "accounts/fireworks/models/llama-v3-8b-instruct".to_string(),
                    "Llama 3 8B Instruct".to_string(),
                    true,
                ),
                (
                    "accounts/fireworks/models/llama-v3-70b-instruct".to_string(),
                    "Llama 3 70B Instruct".to_string(),
                    true,
                ),
                (
                    "accounts/fireworks/models/mixtral-8x7b-instruct".to_string(),
                    "Mixtral 8x7B".to_string(),
                    false,
                ),
                (
                    "accounts/fireworks/models/mistral-7b-instruct-v2".to_string(),
                    "Mistral 7B".to_string(),
                    false,
                ),
            ],
            ChatModelProvider::Perplexity => vec![
                ("sonar".to_string(), "Sonar".to_string(), true),
                ("sonar-pro".to_string(), "Sonar Pro".to_string(), true),
                (
                    "sonar-reasoning".to_string(),
                    "Sonar Reasoning".to_string(),
                    false,
                ),
                (
                    "sonar-reasoning-pro".to_string(),
                    "Sonar Reasoning Pro".to_string(),
                    false,
                ),
                (
                    "sonar-deep-research".to_string(),
                    "Sonar Deep Research".to_string(),
                    false,
                ),
            ],
            ChatModelProvider::Baidu => vec![
                ("ernie-4.0".to_string(), "ERNIE 4.0".to_string(), true),
                ("ernie-3.5".to_string(), "ERNIE 3.5".to_string(), true),
                ("ernie-speed".to_string(), "ERNIE Speed".to_string(), false),
                ("ernie-lite".to_string(), "ERNIE Lite".to_string(), false),
                ("ernie-tiny".to_string(), "ERNIE Tiny".to_string(), false),
            ],
            ChatModelProvider::Alibaba => vec![
                ("qwen-max".to_string(), "Qwen Max".to_string(), true),
                ("qwen-plus".to_string(), "Qwen Plus".to_string(), true),
                ("qwen-turbo".to_string(), "Qwen Turbo".to_string(), false),
                (
                    "qwen-max-longcontext".to_string(),
                    "Qwen Max Long Context".to_string(),
                    false,
                ),
                (
                    "qwen-72b-chat".to_string(),
                    "Qwen 72B Chat".to_string(),
                    false,
                ),
                (
                    "qwen-14b-chat".to_string(),
                    "Qwen 14B Chat".to_string(),
                    false,
                ),
                (
                    "qwen-7b-chat".to_string(),
                    "Qwen 7B Chat".to_string(),
                    false,
                ),
                (
                    "qwen-vl-plus".to_string(),
                    "Qwen VL Plus".to_string(),
                    false,
                ),
            ],
            ChatModelProvider::Tencent => vec![
                ("hunyuan-pro".to_string(), "Hunyuan Pro".to_string(), true),
                (
                    "hunyuan-standard".to_string(),
                    "Hunyuan Standard".to_string(),
                    true,
                ),
                (
                    "hunyuan-lite".to_string(),
                    "Hunyuan Lite".to_string(),
                    false,
                ),
            ],
            ChatModelProvider::Zhipu => vec![
                ("glm-4-plus".to_string(), "GLM-4 Plus".to_string(), true),
                ("glm-4".to_string(), "GLM-4".to_string(), true),
                ("glm-4-air".to_string(), "GLM-4 Air".to_string(), false),
                ("glm-4-flash".to_string(), "GLM-4 Flash".to_string(), false),
                ("glm-3-turbo".to_string(), "GLM-3 Turbo".to_string(), false),
                ("glm-4v".to_string(), "GLM-4V".to_string(), false),
            ],
            ChatModelProvider::MiniMax => vec![
                ("abab6.5-chat".to_string(), "abab6.5 Chat".to_string(), true),
                (
                    "abab6.5s-chat".to_string(),
                    "abab6.5s Chat".to_string(),
                    true,
                ),
                (
                    "abab5.5-chat".to_string(),
                    "abab5.5 Chat".to_string(),
                    false,
                ),
                (
                    "abab5.5s-chat".to_string(),
                    "abab5.5s Chat".to_string(),
                    false,
                ),
            ],
            ChatModelProvider::Moonshot => vec![
                (
                    "moonshot-v1-128k".to_string(),
                    "Moonshot v1 128K".to_string(),
                    true,
                ),
                (
                    "moonshot-v1-32k".to_string(),
                    "Moonshot v1 32K".to_string(),
                    true,
                ),
                (
                    "moonshot-v1-8k".to_string(),
                    "Moonshot v1 8K".to_string(),
                    false,
                ),
                (
                    "kimi-k2-0711-preview".to_string(),
                    "Kimi K2".to_string(),
                    false,
                ),
            ],
            ChatModelProvider::Baichuan => vec![
                ("Baichuan4".to_string(), "Baichuan 4".to_string(), true),
                (
                    "Baichuan3-Turbo".to_string(),
                    "Baichuan 3 Turbo".to_string(),
                    true,
                ),
                ("Baichuan3".to_string(), "Baichuan 3".to_string(), false),
                (
                    "Baichuan2-Turbo".to_string(),
                    "Baichuan 2 Turbo".to_string(),
                    false,
                ),
                ("Baichuan2".to_string(), "Baichuan 2".to_string(), false),
            ],
            ChatModelProvider::Yi => vec![
                ("yi-large".to_string(), "Yi Large".to_string(), true),
                ("yi-34b-chat".to_string(), "Yi 34B Chat".to_string(), true),
                (
                    "yi-34b-chat-200k".to_string(),
                    "Yi 34B Chat 200K".to_string(),
                    false,
                ),
                ("yi-9b-chat".to_string(), "Yi 9B Chat".to_string(), false),
                ("yi-6b-chat".to_string(), "Yi 6B Chat".to_string(), false),
            ],
            ChatModelProvider::Custom => vec![],
        }
    }
}
