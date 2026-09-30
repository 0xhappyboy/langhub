use std::error::Error;
use std::fmt;
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    pub name: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
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
#[derive(Debug)]
pub enum LangHubError {
    LLMError(String),
    PromptError(String),
    ParseError(String),
    ChainError(String),
    IoError(std::io::Error),
    JsonError(serde_json::Error),
}
impl fmt::Display for LangHubError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LangHubError::LLMError(msg) => write!(f, "LLM error: {}", msg),
            LangHubError::PromptError(msg) => write!(f, "Prompt error: {}", msg),
            LangHubError::ParseError(msg) => write!(f, "Parse error: {}", msg),
            LangHubError::ChainError(msg) => write!(f, "Chain error: {}", msg),
            LangHubError::IoError(err) => write!(f, "IO error: {}", err),
            LangHubError::JsonError(err) => write!(f, "JSON error: {}", err),
        }
    }
}
impl Error for LangHubError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            LangHubError::IoError(err) => Some(err),
            LangHubError::JsonError(err) => Some(err),
            _ => None,
        }
    }
}
impl From<std::io::Error> for LangHubError {
    fn from(err: std::io::Error) -> Self {
        LangHubError::IoError(err)
    }
}
impl From<serde_json::Error> for LangHubError {
    fn from(err: serde_json::Error) -> Self {
        LangHubError::JsonError(err)
    }
}
impl From<String> for LangHubError {
    fn from(msg: String) -> Self {
        LangHubError::LLMError(msg)
    }
}
impl From<&str> for LangHubError {
    fn from(msg: &str) -> Self {
        LangHubError::LLMError(msg.to_string())
    }
}
pub type Result<T> = std::result::Result<T, LangHubError>;
use crate::chat::ToolCall;
use serde::{Deserialize, Serialize};
/// LLM model vendor/provider type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LLMVendor {
    OpenAI,
    Anthropic,
    Google,
    DeepSeek,
    Cohere,
    HuggingFace,
    Microsoft,
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
impl fmt::Display for LLMVendor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LLMVendor::OpenAI => write!(f, "OpenAI"),
            LLMVendor::Anthropic => write!(f, "Anthropic"),
            LLMVendor::Google => write!(f, "Google"),
            LLMVendor::DeepSeek => write!(f, "DeepSeek"),
            LLMVendor::Cohere => write!(f, "Cohere"),
            LLMVendor::HuggingFace => write!(f, "HuggingFace"),
            LLMVendor::Microsoft => write!(f, "Microsoft"),
            LLMVendor::Mistral => write!(f, "Mistral"),
            LLMVendor::Groq => write!(f, "Groq"),
            LLMVendor::Together => write!(f, "Together"),
            LLMVendor::Replicate => write!(f, "Replicate"),
            LLMVendor::Fireworks => write!(f, "Fireworks"),
            LLMVendor::Perplexity => write!(f, "Perplexity"),
            LLMVendor::Baidu => write!(f, "Baidu"),
            LLMVendor::Alibaba => write!(f, "Alibaba"),
            LLMVendor::Tencent => write!(f, "Tencent"),
            LLMVendor::Zhipu => write!(f, "Zhipu"),
            LLMVendor::MiniMax => write!(f, "MiniMax"),
            LLMVendor::Moonshot => write!(f, "Moonshot"),
            LLMVendor::Baichuan => write!(f, "Baichuan"),
            LLMVendor::Yi => write!(f, "Yi"),
            LLMVendor::Custom => write!(f, "Custom"),
        }
    }
}
impl LLMVendor {
    /// Returns all supported LLM vendors.
    pub fn all() -> Vec<LLMVendor> {
        vec![
            LLMVendor::OpenAI,
            LLMVendor::Anthropic,
            LLMVendor::Google,
            LLMVendor::DeepSeek,
            LLMVendor::Cohere,
            LLMVendor::HuggingFace,
            LLMVendor::Microsoft,
            LLMVendor::Mistral,
            LLMVendor::Groq,
            LLMVendor::Together,
            LLMVendor::Replicate,
            LLMVendor::Fireworks,
            LLMVendor::Perplexity,
            LLMVendor::Baidu,
            LLMVendor::Alibaba,
            LLMVendor::Tencent,
            LLMVendor::Zhipu,
            LLMVendor::MiniMax,
            LLMVendor::Moonshot,
            LLMVendor::Baichuan,
            LLMVendor::Yi,
            LLMVendor::Custom,
        ]
    }
    /// Returns a short human-readable description of this vendor (English).
    pub fn description(&self) -> &'static str {
        match self {
            LLMVendor::OpenAI => "OpenAI: GPT and o-series reasoning models",
            LLMVendor::Anthropic => "Anthropic: Claude safety-focused models",
            LLMVendor::Google => "Google: Gemini multimodal models",
            LLMVendor::DeepSeek => "DeepSeek: chat, coder and reasoning models",
            LLMVendor::Cohere => "Cohere: enterprise Command models",
            LLMVendor::HuggingFace => "HuggingFace: hosted open-source models",
            LLMVendor::Microsoft => "Microsoft: Azure-hosted OpenAI models",
            LLMVendor::Mistral => "Mistral AI: efficient European models",
            LLMVendor::Groq => "Groq: ultra-low-latency LPU inference",
            LLMVendor::Together => "Together.ai: hosted open-source models",
            LLMVendor::Replicate => "Replicate: cloud open-source model runner",
            LLMVendor::Fireworks => "Fireworks AI: fast open-source inference",
            LLMVendor::Perplexity => "Perplexity: online-search-augmented models",
            LLMVendor::Baidu => "Baidu: ERNIE Chinese-first models",
            LLMVendor::Alibaba => "Alibaba Cloud: Qwen Tongyi models",
            LLMVendor::Tencent => "Tencent: Hunyuan Chinese models",
            LLMVendor::Zhipu => "Zhipu: GLM Chinese-first models",
            LLMVendor::MiniMax => "MiniMax: abab long-context models",
            LLMVendor::Moonshot => "Moonshot: Kimi long-context models",
            LLMVendor::Baichuan => "Baichuan: Chinese-first models",
            LLMVendor::Yi => "01.AI: Yi bilingual models",
            LLMVendor::Custom => "Custom: user-defined OpenAI-compatible endpoint",
        }
    }
    /// Returns a short human-readable description of this vendor (Chinese).
    pub fn description_zh(&self) -> &'static str {
        match self {
            LLMVendor::OpenAI => "OpenAI：GPT 与 o 系列推理模型",
            LLMVendor::Anthropic => "Anthropic：Claude 安全导向模型",
            LLMVendor::Google => "Google：Gemini 多模态模型",
            LLMVendor::DeepSeek => "DeepSeek：对话、代码与推理模型",
            LLMVendor::Cohere => "Cohere：企业级 Command 模型",
            LLMVendor::HuggingFace => "HuggingFace：托管开源模型",
            LLMVendor::Microsoft => "Microsoft：Azure 托管的 OpenAI 模型",
            LLMVendor::Mistral => "Mistral AI：高效欧洲模型",
            LLMVendor::Groq => "Groq：超低延迟 LPU 推理",
            LLMVendor::Together => "Together.ai：托管开源模型",
            LLMVendor::Replicate => "Replicate：云端开源模型运行",
            LLMVendor::Fireworks => "Fireworks AI：开源模型高速推理",
            LLMVendor::Perplexity => "Perplexity：联网搜索增强模型",
            LLMVendor::Baidu => "百度：文心 ERNIE 中文优先模型",
            LLMVendor::Alibaba => "阿里云：通义千问 Qwen 系列",
            LLMVendor::Tencent => "腾讯：混元中文模型",
            LLMVendor::Zhipu => "智谱：GLM 中文优先模型",
            LLMVendor::MiniMax => "MiniMax：abab 长上下文模型",
            LLMVendor::Moonshot => "月之暗面：Kimi 长上下文模型",
            LLMVendor::Baichuan => "百川智能：中文优先模型",
            LLMVendor::Yi => "零一万物：Yi 中英双语模型",
            LLMVendor::Custom => "自定义：用户自定义 OpenAI 兼容端点",
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelProvider {
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
impl fmt::Display for ModelProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelProvider::OpenAI => write!(f, "OpenAI"),
            ModelProvider::Anthropic => write!(f, "Anthropic"),
            ModelProvider::Google => write!(f, "Google"),
            ModelProvider::DeepSeek => write!(f, "DeepSeek"),
            ModelProvider::Cohere => write!(f, "Cohere"),
            ModelProvider::HuggingFace => write!(f, "HuggingFace"),
            ModelProvider::Azure => write!(f, "Azure"),
            ModelProvider::Mistral => write!(f, "Mistral"),
            ModelProvider::Groq => write!(f, "Groq"),
            ModelProvider::Together => write!(f, "Together"),
            ModelProvider::Replicate => write!(f, "Replicate"),
            ModelProvider::Fireworks => write!(f, "Fireworks"),
            ModelProvider::Perplexity => write!(f, "Perplexity"),
            ModelProvider::Baidu => write!(f, "Baidu"),
            ModelProvider::Alibaba => write!(f, "Alibaba"),
            ModelProvider::Tencent => write!(f, "Tencent"),
            ModelProvider::Zhipu => write!(f, "Zhipu"),
            ModelProvider::MiniMax => write!(f, "MiniMax"),
            ModelProvider::Moonshot => write!(f, "Moonshot"),
            ModelProvider::Baichuan => write!(f, "Baichuan"),
            ModelProvider::Yi => write!(f, "Yi"),
            ModelProvider::Custom => write!(f, "Custom"),
        }
    }
}
impl ModelProvider {
    pub fn all() -> Vec<ModelProvider> {
        vec![
            ModelProvider::OpenAI,
            ModelProvider::Anthropic,
            ModelProvider::Google,
            ModelProvider::DeepSeek,
            ModelProvider::Cohere,
            ModelProvider::HuggingFace,
            ModelProvider::Azure,
            ModelProvider::Mistral,
            ModelProvider::Groq,
            ModelProvider::Together,
            ModelProvider::Replicate,
            ModelProvider::Fireworks,
            ModelProvider::Perplexity,
            ModelProvider::Baidu,
            ModelProvider::Alibaba,
            ModelProvider::Tencent,
            ModelProvider::Zhipu,
            ModelProvider::MiniMax,
            ModelProvider::Moonshot,
            ModelProvider::Baichuan,
            ModelProvider::Yi,
            ModelProvider::Custom,
        ]
    }
    /// Returns the vendor of this model provider.
    ///
    /// This mirrors the vendor concept used by `ImageModelProvider`,
    /// `VideoModelProvider` and `AudioModelProvider`, so the frontend can
    /// group providers by vendor consistently across all model families.
    pub fn vendor(&self) -> LLMVendor {
        match self {
            ModelProvider::OpenAI => LLMVendor::OpenAI,
            ModelProvider::Anthropic => LLMVendor::Anthropic,
            ModelProvider::Google => LLMVendor::Google,
            ModelProvider::DeepSeek => LLMVendor::DeepSeek,
            ModelProvider::Cohere => LLMVendor::Cohere,
            ModelProvider::HuggingFace => LLMVendor::HuggingFace,
            ModelProvider::Azure => LLMVendor::Microsoft,
            ModelProvider::Mistral => LLMVendor::Mistral,
            ModelProvider::Groq => LLMVendor::Groq,
            ModelProvider::Together => LLMVendor::Together,
            ModelProvider::Replicate => LLMVendor::Replicate,
            ModelProvider::Fireworks => LLMVendor::Fireworks,
            ModelProvider::Perplexity => LLMVendor::Perplexity,
            ModelProvider::Baidu => LLMVendor::Baidu,
            ModelProvider::Alibaba => LLMVendor::Alibaba,
            ModelProvider::Tencent => LLMVendor::Tencent,
            ModelProvider::Zhipu => LLMVendor::Zhipu,
            ModelProvider::MiniMax => LLMVendor::MiniMax,
            ModelProvider::Moonshot => LLMVendor::Moonshot,
            ModelProvider::Baichuan => LLMVendor::Baichuan,
            ModelProvider::Yi => LLMVendor::Yi,
            ModelProvider::Custom => LLMVendor::Custom,
        }
    }
    /// Returns a short human-readable description of this provider (English).
    pub fn description(&self) -> &'static str {
        match self {
            ModelProvider::OpenAI => {
                "OpenAI: GPT-4, GPT-4o and o-series reasoning models with strong general capability"
            }
            ModelProvider::Anthropic => {
                "Anthropic Claude: long-context, safety-focused models with strong reasoning and tool use"
            }
            ModelProvider::Google => {
                "Google Gemini: multimodal models with very long context and native vision/audio grounding"
            }
            ModelProvider::DeepSeek => {
                "DeepSeek: cost-effective chat, coder and reasoning models with strong math and code ability"
            }
            ModelProvider::Cohere => {
                "Cohere Command: enterprise RAG and tool-use oriented language models"
            }
            ModelProvider::HuggingFace => {
                "HuggingFace: hosted open-source models via Inference API for flexible experimentation"
            }
            ModelProvider::Azure => {
                "Azure OpenAI: Microsoft-hosted OpenAI models with enterprise compliance and regional deployment"
            }
            ModelProvider::Mistral => {
                "Mistral AI: efficient European models from tiny to large, including Codestral for code"
            }
            ModelProvider::Groq => {
                "Groq: ultra-low-latency inference for open-source models via custom LPU hardware"
            }
            ModelProvider::Together => {
                "Together.ai: hosted open-source models with broad selection and competitive pricing"
            }
            ModelProvider::Replicate => {
                "Replicate: run open-source models in the cloud with pay-per-use pricing"
            }
            ModelProvider::Fireworks => {
                "Fireworks AI: fast inference for open-source models with function calling support"
            }
            ModelProvider::Perplexity => {
                "Perplexity Sonar: online-search-augmented models with real-time web grounding"
            }
            ModelProvider::Baidu => {
                "Baidu ERNIE: Chinese-first language models with strong Chinese understanding"
            }
            ModelProvider::Alibaba => {
                "Alibaba Qwen: Tongyi Qianwen models with multilingual and multimodal capability"
            }
            ModelProvider::Tencent => {
                "Tencent Hunyuan: Chinese language models with enterprise-grade deployment"
            }
            ModelProvider::Zhipu => {
                "Zhipu GLM: Chinese-first models with strong bilingual and reasoning capability"
            }
            ModelProvider::MiniMax => {
                "MiniMax abab: long-context Chinese models with strong conversational ability"
            }
            ModelProvider::Moonshot => {
                "Moonshot Kimi: long-context Chinese models with strong document understanding"
            }
            ModelProvider::Baichuan => {
                "Baichuan: Chinese-first models with strong general and medical-domain capability"
            }
            ModelProvider::Yi => {
                "01.AI Yi: bilingual open-source models with strong reasoning and long context"
            }
            ModelProvider::Custom => {
                "Custom: any OpenAI-compatible endpoint with your own API base URL"
            }
        }
    }
    /// Returns a short human-readable description of this provider (Chinese).
    pub fn description_zh(&self) -> &'static str {
        match self {
            ModelProvider::OpenAI => "OpenAI：GPT-4、GPT-4o 与 o 系列推理模型，通用能力最强",
            ModelProvider::Anthropic => {
                "Anthropic Claude：长上下文、安全导向，推理与工具调用能力强"
            }
            ModelProvider::Google => "Google Gemini：多模态模型，超长上下文，原生视觉与音频理解",
            ModelProvider::DeepSeek => "DeepSeek：高性价比对话、代码与推理模型，数学和编程能力突出",
            ModelProvider::Cohere => "Cohere Command：面向企业 RAG 与工具调用的语言模型",
            ModelProvider::HuggingFace => {
                "HuggingFace：通过 Inference API 托管开源模型，便于灵活实验"
            }
            ModelProvider::Azure => "Azure OpenAI：微软托管的 OpenAI 模型，企业合规与区域化部署",
            ModelProvider::Mistral => {
                "Mistral AI：高效的欧洲模型，覆盖 tiny 到 large，含 Codestral 代码模型"
            }
            ModelProvider::Groq => "Groq：基于自研 LPU 硬件的超低延迟开源模型推理",
            ModelProvider::Together => "Together.ai：托管开源模型，选择丰富、价格有竞争力",
            ModelProvider::Replicate => "Replicate：云端运行开源模型，按量付费",
            ModelProvider::Fireworks => "Fireworks AI：开源模型高速推理，支持函数调用",
            ModelProvider::Perplexity => "Perplexity Sonar：联网搜索增强模型，实时 Web 接地",
            ModelProvider::Baidu => "百度文心 ERNIE：中文优先语言模型，中文理解强",
            ModelProvider::Alibaba => "阿里云通义千问 Qwen：多语言、多模态能力",
            ModelProvider::Tencent => "腾讯混元：中文语言模型，面向企业级部署",
            ModelProvider::Zhipu => "智谱 GLM：中文优先模型，中英双语与推理能力强",
            ModelProvider::MiniMax => "MiniMax abab：长上下文中文模型，对话能力强",
            ModelProvider::Moonshot => "月之暗面 Kimi：长上下文中文模型，文档理解强",
            ModelProvider::Baichuan => "百川智能：中文优先模型，通用与医疗领域能力强",
            ModelProvider::Yi => "零一万物 Yi：中英双语开源模型，推理与长上下文能力强",
            ModelProvider::Custom => "自定义：任意 OpenAI 兼容端点，可配置自己的 API Base URL",
        }
    }
    pub fn supports_function_calling(&self) -> bool {
        match self {
            ModelProvider::OpenAI => true,
            ModelProvider::Anthropic => true,
            ModelProvider::Google => true,
            ModelProvider::DeepSeek => true,
            ModelProvider::Cohere => true,
            ModelProvider::Zhipu => true,
            ModelProvider::Moonshot => true,
            ModelProvider::Mistral => true,
            ModelProvider::Groq => true,
            ModelProvider::Together => true,
            ModelProvider::Fireworks => true,
            ModelProvider::Perplexity => true,
            ModelProvider::Baidu => true,
            ModelProvider::Alibaba => true,
            ModelProvider::Tencent => true,
            ModelProvider::MiniMax => true,
            ModelProvider::Baichuan => true,
            ModelProvider::Yi => true,
            ModelProvider::Custom => true,
            _ => false,
        }
    }
    pub fn supports_json_mode(&self) -> bool {
        match self {
            ModelProvider::OpenAI => true,
            ModelProvider::Anthropic => true,
            ModelProvider::DeepSeek => true,
            ModelProvider::Google => true,
            ModelProvider::Mistral => true,
            ModelProvider::Groq => true,
            ModelProvider::Together => true,
            ModelProvider::Fireworks => true,
            ModelProvider::Perplexity => true,
            ModelProvider::Baidu => true,
            ModelProvider::Alibaba => true,
            ModelProvider::Tencent => true,
            ModelProvider::Zhipu => true,
            ModelProvider::MiniMax => true,
            ModelProvider::Moonshot => true,
            ModelProvider::Baichuan => true,
            ModelProvider::Yi => true,
            ModelProvider::Custom => true,
            _ => false,
        }
    }
}
/// Video model vendor/provider type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VideoVendor {
    ByteDance,
    Alibaba,
    Kuaishou,
    Google,
    Runway,
    MiniMax,
    Lightricks,
    Xai,
    Pruna,
    Custom,
}
impl fmt::Display for VideoVendor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VideoVendor::ByteDance => write!(f, "ByteDance"),
            VideoVendor::Alibaba => write!(f, "Alibaba"),
            VideoVendor::Kuaishou => write!(f, "Kuaishou"),
            VideoVendor::Google => write!(f, "Google"),
            VideoVendor::Runway => write!(f, "Runway"),
            VideoVendor::MiniMax => write!(f, "MiniMax"),
            VideoVendor::Lightricks => write!(f, "Lightricks"),
            VideoVendor::Xai => write!(f, "xAI"),
            VideoVendor::Pruna => write!(f, "Pruna"),
            VideoVendor::Custom => write!(f, "Custom"),
        }
    }
}
impl VideoVendor {
    pub fn all() -> Vec<VideoVendor> {
        vec![
            VideoVendor::ByteDance,
            VideoVendor::Alibaba,
            VideoVendor::Kuaishou,
            VideoVendor::Google,
            VideoVendor::Runway,
            VideoVendor::MiniMax,
            VideoVendor::Lightricks,
            VideoVendor::Xai,
            VideoVendor::Pruna,
            VideoVendor::Custom,
        ]
    }
    /// Returns a short human-readable description of this vendor (English).
    pub fn description(&self) -> &'static str {
        match self {
            VideoVendor::ByteDance => "ByteDance: Seedance video generation family",
            VideoVendor::Alibaba => "Alibaba Cloud: Wan and HappyHorse video generation",
            VideoVendor::Kuaishou => "Kuaishou: Kling video generation family",
            VideoVendor::Google => "Google: Veo and Gemini Omni Flash video generation",
            VideoVendor::Runway => "Runway: Gen and Aleph creative video models",
            VideoVendor::MiniMax => "MiniMax: Hailuo video generation family",
            VideoVendor::Lightricks => "Lightricks: LTX efficient video generation",
            VideoVendor::Xai => "xAI: Grok Imagine video generation",
            VideoVendor::Pruna => "Pruna: optimized cost-effective video generation",
            VideoVendor::Custom => "Custom: user-defined video generation endpoint",
        }
    }
    /// Returns a short human-readable description of this vendor (Chinese).
    pub fn description_zh(&self) -> &'static str {
        match self {
            VideoVendor::ByteDance => "字节跳动：Seedance 视频生成系列",
            VideoVendor::Alibaba => "阿里云：万相与 HappyHorse 视频生成",
            VideoVendor::Kuaishou => "快手：可灵视频生成系列",
            VideoVendor::Google => "Google：Veo 与 Gemini Omni Flash 视频生成",
            VideoVendor::Runway => "Runway：Gen 与 Aleph 创意视频模型",
            VideoVendor::MiniMax => "MiniMax：海螺视频生成系列",
            VideoVendor::Lightricks => "Lightricks：LTX 高效视频生成",
            VideoVendor::Xai => "xAI：Grok Imagine 视频生成",
            VideoVendor::Pruna => "Pruna：优化型高性价比视频生成",
            VideoVendor::Custom => "自定义：用户自定义视频生成端点",
        }
    }
}
/// Image model vendor/provider type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ImageVendor {
    ByteDance,
    Alibaba,
    StabilityAI,
    BlackForestLabs,
    Google,
    OpenAI,
    Custom,
}
impl fmt::Display for ImageVendor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ImageVendor::ByteDance => write!(f, "ByteDance"),
            ImageVendor::Alibaba => write!(f, "Alibaba"),
            ImageVendor::StabilityAI => write!(f, "StabilityAI"),
            ImageVendor::BlackForestLabs => write!(f, "BlackForestLabs"),
            ImageVendor::Google => write!(f, "Google"),
            ImageVendor::OpenAI => write!(f, "OpenAI"),
            ImageVendor::Custom => write!(f, "Custom"),
        }
    }
}
impl ImageVendor {
    pub fn all() -> Vec<ImageVendor> {
        vec![
            ImageVendor::ByteDance,
            ImageVendor::Alibaba,
            ImageVendor::StabilityAI,
            ImageVendor::BlackForestLabs,
            ImageVendor::Google,
            ImageVendor::OpenAI,
            ImageVendor::Custom,
        ]
    }
    /// Returns a short human-readable description of this vendor (English).
    pub fn description(&self) -> &'static str {
        match self {
            ImageVendor::ByteDance => "ByteDance: Seedream text-to-image family",
            ImageVendor::Alibaba => "Alibaba Cloud: Wan Image text-to-image family",
            ImageVendor::StabilityAI => "Stability AI: Stable Image Ultra and SD 3.5 family",
            ImageVendor::BlackForestLabs => "Black Forest Labs: FLUX text-to-image family",
            ImageVendor::Google => "Google: Imagen text-to-image family",
            ImageVendor::OpenAI => "OpenAI: DALL·E text-to-image family",
            ImageVendor::Custom => "Custom: user-defined image generation endpoint",
        }
    }
    /// Returns a short human-readable description of this vendor (Chinese).
    pub fn description_zh(&self) -> &'static str {
        match self {
            ImageVendor::ByteDance => "字节跳动：Seedream 文生图系列",
            ImageVendor::Alibaba => "阿里云：万相文生图系列",
            ImageVendor::StabilityAI => "Stability AI：Stable Image Ultra 与 SD 3.5 系列",
            ImageVendor::BlackForestLabs => "Black Forest Labs：FLUX 文生图系列",
            ImageVendor::Google => "Google：Imagen 文生图系列",
            ImageVendor::OpenAI => "OpenAI：DALL·E 文生图系列",
            ImageVendor::Custom => "自定义：用户自定义图像生成端点",
        }
    }
}
/// Audio model vendor/provider type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AudioVendor {
    Alibaba,
    ByteDance,
    StepFun,
    Google,
    ElevenLabs,
    Suno,
    StabilityAI,
    Custom,
}
impl fmt::Display for AudioVendor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AudioVendor::Alibaba => write!(f, "Alibaba"),
            AudioVendor::ByteDance => write!(f, "ByteDance"),
            AudioVendor::StepFun => write!(f, "StepFun"),
            AudioVendor::Google => write!(f, "Google"),
            AudioVendor::ElevenLabs => write!(f, "ElevenLabs"),
            AudioVendor::Suno => write!(f, "Suno"),
            AudioVendor::StabilityAI => write!(f, "StabilityAI"),
            AudioVendor::Custom => write!(f, "Custom"),
        }
    }
}
impl AudioVendor {
    pub fn all() -> Vec<AudioVendor> {
        vec![
            AudioVendor::Alibaba,
            AudioVendor::ByteDance,
            AudioVendor::StepFun,
            AudioVendor::Google,
            AudioVendor::ElevenLabs,
            AudioVendor::Suno,
            AudioVendor::StabilityAI,
            AudioVendor::Custom,
        ]
    }
    /// Returns a short human-readable description of this vendor (English).
    pub fn description(&self) -> &'static str {
        match self {
            AudioVendor::Alibaba => "Alibaba Cloud: Qwen-Audio TTS family",
            AudioVendor::ByteDance => "ByteDance: Seed Audio generation family",
            AudioVendor::StepFun => "StepFun: StepAudio generation family",
            AudioVendor::Google => "Google: Gemini TTS and Lyria music generation",
            AudioVendor::ElevenLabs => "ElevenLabs: top-tier TTS and voice cloning",
            AudioVendor::Suno => "Suno: music generation family",
            AudioVendor::StabilityAI => {
                "Stability AI: Stable Audio music and sound-effect generation"
            }
            AudioVendor::Custom => "Custom: user-defined audio generation endpoint",
        }
    }
    /// Returns a short human-readable description of this vendor (Chinese).
    pub fn description_zh(&self) -> &'static str {
        match self {
            AudioVendor::Alibaba => "阿里云：Qwen-Audio TTS 系列",
            AudioVendor::ByteDance => "字节跳动：Seed Audio 音频生成系列",
            AudioVendor::StepFun => "阶跃星辰：StepAudio 音频生成系列",
            AudioVendor::Google => "Google：Gemini TTS 与 Lyria 音乐生成",
            AudioVendor::ElevenLabs => "ElevenLabs：顶级 TTS 与语音克隆",
            AudioVendor::Suno => "Suno：音乐生成系列",
            AudioVendor::StabilityAI => "Stability AI：Stable Audio 音乐与音效生成",
            AudioVendor::Custom => "自定义：用户自定义音频生成端点",
        }
    }
}
