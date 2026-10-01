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
