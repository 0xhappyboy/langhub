use std::error::Error;
use std::fmt;

#[derive(Debug)]
pub enum LangChainError {
    LLMError(String),
    PromptError(String),
    ParseError(String),
    ChainError(String),
    IoError(std::io::Error),
    JsonError(serde_json::Error),
}

impl fmt::Display for LangChainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LangChainError::LLMError(msg) => write!(f, "LLM error: {}", msg),
            LangChainError::PromptError(msg) => write!(f, "Prompt error: {}", msg),
            LangChainError::ParseError(msg) => write!(f, "Parse error: {}", msg),
            LangChainError::ChainError(msg) => write!(f, "Chain error: {}", msg),
            LangChainError::IoError(err) => write!(f, "IO error: {}", err),
            LangChainError::JsonError(err) => write!(f, "JSON error: {}", err),
        }
    }
}

impl Error for LangChainError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            LangChainError::IoError(err) => Some(err),
            LangChainError::JsonError(err) => Some(err),
            _ => None,
        }
    }
}

impl From<std::io::Error> for LangChainError {
    fn from(err: std::io::Error) -> Self {
        LangChainError::IoError(err)
    }
}

impl From<serde_json::Error> for LangChainError {
    fn from(err: serde_json::Error) -> Self {
        LangChainError::JsonError(err)
    }
}

impl From<String> for LangChainError {
    fn from(msg: String) -> Self {
        LangChainError::LLMError(msg)
    }
}

impl From<&str> for LangChainError {
    fn from(msg: &str) -> Self {
        LangChainError::LLMError(msg.to_string())
    }
}

pub type Result<T> = std::result::Result<T, LangChainError>;

use serde::{Deserialize, Serialize};

/// Model vendor enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelProvider {
    /// OpenAI (GPT-4, GPT-3.5)
    OpenAI,
    /// Anthropic (Claude 3)
    Anthropic,
    /// Google (Gemini)
    Google,
    /// DeepSeek (DeepSeek-V3, R1)
    DeepSeek,
    /// Cohere (Command)
    Cohere,
    /// HuggingFace (Open source models)
    HuggingFace,
    /// Azure OpenAI
    Azure,
    /// Mistral AI
    Mistral,
    /// Groq (LPU accelerated)
    Groq,
    /// Together AI
    Together,
    /// Replicate
    Replicate,
    /// Fireworks AI
    Fireworks,
    /// Perplexity
    Perplexity,
    /// Baidu Wenxin Yiyan (ERNIE)
    Baidu,
    /// Alibaba Tongyi Qianwen
    Alibaba,
    /// Tencent Hunyuan
    Tencent,
    /// Zhipu AI (ChatGLM)
    Zhipu,
    /// MiniMax
    MiniMax,
    /// Moonshot AI (Kimi)
    Moonshot,
    /// Baichuan Intelligent
    Baichuan,
    /// Yi (01.AI)
    Yi,
    /// Local models (Ollama, llama.cpp)
    Local,
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
            ModelProvider::Local => write!(f, "Local"),
        }
    }
}

impl ModelProvider {
    /// Get all supported providers
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
            ModelProvider::Local,
        ]
    }

    /// Check if provider supports function calling
    pub fn supports_function_calling(&self) -> bool {
        match self {
            ModelProvider::OpenAI => true,
            ModelProvider::Anthropic => true,
            ModelProvider::Google => true,
            ModelProvider::DeepSeek => true,
            ModelProvider::Cohere => true,
            ModelProvider::Zhipu => true,
            ModelProvider::Moonshot => true,
            _ => false,
        }
    }

    /// Check if provider supports JSON mode
    pub fn supports_json_mode(&self) -> bool {
        match self {
            ModelProvider::OpenAI => true,
            ModelProvider::Anthropic => true,
            ModelProvider::DeepSeek => true,
            ModelProvider::Google => true,
            _ => false,
        }
    }
}
