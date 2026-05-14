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

use serde::{Deserialize, Serialize};

use crate::llms::ToolCall;

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
            _ => false,
        }
    }
}
