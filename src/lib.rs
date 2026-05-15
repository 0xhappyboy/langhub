pub mod llms;
pub mod tools;
pub mod types;

use crate::llms::*;
use crate::types::{ChatMessage, LangHubError, ModelProvider, Result};

/// llm clinet enum
#[derive(Clone)]
pub enum LLMClient {
    OpenAI(OpenAI),
    Anthropic(Anthropic),
    DeepSeek(DeepSeek),
    Google(GoogleAI),
    Cohere(Cohere),
    HuggingFace(HuggingFace),
    Azure(AzureOpenAI),
    Mistral(Mistral),
    Groq(Groq),
    Together(Together),
    Replicate(Replicate),
    Fireworks(Fireworks),
    Perplexity(Perplexity),
    Baidu(BaiduWenxin),
    Alibaba(AlibabaTongyi),
    Tencent(TencentHunyuan),
    Zhipu(ZhipuAI),
    MiniMax(MiniMax),
    Moonshot(Moonshot),
    Baichuan(Baichuan),
    Yi(Yi),
}

impl LLMClient {
    pub fn new(provider: ModelProvider) -> Result<Self> {
        match provider {
            ModelProvider::OpenAI => {
                let api_key = std::env::var("OPENAI_API_KEY")
                    .map_err(|_| LangHubError::LLMError("OPENAI_API_KEY not set".to_string()))?;
                Ok(LLMClient::OpenAI(OpenAI::new(api_key).gpt4_turbo()))
            }
            ModelProvider::Anthropic => {
                let api_key = std::env::var("ANTHROPIC_API_KEY")
                    .map_err(|_| LangHubError::LLMError("ANTHROPIC_API_KEY not set".to_string()))?;
                Ok(LLMClient::Anthropic(
                    Anthropic::new(api_key).claude3_sonnet(),
                ))
            }
            ModelProvider::DeepSeek => {
                let api_key = std::env::var("DEEPSEEK_API_KEY")
                    .map_err(|_| LangHubError::LLMError("DEEPSEEK_API_KEY not set".to_string()))?;
                Ok(LLMClient::DeepSeek(DeepSeek::new(api_key).chat_model()))
            }
            ModelProvider::Google => {
                let api_key = std::env::var("GOOGLE_API_KEY")
                    .map_err(|_| LangHubError::LLMError("GOOGLE_API_KEY not set".to_string()))?;
                Ok(LLMClient::Google(GoogleAI::new(api_key).gemini15_pro()))
            }
            ModelProvider::Cohere => {
                let api_key = std::env::var("COHERE_API_KEY")
                    .map_err(|_| LangHubError::LLMError("COHERE_API_KEY not set".to_string()))?;
                Ok(LLMClient::Cohere(Cohere::new(api_key).command()))
            }
            ModelProvider::HuggingFace => {
                let api_key = std::env::var("HUGGINGFACE_API_KEY").map_err(|_| {
                    LangHubError::LLMError("HUGGINGFACE_API_KEY not set".to_string())
                })?;
                Ok(LLMClient::HuggingFace(
                    HuggingFace::new(api_key).llama3_8b(),
                ))
            }
            ModelProvider::Azure => {
                let api_key = std::env::var("AZURE_API_KEY")
                    .map_err(|_| LangHubError::LLMError("AZURE_API_KEY not set".to_string()))?;
                let endpoint = std::env::var("AZURE_ENDPOINT")
                    .map_err(|_| LangHubError::LLMError("AZURE_ENDPOINT not set".to_string()))?;
                let deployment = std::env::var("AZURE_DEPLOYMENT_NAME").map_err(|_| {
                    LangHubError::LLMError("AZURE_DEPLOYMENT_NAME not set".to_string())
                })?;
                Ok(LLMClient::Azure(AzureOpenAI::new(
                    api_key, endpoint, deployment,
                )))
            }
            ModelProvider::Mistral => {
                let api_key = std::env::var("MISTRAL_API_KEY")
                    .map_err(|_| LangHubError::LLMError("MISTRAL_API_KEY not set".to_string()))?;
                Ok(LLMClient::Mistral(Mistral::new(api_key).small()))
            }
            ModelProvider::Groq => {
                let api_key = std::env::var("GROQ_API_KEY")
                    .map_err(|_| LangHubError::LLMError("GROQ_API_KEY not set".to_string()))?;
                Ok(LLMClient::Groq(Groq::new(api_key).mixtral()))
            }
            ModelProvider::Together => {
                let api_key = std::env::var("TOGETHER_API_KEY")
                    .map_err(|_| LangHubError::LLMError("TOGETHER_API_KEY not set".to_string()))?;
                Ok(LLMClient::Together(Together::new(api_key).mixtral()))
            }
            ModelProvider::Replicate => {
                let api_key = std::env::var("REPLICATE_API_KEY")
                    .map_err(|_| LangHubError::LLMError("REPLICATE_API_KEY not set".to_string()))?;
                Ok(LLMClient::Replicate(Replicate::new(api_key).mixtral()))
            }
            ModelProvider::Fireworks => {
                let api_key = std::env::var("FIREWORKS_API_KEY")
                    .map_err(|_| LangHubError::LLMError("FIREWORKS_API_KEY not set".to_string()))?;
                Ok(LLMClient::Fireworks(Fireworks::new(api_key).mixtral()))
            }
            ModelProvider::Perplexity => {
                let api_key = std::env::var("PERPLEXITY_API_KEY").map_err(|_| {
                    LangHubError::LLMError("PERPLEXITY_API_KEY not set".to_string())
                })?;
                Ok(LLMClient::Perplexity(
                    Perplexity::new(api_key).sonar_medium(),
                ))
            }
            ModelProvider::Baidu => {
                let api_key = std::env::var("BAIDU_API_KEY")
                    .map_err(|_| LangHubError::LLMError("BAIDU_API_KEY not set".to_string()))?;
                let secret_key = std::env::var("BAIDU_SECRET_KEY")
                    .map_err(|_| LangHubError::LLMError("BAIDU_SECRET_KEY not set".to_string()))?;
                Ok(LLMClient::Baidu(
                    BaiduWenxin::new(api_key, secret_key).ernie4_0(),
                ))
            }
            ModelProvider::Alibaba => {
                let api_key = std::env::var("ALIBABA_API_KEY")
                    .map_err(|_| LangHubError::LLMError("ALIBABA_API_KEY not set".to_string()))?;
                Ok(LLMClient::Alibaba(AlibabaTongyi::new(api_key).qwen_plus()))
            }
            ModelProvider::Tencent => {
                let secret_id = std::env::var("TENCENT_SECRET_ID")
                    .map_err(|_| LangHubError::LLMError("TENCENT_SECRET_ID not set".to_string()))?;
                let secret_key = std::env::var("TENCENT_SECRET_KEY").map_err(|_| {
                    LangHubError::LLMError("TENCENT_SECRET_KEY not set".to_string())
                })?;
                Ok(LLMClient::Tencent(
                    TencentHunyuan::new(secret_id, secret_key).hunyuan_pro(),
                ))
            }
            ModelProvider::Zhipu => {
                let api_key = std::env::var("ZHIPU_API_KEY")
                    .map_err(|_| LangHubError::LLMError("ZHIPU_API_KEY not set".to_string()))?;
                Ok(LLMClient::Zhipu(ZhipuAI::new(api_key).glm4()))
            }
            ModelProvider::MiniMax => {
                let api_key = std::env::var("MINIMAX_API_KEY")
                    .map_err(|_| LangHubError::LLMError("MINIMAX_API_KEY not set".to_string()))?;
                let group_id = std::env::var("MINIMAX_GROUP_ID")
                    .map_err(|_| LangHubError::LLMError("MINIMAX_GROUP_ID not set".to_string()))?;
                Ok(LLMClient::MiniMax(
                    MiniMax::new(api_key, group_id).abab6_5(),
                ))
            }
            ModelProvider::Moonshot => {
                let api_key = std::env::var("MOONSHOT_API_KEY")
                    .map_err(|_| LangHubError::LLMError("MOONSHOT_API_KEY not set".to_string()))?;
                Ok(LLMClient::Moonshot(Moonshot::new(api_key).kimi_128k()))
            }
            ModelProvider::Baichuan => {
                let api_key = std::env::var("BAICHUAN_API_KEY")
                    .map_err(|_| LangHubError::LLMError("BAICHUAN_API_KEY not set".to_string()))?;
                Ok(LLMClient::Baichuan(Baichuan::new(api_key).baichuan4()))
            }
            ModelProvider::Yi => {
                let api_key = std::env::var("YI_API_KEY")
                    .map_err(|_| LangHubError::LLMError("YI_API_KEY not set".to_string()))?;
                Ok(LLMClient::Yi(Yi::new(api_key).yi34b()))
            }
            ModelProvider::Local => Err(LangHubError::LLMError(
                "Local models not yet supported".to_string(),
            )),
        }
    }

    pub async fn generate(&self, prompt: &str) -> Result<String> {
        match self {
            LLMClient::OpenAI(m) => m.generate(prompt).await.map(|r| r.text),
            LLMClient::Anthropic(m) => m.generate(prompt).await.map(|r| r.text),
            LLMClient::DeepSeek(m) => m.generate(prompt).await.map(|r| r.text),
            LLMClient::Google(m) => m.generate(prompt).await.map(|r| r.text),
            LLMClient::Cohere(m) => m.generate(prompt).await.map(|r| r.text),
            LLMClient::HuggingFace(m) => m.generate(prompt).await.map(|r| r.text),
            LLMClient::Azure(m) => m.generate(prompt).await.map(|r| r.text),
            LLMClient::Mistral(m) => m.generate(prompt).await.map(|r| r.text),
            LLMClient::Groq(m) => m.generate(prompt).await.map(|r| r.text),
            LLMClient::Together(m) => m.generate(prompt).await.map(|r| r.text),
            LLMClient::Replicate(m) => m.generate(prompt).await.map(|r| r.text),
            LLMClient::Fireworks(m) => m.generate(prompt).await.map(|r| r.text),
            LLMClient::Perplexity(m) => m.generate(prompt).await.map(|r| r.text),
            LLMClient::Baidu(m) => m.generate(prompt).await.map(|r| r.text),
            LLMClient::Alibaba(m) => m.generate(prompt).await.map(|r| r.text),
            LLMClient::Tencent(m) => m.generate(prompt).await.map(|r| r.text),
            LLMClient::Zhipu(m) => m.generate(prompt).await.map(|r| r.text),
            LLMClient::MiniMax(m) => m.generate(prompt).await.map(|r| r.text),
            LLMClient::Moonshot(m) => m.generate(prompt).await.map(|r| r.text),
            LLMClient::Baichuan(m) => m.generate(prompt).await.map(|r| r.text),
            LLMClient::Yi(m) => m.generate(prompt).await.map(|r| r.text),
        }
    }

    pub async fn chat(&self, messages: Vec<ChatMessage>) -> Result<String> {
        match self {
            LLMClient::OpenAI(m) => m.chat(messages).await.map(|r| r.text),
            LLMClient::Anthropic(m) => m.chat(messages).await.map(|r| r.text),
            LLMClient::DeepSeek(m) => m.chat(messages).await.map(|r| r.text),
            LLMClient::Google(m) => m.chat(messages).await.map(|r| r.text),
            LLMClient::Cohere(m) => m.chat(messages).await.map(|r| r.text),
            LLMClient::HuggingFace(m) => m.chat(messages).await.map(|r| r.text),
            LLMClient::Azure(m) => m.chat(messages).await.map(|r| r.text),
            LLMClient::Mistral(m) => m.chat(messages).await.map(|r| r.text),
            LLMClient::Groq(m) => m.chat(messages).await.map(|r| r.text),
            LLMClient::Together(m) => m.chat(messages).await.map(|r| r.text),
            LLMClient::Replicate(m) => m.chat(messages).await.map(|r| r.text),
            LLMClient::Fireworks(m) => m.chat(messages).await.map(|r| r.text),
            LLMClient::Perplexity(m) => m.chat(messages).await.map(|r| r.text),
            LLMClient::Baidu(m) => m.chat(messages).await.map(|r| r.text),
            LLMClient::Alibaba(m) => m.chat(messages).await.map(|r| r.text),
            LLMClient::Tencent(m) => m.chat(messages).await.map(|r| r.text),
            LLMClient::Zhipu(m) => m.chat(messages).await.map(|r| r.text),
            LLMClient::MiniMax(m) => m.chat(messages).await.map(|r| r.text),
            LLMClient::Moonshot(m) => m.chat(messages).await.map(|r| r.text),
            LLMClient::Baichuan(m) => m.chat(messages).await.map(|r| r.text),
            LLMClient::Yi(m) => m.chat(messages).await.map(|r| r.text),
        }
    }
}
