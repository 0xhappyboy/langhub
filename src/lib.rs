//! LangHub - An LLM application development library.
pub mod audio;
pub mod chat;
pub mod image;
pub mod tools;
pub mod types;
pub mod video;
use crate::audio::*;
use crate::chat::*;
use crate::image::*;
use crate::types::ImageVendor;
use crate::types::{ChatMessage, LangHubError, ModelProvider, Result};
use crate::video::*;
/// Configuration for LLM client initialization
///
/// # Example
/// ```
/// use langhub::LLMConfig;
///
/// let config = LLMConfig::new()
///     .openai("sk-xxx".to_string())
///     .anthropic("anthropic-api-key".to_string());
/// ```
#[derive(Debug, Clone, Default)]
pub struct LLMConfig {
    /// OpenAI API key
    pub openai_api_key: Option<String>,
    /// Anthropic API key
    pub anthropic_api_key: Option<String>,
    /// DeepSeek API key
    pub deepseek_api_key: Option<String>,
    /// Google AI Studio API key
    pub google_api_key: Option<String>,
    /// Cohere API key
    pub cohere_api_key: Option<String>,
    /// HuggingFace API key
    pub huggingface_api_key: Option<String>,
    /// Azure OpenAI API key
    pub azure_api_key: Option<String>,
    /// Azure OpenAI endpoint URL
    pub azure_endpoint: Option<String>,
    /// Azure OpenAI deployment name
    pub azure_deployment_name: Option<String>,
    /// Mistral AI API key
    pub mistral_api_key: Option<String>,
    /// Groq API key
    pub groq_api_key: Option<String>,
    /// Together.ai API key
    pub together_api_key: Option<String>,
    /// Replicate API key
    pub replicate_api_key: Option<String>,
    /// Fireworks AI API key
    pub fireworks_api_key: Option<String>,
    /// Perplexity API key
    pub perplexity_api_key: Option<String>,
    /// Baidu Wenxin API key
    pub baidu_api_key: Option<String>,
    /// Baidu Wenxin secret key
    pub baidu_secret_key: Option<String>,
    /// Alibaba Tongyi API key
    pub alibaba_api_key: Option<String>,
    /// Tencent Hunyuan secret ID
    pub tencent_secret_id: Option<String>,
    /// Tencent Hunyuan secret key
    pub tencent_secret_key: Option<String>,
    /// Zhipu AI API key
    pub zhipu_api_key: Option<String>,
    /// MiniMax API key
    pub minimax_api_key: Option<String>,
    /// MiniMax group ID
    pub minimax_group_id: Option<String>,
    /// Moonshot AI API key
    pub moonshot_api_key: Option<String>,
    /// Baichuan AI API key
    pub baichuan_api_key: Option<String>,
    /// Yi AI API key
    pub yi_api_key: Option<String>,
    /// Custom API base URL
    pub custom_api_base: Option<String>,
}
impl LLMConfig {
    /// Creates a new empty configuration
    ///
    /// # Example
    /// ```
    /// let config = LLMConfig::new();
    /// ```
    pub fn new() -> Self {
        Self::default()
    }
    /// Sets the OpenAI API key
    ///
    /// # Arguments
    /// * `api_key` - OpenAI API key
    pub fn openai(mut self, api_key: String) -> Self {
        self.openai_api_key = Some(api_key);
        self
    }
    /// Sets the Anthropic API key
    ///
    /// # Arguments
    /// * `api_key` - Anthropic API key
    pub fn anthropic(mut self, api_key: String) -> Self {
        self.anthropic_api_key = Some(api_key);
        self
    }
    /// Sets the DeepSeek API key
    ///
    /// # Arguments
    /// * `api_key` - DeepSeek API key
    pub fn deepseek(mut self, api_key: String) -> Self {
        self.deepseek_api_key = Some(api_key);
        self
    }
    /// Sets the Google API key
    ///
    /// # Arguments
    /// * `api_key` - Google AI Studio API key
    pub fn google(mut self, api_key: String) -> Self {
        self.google_api_key = Some(api_key);
        self
    }
    /// Sets the Cohere API key
    ///
    /// # Arguments
    /// * `api_key` - Cohere API key
    pub fn cohere(mut self, api_key: String) -> Self {
        self.cohere_api_key = Some(api_key);
        self
    }
    /// Sets the HuggingFace API key
    ///
    /// # Arguments
    /// * `api_key` - HuggingFace API key
    pub fn huggingface(mut self, api_key: String) -> Self {
        self.huggingface_api_key = Some(api_key);
        self
    }
    /// Sets the Azure OpenAI configuration
    ///
    /// # Arguments
    /// * `api_key` - Azure OpenAI API key
    /// * `endpoint` - Azure OpenAI endpoint URL
    /// * `deployment_name` - Azure OpenAI deployment name
    pub fn azure(mut self, api_key: String, endpoint: String, deployment_name: String) -> Self {
        self.azure_api_key = Some(api_key);
        self.azure_endpoint = Some(endpoint);
        self.azure_deployment_name = Some(deployment_name);
        self
    }
    /// Sets the Mistral API key
    ///
    /// # Arguments
    /// * `api_key` - Mistral AI API key
    pub fn mistral(mut self, api_key: String) -> Self {
        self.mistral_api_key = Some(api_key);
        self
    }
    /// Sets the Groq API key
    ///
    /// # Arguments
    /// * `api_key` - Groq API key
    pub fn groq(mut self, api_key: String) -> Self {
        self.groq_api_key = Some(api_key);
        self
    }
    /// Sets the Together.ai API key
    ///
    /// # Arguments
    /// * `api_key` - Together.ai API key
    pub fn together(mut self, api_key: String) -> Self {
        self.together_api_key = Some(api_key);
        self
    }
    /// Sets the Replicate API key
    ///
    /// # Arguments
    /// * `api_key` - Replicate API key
    pub fn replicate(mut self, api_key: String) -> Self {
        self.replicate_api_key = Some(api_key);
        self
    }
    /// Sets the Fireworks AI API key
    ///
    /// # Arguments
    /// * `api_key` - Fireworks AI API key
    pub fn fireworks(mut self, api_key: String) -> Self {
        self.fireworks_api_key = Some(api_key);
        self
    }
    /// Sets the Perplexity API key
    ///
    /// # Arguments
    /// * `api_key` - Perplexity API key
    pub fn perplexity(mut self, api_key: String) -> Self {
        self.perplexity_api_key = Some(api_key);
        self
    }
    /// Sets the Baidu Wenxin configuration
    ///
    /// # Arguments
    /// * `api_key` - Baidu API key
    /// * `secret_key` - Baidu secret key
    pub fn baidu(mut self, api_key: String, secret_key: String) -> Self {
        self.baidu_api_key = Some(api_key);
        self.baidu_secret_key = Some(secret_key);
        self
    }
    /// Sets the Alibaba Tongyi API key
    ///
    /// # Arguments
    /// * `api_key` - Alibaba API key
    pub fn alibaba(mut self, api_key: String) -> Self {
        self.alibaba_api_key = Some(api_key);
        self
    }
    /// Sets the Tencent Hunyuan configuration
    ///
    /// # Arguments
    /// * `secret_id` - Tencent secret ID
    /// * `secret_key` - Tencent secret key
    pub fn tencent(mut self, secret_id: String, secret_key: String) -> Self {
        self.tencent_secret_id = Some(secret_id);
        self.tencent_secret_key = Some(secret_key);
        self
    }
    /// Sets the Zhipu AI API key
    ///
    /// # Arguments
    /// * `api_key` - Zhipu AI API key
    pub fn zhipu(mut self, api_key: String) -> Self {
        self.zhipu_api_key = Some(api_key);
        self
    }
    /// Sets the MiniMax configuration
    ///
    /// # Arguments
    /// * `api_key` - MiniMax API key
    /// * `group_id` - MiniMax group ID
    pub fn minimax(mut self, api_key: String, group_id: String) -> Self {
        self.minimax_api_key = Some(api_key);
        self.minimax_group_id = Some(group_id);
        self
    }
    /// Sets the Moonshot API key
    ///
    /// # Arguments
    /// * `api_key` - Moonshot AI API key
    pub fn moonshot(mut self, api_key: String) -> Self {
        self.moonshot_api_key = Some(api_key);
        self
    }
    /// Sets the Baichuan API key
    ///
    /// # Arguments
    /// * `api_key` - Baichuan AI API key
    pub fn baichuan(mut self, api_key: String) -> Self {
        self.baichuan_api_key = Some(api_key);
        self
    }
    /// Sets the Yi API key
    ///
    /// # Arguments
    /// * `api_key` - Yi AI API key
    pub fn yi(mut self, api_key: String) -> Self {
        self.yi_api_key = Some(api_key);
        self
    }
    /// Sets the custom API base URL
    ///
    /// # Arguments
    /// * `api_base` - Custom API base URL
    pub fn custom_api_base(mut self, api_base: String) -> Self {
        self.custom_api_base = Some(api_base);
        self
    }
}
/// Unified LLM client for multiple AI providers
///
/// This enum represents a client for any supported LLM provider.
/// Use `LLMClient::new_with_config()` to create an instance.
///
/// # Example
/// ```
/// use langhub::{LLMClient, LLMConfig, ModelProvider};
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let config = LLMConfig::new().openai("sk-xxx".to_string());
/// let client = LLMClient::new_with_config(ModelProvider::OpenAI, &config)?;
/// let response = client.generate("Hello, world!").await?;
/// println!("{}", response);
/// # Ok(())
/// # }
/// ```
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
    Custom(CustomLLM),
}
impl LLMClient {
    /// Creates a new LLM client with the given provider using optional API keys
    ///
    /// # Arguments
    /// * `provider` - The LLM provider to use
    /// * `api_key` - Optional API key for the provider (some providers may require additional keys)
    /// * `extra_keys` - Optional additional keys for providers that need them (e.g., secret_key, endpoint, group_id)
    ///
    /// # Returns
    /// A `Result` containing the LLM client or an error if required keys are missing
    ///
    /// # Example
    /// ```
    /// use langhub::{LLMClient, ModelProvider};
    ///
    /// // OpenAI with just API key
    /// let client = LLMClient::new_with_key(ModelProvider::OpenAI, Some("sk-xxx".to_string()), None).unwrap();
    ///
    /// // Baidu with API key and secret key
    /// let extra = std::collections::HashMap::from([
    ///     ("secret_key".to_string(), "your_secret_key".to_string())
    /// ]);
    /// let client = LLMClient::new_with_key(ModelProvider::Baidu, Some("api_key".to_string()), Some(extra)).unwrap();
    /// ```
    pub fn new_with_key(
        provider: ModelProvider,
        api_key: Option<String>,
        extra_keys: Option<std::collections::HashMap<String, String>>,
    ) -> Result<Self> {
        let mut config = LLMConfig::new();
        let extra = extra_keys.unwrap_or_default();
        match provider {
            ModelProvider::OpenAI => {
                let key = api_key.ok_or_else(|| {
                    LangHubError::LLMError("OpenAI API key not provided".to_string())
                })?;
                config = config.openai(key);
            }
            ModelProvider::Anthropic => {
                let key = api_key.ok_or_else(|| {
                    LangHubError::LLMError("Anthropic API key not provided".to_string())
                })?;
                config = config.anthropic(key);
            }
            ModelProvider::DeepSeek => {
                let key = api_key.ok_or_else(|| {
                    LangHubError::LLMError("DeepSeek API key not provided".to_string())
                })?;
                config = config.deepseek(key);
            }
            ModelProvider::Google => {
                let key = api_key.ok_or_else(|| {
                    LangHubError::LLMError("Google API key not provided".to_string())
                })?;
                config = config.google(key);
            }
            ModelProvider::Cohere => {
                let key = api_key.ok_or_else(|| {
                    LangHubError::LLMError("Cohere API key not provided".to_string())
                })?;
                config = config.cohere(key);
            }
            ModelProvider::HuggingFace => {
                let key = api_key.ok_or_else(|| {
                    LangHubError::LLMError("HuggingFace API key not provided".to_string())
                })?;
                config = config.huggingface(key);
            }
            ModelProvider::Azure => {
                let key = api_key.ok_or_else(|| {
                    LangHubError::LLMError("Azure API key not provided".to_string())
                })?;
                let endpoint = extra.get("endpoint").ok_or_else(|| {
                    LangHubError::LLMError("Azure endpoint not provided".to_string())
                })?;
                let deployment = extra.get("deployment_name").ok_or_else(|| {
                    LangHubError::LLMError("Azure deployment name not provided".to_string())
                })?;
                config = config.azure(key, endpoint.clone(), deployment.clone());
            }
            ModelProvider::Mistral => {
                let key = api_key.ok_or_else(|| {
                    LangHubError::LLMError("Mistral API key not provided".to_string())
                })?;
                config = config.mistral(key);
            }
            ModelProvider::Groq => {
                let key = api_key.ok_or_else(|| {
                    LangHubError::LLMError("Groq API key not provided".to_string())
                })?;
                config = config.groq(key);
            }
            ModelProvider::Together => {
                let key = api_key.ok_or_else(|| {
                    LangHubError::LLMError("Together API key not provided".to_string())
                })?;
                config = config.together(key);
            }
            ModelProvider::Replicate => {
                let key = api_key.ok_or_else(|| {
                    LangHubError::LLMError("Replicate API key not provided".to_string())
                })?;
                config = config.replicate(key);
            }
            ModelProvider::Fireworks => {
                let key = api_key.ok_or_else(|| {
                    LangHubError::LLMError("Fireworks API key not provided".to_string())
                })?;
                config = config.fireworks(key);
            }
            ModelProvider::Perplexity => {
                let key = api_key.ok_or_else(|| {
                    LangHubError::LLMError("Perplexity API key not provided".to_string())
                })?;
                config = config.perplexity(key);
            }
            ModelProvider::Baidu => {
                let key = api_key.ok_or_else(|| {
                    LangHubError::LLMError("Baidu API key not provided".to_string())
                })?;
                let secret = extra.get("secret_key").ok_or_else(|| {
                    LangHubError::LLMError("Baidu secret key not provided".to_string())
                })?;
                config = config.baidu(key, secret.clone());
            }
            ModelProvider::Alibaba => {
                let key = api_key.ok_or_else(|| {
                    LangHubError::LLMError("Alibaba API key not provided".to_string())
                })?;
                config = config.alibaba(key);
            }
            ModelProvider::Tencent => {
                let secret_id = extra.get("secret_id").ok_or_else(|| {
                    LangHubError::LLMError("Tencent secret ID not provided".to_string())
                })?;
                let secret_key = extra.get("secret_key").ok_or_else(|| {
                    LangHubError::LLMError("Tencent secret key not provided".to_string())
                })?;
                config = config.tencent(secret_id.clone(), secret_key.clone());
            }
            ModelProvider::Zhipu => {
                let key = api_key.ok_or_else(|| {
                    LangHubError::LLMError("Zhipu API key not provided".to_string())
                })?;
                config = config.zhipu(key);
            }
            ModelProvider::MiniMax => {
                let key = api_key.ok_or_else(|| {
                    LangHubError::LLMError("MiniMax API key not provided".to_string())
                })?;
                let group_id = extra.get("group_id").ok_or_else(|| {
                    LangHubError::LLMError("MiniMax group ID not provided".to_string())
                })?;
                config = config.minimax(key, group_id.clone());
            }
            ModelProvider::Moonshot => {
                let key = api_key.ok_or_else(|| {
                    LangHubError::LLMError("Moonshot API key not provided".to_string())
                })?;
                config = config.moonshot(key);
            }
            ModelProvider::Baichuan => {
                let key = api_key.ok_or_else(|| {
                    LangHubError::LLMError("Baichuan API key not provided".to_string())
                })?;
                config = config.baichuan(key);
            }
            ModelProvider::Yi => {
                let key = api_key
                    .ok_or_else(|| LangHubError::LLMError("Yi API key not provided".to_string()))?;
                config = config.yi(key);
            }
            ModelProvider::Custom => {
                let api_key = api_key.ok_or_else(|| {
                    LangHubError::LLMError("Custom model API key not provided".to_string())
                })?;
                let api_base = extra.get("api_base").ok_or_else(|| {
                    LangHubError::LLMError("Custom model API base URL not provided".to_string())
                })?;
                let client = CustomLLM::new(api_key, api_base.clone());
                return Ok(LLMClient::Custom(client));
            }
        }
        Self::new_with_config(provider, &config)
    }
    /// Creates a new LLM client with the given provider and configuration
    ///
    /// # Arguments
    /// * `provider` - The LLM provider to use
    /// * `config` - Configuration containing API keys and credentials
    ///
    /// # Returns
    /// A `Result` containing the LLM client or an error if required keys are missing
    ///
    /// # Errors
    /// Returns `LangHubError::LLMError` if the required API key for the provider is not provided
    ///
    /// # Example
    /// ```
    /// use langhub::{LLMClient, LLMConfig, ModelProvider};
    ///
    /// let config = LLMConfig::new()
    ///     .openai("sk-xxx".to_string())
    ///     .anthropic("anth-xxx".to_string());
    ///
    /// let openai_client = LLMClient::new_with_config(ModelProvider::OpenAI, &config).unwrap();
    /// let anthropic_client = LLMClient::new_with_config(ModelProvider::Anthropic, &config).unwrap();
    /// ```
    pub fn new_with_config(provider: ModelProvider, config: &LLMConfig) -> Result<Self> {
        match provider {
            ModelProvider::OpenAI => {
                let api_key = config.openai_api_key.as_ref().ok_or_else(|| {
                    LangHubError::LLMError("OpenAI API key not provided".to_string())
                })?;
                Ok(LLMClient::OpenAI(OpenAI::new(api_key.clone()).gpt4_turbo()))
            }
            ModelProvider::Anthropic => {
                let api_key = config.anthropic_api_key.as_ref().ok_or_else(|| {
                    LangHubError::LLMError("Anthropic API key not provided".to_string())
                })?;
                Ok(LLMClient::Anthropic(
                    Anthropic::new(api_key.clone()).claude3_sonnet(),
                ))
            }
            ModelProvider::DeepSeek => {
                let api_key = config.deepseek_api_key.as_ref().ok_or_else(|| {
                    LangHubError::LLMError("DeepSeek API key not provided".to_string())
                })?;
                Ok(LLMClient::DeepSeek(
                    DeepSeek::new(api_key.clone()).chat_model(),
                ))
            }
            ModelProvider::Google => {
                let api_key = config.google_api_key.as_ref().ok_or_else(|| {
                    LangHubError::LLMError("Google API key not provided".to_string())
                })?;
                Ok(LLMClient::Google(
                    GoogleAI::new(api_key.clone()).gemini15_pro(),
                ))
            }
            ModelProvider::Cohere => {
                let api_key = config.cohere_api_key.as_ref().ok_or_else(|| {
                    LangHubError::LLMError("Cohere API key not provided".to_string())
                })?;
                Ok(LLMClient::Cohere(Cohere::new(api_key.clone()).command()))
            }
            ModelProvider::HuggingFace => {
                let api_key = config.huggingface_api_key.as_ref().ok_or_else(|| {
                    LangHubError::LLMError("HuggingFace API key not provided".to_string())
                })?;
                Ok(LLMClient::HuggingFace(
                    HuggingFace::new(api_key.clone()).llama3_8b(),
                ))
            }
            ModelProvider::Azure => {
                let api_key = config.azure_api_key.as_ref().ok_or_else(|| {
                    LangHubError::LLMError("Azure API key not provided".to_string())
                })?;
                let endpoint = config.azure_endpoint.as_ref().ok_or_else(|| {
                    LangHubError::LLMError("Azure endpoint not provided".to_string())
                })?;
                let deployment = config.azure_deployment_name.as_ref().ok_or_else(|| {
                    LangHubError::LLMError("Azure deployment name not provided".to_string())
                })?;
                Ok(LLMClient::Azure(AzureOpenAI::new(
                    api_key.clone(),
                    endpoint.clone(),
                    deployment.clone(),
                )))
            }
            ModelProvider::Mistral => {
                let api_key = config.mistral_api_key.as_ref().ok_or_else(|| {
                    LangHubError::LLMError("Mistral API key not provided".to_string())
                })?;
                Ok(LLMClient::Mistral(Mistral::new(api_key.clone()).small()))
            }
            ModelProvider::Groq => {
                let api_key = config.groq_api_key.as_ref().ok_or_else(|| {
                    LangHubError::LLMError("Groq API key not provided".to_string())
                })?;
                Ok(LLMClient::Groq(Groq::new(api_key.clone()).mixtral()))
            }
            ModelProvider::Together => {
                let api_key = config.together_api_key.as_ref().ok_or_else(|| {
                    LangHubError::LLMError("Together API key not provided".to_string())
                })?;
                Ok(LLMClient::Together(
                    Together::new(api_key.clone()).mixtral(),
                ))
            }
            ModelProvider::Replicate => {
                let api_key = config.replicate_api_key.as_ref().ok_or_else(|| {
                    LangHubError::LLMError("Replicate API key not provided".to_string())
                })?;
                Ok(LLMClient::Replicate(
                    Replicate::new(api_key.clone()).mixtral(),
                ))
            }
            ModelProvider::Fireworks => {
                let api_key = config.fireworks_api_key.as_ref().ok_or_else(|| {
                    LangHubError::LLMError("Fireworks API key not provided".to_string())
                })?;
                Ok(LLMClient::Fireworks(
                    Fireworks::new(api_key.clone()).mixtral(),
                ))
            }
            ModelProvider::Perplexity => {
                let api_key = config.perplexity_api_key.as_ref().ok_or_else(|| {
                    LangHubError::LLMError("Perplexity API key not provided".to_string())
                })?;
                Ok(LLMClient::Perplexity(
                    Perplexity::new(api_key.clone()).sonar_medium(),
                ))
            }
            ModelProvider::Baidu => {
                let api_key = config.baidu_api_key.as_ref().ok_or_else(|| {
                    LangHubError::LLMError("Baidu API key not provided".to_string())
                })?;
                let secret_key = config.baidu_secret_key.as_ref().ok_or_else(|| {
                    LangHubError::LLMError("Baidu secret key not provided".to_string())
                })?;
                Ok(LLMClient::Baidu(
                    BaiduWenxin::new(api_key.clone(), secret_key.clone()).ernie4_0(),
                ))
            }
            ModelProvider::Alibaba => {
                let api_key = config.alibaba_api_key.as_ref().ok_or_else(|| {
                    LangHubError::LLMError("Alibaba API key not provided".to_string())
                })?;
                Ok(LLMClient::Alibaba(
                    AlibabaTongyi::new(api_key.clone()).qwen_plus(),
                ))
            }
            ModelProvider::Tencent => {
                let secret_id = config.tencent_secret_id.as_ref().ok_or_else(|| {
                    LangHubError::LLMError("Tencent secret ID not provided".to_string())
                })?;
                let secret_key = config.tencent_secret_key.as_ref().ok_or_else(|| {
                    LangHubError::LLMError("Tencent secret key not provided".to_string())
                })?;
                Ok(LLMClient::Tencent(
                    TencentHunyuan::new(secret_id.clone(), secret_key.clone()).hunyuan_pro(),
                ))
            }
            ModelProvider::Zhipu => {
                let api_key = config.zhipu_api_key.as_ref().ok_or_else(|| {
                    LangHubError::LLMError("Zhipu API key not provided".to_string())
                })?;
                Ok(LLMClient::Zhipu(ZhipuAI::new(api_key.clone()).glm4()))
            }
            ModelProvider::MiniMax => {
                let api_key = config.minimax_api_key.as_ref().ok_or_else(|| {
                    LangHubError::LLMError("MiniMax API key not provided".to_string())
                })?;
                let group_id = config.minimax_group_id.as_ref().ok_or_else(|| {
                    LangHubError::LLMError("MiniMax group ID not provided".to_string())
                })?;
                Ok(LLMClient::MiniMax(
                    MiniMax::new(api_key.clone(), group_id.clone()).abab6_5(),
                ))
            }
            ModelProvider::Moonshot => {
                let api_key = config.moonshot_api_key.as_ref().ok_or_else(|| {
                    LangHubError::LLMError("Moonshot API key not provided".to_string())
                })?;
                Ok(LLMClient::Moonshot(
                    Moonshot::new(api_key.clone()).kimi_128k(),
                ))
            }
            ModelProvider::Baichuan => {
                let api_key = config.baichuan_api_key.as_ref().ok_or_else(|| {
                    LangHubError::LLMError("Baichuan API key not provided".to_string())
                })?;
                Ok(LLMClient::Baichuan(
                    Baichuan::new(api_key.clone()).baichuan4(),
                ))
            }
            ModelProvider::Yi => {
                let api_key = config
                    .yi_api_key
                    .as_ref()
                    .ok_or_else(|| LangHubError::LLMError("Yi API key not provided".to_string()))?;
                Ok(LLMClient::Yi(Yi::new(api_key.clone()).yi34b()))
            }
            ModelProvider::Custom => {
                let api_key = config.openai_api_key.as_ref().ok_or_else(|| {
                    LangHubError::LLMError("Custom model API key not provided".to_string())
                })?;
                let api_base = config.custom_api_base.as_ref().ok_or_else(|| {
                    LangHubError::LLMError("Custom model API base URL not provided".to_string())
                })?;
                Ok(LLMClient::Custom(CustomLLM::new(
                    api_key.clone(),
                    api_base.clone(),
                )))
            }
        }
    }
    /// Generates a text completion from a prompt
    ///
    /// # Arguments
    /// * `prompt` - The input prompt string
    ///
    /// # Returns
    /// A `Result` containing the generated text or an error
    ///
    /// # Example
    /// ```
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// # let config = langhub::LLMConfig::new().openai("sk-xxx".to_string());
    /// # let client = langhub::LLMClient::new_with_config(langhub::types::ModelProvider::OpenAI, &config)?;
    /// let response = client.generate("What is the capital of France?").await?;
    /// println!("{}", response); // "The capital of France is Paris."
    /// # Ok(())
    /// # }
    /// ```
    pub async fn generate(&self, prompt: &str) -> Result<LLMResult> {
        match self {
            LLMClient::OpenAI(m) => m.generate(prompt).await,
            LLMClient::Anthropic(m) => m.generate(prompt).await,
            LLMClient::DeepSeek(m) => m.generate(prompt).await,
            LLMClient::Google(m) => m.generate(prompt).await,
            LLMClient::Cohere(m) => m.generate(prompt).await,
            LLMClient::HuggingFace(m) => m.generate(prompt).await,
            LLMClient::Azure(m) => m.generate(prompt).await,
            LLMClient::Mistral(m) => m.generate(prompt).await,
            LLMClient::Groq(m) => m.generate(prompt).await,
            LLMClient::Together(m) => m.generate(prompt).await,
            LLMClient::Replicate(m) => m.generate(prompt).await,
            LLMClient::Fireworks(m) => m.generate(prompt).await,
            LLMClient::Perplexity(m) => m.generate(prompt).await,
            LLMClient::Baidu(m) => m.generate(prompt).await,
            LLMClient::Alibaba(m) => m.generate(prompt).await,
            LLMClient::Tencent(m) => m.generate(prompt).await,
            LLMClient::Zhipu(m) => m.generate(prompt).await,
            LLMClient::MiniMax(m) => m.generate(prompt).await,
            LLMClient::Moonshot(m) => m.generate(prompt).await,
            LLMClient::Baichuan(m) => m.generate(prompt).await,
            LLMClient::Yi(m) => m.generate(prompt).await,
            LLMClient::Custom(m) => m.generate(prompt).await,
        }
    }
    /// Generates a chat completion from a conversation history
    ///
    /// # Arguments
    /// * `messages` - A vector of chat messages representing the conversation
    ///
    /// # Returns
    /// A `Result` containing the llm's response or an error
    ///
    /// # Example
    /// ```
    /// use langhub::types::ChatMessage;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// # let config = langhub::LLMConfig::new().openai("sk-xxx".to_string());
    /// # let client = langhub::LLMClient::new_with_config(langhub::types::ModelProvider::OpenAI, &config)?;
    /// let messages = vec![
    ///     ChatMessage::user("Hello, who are you?"),
    ///     ChatMessage::llm("I am an llm."),
    ///     ChatMessage::user("What can you do?"),
    /// ];
    /// let response = client.chat(messages).await?;
    /// println!("{}", response);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn chat(&self, messages: Vec<ChatMessage>) -> Result<LLMResult> {
        match self {
            LLMClient::OpenAI(m) => m.chat(messages).await,
            LLMClient::Anthropic(m) => m.chat(messages).await,
            LLMClient::DeepSeek(m) => m.chat(messages).await,
            LLMClient::Google(m) => m.chat(messages).await,
            LLMClient::Cohere(m) => m.chat(messages).await,
            LLMClient::HuggingFace(m) => m.chat(messages).await,
            LLMClient::Azure(m) => m.chat(messages).await,
            LLMClient::Mistral(m) => m.chat(messages).await,
            LLMClient::Groq(m) => m.chat(messages).await,
            LLMClient::Together(m) => m.chat(messages).await,
            LLMClient::Replicate(m) => m.chat(messages).await,
            LLMClient::Fireworks(m) => m.chat(messages).await,
            LLMClient::Perplexity(m) => m.chat(messages).await,
            LLMClient::Baidu(m) => m.chat(messages).await,
            LLMClient::Alibaba(m) => m.chat(messages).await,
            LLMClient::Tencent(m) => m.chat(messages).await,
            LLMClient::Zhipu(m) => m.chat(messages).await,
            LLMClient::MiniMax(m) => m.chat(messages).await,
            LLMClient::Moonshot(m) => m.chat(messages).await,
            LLMClient::Baichuan(m) => m.chat(messages).await,
            LLMClient::Yi(m) => m.chat(messages).await,
            LLMClient::Custom(m) => m.chat(messages).await,
        }
    }
}
/// Configuration for video LLM client initialization
///
/// # Example
/// ```
/// use langhub::VideoLLMConfig;
///
/// let config = VideoLLMConfig::new()
///     .seedance("your-ark-api-key".to_string())
///     .wan("your-dashscope-api-key".to_string());
/// ```
#[derive(Debug, Clone, Default)]
pub struct VideoLLMConfig {
    /// Seedance API key (Volcengine Ark)
    pub seedance_api_key: Option<String>,
    /// Seedance custom base URL
    pub seedance_base_url: Option<String>,
    /// Wan API key (Alibaba Cloud Bailian)
    pub wan_api_key: Option<String>,
    /// Wan custom base URL
    pub wan_base_url: Option<String>,
    /// Kling API key
    pub kling_api_key: Option<String>,
    /// Kling secret key
    pub kling_secret_key: Option<String>,
    /// Kling custom base URL
    pub kling_base_url: Option<String>,
    /// Veo API key (Google AI Studio)
    pub veo_api_key: Option<String>,
    /// Veo custom base URL
    pub veo_base_url: Option<String>,
    /// Runway API key
    pub runway_api_key: Option<String>,
    /// Runway custom base URL
    pub runway_base_url: Option<String>,
    /// MiniMax H3 API key
    pub minimax_h3_api_key: Option<String>,
    /// MiniMax H3 group ID
    pub minimax_h3_group_id: Option<String>,
    /// MiniMax H3 custom base URL
    pub minimax_h3_base_url: Option<String>,
    /// HappyHorse API key
    pub happyhorse_api_key: Option<String>,
    /// HappyHorse custom base URL
    pub happyhorse_base_url: Option<String>,
    /// LTX API key
    pub ltx_api_key: Option<String>,
    /// LTX custom base URL
    pub ltx_base_url: Option<String>,
    /// Grok Imagine API key
    pub grok_api_key: Option<String>,
    /// Grok Imagine custom base URL
    pub grok_base_url: Option<String>,
    /// Pruna API key
    pub pruna_api_key: Option<String>,
    /// Pruna custom base URL
    pub pruna_base_url: Option<String>,
    /// Gemini API key (for Omni Flash)
    pub gemini_api_key: Option<String>,
    /// Gemini custom base URL
    pub gemini_base_url: Option<String>,
}
impl VideoLLMConfig {
    /// Creates a new empty video configuration
    ///
    /// # Example
    /// ```
    /// let config = VideoLLMConfig::new();
    /// ```
    pub fn new() -> Self {
        Self::default()
    }
    /// Sets the Seedance API key
    ///
    /// # Arguments
    /// * `api_key` - Seedance API key
    pub fn seedance(mut self, api_key: String) -> Self {
        self.seedance_api_key = Some(api_key);
        self
    }
    /// Sets the Wan API key
    ///
    /// # Arguments
    /// * `api_key` - Wan API key
    pub fn wan(mut self, api_key: String) -> Self {
        self.wan_api_key = Some(api_key);
        self
    }
    /// Sets the Kling API key and secret key
    ///
    /// # Arguments
    /// * `api_key` - Kling API key
    /// * `secret_key` - Kling secret key
    pub fn kling(mut self, api_key: String, secret_key: String) -> Self {
        self.kling_api_key = Some(api_key);
        self.kling_secret_key = Some(secret_key);
        self
    }
    /// Sets the Veo API key
    ///
    /// # Arguments
    /// * `api_key` - Veo API key
    pub fn veo(mut self, api_key: String) -> Self {
        self.veo_api_key = Some(api_key);
        self
    }
    /// Sets the Runway API key
    ///
    /// # Arguments
    /// * `api_key` - Runway API key
    pub fn runway(mut self, api_key: String) -> Self {
        self.runway_api_key = Some(api_key);
        self
    }
    /// Sets the MiniMax H3 API key and group ID
    ///
    /// # Arguments
    /// * `api_key` - MiniMax H3 API key
    /// * `group_id` - MiniMax H3 group ID
    pub fn minimax_h3(mut self, api_key: String, group_id: String) -> Self {
        self.minimax_h3_api_key = Some(api_key);
        self.minimax_h3_group_id = Some(group_id);
        self
    }
    /// Sets the HappyHorse API key
    ///
    /// # Arguments
    /// * `api_key` - HappyHorse API key
    pub fn happyhorse(mut self, api_key: String) -> Self {
        self.happyhorse_api_key = Some(api_key);
        self
    }
    /// Sets the LTX API key
    ///
    /// # Arguments
    /// * `api_key` - LTX API key
    pub fn ltx(mut self, api_key: String) -> Self {
        self.ltx_api_key = Some(api_key);
        self
    }
    /// Sets the Grok Imagine API key
    ///
    /// # Arguments
    /// * `api_key` - Grok Imagine API key
    pub fn grok(mut self, api_key: String) -> Self {
        self.grok_api_key = Some(api_key);
        self
    }
    /// Sets the Pruna API key
    ///
    /// # Arguments
    /// * `api_key` - Pruna API key
    pub fn pruna(mut self, api_key: String) -> Self {
        self.pruna_api_key = Some(api_key);
        self
    }
    /// Sets the Gemini API key
    ///
    /// # Arguments
    /// * `api_key` - Gemini API key
    pub fn gemini(mut self, api_key: String) -> Self {
        self.gemini_api_key = Some(api_key);
        self
    }
}
/// Unified video generation client for multiple providers
///
/// This is the video counterpart of `LLMClient`. It represents a client for
/// any supported text-to-video provider.
///
/// # Example
/// ```
/// use langhub::{VideoLLMClient, VideoLLMConfig, VideoModelProvider};
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let config = VideoLLMConfig::new().seedance("your-api-key".to_string());
/// let client = VideoLLMClient::new_with_config(VideoModelProvider::Seedance, &config)?;
/// let result = client.generate("A cat walking on the beach").await?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub enum VideoLLMClient {
    Seedance(Seedance),
    Wan(WanVideo),
    Kling(KlingVideo),
    Veo(Veo),
    Runway(RunwayVideo),
    MiniMaxH3(MiniMaxH3),
    HappyHorse(HappyHorse),
    Ltx(LtxVideo),
    GrokImagine(GrokImagine),
    Pruna(PrunaVideo),
    GeminiOmniFlash(GeminiOmniFlash),
}
impl VideoLLMClient {
    /// Creates a new video client with the given provider using optional API keys
    ///
    /// # Arguments
    /// * `provider` - The video model provider to use
    /// * `api_key` - Optional API key for the provider
    /// * `extra_keys` - Optional additional keys for providers that need them
    ///
    /// # Returns
    /// A `Result` containing the video client or an error if required keys are missing
    ///
    /// # Example
    /// ```
    /// use langhub::{VideoLLMClient, VideoModelProvider};
    ///
    /// let client = VideoLLMClient::new_with_key(
    ///     VideoModelProvider::Seedance,
    ///     Some("your-api-key".to_string()),
    ///     None,
    /// ).unwrap();
    /// ```
    pub fn new_with_key(
        provider: VideoModelProvider,
        api_key: Option<String>,
        extra_keys: Option<std::collections::HashMap<String, String>>,
    ) -> Result<Self> {
        let extra = extra_keys.unwrap_or_default();
        match provider {
            VideoModelProvider::Seedance => {
                let key = api_key.ok_or_else(|| {
                    LangHubError::LLMError("Seedance API key not provided".to_string())
                })?;
                let mut client = Seedance::new(key);
                if let Some(base) = extra.get("base_url") {
                    client = client.with_base_url(base);
                }
                Ok(VideoLLMClient::Seedance(client))
            }
            VideoModelProvider::Wan => {
                let key = api_key.ok_or_else(|| {
                    LangHubError::LLMError("Wan API key not provided".to_string())
                })?;
                let mut client = WanVideo::new(key);
                if let Some(base) = extra.get("base_url") {
                    client = client.with_base_url(base);
                }
                Ok(VideoLLMClient::Wan(client))
            }
            VideoModelProvider::Kling => {
                let key = api_key.ok_or_else(|| {
                    LangHubError::LLMError("Kling API key not provided".to_string())
                })?;
                let mut client = KlingVideo::new(key);
                if let Some(secret) = extra.get("secret_key") {
                    client = client.with_secret_key(secret.clone());
                }
                if let Some(base) = extra.get("base_url") {
                    client = client.with_base_url(base);
                }
                Ok(VideoLLMClient::Kling(client))
            }
            VideoModelProvider::Veo => {
                let key = api_key.ok_or_else(|| {
                    LangHubError::LLMError("Veo API key not provided".to_string())
                })?;
                let mut client = Veo::new(key);
                if let Some(base) = extra.get("base_url") {
                    client = client.with_base_url(base);
                }
                Ok(VideoLLMClient::Veo(client))
            }
            VideoModelProvider::Runway => {
                let key = api_key.ok_or_else(|| {
                    LangHubError::LLMError("Runway API key not provided".to_string())
                })?;
                let mut client = RunwayVideo::new(key);
                if let Some(base) = extra.get("base_url") {
                    client = client.with_base_url(base);
                }
                Ok(VideoLLMClient::Runway(client))
            }
            VideoModelProvider::MiniMaxH3 => {
                let key = api_key.ok_or_else(|| {
                    LangHubError::LLMError("MiniMax H3 API key not provided".to_string())
                })?;
                let mut client = MiniMaxH3::new(key);
                if let Some(group) = extra.get("group_id") {
                    client = client.with_group_id(group.clone());
                }
                if let Some(base) = extra.get("base_url") {
                    client = client.with_base_url(base);
                }
                Ok(VideoLLMClient::MiniMaxH3(client))
            }
            VideoModelProvider::HappyHorse => {
                let key = api_key.ok_or_else(|| {
                    LangHubError::LLMError("HappyHorse API key not provided".to_string())
                })?;
                let mut client = HappyHorse::new(key);
                if let Some(base) = extra.get("base_url") {
                    client = client.with_base_url(base);
                }
                Ok(VideoLLMClient::HappyHorse(client))
            }
            VideoModelProvider::Ltx => {
                let key = api_key.ok_or_else(|| {
                    LangHubError::LLMError("LTX API key not provided".to_string())
                })?;
                let mut client = LtxVideo::new(key);
                if let Some(base) = extra.get("base_url") {
                    client = client.with_base_url(base);
                }
                Ok(VideoLLMClient::Ltx(client))
            }
            VideoModelProvider::GrokImagine => {
                let key = api_key.ok_or_else(|| {
                    LangHubError::LLMError("Grok Imagine API key not provided".to_string())
                })?;
                let mut client = GrokImagine::new(key);
                if let Some(base) = extra.get("base_url") {
                    client = client.with_base_url(base);
                }
                Ok(VideoLLMClient::GrokImagine(client))
            }
            VideoModelProvider::Pruna => {
                let key = api_key.ok_or_else(|| {
                    LangHubError::LLMError("Pruna API key not provided".to_string())
                })?;
                let mut client = PrunaVideo::new(key);
                if let Some(base) = extra.get("base_url") {
                    client = client.with_base_url(base);
                }
                Ok(VideoLLMClient::Pruna(client))
            }
            VideoModelProvider::GeminiOmniFlash => {
                let key = api_key.ok_or_else(|| {
                    LangHubError::LLMError("Gemini Omni Flash API key not provided".to_string())
                })?;
                let mut client = GeminiOmniFlash::new(key);
                if let Some(base) = extra.get("base_url") {
                    client = client.with_base_url(base);
                }
                Ok(VideoLLMClient::GeminiOmniFlash(client))
            }
        }
    }
    /// Creates a new video client with the given provider and configuration
    ///
    /// # Arguments
    /// * `provider` - The video model provider to use
    /// * `config` - Configuration containing API keys and credentials
    ///
    /// # Returns
    /// A `Result` containing the video client or an error if required keys are missing
    ///
    /// # Errors
    /// Returns `LangHubError::LLMError` if the required API key for the provider is not provided
    ///
    /// # Example
    /// ```
    /// use langhub::{VideoLLMClient, VideoLLMConfig, VideoModelProvider};
    ///
    /// let config = VideoLLMConfig::new()
    ///     .seedance("your-ark-api-key".to_string())
    ///     .wan("your-dashscope-api-key".to_string());
    ///
    /// let seedance_client = VideoLLMClient::new_with_config(VideoModelProvider::Seedance, &config).unwrap();
    /// let wan_client = VideoLLMClient::new_with_config(VideoModelProvider::Wan, &config).unwrap();
    /// ```
    pub fn new_with_config(provider: VideoModelProvider, config: &VideoLLMConfig) -> Result<Self> {
        match provider {
            VideoModelProvider::Seedance => {
                let key = config.seedance_api_key.as_ref().ok_or_else(|| {
                    LangHubError::LLMError("Seedance API key not provided".to_string())
                })?;
                let mut client = Seedance::new(key.clone());
                if let Some(base) = &config.seedance_base_url {
                    client = client.with_base_url(base);
                }
                Ok(VideoLLMClient::Seedance(client))
            }
            VideoModelProvider::Wan => {
                let key = config.wan_api_key.as_ref().ok_or_else(|| {
                    LangHubError::LLMError("Wan API key not provided".to_string())
                })?;
                let mut client = WanVideo::new(key.clone());
                if let Some(base) = &config.wan_base_url {
                    client = client.with_base_url(base);
                }
                Ok(VideoLLMClient::Wan(client))
            }
            VideoModelProvider::Kling => {
                let key = config.kling_api_key.as_ref().ok_or_else(|| {
                    LangHubError::LLMError("Kling API key not provided".to_string())
                })?;
                let mut client = KlingVideo::new(key.clone());
                if let Some(secret) = &config.kling_secret_key {
                    client = client.with_secret_key(secret.clone());
                }
                if let Some(base) = &config.kling_base_url {
                    client = client.with_base_url(base);
                }
                Ok(VideoLLMClient::Kling(client))
            }
            VideoModelProvider::Veo => {
                let key = config.veo_api_key.as_ref().ok_or_else(|| {
                    LangHubError::LLMError("Veo API key not provided".to_string())
                })?;
                let mut client = Veo::new(key.clone());
                if let Some(base) = &config.veo_base_url {
                    client = client.with_base_url(base);
                }
                Ok(VideoLLMClient::Veo(client))
            }
            VideoModelProvider::Runway => {
                let key = config.runway_api_key.as_ref().ok_or_else(|| {
                    LangHubError::LLMError("Runway API key not provided".to_string())
                })?;
                let mut client = RunwayVideo::new(key.clone());
                if let Some(base) = &config.runway_base_url {
                    client = client.with_base_url(base);
                }
                Ok(VideoLLMClient::Runway(client))
            }
            VideoModelProvider::MiniMaxH3 => {
                let key = config.minimax_h3_api_key.as_ref().ok_or_else(|| {
                    LangHubError::LLMError("MiniMax H3 API key not provided".to_string())
                })?;
                let mut client = MiniMaxH3::new(key.clone());
                if let Some(group) = &config.minimax_h3_group_id {
                    client = client.with_group_id(group.clone());
                }
                if let Some(base) = &config.minimax_h3_base_url {
                    client = client.with_base_url(base);
                }
                Ok(VideoLLMClient::MiniMaxH3(client))
            }
            VideoModelProvider::HappyHorse => {
                let key = config.happyhorse_api_key.as_ref().ok_or_else(|| {
                    LangHubError::LLMError("HappyHorse API key not provided".to_string())
                })?;
                let mut client = HappyHorse::new(key.clone());
                if let Some(base) = &config.happyhorse_base_url {
                    client = client.with_base_url(base);
                }
                Ok(VideoLLMClient::HappyHorse(client))
            }
            VideoModelProvider::Ltx => {
                let key = config.ltx_api_key.as_ref().ok_or_else(|| {
                    LangHubError::LLMError("LTX API key not provided".to_string())
                })?;
                let mut client = LtxVideo::new(key.clone());
                if let Some(base) = &config.ltx_base_url {
                    client = client.with_base_url(base);
                }
                Ok(VideoLLMClient::Ltx(client))
            }
            VideoModelProvider::GrokImagine => {
                let key = config.grok_api_key.as_ref().ok_or_else(|| {
                    LangHubError::LLMError("Grok Imagine API key not provided".to_string())
                })?;
                let mut client = GrokImagine::new(key.clone());
                if let Some(base) = &config.grok_base_url {
                    client = client.with_base_url(base);
                }
                Ok(VideoLLMClient::GrokImagine(client))
            }
            VideoModelProvider::Pruna => {
                let key = config.pruna_api_key.as_ref().ok_or_else(|| {
                    LangHubError::LLMError("Pruna API key not provided".to_string())
                })?;
                let mut client = PrunaVideo::new(key.clone());
                if let Some(base) = &config.pruna_base_url {
                    client = client.with_base_url(base);
                }
                Ok(VideoLLMClient::Pruna(client))
            }
            VideoModelProvider::GeminiOmniFlash => {
                let key = config.gemini_api_key.as_ref().ok_or_else(|| {
                    LangHubError::LLMError("Gemini API key not provided".to_string())
                })?;
                let mut client = GeminiOmniFlash::new(key.clone());
                if let Some(base) = &config.gemini_base_url {
                    client = client.with_base_url(base);
                }
                Ok(VideoLLMClient::GeminiOmniFlash(client))
            }
        }
    }
    /// Generates a video from a text prompt
    ///
    /// # Arguments
    /// * `prompt` - The input text prompt string
    ///
    /// # Returns
    /// A `Result` containing the generated video result or an error
    ///
    /// # Example
    /// ```
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// # let config = langhub::VideoLLMConfig::new().seedance("your-api-key".to_string());
    /// # let client = langhub::VideoLLMClient::new_with_config(langhub::video::VideoModelProvider::Seedance, &config)?;
    /// let result = client.generate("A cat walking on the beach").await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn generate(&self, prompt: &str) -> Result<VideoLLMResult> {
        match self {
            VideoLLMClient::Seedance(m) => m.generate(prompt).await,
            VideoLLMClient::Wan(m) => m.generate(prompt).await,
            VideoLLMClient::Kling(m) => m.generate(prompt).await,
            VideoLLMClient::Veo(m) => m.generate(prompt).await,
            VideoLLMClient::Runway(m) => m.generate(prompt).await,
            VideoLLMClient::MiniMaxH3(m) => m.generate(prompt).await,
            VideoLLMClient::HappyHorse(m) => m.generate(prompt).await,
            VideoLLMClient::Ltx(m) => m.generate(prompt).await,
            VideoLLMClient::GrokImagine(m) => m.generate(prompt).await,
            VideoLLMClient::Pruna(m) => m.generate(prompt).await,
            VideoLLMClient::GeminiOmniFlash(m) => m.generate(prompt).await,
        }
    }
    /// Generates a video with options
    ///
    /// # Arguments
    /// * `prompt` - The input text prompt string
    /// * `options` - Generation options such as duration, resolution, aspect ratio
    ///
    /// # Returns
    /// A `Result` containing the generated video result or an error
    pub async fn generate_with_options(
        &self,
        prompt: &str,
        options: VideoLLMOptions,
    ) -> Result<VideoLLMResult> {
        match self {
            VideoLLMClient::Seedance(m) => m.generate_with_options(prompt, options).await,
            VideoLLMClient::Wan(m) => m.generate_with_options(prompt, options).await,
            VideoLLMClient::Kling(m) => m.generate_with_options(prompt, options).await,
            VideoLLMClient::Veo(m) => m.generate_with_options(prompt, options).await,
            VideoLLMClient::Runway(m) => m.generate_with_options(prompt, options).await,
            VideoLLMClient::MiniMaxH3(m) => m.generate_with_options(prompt, options).await,
            VideoLLMClient::HappyHorse(m) => m.generate_with_options(prompt, options).await,
            VideoLLMClient::Ltx(m) => m.generate_with_options(prompt, options).await,
            VideoLLMClient::GrokImagine(m) => m.generate_with_options(prompt, options).await,
            VideoLLMClient::Pruna(m) => m.generate_with_options(prompt, options).await,
            VideoLLMClient::GeminiOmniFlash(m) => m.generate_with_options(prompt, options).await,
        }
    }
    /// Submits an asynchronous video generation task
    ///
    /// # Arguments
    /// * `prompt` - The input text prompt string
    /// * `options` - Generation options
    ///
    /// # Returns
    /// A `Result` containing a `VideoTask` handle for polling
    pub async fn submit_task(&self, prompt: &str, options: VideoLLMOptions) -> Result<VideoTask> {
        match self {
            VideoLLMClient::Seedance(m) => m.submit_task(prompt, options).await,
            VideoLLMClient::Wan(m) => m.submit_task(prompt, options).await,
            VideoLLMClient::Kling(m) => m.submit_task(prompt, options).await,
            VideoLLMClient::Veo(m) => m.submit_task(prompt, options).await,
            VideoLLMClient::Runway(m) => m.submit_task(prompt, options).await,
            VideoLLMClient::MiniMaxH3(m) => m.submit_task(prompt, options).await,
            VideoLLMClient::HappyHorse(m) => m.submit_task(prompt, options).await,
            VideoLLMClient::Ltx(m) => m.submit_task(prompt, options).await,
            VideoLLMClient::GrokImagine(m) => m.submit_task(prompt, options).await,
            VideoLLMClient::Pruna(m) => m.submit_task(prompt, options).await,
            VideoLLMClient::GeminiOmniFlash(m) => m.submit_task(prompt, options).await,
        }
    }
    /// Polls an asynchronous video generation task
    ///
    /// # Arguments
    /// * `task_id` - The task ID returned by `submit_task`
    ///
    /// # Returns
    /// A `Result` containing the current `VideoTask` state
    pub async fn poll_task(&self, task_id: &str) -> Result<VideoTask> {
        match self {
            VideoLLMClient::Seedance(m) => m.poll_task(task_id).await,
            VideoLLMClient::Wan(m) => m.poll_task(task_id).await,
            VideoLLMClient::Kling(m) => m.poll_task(task_id).await,
            VideoLLMClient::Veo(m) => m.poll_task(task_id).await,
            VideoLLMClient::Runway(m) => m.poll_task(task_id).await,
            VideoLLMClient::MiniMaxH3(m) => m.poll_task(task_id).await,
            VideoLLMClient::HappyHorse(m) => m.poll_task(task_id).await,
            VideoLLMClient::Ltx(m) => m.poll_task(task_id).await,
            VideoLLMClient::GrokImagine(m) => m.poll_task(task_id).await,
            VideoLLMClient::Pruna(m) => m.poll_task(task_id).await,
            VideoLLMClient::GeminiOmniFlash(m) => m.poll_task(task_id).await,
        }
    }
    /// Gets the provider enum for this client
    ///
    /// # Returns
    /// The `VideoModelProvider` variant corresponding to this client
    pub fn get_provider_enum(&self) -> VideoModelProvider {
        match self {
            VideoLLMClient::Seedance(_) => VideoModelProvider::Seedance,
            VideoLLMClient::Wan(_) => VideoModelProvider::Wan,
            VideoLLMClient::Kling(_) => VideoModelProvider::Kling,
            VideoLLMClient::Veo(_) => VideoModelProvider::Veo,
            VideoLLMClient::Runway(_) => VideoModelProvider::Runway,
            VideoLLMClient::MiniMaxH3(_) => VideoModelProvider::MiniMaxH3,
            VideoLLMClient::HappyHorse(_) => VideoModelProvider::HappyHorse,
            VideoLLMClient::Ltx(_) => VideoModelProvider::Ltx,
            VideoLLMClient::GrokImagine(_) => VideoModelProvider::GrokImagine,
            VideoLLMClient::Pruna(_) => VideoModelProvider::Pruna,
            VideoLLMClient::GeminiOmniFlash(_) => VideoModelProvider::GeminiOmniFlash,
        }
    }
    /// Gets the vendor of this client's model
    ///
    /// # Returns
    /// The `VideoVendor` variant corresponding to this client's provider
    pub fn get_vendor(&self) -> crate::types::VideoVendor {
        self.get_provider_enum().vendor()
    }
}
/// Configuration for image LLM client initialization
///
/// # Example
/// ```
/// use langhub::ImageLLMConfig;
///
/// let config = ImageLLMConfig::new()
///     .seedream("your-ark-api-key".to_string())
///     .dalle("your-openai-api-key".to_string());
/// ```
#[derive(Debug, Clone, Default)]
pub struct ImageLLMConfig {
    /// Seedream API key (Volcengine Ark)
    pub seedream_api_key: Option<String>,
    /// Seedream custom base URL
    pub seedream_base_url: Option<String>,
    /// Wan text-to-image API key (Alibaba Cloud Bailian)
    pub wan_image_api_key: Option<String>,
    /// Wan text-to-image custom base URL
    pub wan_image_base_url: Option<String>,
    /// Stability AI API key
    pub stability_api_key: Option<String>,
    /// Stability AI custom base URL
    pub stability_base_url: Option<String>,
    /// Black Forest Labs (FLUX) API key
    pub flux_api_key: Option<String>,
    /// Black Forest Labs (FLUX) custom base URL
    pub flux_base_url: Option<String>,
    /// Google Imagen API key
    pub imagen_api_key: Option<String>,
    /// Google Imagen custom base URL
    pub imagen_base_url: Option<String>,
    /// OpenAI DALL·E API key
    pub dalle_api_key: Option<String>,
    /// OpenAI DALL·E custom base URL
    pub dalle_base_url: Option<String>,
}
impl ImageLLMConfig {
    /// Creates a new empty image configuration
    ///
    /// # Example
    /// ```
    /// let config = ImageLLMConfig::new();
    /// ```
    pub fn new() -> Self {
        Self::default()
    }
    /// Sets the Seedream API key
    ///
    /// # Arguments
    /// * `api_key` - Seedream API key
    pub fn seedream(mut self, api_key: String) -> Self {
        self.seedream_api_key = Some(api_key);
        self
    }
    /// Sets the Wan text-to-image API key
    ///
    /// # Arguments
    /// * `api_key` - Wan text-to-image API key
    pub fn wan_image(mut self, api_key: String) -> Self {
        self.wan_image_api_key = Some(api_key);
        self
    }
    /// Sets the Stability AI API key
    ///
    /// # Arguments
    /// * `api_key` - Stability AI API key
    pub fn stability(mut self, api_key: String) -> Self {
        self.stability_api_key = Some(api_key);
        self
    }
    /// Sets the Black Forest Labs (FLUX) API key
    ///
    /// # Arguments
    /// * `api_key` - FLUX API key
    pub fn flux(mut self, api_key: String) -> Self {
        self.flux_api_key = Some(api_key);
        self
    }
    /// Sets the Google Imagen API key
    ///
    /// # Arguments
    /// * `api_key` - Imagen API key
    pub fn imagen(mut self, api_key: String) -> Self {
        self.imagen_api_key = Some(api_key);
        self
    }
    /// Sets the OpenAI DALL·E API key
    ///
    /// # Arguments
    /// * `api_key` - DALL·E API key
    pub fn dalle(mut self, api_key: String) -> Self {
        self.dalle_api_key = Some(api_key);
        self
    }
}
/// Unified image generation client for multiple providers
///
/// # Example
/// ```
/// use langhub::{ImageLLMClient, ImageLLMConfig, ImageModelProvider};
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let config = ImageLLMConfig::new().seedream("your-api-key".to_string());
/// let client = ImageLLMClient::new_with_config(ImageModelProvider::Seedream, &config)?;
/// let result = client.generate("A cat sitting on a windowsill").await?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub enum ImageLLMClient {
    Seedream(Seedream),
    WanImage(WanImage),
    StabilityImage(StabilityImage),
    Flux(FluxImage),
    Imagen(Imagen),
    DallE(DallE),
}
impl ImageLLMClient {
    /// Creates a new image client with the given provider using optional API keys
    ///
    /// # Arguments
    /// * `provider` - The image model provider to use
    /// * `api_key` - Optional API key for the provider
    /// * `extra_keys` - Optional additional keys for providers that need them
    ///
    /// # Returns
    /// A `Result` containing the image client or an error if required keys are missing
    ///
    /// # Example
    /// ```
    /// use langhub::{ImageLLMClient, ImageModelProvider};
    ///
    /// let client = ImageLLMClient::new_with_key(
    ///     ImageModelProvider::Seedream,
    ///     Some("your-api-key".to_string()),
    ///     None,
    /// ).unwrap();
    /// ```
    pub fn new_with_key(
        provider: ImageModelProvider,
        api_key: Option<String>,
        extra_keys: Option<std::collections::HashMap<String, String>>,
    ) -> Result<Self> {
        let extra = extra_keys.unwrap_or_default();
        match provider {
            ImageModelProvider::Seedream => {
                let key = api_key.ok_or_else(|| {
                    LangHubError::LLMError("Seedream API key not provided".to_string())
                })?;
                let mut client = Seedream::new(key);
                if let Some(base) = extra.get("base_url") {
                    client = client.with_base_url(base);
                }
                Ok(ImageLLMClient::Seedream(client))
            }
            ImageModelProvider::WanImage => {
                let key = api_key.ok_or_else(|| {
                    LangHubError::LLMError("Wan image API key not provided".to_string())
                })?;
                let mut client = WanImage::new(key);
                if let Some(base) = extra.get("base_url") {
                    client = client.with_base_url(base);
                }
                Ok(ImageLLMClient::WanImage(client))
            }
            ImageModelProvider::StabilityImage => {
                let key = api_key.ok_or_else(|| {
                    LangHubError::LLMError("Stability AI API key not provided".to_string())
                })?;
                let mut client = StabilityImage::new(key);
                if let Some(base) = extra.get("base_url") {
                    client = client.with_base_url(base);
                }
                Ok(ImageLLMClient::StabilityImage(client))
            }
            ImageModelProvider::Flux => {
                let key = api_key.ok_or_else(|| {
                    LangHubError::LLMError("FLUX API key not provided".to_string())
                })?;
                let mut client = FluxImage::new(key);
                if let Some(base) = extra.get("base_url") {
                    client = client.with_base_url(base);
                }
                Ok(ImageLLMClient::Flux(client))
            }
            ImageModelProvider::Imagen => {
                let key = api_key.ok_or_else(|| {
                    LangHubError::LLMError("Imagen API key not provided".to_string())
                })?;
                let mut client = Imagen::new(key);
                if let Some(base) = extra.get("base_url") {
                    client = client.with_base_url(base);
                }
                Ok(ImageLLMClient::Imagen(client))
            }
            ImageModelProvider::DallE => {
                let key = api_key.ok_or_else(|| {
                    LangHubError::LLMError("DALL·E API key not provided".to_string())
                })?;
                let mut client = DallE::new(key);
                if let Some(base) = extra.get("base_url") {
                    client = client.with_base_url(base);
                }
                Ok(ImageLLMClient::DallE(client))
            }
        }
    }
    /// Creates a new image client with the given provider and configuration
    ///
    /// # Arguments
    /// * `provider` - The image model provider to use
    /// * `config` - Configuration containing API keys and credentials
    ///
    /// # Returns
    /// A `Result` containing the image client or an error if required keys are missing
    ///
    /// # Errors
    /// Returns `LangHubError::LLMError` if the required API key for the provider is not provided
    ///
    /// # Example
    /// ```
    /// use langhub::{ImageLLMClient, ImageLLMConfig, ImageModelProvider};
    ///
    /// let config = ImageLLMConfig::new()
    ///     .seedream("your-ark-api-key".to_string())
    ///     .dalle("your-openai-api-key".to_string());
    ///
    /// let seedream_client = ImageLLMClient::new_with_config(ImageModelProvider::Seedream, &config).unwrap();
    /// let dalle_client = ImageLLMClient::new_with_config(ImageModelProvider::DallE, &config).unwrap();
    /// ```
    pub fn new_with_config(provider: ImageModelProvider, config: &ImageLLMConfig) -> Result<Self> {
        match provider {
            ImageModelProvider::Seedream => {
                let key = config.seedream_api_key.as_ref().ok_or_else(|| {
                    LangHubError::LLMError("Seedream API key not provided".to_string())
                })?;
                let mut client = Seedream::new(key.clone());
                if let Some(base) = &config.seedream_base_url {
                    client = client.with_base_url(base);
                }
                Ok(ImageLLMClient::Seedream(client))
            }
            ImageModelProvider::WanImage => {
                let key = config.wan_image_api_key.as_ref().ok_or_else(|| {
                    LangHubError::LLMError("Wan image API key not provided".to_string())
                })?;
                let mut client = WanImage::new(key.clone());
                if let Some(base) = &config.wan_image_base_url {
                    client = client.with_base_url(base);
                }
                Ok(ImageLLMClient::WanImage(client))
            }
            ImageModelProvider::StabilityImage => {
                let key = config.stability_api_key.as_ref().ok_or_else(|| {
                    LangHubError::LLMError("Stability AI API key not provided".to_string())
                })?;
                let mut client = StabilityImage::new(key.clone());
                if let Some(base) = &config.stability_base_url {
                    client = client.with_base_url(base);
                }
                Ok(ImageLLMClient::StabilityImage(client))
            }
            ImageModelProvider::Flux => {
                let key = config.flux_api_key.as_ref().ok_or_else(|| {
                    LangHubError::LLMError("FLUX API key not provided".to_string())
                })?;
                let mut client = FluxImage::new(key.clone());
                if let Some(base) = &config.flux_base_url {
                    client = client.with_base_url(base);
                }
                Ok(ImageLLMClient::Flux(client))
            }
            ImageModelProvider::Imagen => {
                let key = config.imagen_api_key.as_ref().ok_or_else(|| {
                    LangHubError::LLMError("Imagen API key not provided".to_string())
                })?;
                let mut client = Imagen::new(key.clone());
                if let Some(base) = &config.imagen_base_url {
                    client = client.with_base_url(base);
                }
                Ok(ImageLLMClient::Imagen(client))
            }
            ImageModelProvider::DallE => {
                let key = config.dalle_api_key.as_ref().ok_or_else(|| {
                    LangHubError::LLMError("DALL·E API key not provided".to_string())
                })?;
                let mut client = DallE::new(key.clone());
                if let Some(base) = &config.dalle_base_url {
                    client = client.with_base_url(base);
                }
                Ok(ImageLLMClient::DallE(client))
            }
        }
    }
    /// Generates images from a text prompt
    ///
    /// # Arguments
    /// * `prompt` - The input text prompt string
    ///
    /// # Returns
    /// A `Result` containing the generated image result or an error
    ///
    /// # Example
    /// ```
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// # let config = langhub::ImageLLMConfig::new().seedream("your-api-key".to_string());
    /// # let client = langhub::ImageLLMClient::new_with_config(langhub::image::ImageModelProvider::Seedream, &config)?;
    /// let result = client.generate("A cat sitting on a windowsill").await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn generate(&self, prompt: &str) -> Result<ImageLLMResult> {
        match self {
            ImageLLMClient::Seedream(m) => m.generate(prompt).await,
            ImageLLMClient::WanImage(m) => m.generate(prompt).await,
            ImageLLMClient::StabilityImage(m) => m.generate(prompt).await,
            ImageLLMClient::Flux(m) => m.generate(prompt).await,
            ImageLLMClient::Imagen(m) => m.generate(prompt).await,
            ImageLLMClient::DallE(m) => m.generate(prompt).await,
        }
    }
    /// Generates images with options
    ///
    /// # Arguments
    /// * `prompt` - The input text prompt string
    /// * `options` - Generation options such as n, resolution, aspect ratio
    ///
    /// # Returns
    /// A `Result` containing the generated image result or an error
    pub async fn generate_with_options(
        &self,
        prompt: &str,
        options: ImageLLMOptions,
    ) -> Result<ImageLLMResult> {
        match self {
            ImageLLMClient::Seedream(m) => m.generate_with_options(prompt, options).await,
            ImageLLMClient::WanImage(m) => m.generate_with_options(prompt, options).await,
            ImageLLMClient::StabilityImage(m) => m.generate_with_options(prompt, options).await,
            ImageLLMClient::Flux(m) => m.generate_with_options(prompt, options).await,
            ImageLLMClient::Imagen(m) => m.generate_with_options(prompt, options).await,
            ImageLLMClient::DallE(m) => m.generate_with_options(prompt, options).await,
        }
    }
    /// Submits an asynchronous image generation task
    ///
    /// # Arguments
    /// * `prompt` - The input text prompt string
    /// * `options` - Generation options
    ///
    /// # Returns
    /// A `Result` containing an `ImageTask` handle for polling
    pub async fn submit_task(&self, prompt: &str, options: ImageLLMOptions) -> Result<ImageTask> {
        match self {
            ImageLLMClient::Seedream(m) => m.submit_task(prompt, options).await,
            ImageLLMClient::WanImage(m) => m.submit_task(prompt, options).await,
            ImageLLMClient::StabilityImage(m) => m.submit_task(prompt, options).await,
            ImageLLMClient::Flux(m) => m.submit_task(prompt, options).await,
            ImageLLMClient::Imagen(m) => m.submit_task(prompt, options).await,
            ImageLLMClient::DallE(m) => m.submit_task(prompt, options).await,
        }
    }
    /// Polls an asynchronous image generation task
    ///
    /// # Arguments
    /// * `task_id` - The task ID returned by `submit_task`
    ///
    /// # Returns
    /// A `Result` containing the current `ImageTask` state
    pub async fn poll_task(&self, task_id: &str) -> Result<ImageTask> {
        match self {
            ImageLLMClient::Seedream(m) => m.poll_task(task_id).await,
            ImageLLMClient::WanImage(m) => m.poll_task(task_id).await,
            ImageLLMClient::StabilityImage(m) => m.poll_task(task_id).await,
            ImageLLMClient::Flux(m) => m.poll_task(task_id).await,
            ImageLLMClient::Imagen(m) => m.poll_task(task_id).await,
            ImageLLMClient::DallE(m) => m.poll_task(task_id).await,
        }
    }
    /// Gets the provider enum for this client
    ///
    /// # Returns
    /// The `ImageModelProvider` variant corresponding to this client
    pub fn get_provider_enum(&self) -> ImageModelProvider {
        match self {
            ImageLLMClient::Seedream(_) => ImageModelProvider::Seedream,
            ImageLLMClient::WanImage(_) => ImageModelProvider::WanImage,
            ImageLLMClient::StabilityImage(_) => ImageModelProvider::StabilityImage,
            ImageLLMClient::Flux(_) => ImageModelProvider::Flux,
            ImageLLMClient::Imagen(_) => ImageModelProvider::Imagen,
            ImageLLMClient::DallE(_) => ImageModelProvider::DallE,
        }
    }
    /// Gets the vendor of this client's model
    ///
    /// # Returns
    /// The `ImageVendor` variant corresponding to this client's provider
    pub fn get_vendor(&self) -> ImageVendor {
        self.get_provider_enum().vendor()
    }
}
/// Configuration for audio LLM client initialization
///
/// # Example
/// ```
/// use langhub::AudioLLMConfig;
///
/// let config = AudioLLMConfig::new()
///     .qwen_tts("your-dashscope-api-key".to_string())
///     .elevenlabs("your-elevenlabs-api-key".to_string());
/// ```
#[derive(Debug, Clone, Default)]
pub struct AudioLLMConfig {
    /// Alibaba Qwen-Audio TTS API key (DashScope)
    pub qwen_tts_api_key: Option<String>,
    /// Alibaba Qwen-Audio TTS custom base URL
    pub qwen_tts_base_url: Option<String>,
    /// ByteDance Seed Audio API key (Volcengine Ark)
    pub seed_audio_api_key: Option<String>,
    /// ByteDance Seed Audio custom base URL
    pub seed_audio_base_url: Option<String>,
    /// StepFun StepAudio API key
    pub step_audio_api_key: Option<String>,
    /// StepFun StepAudio custom base URL
    pub step_audio_base_url: Option<String>,
    /// Google Gemini TTS API key
    pub gemini_tts_api_key: Option<String>,
    /// Google Gemini TTS custom base URL
    pub gemini_tts_base_url: Option<String>,
    /// ElevenLabs API key
    pub elevenlabs_api_key: Option<String>,
    /// ElevenLabs custom base URL
    pub elevenlabs_base_url: Option<String>,
    /// Google Lyria API key
    pub lyria_api_key: Option<String>,
    /// Google Lyria custom base URL
    pub lyria_base_url: Option<String>,
    /// Suno API key
    pub suno_api_key: Option<String>,
    /// Suno custom base URL
    pub suno_base_url: Option<String>,
    /// Stability AI Stable Audio API key
    pub stable_audio_api_key: Option<String>,
    /// Stability AI Stable Audio custom base URL
    pub stable_audio_base_url: Option<String>,
}
impl AudioLLMConfig {
    /// Creates a new empty audio configuration
    ///
    /// # Example
    /// ```
    /// let config = AudioLLMConfig::new();
    /// ```
    pub fn new() -> Self {
        Self::default()
    }
    /// Sets the Qwen-Audio TTS API key
    ///
    /// # Arguments
    /// * `api_key` - Qwen-Audio TTS API key
    pub fn qwen_tts(mut self, api_key: String) -> Self {
        self.qwen_tts_api_key = Some(api_key);
        self
    }
    /// Sets the Seed Audio API key
    ///
    /// # Arguments
    /// * `api_key` - Seed Audio API key
    pub fn seed_audio(mut self, api_key: String) -> Self {
        self.seed_audio_api_key = Some(api_key);
        self
    }
    /// Sets the StepAudio API key
    ///
    /// # Arguments
    /// * `api_key` - StepAudio API key
    pub fn step_audio(mut self, api_key: String) -> Self {
        self.step_audio_api_key = Some(api_key);
        self
    }
    /// Sets the Gemini TTS API key
    ///
    /// # Arguments
    /// * `api_key` - Gemini TTS API key
    pub fn gemini_tts(mut self, api_key: String) -> Self {
        self.gemini_tts_api_key = Some(api_key);
        self
    }
    /// Sets the ElevenLabs API key
    ///
    /// # Arguments
    /// * `api_key` - ElevenLabs API key
    pub fn elevenlabs(mut self, api_key: String) -> Self {
        self.elevenlabs_api_key = Some(api_key);
        self
    }
    /// Sets the Lyria API key
    ///
    /// # Arguments
    /// * `api_key` - Lyria API key
    pub fn lyria(mut self, api_key: String) -> Self {
        self.lyria_api_key = Some(api_key);
        self
    }
    /// Sets the Suno API key
    ///
    /// # Arguments
    /// * `api_key` - Suno API key
    pub fn suno(mut self, api_key: String) -> Self {
        self.suno_api_key = Some(api_key);
        self
    }
    /// Sets the Stable Audio API key
    ///
    /// # Arguments
    /// * `api_key` - Stable Audio API key
    pub fn stable_audio(mut self, api_key: String) -> Self {
        self.stable_audio_api_key = Some(api_key);
        self
    }
}
/// Unified audio generation client for multiple providers
///
/// This is the audio counterpart of `LLMClient`. It represents a client for
/// any supported audio provider (TTS, soundscape, or music).
///
/// # Example
/// ```
/// use langhub::{AudioLLMClient, AudioLLMConfig, AudioModelProvider};
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let config = AudioLLMConfig::new().qwen_tts("your-api-key".to_string());
/// let client = AudioLLMClient::new_with_config(AudioModelProvider::QwenTts, &config)?;
/// let result = client.generate("Hello, world!").await?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub enum AudioLLMClient {
    QwenTts(QwenTts),
    SeedAudio(SeedAudio),
    StepAudio(StepAudio),
    GeminiTts(GeminiTts),
    ElevenLabs(ElevenLabs),
    Lyria(Lyria),
    Suno(Suno),
    StableAudio(StableAudio),
}
impl AudioLLMClient {
    /// Creates a new audio client with the given provider using optional API keys
    ///
    /// # Arguments
    /// * `provider` - The audio model provider to use
    /// * `api_key` - Optional API key for the provider
    /// * `extra_keys` - Optional additional keys for providers that need them
    ///
    /// # Returns
    /// A `Result` containing the audio client or an error if required keys are missing
    ///
    /// # Example
    /// ```
    /// use langhub::{AudioLLMClient, AudioModelProvider};
    ///
    /// let client = AudioLLMClient::new_with_key(
    ///     AudioModelProvider::QwenTts,
    ///     Some("your-api-key".to_string()),
    ///     None,
    /// ).unwrap();
    /// ```
    pub fn new_with_key(
        provider: AudioModelProvider,
        api_key: Option<String>,
        extra_keys: Option<std::collections::HashMap<String, String>>,
    ) -> Result<Self> {
        let extra = extra_keys.unwrap_or_default();
        match provider {
            AudioModelProvider::QwenTts => {
                let key = api_key.ok_or_else(|| {
                    LangHubError::LLMError("Qwen TTS API key not provided".to_string())
                })?;
                let mut client = QwenTts::new(key);
                if let Some(base) = extra.get("base_url") {
                    client = client.with_base_url(base);
                }
                Ok(AudioLLMClient::QwenTts(client))
            }
            AudioModelProvider::SeedAudio => {
                let key = api_key.ok_or_else(|| {
                    LangHubError::LLMError("Seed Audio API key not provided".to_string())
                })?;
                let mut client = SeedAudio::new(key);
                if let Some(base) = extra.get("base_url") {
                    client = client.with_base_url(base);
                }
                Ok(AudioLLMClient::SeedAudio(client))
            }
            AudioModelProvider::StepAudio => {
                let key = api_key.ok_or_else(|| {
                    LangHubError::LLMError("StepAudio API key not provided".to_string())
                })?;
                let mut client = StepAudio::new(key);
                if let Some(base) = extra.get("base_url") {
                    client = client.with_base_url(base);
                }
                Ok(AudioLLMClient::StepAudio(client))
            }
            AudioModelProvider::GeminiTts => {
                let key = api_key.ok_or_else(|| {
                    LangHubError::LLMError("Gemini TTS API key not provided".to_string())
                })?;
                let mut client = GeminiTts::new(key);
                if let Some(base) = extra.get("base_url") {
                    client = client.with_base_url(base);
                }
                Ok(AudioLLMClient::GeminiTts(client))
            }
            AudioModelProvider::ElevenLabs => {
                let key = api_key.ok_or_else(|| {
                    LangHubError::LLMError("ElevenLabs API key not provided".to_string())
                })?;
                let mut client = ElevenLabs::new(key);
                if let Some(base) = extra.get("base_url") {
                    client = client.with_base_url(base);
                }
                Ok(AudioLLMClient::ElevenLabs(client))
            }
            AudioModelProvider::Lyria => {
                let key = api_key.ok_or_else(|| {
                    LangHubError::LLMError("Lyria API key not provided".to_string())
                })?;
                let mut client = Lyria::new(key);
                if let Some(base) = extra.get("base_url") {
                    client = client.with_base_url(base);
                }
                Ok(AudioLLMClient::Lyria(client))
            }
            AudioModelProvider::Suno => {
                let key = api_key.ok_or_else(|| {
                    LangHubError::LLMError("Suno API key not provided".to_string())
                })?;
                let mut client = Suno::new(key);
                if let Some(base) = extra.get("base_url") {
                    client = client.with_base_url(base);
                }
                Ok(AudioLLMClient::Suno(client))
            }
            AudioModelProvider::StableAudio => {
                let key = api_key.ok_or_else(|| {
                    LangHubError::LLMError("Stable Audio API key not provided".to_string())
                })?;
                let mut client = StableAudio::new(key);
                if let Some(base) = extra.get("base_url") {
                    client = client.with_base_url(base);
                }
                Ok(AudioLLMClient::StableAudio(client))
            }
        }
    }
    /// Creates a new audio client with the given provider and configuration
    ///
    /// # Arguments
    /// * `provider` - The audio model provider to use
    /// * `config` - Configuration containing API keys and credentials
    ///
    /// # Returns
    /// A `Result` containing the audio client or an error if required keys are missing
    ///
    /// # Errors
    /// Returns `LangHubError::LLMError` if the required API key for the provider is not provided
    ///
    /// # Example
    /// ```
    /// use langhub::{AudioLLMClient, AudioLLMConfig, AudioModelProvider};
    ///
    /// let config = AudioLLMConfig::new()
    ///     .qwen_tts("your-dashscope-api-key".to_string())
    ///     .elevenlabs("your-elevenlabs-api-key".to_string());
    ///
    /// let qwen_client = AudioLLMClient::new_with_config(AudioModelProvider::QwenTts, &config).unwrap();
    /// let eleven_client = AudioLLMClient::new_with_config(AudioModelProvider::ElevenLabs, &config).unwrap();
    /// ```
    pub fn new_with_config(provider: AudioModelProvider, config: &AudioLLMConfig) -> Result<Self> {
        match provider {
            AudioModelProvider::QwenTts => {
                let key = config.qwen_tts_api_key.as_ref().ok_or_else(|| {
                    LangHubError::LLMError("Qwen TTS API key not provided".to_string())
                })?;
                let mut client = QwenTts::new(key.clone());
                if let Some(base) = &config.qwen_tts_base_url {
                    client = client.with_base_url(base);
                }
                Ok(AudioLLMClient::QwenTts(client))
            }
            AudioModelProvider::SeedAudio => {
                let key = config.seed_audio_api_key.as_ref().ok_or_else(|| {
                    LangHubError::LLMError("Seed Audio API key not provided".to_string())
                })?;
                let mut client = SeedAudio::new(key.clone());
                if let Some(base) = &config.seed_audio_base_url {
                    client = client.with_base_url(base);
                }
                Ok(AudioLLMClient::SeedAudio(client))
            }
            AudioModelProvider::StepAudio => {
                let key = config.step_audio_api_key.as_ref().ok_or_else(|| {
                    LangHubError::LLMError("StepAudio API key not provided".to_string())
                })?;
                let mut client = StepAudio::new(key.clone());
                if let Some(base) = &config.step_audio_base_url {
                    client = client.with_base_url(base);
                }
                Ok(AudioLLMClient::StepAudio(client))
            }
            AudioModelProvider::GeminiTts => {
                let key = config.gemini_tts_api_key.as_ref().ok_or_else(|| {
                    LangHubError::LLMError("Gemini TTS API key not provided".to_string())
                })?;
                let mut client = GeminiTts::new(key.clone());
                if let Some(base) = &config.gemini_tts_base_url {
                    client = client.with_base_url(base);
                }
                Ok(AudioLLMClient::GeminiTts(client))
            }
            AudioModelProvider::ElevenLabs => {
                let key = config.elevenlabs_api_key.as_ref().ok_or_else(|| {
                    LangHubError::LLMError("ElevenLabs API key not provided".to_string())
                })?;
                let mut client = ElevenLabs::new(key.clone());
                if let Some(base) = &config.elevenlabs_base_url {
                    client = client.with_base_url(base);
                }
                Ok(AudioLLMClient::ElevenLabs(client))
            }
            AudioModelProvider::Lyria => {
                let key = config.lyria_api_key.as_ref().ok_or_else(|| {
                    LangHubError::LLMError("Lyria API key not provided".to_string())
                })?;
                let mut client = Lyria::new(key.clone());
                if let Some(base) = &config.lyria_base_url {
                    client = client.with_base_url(base);
                }
                Ok(AudioLLMClient::Lyria(client))
            }
            AudioModelProvider::Suno => {
                let key = config.suno_api_key.as_ref().ok_or_else(|| {
                    LangHubError::LLMError("Suno API key not provided".to_string())
                })?;
                let mut client = Suno::new(key.clone());
                if let Some(base) = &config.suno_base_url {
                    client = client.with_base_url(base);
                }
                Ok(AudioLLMClient::Suno(client))
            }
            AudioModelProvider::StableAudio => {
                let key = config.stable_audio_api_key.as_ref().ok_or_else(|| {
                    LangHubError::LLMError("Stable Audio API key not provided".to_string())
                })?;
                let mut client = StableAudio::new(key.clone());
                if let Some(base) = &config.stable_audio_base_url {
                    client = client.with_base_url(base);
                }
                Ok(AudioLLMClient::StableAudio(client))
            }
        }
    }
    /// Generates audio from a text prompt
    ///
    /// # Arguments
    /// * `prompt` - The input text prompt string
    ///
    /// # Returns
    /// A `Result` containing the generated audio result or an error
    ///
    /// # Example
    /// ```
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// # let config = langhub::AudioLLMConfig::new().qwen_tts("your-api-key".to_string());
    /// # let client = langhub::AudioLLMClient::new_with_config(langhub::audio::AudioModelProvider::QwenTts, &config)?;
    /// let result = client.generate("Hello, world!").await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn generate(&self, prompt: &str) -> Result<AudioLLMResult> {
        match self {
            AudioLLMClient::QwenTts(m) => m.generate(prompt).await,
            AudioLLMClient::SeedAudio(m) => m.generate(prompt).await,
            AudioLLMClient::StepAudio(m) => m.generate(prompt).await,
            AudioLLMClient::GeminiTts(m) => m.generate(prompt).await,
            AudioLLMClient::ElevenLabs(m) => m.generate(prompt).await,
            AudioLLMClient::Lyria(m) => m.generate(prompt).await,
            AudioLLMClient::Suno(m) => m.generate(prompt).await,
            AudioLLMClient::StableAudio(m) => m.generate(prompt).await,
        }
    }
    /// Generates audio with options
    ///
    /// # Arguments
    /// * `prompt` - The input text prompt string
    /// * `options` - Generation options such as voice, emotion, format, duration
    ///
    /// # Returns
    /// A `Result` containing the generated audio result or an error
    pub async fn generate_with_options(
        &self,
        prompt: &str,
        options: AudioLLMOptions,
    ) -> Result<AudioLLMResult> {
        match self {
            AudioLLMClient::QwenTts(m) => m.generate_with_options(prompt, options).await,
            AudioLLMClient::SeedAudio(m) => m.generate_with_options(prompt, options).await,
            AudioLLMClient::StepAudio(m) => m.generate_with_options(prompt, options).await,
            AudioLLMClient::GeminiTts(m) => m.generate_with_options(prompt, options).await,
            AudioLLMClient::ElevenLabs(m) => m.generate_with_options(prompt, options).await,
            AudioLLMClient::Lyria(m) => m.generate_with_options(prompt, options).await,
            AudioLLMClient::Suno(m) => m.generate_with_options(prompt, options).await,
            AudioLLMClient::StableAudio(m) => m.generate_with_options(prompt, options).await,
        }
    }
    /// Submits an asynchronous audio generation task
    ///
    /// # Arguments
    /// * `prompt` - The input text prompt string
    /// * `options` - Generation options
    ///
    /// # Returns
    /// A `Result` containing an `AudioTask` handle for polling
    pub async fn submit_task(&self, prompt: &str, options: AudioLLMOptions) -> Result<AudioTask> {
        match self {
            AudioLLMClient::QwenTts(m) => m.submit_task(prompt, options).await,
            AudioLLMClient::SeedAudio(m) => m.submit_task(prompt, options).await,
            AudioLLMClient::StepAudio(m) => m.submit_task(prompt, options).await,
            AudioLLMClient::GeminiTts(m) => m.submit_task(prompt, options).await,
            AudioLLMClient::ElevenLabs(m) => m.submit_task(prompt, options).await,
            AudioLLMClient::Lyria(m) => m.submit_task(prompt, options).await,
            AudioLLMClient::Suno(m) => m.submit_task(prompt, options).await,
            AudioLLMClient::StableAudio(m) => m.submit_task(prompt, options).await,
        }
    }
    /// Polls an asynchronous audio generation task
    ///
    /// # Arguments
    /// * `task_id` - The task ID returned by `submit_task`
    ///
    /// # Returns
    /// A `Result` containing the current `AudioTask` state
    pub async fn poll_task(&self, task_id: &str) -> Result<AudioTask> {
        match self {
            AudioLLMClient::QwenTts(m) => m.poll_task(task_id).await,
            AudioLLMClient::SeedAudio(m) => m.poll_task(task_id).await,
            AudioLLMClient::StepAudio(m) => m.poll_task(task_id).await,
            AudioLLMClient::GeminiTts(m) => m.poll_task(task_id).await,
            AudioLLMClient::ElevenLabs(m) => m.poll_task(task_id).await,
            AudioLLMClient::Lyria(m) => m.poll_task(task_id).await,
            AudioLLMClient::Suno(m) => m.poll_task(task_id).await,
            AudioLLMClient::StableAudio(m) => m.poll_task(task_id).await,
        }
    }
    /// Gets the provider enum for this client
    ///
    /// # Returns
    /// The `AudioModelProvider` variant corresponding to this client
    pub fn get_provider_enum(&self) -> AudioModelProvider {
        match self {
            AudioLLMClient::QwenTts(_) => AudioModelProvider::QwenTts,
            AudioLLMClient::SeedAudio(_) => AudioModelProvider::SeedAudio,
            AudioLLMClient::StepAudio(_) => AudioModelProvider::StepAudio,
            AudioLLMClient::GeminiTts(_) => AudioModelProvider::GeminiTts,
            AudioLLMClient::ElevenLabs(_) => AudioModelProvider::ElevenLabs,
            AudioLLMClient::Lyria(_) => AudioModelProvider::Lyria,
            AudioLLMClient::Suno(_) => AudioModelProvider::Suno,
            AudioLLMClient::StableAudio(_) => AudioModelProvider::StableAudio,
        }
    }
    /// Gets the vendor of this client's model
    ///
    /// # Returns
    /// The `AudioVendor` variant corresponding to this client's provider
    pub fn get_vendor(&self) -> crate::types::AudioVendor {
        self.get_provider_enum().vendor()
    }
}
