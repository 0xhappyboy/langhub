//! LangHub - An LLM application development library.
pub mod llms;
pub mod tools;
pub mod types;

use crate::llms::*;
use crate::types::{ChatMessage, LangHubError, ModelProvider, Result};

/// Configuration for LLM client initialization
///
/// This struct holds all the API keys and credentials needed to initialize
/// LLM clients for different providers. All fields are optional, and you only
/// need to set the ones for the provider you plan to use.
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
