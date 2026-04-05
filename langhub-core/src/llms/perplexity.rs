/// Perplexity AI
use crate::types::*;

use super::{ChatMessage, LLM, LLMOptions, LLMResult};
use serde_json::json;
use std::future::Future;
use std::pin::Pin;

#[derive(Debug, Clone)]
pub enum PerplexityModel {
    SonarSmall,
    SonarMedium,
    SonarLarge,
    Llama3_8b,
    Llama3_70b,
    Mixtral_8x7b,
}

impl PerplexityModel {
    fn as_str(&self) -> &'static str {
        match self {
            PerplexityModel::SonarSmall => "llama-3-sonar-small-32k-online",
            PerplexityModel::SonarMedium => "llama-3-sonar-medium-32k-online",
            PerplexityModel::SonarLarge => "llama-3-sonar-large-32k-online",
            PerplexityModel::Llama3_8b => "llama-3-8b-instruct",
            PerplexityModel::Llama3_70b => "llama-3-70b-instruct",
            PerplexityModel::Mixtral_8x7b => "mixtral-8x7b-instruct",
        }
    }
}

impl From<PerplexityModel> for String {
    fn from(model: PerplexityModel) -> Self {
        model.as_str().to_string()
    }
}

pub struct Perplexity {
    api_key: String,
    model: PerplexityModel,
    base_url: String,
    client: reqwest::Client,
    default_options: LLMOptions,
}

impl Perplexity {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            model: PerplexityModel::SonarMedium,
            base_url: "https://api.perplexity.ai".to_string(),
            client: reqwest::Client::new(),
            default_options: LLMOptions::default(),
        }
    }

    pub fn with_model(mut self, model: PerplexityModel) -> Self {
        self.model = model;
        self
    }

    pub fn sonar_small(self) -> Self {
        self.with_model(PerplexityModel::SonarSmall)
    }

    pub fn sonar_medium(self) -> Self {
        self.with_model(PerplexityModel::SonarMedium)
    }

    pub fn sonar_large(self) -> Self {
        self.with_model(PerplexityModel::SonarLarge)
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.default_options.temperature = Some(temperature);
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.default_options.max_tokens = Some(max_tokens);
        self
    }

    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.default_options.top_p = Some(top_p);
        self
    }

    pub fn with_top_k(mut self, top_k: u32) -> Self {
        self.default_options.top_k = Some(top_k);
        self
    }

    pub fn with_base_url(mut self, base_url: &str) -> Self {
        self.base_url = base_url.to_string();
        self
    }

    async fn chat_completion(
        &self,
        messages: &[ChatMessage],
        options: &LLMOptions,
    ) -> Result<String> {
        let model_name: String = self.model.clone().into();

        let mut request_body = json!({
            "model": model_name,
            "messages": messages.iter().map(|m| json!({
                "role": m.role,
                "content": m.content,
            })).collect::<Vec<_>>(),
        });

        if let Some(temp) = options.temperature.or(self.default_options.temperature) {
            request_body["temperature"] = json!(temp);
        }
        if let Some(max_tokens) = options.max_tokens.or(self.default_options.max_tokens) {
            request_body["max_tokens"] = json!(max_tokens);
        }
        if let Some(top_p) = options.top_p.or(self.default_options.top_p) {
            request_body["top_p"] = json!(top_p);
        }
        if let Some(top_k) = options.top_k.or(self.default_options.top_k) {
            request_body["top_k"] = json!(top_k);
        }
        if let Some(stop) = options
            .stop_sequences
            .as_ref()
            .or(self.default_options.stop_sequences.as_ref())
        {
            request_body["stop"] = json!(stop);
        }

        let response = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| LangHubError::LLMError(format!("Perplexity request error: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LangHubError::LLMError(format!(
                "Perplexity API error ({}): {}",
                status, error_text
            )));
        }

        let json: serde_json::Value = response
            .json()
            .await
            .map_err(|e| LangHubError::LLMError(format!("JSON parse error: {}", e)))?;

        let text = json["choices"][0]["message"]["content"]
            .as_str()
            .ok_or_else(|| {
                LangHubError::ParseError("Missing 'content' field in response".to_string())
            })?
            .to_string();

        Ok(text)
    }
}

impl LLM for Perplexity {
    fn generate(
        &self,
        prompt: &str,
    ) -> Pin<Box<dyn Future<Output = Result<LLMResult>> + Send + '_>> {
        let prompt = prompt.to_string();
        let options = self.default_options.clone();
        Box::pin(async move {
            let messages = vec![ChatMessage::user(&prompt)];
            let text = self.chat_completion(&messages, &options).await?;
            Ok(LLMResult {
                text,
                metadata: None,
            })
        })
    }

    fn generate_with_options(
        &self,
        prompt: &str,
        options: LLMOptions,
    ) -> Pin<Box<dyn Future<Output = Result<LLMResult>> + Send + '_>> {
        let prompt = prompt.to_string();
        Box::pin(async move {
            let messages = vec![ChatMessage::user(&prompt)];
            let text = self.chat_completion(&messages, &options).await?;
            Ok(LLMResult {
                text,
                metadata: None,
            })
        })
    }

    fn chat(
        &self,
        messages: Vec<ChatMessage>,
    ) -> Pin<Box<dyn Future<Output = Result<LLMResult>> + Send + '_>> {
        Box::pin(async move {
            let text = self
                .chat_completion(&messages, &LLMOptions::default())
                .await?;
            Ok(LLMResult {
                text,
                metadata: None,
            })
        })
    }

    fn get_model_name(&self) -> &str {
        match self.model {
            PerplexityModel::SonarSmall => "sonar-small",
            PerplexityModel::SonarMedium => "sonar-medium",
            PerplexityModel::SonarLarge => "sonar-large",
            PerplexityModel::Llama3_8b => "llama3-8b",
            PerplexityModel::Llama3_70b => "llama3-70b",
            PerplexityModel::Mixtral_8x7b => "mixtral-8x7b",
        }
    }

    fn get_provider_name(&self) -> &str {
        match self.model {
            PerplexityModel::SonarSmall => "Perplexity-Sonar-Small",
            PerplexityModel::SonarMedium => "Perplexity-Sonar-Medium",
            PerplexityModel::SonarLarge => "Perplexity-Sonar-Large",
            PerplexityModel::Llama3_8b => "Perplexity-Llama3-8B",
            PerplexityModel::Llama3_70b => "Perplexity-Llama3-70B",
            PerplexityModel::Mixtral_8x7b => "Perplexity-Mixtral-8x7B",
        }
    }

    fn get_provider_enum(&self) -> ModelProvider {
        ModelProvider::Perplexity
    }

    fn supports_function_calling(&self) -> bool {
        true
    }

    fn supports_json_mode(&self) -> bool {
        true
    }

    fn max_context_length(&self) -> Option<usize> {
        match self.model {
            PerplexityModel::SonarSmall
            | PerplexityModel::SonarMedium
            | PerplexityModel::SonarLarge => Some(32768),
            _ => Some(8192),
        }
    }
}