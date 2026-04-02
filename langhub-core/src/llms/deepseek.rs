/// DeepSeek (DeepSeek-V3, R1)
use crate::types::*;

use super::{ChatMessage, LLM, LLMOptions, LLMResult};
use serde_json::json;
use std::future::Future;
use std::pin::Pin;

#[derive(Debug, Clone)]
pub enum DeepSeekModel {
    Chat,     // DeepSeek-V3
    Coder,    // DeepSeek-Coder
    Reasoner, // DeepSeek-R1
}

impl DeepSeekModel {
    fn as_str(&self) -> &'static str {
        match self {
            DeepSeekModel::Chat => "deepseek-chat",
            DeepSeekModel::Coder => "deepseek-coder",
            DeepSeekModel::Reasoner => "deepseek-reasoner",
        }
    }
}

impl From<DeepSeekModel> for String {
    fn from(model: DeepSeekModel) -> Self {
        model.as_str().to_string()
    }
}

pub struct DeepSeek {
    api_key: String,
    model: DeepSeekModel,
    base_url: String,
    client: reqwest::Client,
    default_options: LLMOptions,
}

impl DeepSeek {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            model: DeepSeekModel::Chat,
            base_url: "https://api.deepseek.com/v1".to_string(),
            client: reqwest::Client::new(),
            default_options: LLMOptions::default(),
        }
    }

    pub fn with_model(mut self, model: DeepSeekModel) -> Self {
        self.model = model;
        self
    }

    pub fn chat_model(self) -> Self {
        self.with_model(DeepSeekModel::Chat)
    }

    pub fn coder_model(self) -> Self {
        self.with_model(DeepSeekModel::Coder)
    }

    pub fn reasoner_model(self) -> Self {
        self.with_model(DeepSeekModel::Reasoner)
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
        if let Some(freq_penalty) = options
            .frequency_penalty
            .or(self.default_options.frequency_penalty)
        {
            request_body["frequency_penalty"] = json!(freq_penalty);
        }
        if let Some(pres_penalty) = options
            .presence_penalty
            .or(self.default_options.presence_penalty)
        {
            request_body["presence_penalty"] = json!(pres_penalty);
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
            .map_err(|e| LangChainError::LLMError(format!("DeepSeek request error: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LangChainError::LLMError(format!(
                "DeepSeek API error ({}): {}",
                status, error_text
            )));
        }

        let json: serde_json::Value = response
            .json()
            .await
            .map_err(|e| LangChainError::LLMError(format!("JSON parse error: {}", e)))?;

        let text = json["choices"][0]["message"]["content"]
            .as_str()
            .ok_or_else(|| {
                LangChainError::ParseError("Missing 'content' field in response".to_string())
            })?
            .to_string();

        Ok(text)
    }
}

impl LLM for DeepSeek {
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
            DeepSeekModel::Chat => "deepseek-chat",
            DeepSeekModel::Coder => "deepseek-coder",
            DeepSeekModel::Reasoner => "deepseek-reasoner",
        }
    }

    fn get_provider_name(&self) -> &str {
        match self.model {
            DeepSeekModel::Chat => "DeepSeek-V3",
            DeepSeekModel::Coder => "DeepSeek-Coder",
            DeepSeekModel::Reasoner => "DeepSeek-R1",
        }
    }

    fn get_provider_enum(&self) -> ModelProvider {
        ModelProvider::DeepSeek
    }

    fn supports_function_calling(&self) -> bool {
        true
    }

    fn supports_json_mode(&self) -> bool {
        true
    }

    fn max_context_length(&self) -> Option<usize> {
        Some(1_000_000) // DeepSeek supports 1M context
    }
}
