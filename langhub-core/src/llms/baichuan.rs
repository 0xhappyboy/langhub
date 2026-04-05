/// baichuan AI
use crate::types::*;

use super::{ChatMessage, LLM, LLMOptions, LLMResult};
use serde_json::json;
use std::future::Future;
use std::pin::Pin;

#[derive(Debug, Clone)]
pub enum BaichuanModel {
    Baichuan4,      // Baichuan 4
    Baichuan3Turbo, // Baichuan 3 Turbo
    Baichuan3,      // Baichuan 3
    Baichuan2Turbo, // Baichuan 2 Turbo
    Baichuan2,      // Baichuan 2
}

impl BaichuanModel {
    fn as_str(&self) -> &'static str {
        match self {
            BaichuanModel::Baichuan4 => "Baichuan4",
            BaichuanModel::Baichuan3Turbo => "Baichuan3-Turbo",
            BaichuanModel::Baichuan3 => "Baichuan3",
            BaichuanModel::Baichuan2Turbo => "Baichuan2-Turbo",
            BaichuanModel::Baichuan2 => "Baichuan2",
        }
    }
}

impl From<BaichuanModel> for String {
    fn from(model: BaichuanModel) -> Self {
        model.as_str().to_string()
    }
}

pub struct Baichuan {
    api_key: String,
    model: BaichuanModel,
    base_url: String,
    client: reqwest::Client,
    default_options: LLMOptions,
}

impl Baichuan {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            model: BaichuanModel::Baichuan4,
            base_url: "https://api.baichuan-ai.com/v1".to_string(),
            client: reqwest::Client::new(),
            default_options: LLMOptions::default(),
        }
    }

    pub fn with_model(mut self, model: BaichuanModel) -> Self {
        self.model = model;
        self
    }

    pub fn baichuan4(self) -> Self {
        self.with_model(BaichuanModel::Baichuan4)
    }

    pub fn baichuan3_turbo(self) -> Self {
        self.with_model(BaichuanModel::Baichuan3Turbo)
    }

    pub fn baichuan2_turbo(self) -> Self {
        self.with_model(BaichuanModel::Baichuan2Turbo)
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
            .map_err(|e| LangHubError::LLMError(format!("Baichuan request error: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LangHubError::LLMError(format!(
                "Baichuan API error ({}): {}",
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

impl LLM for Baichuan {
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
            BaichuanModel::Baichuan4 => "baichuan-4",
            BaichuanModel::Baichuan3Turbo => "baichuan-3-turbo",
            BaichuanModel::Baichuan3 => "baichuan-3",
            BaichuanModel::Baichuan2Turbo => "baichuan-2-turbo",
            BaichuanModel::Baichuan2 => "baichuan-2",
        }
    }

    fn get_provider_name(&self) -> &str {
        match self.model {
            BaichuanModel::Baichuan4 => "Baichuan-4",
            BaichuanModel::Baichuan3Turbo => "Baichuan-3-Turbo",
            BaichuanModel::Baichuan3 => "Baichuan-3",
            BaichuanModel::Baichuan2Turbo => "Baichuan-2-Turbo",
            BaichuanModel::Baichuan2 => "Baichuan-2",
        }
    }

    fn get_provider_enum(&self) -> ModelProvider {
        ModelProvider::Baichuan
    }

    fn supports_function_calling(&self) -> bool {
        true
    }

    fn supports_json_mode(&self) -> bool {
        true
    }

    fn max_context_length(&self) -> Option<usize> {
        match self.model {
            BaichuanModel::Baichuan4 => Some(128000),
            BaichuanModel::Baichuan3Turbo => Some(32768),
            BaichuanModel::Baichuan2Turbo => Some(32768),
            _ => Some(8192),
        }
    }
}
