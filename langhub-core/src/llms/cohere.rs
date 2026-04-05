/// Cohere (Command)
use crate::types::*;

use super::{ChatMessage, LLM, LLMOptions, LLMResult};
use serde_json::json;
use std::future::Future;
use std::pin::Pin;

#[derive(Debug, Clone)]
pub enum CohereModel {
    Command,
    CommandR,
    CommandLight,
    CommandNightly,
}

impl CohereModel {
    fn as_str(&self) -> &'static str {
        match self {
            CohereModel::Command => "command",
            CohereModel::CommandR => "command-r",
            CohereModel::CommandLight => "command-light",
            CohereModel::CommandNightly => "command-nightly",
        }
    }
}

impl From<CohereModel> for String {
    fn from(model: CohereModel) -> Self {
        model.as_str().to_string()
    }
}

pub struct Cohere {
    api_key: String,
    model: CohereModel,
    base_url: String,
    client: reqwest::Client,
    default_options: LLMOptions,
}

impl Cohere {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            model: CohereModel::Command,
            base_url: "https://api.cohere.ai/v1".to_string(),
            client: reqwest::Client::new(),
            default_options: LLMOptions::default(),
        }
    }

    pub fn with_model(mut self, model: CohereModel) -> Self {
        self.model = model;
        self
    }

    pub fn command(self) -> Self {
        self.with_model(CohereModel::Command)
    }

    pub fn command_r(self) -> Self {
        self.with_model(CohereModel::CommandR)
    }

    pub fn command_light(self) -> Self {
        self.with_model(CohereModel::CommandLight)
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

        let chat_history: Vec<serde_json::Value> = messages
            .iter()
            .take(messages.len().saturating_sub(1))
            .map(|m| {
                json!({
                    "role": m.role,
                    "message": m.content,
                })
            })
            .collect();

        let last_message = messages.last().ok_or_else(|| {
            LangHubError::LLMError("No messages provided".to_string())
        })?;

        let mut request_body = json!({
            "model": model_name,
            "message": last_message.content,
            "chat_history": chat_history,
        });

        if let Some(temp) = options.temperature.or(self.default_options.temperature) {
            request_body["temperature"] = json!(temp);
        }
        if let Some(max_tokens) = options.max_tokens.or(self.default_options.max_tokens) {
            request_body["max_tokens"] = json!(max_tokens);
        }
        if let Some(top_p) = options.top_p.or(self.default_options.top_p) {
            request_body["p"] = json!(top_p);
        }
        if let Some(top_k) = options.top_k.or(self.default_options.top_k) {
            request_body["k"] = json!(top_k);
        }
        if let Some(stop) = options
            .stop_sequences
            .as_ref()
            .or(self.default_options.stop_sequences.as_ref())
        {
            request_body["stop_sequences"] = json!(stop);
        }

        let response = self
            .client
            .post(format!("{}/chat", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| LangHubError::LLMError(format!("Cohere request error: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LangHubError::LLMError(format!(
                "Cohere API error ({}): {}",
                status, error_text
            )));
        }

        let json: serde_json::Value = response
            .json()
            .await
            .map_err(|e| LangHubError::LLMError(format!("JSON parse error: {}", e)))?;

        let text = json["text"]
            .as_str()
            .ok_or_else(|| {
                LangHubError::ParseError("Missing 'text' field in response".to_string())
            })?
            .to_string();

        Ok(text)
    }
}

impl LLM for Cohere {
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
            CohereModel::Command => "cohere-command",
            CohereModel::CommandR => "cohere-command-r",
            CohereModel::CommandLight => "cohere-command-light",
            CohereModel::CommandNightly => "cohere-command-nightly",
        }
    }

    fn get_provider_name(&self) -> &str {
        match self.model {
            CohereModel::Command => "Cohere-Command",
            CohereModel::CommandR => "Cohere-Command-R",
            CohereModel::CommandLight => "Cohere-Command-Light",
            CohereModel::CommandNightly => "Cohere-Command-Nightly",
        }
    }

    fn get_provider_enum(&self) -> ModelProvider {
        ModelProvider::Cohere
    }

    fn supports_function_calling(&self) -> bool {
        true
    }

    fn supports_json_mode(&self) -> bool {
        true
    }

    fn max_context_length(&self) -> Option<usize> {
        match self.model {
            CohereModel::Command => Some(4096),
            CohereModel::CommandR => Some(128000),
            CohereModel::CommandLight => Some(4096),
            CohereModel::CommandNightly => Some(4096),
        }
    }
}