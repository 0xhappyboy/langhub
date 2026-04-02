/// Anthropic (Claude 3)
use crate::types::*;

use super::{ChatMessage, LLM, LLMOptions, LLMResult};
use serde_json::json;
use std::future::Future;
use std::pin::Pin;

#[derive(Debug, Clone)]
pub enum AnthropicModel {
    Claude3Opus,   // Claude 3 Opus
    Claude3Sonnet, // Claude 3 Sonnet
    Claude3Haiku,  // Claude 3 Haiku
    Claude21,      // Claude 2.1
    Claude2,       // Claude 2.0
}

impl AnthropicModel {
    fn as_str(&self) -> &'static str {
        match self {
            AnthropicModel::Claude3Opus => "claude-3-opus-20240229",
            AnthropicModel::Claude3Sonnet => "claude-3-sonnet-20240229",
            AnthropicModel::Claude3Haiku => "claude-3-haiku-20240307",
            AnthropicModel::Claude21 => "claude-2.1",
            AnthropicModel::Claude2 => "claude-2.0",
        }
    }
}

impl From<AnthropicModel> for String {
    fn from(model: AnthropicModel) -> Self {
        model.as_str().to_string()
    }
}

pub struct Anthropic {
    api_key: String,
    model: AnthropicModel,
    base_url: String,
    client: reqwest::Client,
    default_options: LLMOptions,
}

impl Anthropic {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            model: AnthropicModel::Claude3Sonnet,
            base_url: "https://api.anthropic.com/v1".to_string(),
            client: reqwest::Client::new(),
            default_options: LLMOptions::default(),
        }
    }

    pub fn with_model(mut self, model: AnthropicModel) -> Self {
        self.model = model;
        self
    }

    pub fn claude3_opus(self) -> Self {
        self.with_model(AnthropicModel::Claude3Opus)
    }

    pub fn claude3_sonnet(self) -> Self {
        self.with_model(AnthropicModel::Claude3Sonnet)
    }

    pub fn claude3_haiku(self) -> Self {
        self.with_model(AnthropicModel::Claude3Haiku)
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

        let anthropic_messages: Vec<serde_json::Value> = messages
            .iter()
            .filter(|m| m.role != "system")
            .map(|m| {
                json!({
                    "role": m.role,
                    "content": m.content,
                })
            })
            .collect();

        let mut request_body = json!({
            "model": model_name,
            "messages": anthropic_messages,
            "max_tokens": options.max_tokens.or(self.default_options.max_tokens).unwrap_or(4096),
        });

        // Add system prompt if present
        if let Some(system_msg) = messages.iter().find(|m| m.role == "system") {
            request_body["system"] = json!(system_msg.content);
        }

        if let Some(temp) = options.temperature.or(self.default_options.temperature) {
            request_body["temperature"] = json!(temp);
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
            request_body["stop_sequences"] = json!(stop);
        }

        let response = self
            .client
            .post(format!("{}/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| LangHubError::LLMError(format!("Anthropic request error: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LangHubError::LLMError(format!(
                "Anthropic API error ({}): {}",
                status, error_text
            )));
        }

        let json: serde_json::Value = response
            .json()
            .await
            .map_err(|e| LangHubError::LLMError(format!("JSON parse error: {}", e)))?;

        let text = json["content"][0]["text"]
            .as_str()
            .ok_or_else(|| {
                LangHubError::ParseError("Missing 'text' field in response".to_string())
            })?
            .to_string();

        Ok(text)
    }
}

impl LLM for Anthropic {
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
            AnthropicModel::Claude3Opus => "claude-3-opus",
            AnthropicModel::Claude3Sonnet => "claude-3-sonnet",
            AnthropicModel::Claude3Haiku => "claude-3-haiku",
            AnthropicModel::Claude21 => "claude-2.1",
            AnthropicModel::Claude2 => "claude-2.0",
        }
    }

    fn get_provider_name(&self) -> &str {
        match self.model {
            AnthropicModel::Claude3Opus => "Anthropic-Claude3-Opus",
            AnthropicModel::Claude3Sonnet => "Anthropic-Claude3-Sonnet",
            AnthropicModel::Claude3Haiku => "Anthropic-Claude3-Haiku",
            AnthropicModel::Claude21 => "Anthropic-Claude2.1",
            AnthropicModel::Claude2 => "Anthropic-Claude2.0",
        }
    }

    fn get_provider_enum(&self) -> ModelProvider {
        ModelProvider::Anthropic
    }

    fn supports_function_calling(&self) -> bool {
        true
    }

    fn supports_json_mode(&self) -> bool {
        true
    }

    fn max_context_length(&self) -> Option<usize> {
        match self.model {
            AnthropicModel::Claude3Opus => Some(200000),
            AnthropicModel::Claude3Sonnet => Some(200000),
            AnthropicModel::Claude3Haiku => Some(200000),
            AnthropicModel::Claude21 => Some(100000),
            AnthropicModel::Claude2 => Some(100000),
        }
    }
}
