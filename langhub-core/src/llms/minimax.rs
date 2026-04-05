/// MiniMax (Abab)
use crate::types::*;

use super::{ChatMessage, LLM, LLMOptions, LLMResult};
use serde_json::json;
use std::future::Future;
use std::pin::Pin;

#[derive(Debug, Clone)]
pub enum MiniMaxModel {
    Abab6_5,  // Abab 6.5
    Abab6_5S, // Abab 6.5s
    Abab5_5,  // Abab 5.5
    Abab5_5S, // Abab 5.5s
}

impl MiniMaxModel {
    fn as_str(&self) -> &'static str {
        match self {
            MiniMaxModel::Abab6_5 => "abab6.5-chat",
            MiniMaxModel::Abab6_5S => "abab6.5s-chat",
            MiniMaxModel::Abab5_5 => "abab5.5-chat",
            MiniMaxModel::Abab5_5S => "abab5.5s-chat",
        }
    }
}

impl From<MiniMaxModel> for String {
    fn from(model: MiniMaxModel) -> Self {
        model.as_str().to_string()
    }
}

pub struct MiniMax {
    api_key: String,
    group_id: String,
    model: MiniMaxModel,
    base_url: String,
    client: reqwest::Client,
    default_options: LLMOptions,
}

impl MiniMax {
    pub fn new(api_key: String, group_id: String) -> Self {
        Self {
            api_key,
            group_id,
            model: MiniMaxModel::Abab6_5,
            base_url: "https://api.minimax.chat/v1".to_string(),
            client: reqwest::Client::new(),
            default_options: LLMOptions::default(),
        }
    }

    pub fn with_model(mut self, model: MiniMaxModel) -> Self {
        self.model = model;
        self
    }

    pub fn abab6_5(self) -> Self {
        self.with_model(MiniMaxModel::Abab6_5)
    }

    pub fn abab6_5s(self) -> Self {
        self.with_model(MiniMaxModel::Abab6_5S)
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

    async fn chat_completion(
        &self,
        messages: &[ChatMessage],
        options: &LLMOptions,
    ) -> Result<String> {
        let model_name: String = self.model.clone().into();

        let mut messages_json: Vec<serde_json::Value> = messages
            .iter()
            .map(|m| {
                let sender_type = if m.role == "assistant" { "BOT" } else { "USER" };
                json!({
                    "sender_type": sender_type,
                    "text": m.content,
                })
            })
            .collect();

        let mut request_body = json!({
            "model": model_name,
            "messages": messages_json,
            "group_id": self.group_id,
        });

        if let Some(temp) = options.temperature.or(self.default_options.temperature) {
            request_body["temperature"] = json!(temp);
        }
        if let Some(max_tokens) = options.max_tokens.or(self.default_options.max_tokens) {
            request_body["tokens_to_generate"] = json!(max_tokens);
        }
        if let Some(top_p) = options.top_p.or(self.default_options.top_p) {
            request_body["top_p"] = json!(top_p);
        }

        let response = self
            .client
            .post(format!("{}/text/chatcompletion", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| LangHubError::LLMError(format!("MiniMax request error: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LangHubError::LLMError(format!(
                "MiniMax API error ({}): {}",
                status, error_text
            )));
        }

        let json: serde_json::Value = response
            .json()
            .await
            .map_err(|e| LangHubError::LLMError(format!("JSON parse error: {}", e)))?;

        let text = json["reply"]
            .as_str()
            .ok_or_else(|| {
                LangHubError::ParseError("Missing 'reply' field in response".to_string())
            })?
            .to_string();

        Ok(text)
    }
}

impl LLM for MiniMax {
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
            MiniMaxModel::Abab6_5 => "abab6.5",
            MiniMaxModel::Abab6_5S => "abab6.5s",
            MiniMaxModel::Abab5_5 => "abab5.5",
            MiniMaxModel::Abab5_5S => "abab5.5s",
        }
    }

    fn get_provider_name(&self) -> &str {
        match self.model {
            MiniMaxModel::Abab6_5 => "MiniMax-Abab6.5",
            MiniMaxModel::Abab6_5S => "MiniMax-Abab6.5s",
            MiniMaxModel::Abab5_5 => "MiniMax-Abab5.5",
            MiniMaxModel::Abab5_5S => "MiniMax-Abab5.5s",
        }
    }

    fn get_provider_enum(&self) -> ModelProvider {
        ModelProvider::MiniMax
    }

    fn supports_function_calling(&self) -> bool {
        true
    }

    fn supports_json_mode(&self) -> bool {
        true
    }

    fn max_context_length(&self) -> Option<usize> {
        match self.model {
            MiniMaxModel::Abab6_5 => Some(32768),
            MiniMaxModel::Abab6_5S => Some(8192),
            _ => Some(8192),
        }
    }
}
