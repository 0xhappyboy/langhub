/// Together AI
use crate::types::*;

use super::{ChatMessage, LLM, LLMOptions, LLMResult};
use serde_json::json;
use std::future::Future;
use std::pin::Pin;

#[derive(Debug, Clone)]
pub enum TogetherModel {
    Llama3_8b,
    Llama3_70b,
    Llama2_70b,
    Mixtral_8x7b,
    Mistral_7b,
    Gemma_7b,
    Custom(String),
}

impl TogetherModel {
    fn as_str(&self) -> String {
        match self {
            TogetherModel::Llama3_8b => "meta-llama/Llama-3-8b-chat-hf".to_string(),
            TogetherModel::Llama3_70b => "meta-llama/Llama-3-70b-chat-hf".to_string(),
            TogetherModel::Llama2_70b => "meta-llama/Llama-2-70b-chat-hf".to_string(),
            TogetherModel::Mixtral_8x7b => "mistralai/Mixtral-8x7B-Instruct-v0.1".to_string(),
            TogetherModel::Mistral_7b => "mistralai/Mistral-7B-Instruct-v0.2".to_string(),
            TogetherModel::Gemma_7b => "google/gemma-7b-it".to_string(),
            TogetherModel::Custom(name) => name.clone(),
        }
    }
}

impl From<TogetherModel> for String {
    fn from(model: TogetherModel) -> Self {
        model.as_str()
    }
}

pub struct Together {
    api_key: String,
    model: TogetherModel,
    base_url: String,
    client: reqwest::Client,
    default_options: LLMOptions,
}

impl Together {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            model: TogetherModel::Mixtral_8x7b,
            base_url: "https://api.together.xyz/v1".to_string(),
            client: reqwest::Client::new(),
            default_options: LLMOptions::default(),
        }
    }

    pub fn with_model(mut self, model: TogetherModel) -> Self {
        self.model = model;
        self
    }

    pub fn llama3_8b(self) -> Self {
        self.with_model(TogetherModel::Llama3_8b)
    }

    pub fn llama3_70b(self) -> Self {
        self.with_model(TogetherModel::Llama3_70b)
    }

    pub fn mixtral(self) -> Self {
        self.with_model(TogetherModel::Mixtral_8x7b)
    }

    pub fn with_custom_model(mut self, model_name: &str) -> Self {
        self.model = TogetherModel::Custom(model_name.to_string());
        self
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

    pub fn with_repetition_penalty(mut self, penalty: f32) -> Self {
        self.default_options.repetition_penalty = Some(penalty);
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
        if let Some(rep_penalty) = options
            .repetition_penalty
            .or(self.default_options.repetition_penalty)
        {
            request_body["repetition_penalty"] = json!(rep_penalty);
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
            .map_err(|e| LangHubError::LLMError(format!("Together request error: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LangHubError::LLMError(format!(
                "Together API error ({}): {}",
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

impl LLM for Together {
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
        match &self.model {
            TogetherModel::Llama3_8b => "llama3-8b",
            TogetherModel::Llama3_70b => "llama3-70b",
            TogetherModel::Llama2_70b => "llama2-70b",
            TogetherModel::Mixtral_8x7b => "mixtral-8x7b",
            TogetherModel::Mistral_7b => "mistral-7b",
            TogetherModel::Gemma_7b => "gemma-7b",
            TogetherModel::Custom(name) => name,
        }
    }

    fn get_provider_name(&self) -> &str {
        match &self.model {
            TogetherModel::Llama3_8b => "Together-Llama3-8B",
            TogetherModel::Llama3_70b => "Together-Llama3-70B",
            TogetherModel::Llama2_70b => "Together-Llama2-70B",
            TogetherModel::Mixtral_8x7b => "Together-Mixtral-8x7B",
            TogetherModel::Mistral_7b => "Together-Mistral-7B",
            TogetherModel::Gemma_7b => "Together-Gemma-7B",
            TogetherModel::Custom(name) => &format!("Together-{}", name),
        }
    }

    fn get_provider_enum(&self) -> ModelProvider {
        ModelProvider::Together
    }

    fn supports_function_calling(&self) -> bool {
        true
    }

    fn supports_json_mode(&self) -> bool {
        true
    }

    fn max_context_length(&self) -> Option<usize> {
        match self.model {
            TogetherModel::Llama3_8b | TogetherModel::Llama3_70b => Some(8192),
            TogetherModel::Mixtral_8x7b => Some(32768),
            _ => Some(4096),
        }
    }
}