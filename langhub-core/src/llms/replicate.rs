/// Replicate
use crate::types::*;

use super::{ChatMessage, LLM, LLMOptions, LLMResult};
use serde_json::json;
use std::future::Future;
use std::pin::Pin;

#[derive(Debug, Clone)]
pub enum ReplicateModel {
    Llama3_8b,
    Llama3_70b,
    Mixtral_8x7b,
    Mistral_7b,
    Custom(String),
}

impl ReplicateModel {
    fn as_str(&self) -> String {
        match self {
            ReplicateModel::Llama3_8b => "meta/meta-llama-3-8b-instruct".to_string(),
            ReplicateModel::Llama3_70b => "meta/meta-llama-3-70b-instruct".to_string(),
            ReplicateModel::Mixtral_8x7b => "mistralai/mixtral-8x7b-instruct-v0.1".to_string(),
            ReplicateModel::Mistral_7b => "mistralai/mistral-7b-instruct-v0.2".to_string(),
            ReplicateModel::Custom(name) => name.clone(),
        }
    }
}

impl From<ReplicateModel> for String {
    fn from(model: ReplicateModel) -> Self {
        model.as_str()
    }
}

pub struct Replicate {
    api_key: String,
    model: ReplicateModel,
    base_url: String,
    client: reqwest::Client,
    default_options: LLMOptions,
}

impl Replicate {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            model: ReplicateModel::Mixtral_8x7b,
            base_url: "https://api.replicate.com/v1".to_string(),
            client: reqwest::Client::new(),
            default_options: LLMOptions::default(),
        }
    }

    pub fn with_model(mut self, model: ReplicateModel) -> Self {
        self.model = model;
        self
    }

    pub fn llama3_8b(self) -> Self {
        self.with_model(ReplicateModel::Llama3_8b)
    }

    pub fn llama3_70b(self) -> Self {
        self.with_model(ReplicateModel::Llama3_70b)
    }

    pub fn mixtral(self) -> Self {
        self.with_model(ReplicateModel::Mixtral_8x7b)
    }

    pub fn with_custom_model(mut self, model_name: &str) -> Self {
        self.model = ReplicateModel::Custom(model_name.to_string());
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

        // Build prompt from messages
        let mut prompt = String::new();
        for msg in messages {
            if msg.role == "system" {
                prompt.push_str(&format!("<|system|>\n{}\n", msg.content));
            } else if msg.role == "user" {
                prompt.push_str(&format!("<|user|>\n{}\n", msg.content));
            } else if msg.role == "assistant" {
                prompt.push_str(&format!("<|assistant|>\n{}\n", msg.content));
            }
        }
        prompt.push_str("<|assistant|>\n");

        let mut input = json!({});
        
        if let Some(temp) = options.temperature.or(self.default_options.temperature) {
            input["temperature"] = json!(temp);
        }
        if let Some(max_tokens) = options.max_tokens.or(self.default_options.max_tokens) {
            input["max_new_tokens"] = json!(max_tokens);
        }
        if let Some(top_p) = options.top_p.or(self.default_options.top_p) {
            input["top_p"] = json!(top_p);
        }
        if let Some(top_k) = options.top_k.or(self.default_options.top_k) {
            input["top_k"] = json!(top_k);
        }

        input["prompt"] = json!(prompt);

        let response = self
            .client
            .post(format!("{}/models/{}/predictions", self.base_url, model_name))
            .header("Authorization", format!("Token {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&json!({ "input": input }))
            .send()
            .await
            .map_err(|e| LangHubError::LLMError(format!("Replicate request error: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LangHubError::LLMError(format!(
                "Replicate API error ({}): {}",
                status, error_text
            )));
        }

        let prediction: serde_json::Value = response
            .json()
            .await
            .map_err(|e| LangHubError::LLMError(format!("JSON parse error: {}", e)))?;

        let prediction_url = prediction["urls"]["get"]
            .as_str()
            .ok_or_else(|| LangHubError::ParseError("Missing prediction URL".to_string()))?;

        // Wait for prediction to complete
        let mut output = None;
        for _ in 0..60 {
            let status_response = self
                .client
                .get(prediction_url)
                .header("Authorization", format!("Token {}", self.api_key))
                .send()
                .await
                .map_err(|e| LangHubError::LLMError(format!("Status check error: {}", e)))?;

            let status_json: serde_json::Value = status_response
                .json()
                .await
                .map_err(|e| LangHubError::LLMError(format!("JSON parse error: {}", e)))?;

            if status_json["status"] == "succeeded" {
                output = status_json["output"].as_str().map(String::from);
                break;
            } else if status_json["status"] == "failed" {
                return Err(LangHubError::LLMError("Prediction failed".to_string()));
            }
            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        }

        let text = output.ok_or_else(|| {
            LangHubError::LLMError("Prediction timeout".to_string())
        })?;

        Ok(text)
    }
}

impl LLM for Replicate {
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
            ReplicateModel::Llama3_8b => "llama3-8b",
            ReplicateModel::Llama3_70b => "llama3-70b",
            ReplicateModel::Mixtral_8x7b => "mixtral-8x7b",
            ReplicateModel::Mistral_7b => "mistral-7b",
            ReplicateModel::Custom(name) => name,
        }
    }

    fn get_provider_name(&self) -> &str {
        match &self.model {
            ReplicateModel::Llama3_8b => "Replicate-Llama3-8B",
            ReplicateModel::Llama3_70b => "Replicate-Llama3-70B",
            ReplicateModel::Mixtral_8x7b => "Replicate-Mixtral-8x7B",
            ReplicateModel::Mistral_7b => "Replicate-Mistral-7B",
            ReplicateModel::Custom(name) => &format!("Replicate-{}", name),
        }
    }

    fn get_provider_enum(&self) -> ModelProvider {
        ModelProvider::Replicate
    }

    fn supports_function_calling(&self) -> bool {
        false
    }

    fn supports_json_mode(&self) -> bool {
        false
    }

    fn max_context_length(&self) -> Option<usize> {
        Some(4096)
    }
}