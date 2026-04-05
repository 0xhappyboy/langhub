/// Azure OpenAI
use crate::types::*;

use super::{ChatMessage, LLM, LLMOptions, LLMResult, ResponseFormat};
use serde_json::json;
use std::future::Future;
use std::pin::Pin;

#[derive(Debug, Clone)]
pub enum AzureModel {
    Gpt4,
    Gpt4Turbo,
    Gpt35Turbo,
    Custom(String),
}

impl AzureModel {
    fn as_str(&self) -> String {
        match self {
            AzureModel::Gpt4 => "gpt-4".to_string(),
            AzureModel::Gpt4Turbo => "gpt-4-turbo".to_string(),
            AzureModel::Gpt35Turbo => "gpt-35-turbo".to_string(),
            AzureModel::Custom(name) => name.clone(),
        }
    }
}

impl From<AzureModel> for String {
    fn from(model: AzureModel) -> Self {
        model.as_str()
    }
}

pub struct AzureOpenAI {
    api_key: String,
    endpoint: String,
    deployment_name: String,
    api_version: String,
    model: AzureModel,
    client: reqwest::Client,
    default_options: LLMOptions,
}

impl AzureOpenAI {
    pub fn new(api_key: String, endpoint: String, deployment_name: String) -> Self {
        Self {
            api_key,
            endpoint,
            deployment_name,
            api_version: "2024-02-15-preview".to_string(),
            model: AzureModel::Gpt4Turbo,
            client: reqwest::Client::new(),
            default_options: LLMOptions::default(),
        }
    }

    pub fn with_model(mut self, model: AzureModel) -> Self {
        self.model = model;
        self
    }

    pub fn with_api_version(mut self, api_version: &str) -> Self {
        self.api_version = api_version.to_string();
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

    pub fn with_json_mode(mut self) -> Self {
        self.default_options.response_format = Some(ResponseFormat::Json);
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
        if let Some(response_format) = &options.response_format {
            match response_format {
                ResponseFormat::Json => {
                    request_body["response_format"] = json!({ "type": "json_object" });
                }
                _ => {}
            }
        }

        let url = format!(
            "{}/openai/deployments/{}/chat/completions?api-version={}",
            self.endpoint, self.deployment_name, self.api_version
        );

        let response = self
            .client
            .post(&url)
            .header("api-key", &self.api_key)
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| LangHubError::LLMError(format!("Azure OpenAI request error: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LangHubError::LLMError(format!(
                "Azure OpenAI API error ({}): {}",
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

impl LLM for AzureOpenAI {
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
            AzureModel::Gpt4 => "azure-gpt-4",
            AzureModel::Gpt4Turbo => "azure-gpt-4-turbo",
            AzureModel::Gpt35Turbo => "azure-gpt-35-turbo",
            AzureModel::Custom(name) => name,
        }
    }

    fn get_provider_name(&self) -> &str {
        "Azure-OpenAI"
    }

    fn get_provider_enum(&self) -> ModelProvider {
        ModelProvider::Azure
    }

    fn supports_function_calling(&self) -> bool {
        true
    }

    fn supports_json_mode(&self) -> bool {
        true
    }

    fn max_context_length(&self) -> Option<usize> {
        match self.model {
            AzureModel::Gpt4Turbo => Some(128000),
            AzureModel::Gpt4 => Some(8192),
            _ => Some(4096),
        }
    }
}