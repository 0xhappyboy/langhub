/// OpenAI (GPT-4, GPT-3.5, O1)
use crate::types::*;

use super::{ChatMessage, LLM, LLMOptions, LLMResult, ResponseFormat};
use serde_json::json;
use std::future::Future;
use std::pin::Pin;

#[derive(Debug, Clone)]
pub enum OpenAIModel {
    Gpt4,           // GPT-4
    Gpt4Turbo,      // GPT-4 Turbo
    Gpt4Vision,     // GPT-4 Vision
    Gpt432k,        // GPT-4 32K
    Gpt35Turbo,     // GPT-3.5 Turbo
    Gpt35Turbo16k,  // GPT-3.5 Turbo 16K
    O1Preview,      // O1 Preview
    O1Mini,         // O1 Mini
}

impl OpenAIModel {
    fn as_str(&self) -> &'static str {
        match self {
            OpenAIModel::Gpt4 => "gpt-4",
            OpenAIModel::Gpt4Turbo => "gpt-4-turbo-preview",
            OpenAIModel::Gpt4Vision => "gpt-4-vision-preview",
            OpenAIModel::Gpt432k => "gpt-4-32k",
            OpenAIModel::Gpt35Turbo => "gpt-3.5-turbo",
            OpenAIModel::Gpt35Turbo16k => "gpt-3.5-turbo-16k",
            OpenAIModel::O1Preview => "o1-preview",
            OpenAIModel::O1Mini => "o1-mini",
        }
    }
}

impl From<OpenAIModel> for String {
    fn from(model: OpenAIModel) -> Self {
        model.as_str().to_string()
    }
}

pub struct OpenAI {
    api_key: String,
    model: OpenAIModel,
    base_url: String,
    client: reqwest::Client,
    default_options: LLMOptions,
}

impl OpenAI {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            model: OpenAIModel::Gpt4Turbo,
            base_url: "https://api.openai.com/v1".to_string(),
            client: reqwest::Client::new(),
            default_options: LLMOptions::default(),
        }
    }

    pub fn with_model(mut self, model: OpenAIModel) -> Self {
        self.model = model;
        self
    }

    pub fn gpt4(self) -> Self {
        self.with_model(OpenAIModel::Gpt4)
    }

    pub fn gpt4_turbo(self) -> Self {
        self.with_model(OpenAIModel::Gpt4Turbo)
    }

    pub fn gpt35_turbo(self) -> Self {
        self.with_model(OpenAIModel::Gpt35Turbo)
    }

    pub fn o1_preview(self) -> Self {
        self.with_model(OpenAIModel::O1Preview)
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
                ResponseFormat::JsonSchema { schema } => {
                    request_body["response_format"] = json!({
                        "type": "json_schema",
                        "json_schema": schema
                    });
                }
                _ => {}
            }
        }

        let response = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| LangHubError::LLMError(format!("OpenAI request error: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LangHubError::LLMError(format!(
                "OpenAI API error ({}): {}",
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

impl LLM for OpenAI {
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
            OpenAIModel::Gpt4 => "gpt-4",
            OpenAIModel::Gpt4Turbo => "gpt-4-turbo-preview",
            OpenAIModel::Gpt4Vision => "gpt-4-vision-preview",
            OpenAIModel::Gpt432k => "gpt-4-32k",
            OpenAIModel::Gpt35Turbo => "gpt-3.5-turbo",
            OpenAIModel::Gpt35Turbo16k => "gpt-3.5-turbo-16k",
            OpenAIModel::O1Preview => "o1-preview",
            OpenAIModel::O1Mini => "o1-mini",
        }
    }

    fn get_provider_name(&self) -> &str {
        match self.model {
            OpenAIModel::Gpt4 => "OpenAI-GPT4",
            OpenAIModel::Gpt4Turbo => "OpenAI-GPT4-Turbo",
            OpenAIModel::Gpt4Vision => "OpenAI-GPT4-Vision",
            OpenAIModel::Gpt432k => "OpenAI-GPT4-32K",
            OpenAIModel::Gpt35Turbo => "OpenAI-GPT3.5-Turbo",
            OpenAIModel::Gpt35Turbo16k => "OpenAI-GPT3.5-Turbo-16K",
            OpenAIModel::O1Preview => "OpenAI-O1-Preview",
            OpenAIModel::O1Mini => "OpenAI-O1-Mini",
        }
    }

    fn get_provider_enum(&self) -> ModelProvider {
        ModelProvider::OpenAI
    }

    fn supports_function_calling(&self) -> bool {
        true
    }

    fn supports_json_mode(&self) -> bool {
        true
    }

    fn max_context_length(&self) -> Option<usize> {
        match self.model {
            OpenAIModel::Gpt4Turbo => Some(128000),
            OpenAIModel::Gpt4 => Some(8192),
            OpenAIModel::Gpt432k => Some(32768),
            OpenAIModel::Gpt35Turbo16k => Some(16384),
            OpenAIModel::O1Preview => Some(128000),
            OpenAIModel::O1Mini => Some(128000),
            _ => Some(4096),
        }
    }
}