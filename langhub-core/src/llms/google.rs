/// Google (Gemini)
use crate::types::*;

use super::{ChatMessage, LLM, LLMOptions, LLMResult};
use serde_json::json;
use std::future::Future;
use std::pin::Pin;

#[derive(Debug, Clone)]
pub enum GoogleModel {
    Gemini15Pro,        // Gemini 1.5 Pro
    Gemini15Flash,      // Gemini 1.5 Flash
    Gemini15ProVision,  // Gemini 1.5 Pro Vision
    GeminiPro,          // Gemini Pro
    GeminiProVision,    // Gemini Pro Vision
    GeminiUltra,        // Gemini Ultra
}

impl GoogleModel {
    fn as_str(&self) -> &'static str {
        match self {
            GoogleModel::Gemini15Pro => "gemini-1.5-pro",
            GoogleModel::Gemini15Flash => "gemini-1.5-flash",
            GoogleModel::Gemini15ProVision => "gemini-1.5-pro-vision",
            GoogleModel::GeminiPro => "gemini-pro",
            GoogleModel::GeminiProVision => "gemini-pro-vision",
            GoogleModel::GeminiUltra => "gemini-ultra",
        }
    }
}

impl From<GoogleModel> for String {
    fn from(model: GoogleModel) -> Self {
        model.as_str().to_string()
    }
}

pub struct GoogleAI {
    api_key: String,
    model: GoogleModel,
    base_url: String,
    client: reqwest::Client,
    default_options: LLMOptions,
}

impl GoogleAI {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            model: GoogleModel::Gemini15Pro,
            base_url: "https://generativelanguage.googleapis.com/v1beta".to_string(),
            client: reqwest::Client::new(),
            default_options: LLMOptions::default(),
        }
    }

    pub fn with_model(mut self, model: GoogleModel) -> Self {
        self.model = model;
        self
    }

    pub fn gemini15_pro(self) -> Self {
        self.with_model(GoogleModel::Gemini15Pro)
    }

    pub fn gemini15_flash(self) -> Self {
        self.with_model(GoogleModel::Gemini15Flash)
    }

    pub fn gemini_pro(self) -> Self {
        self.with_model(GoogleModel::GeminiPro)
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

        let contents: Vec<serde_json::Value> = messages
            .iter()
            .map(|m| {
                json!({
                    "parts": [{"text": m.content}],
                    "role": if m.role == "user" { "user" } else { "model" },
                })
            })
            .collect();

        let mut generation_config = json!({});

        if let Some(temp) = options.temperature.or(self.default_options.temperature) {
            generation_config["temperature"] = json!(temp);
        }
        if let Some(max_tokens) = options.max_tokens.or(self.default_options.max_tokens) {
            generation_config["maxOutputTokens"] = json!(max_tokens);
        }
        if let Some(top_p) = options.top_p.or(self.default_options.top_p) {
            generation_config["topP"] = json!(top_p);
        }
        if let Some(top_k) = options.top_k.or(self.default_options.top_k) {
            generation_config["topK"] = json!(top_k);
        }

        let request_body = json!({
            "contents": contents,
            "generationConfig": generation_config,
        });

        let url = format!(
            "{}/models/{}:generateContent?key={}",
            self.base_url, model_name, self.api_key
        );

        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| LangHubError::LLMError(format!("Google request error: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LangHubError::LLMError(format!(
                "Google API error ({}): {}",
                status, error_text
            )));
        }

        let json: serde_json::Value = response
            .json()
            .await
            .map_err(|e| LangHubError::LLMError(format!("JSON parse error: {}", e)))?;

        let text = json["candidates"][0]["content"]["parts"][0]["text"]
            .as_str()
            .ok_or_else(|| {
                LangHubError::ParseError("Missing 'text' field in response".to_string())
            })?
            .to_string();

        Ok(text)
    }
}

impl LLM for GoogleAI {
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
            GoogleModel::Gemini15Pro => "gemini-1.5-pro",
            GoogleModel::Gemini15Flash => "gemini-1.5-flash",
            GoogleModel::Gemini15ProVision => "gemini-1.5-pro-vision",
            GoogleModel::GeminiPro => "gemini-pro",
            GoogleModel::GeminiProVision => "gemini-pro-vision",
            GoogleModel::GeminiUltra => "gemini-ultra",
        }
    }

    fn get_provider_name(&self) -> &str {
        match self.model {
            GoogleModel::Gemini15Pro => "Google-Gemini1.5-Pro",
            GoogleModel::Gemini15Flash => "Google-Gemini1.5-Flash",
            GoogleModel::Gemini15ProVision => "Google-Gemini1.5-Pro-Vision",
            GoogleModel::GeminiPro => "Google-Gemini-Pro",
            GoogleModel::GeminiProVision => "Google-Gemini-Pro-Vision",
            GoogleModel::GeminiUltra => "Google-Gemini-Ultra",
        }
    }

    fn get_provider_enum(&self) -> ModelProvider {
        ModelProvider::Google
    }

    fn supports_function_calling(&self) -> bool {
        true
    }

    fn supports_json_mode(&self) -> bool {
        true
    }

    fn max_context_length(&self) -> Option<usize> {
        match self.model {
            GoogleModel::Gemini15Pro => Some(2_000_000),
            GoogleModel::Gemini15Flash => Some(1_000_000),
            GoogleModel::GeminiPro => Some(32768),
            GoogleModel::GeminiUltra => Some(32768),
            _ => Some(32768),
        }
    }
}