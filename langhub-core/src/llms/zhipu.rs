/// Zhipu AI (ChatGLM)
use crate::types::*;

use super::{ChatMessage, LLM, LLMOptions, LLMResult};
use hmac::{Hmac, Mac};
use serde_json::json;
use sha2::Sha256;
use std::future::Future;
use std::pin::Pin;
use std::time::{SystemTime, UNIX_EPOCH};

type HmacSha256 = Hmac<Sha256>;

#[derive(Debug, Clone)]
pub enum ZhipuModel {
    GLM4,      // GLM-4
    GLM4Air,   // GLM-4-Air
    GLM4Flash, // GLM-4-Flash
    GLM3Turbo, // GLM-3-Turbo
    GLM4V,     // GLM-4V (Vision)
}

impl ZhipuModel {
    fn as_str(&self) -> &'static str {
        match self {
            ZhipuModel::GLM4 => "glm-4",
            ZhipuModel::GLM4Air => "glm-4-air",
            ZhipuModel::GLM4Flash => "glm-4-flash",
            ZhipuModel::GLM3Turbo => "glm-3-turbo",
            ZhipuModel::GLM4V => "glm-4v",
        }
    }
}

impl From<ZhipuModel> for String {
    fn from(model: ZhipuModel) -> Self {
        model.as_str().to_string()
    }
}

pub struct ZhipuAI {
    api_key: String,
    model: ZhipuModel,
    base_url: String,
    client: reqwest::Client,
    default_options: LLMOptions,
}

impl ZhipuAI {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            model: ZhipuModel::GLM4,
            base_url: "https://open.bigmodel.cn/api/paas/v4".to_string(),
            client: reqwest::Client::new(),
            default_options: LLMOptions::default(),
        }
    }

    pub fn with_model(mut self, model: ZhipuModel) -> Self {
        self.model = model;
        self
    }

    pub fn glm4(self) -> Self {
        self.with_model(ZhipuModel::GLM4)
    }

    pub fn glm4_air(self) -> Self {
        self.with_model(ZhipuModel::GLM4Air)
    }

    pub fn glm3_turbo(self) -> Self {
        self.with_model(ZhipuModel::GLM3Turbo)
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
            .map_err(|e| LangHubError::LLMError(format!("Zhipu request error: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LangHubError::LLMError(format!(
                "Zhipu API error ({}): {}",
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

impl LLM for ZhipuAI {
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
            ZhipuModel::GLM4 => "glm-4",
            ZhipuModel::GLM4Air => "glm-4-air",
            ZhipuModel::GLM4Flash => "glm-4-flash",
            ZhipuModel::GLM3Turbo => "glm-3-turbo",
            ZhipuModel::GLM4V => "glm-4v",
        }
    }

    fn get_provider_name(&self) -> &str {
        match self.model {
            ZhipuModel::GLM4 => "Zhipu-GLM-4",
            ZhipuModel::GLM4Air => "Zhipu-GLM-4-Air",
            ZhipuModel::GLM4Flash => "Zhipu-GLM-4-Flash",
            ZhipuModel::GLM3Turbo => "Zhipu-GLM-3-Turbo",
            ZhipuModel::GLM4V => "Zhipu-GLM-4V",
        }
    }

    fn get_provider_enum(&self) -> ModelProvider {
        ModelProvider::Zhipu
    }

    fn supports_function_calling(&self) -> bool {
        true
    }

    fn supports_json_mode(&self) -> bool {
        true
    }

    fn max_context_length(&self) -> Option<usize> {
        match self.model {
            ZhipuModel::GLM4 => Some(128000),
            ZhipuModel::GLM4Air => Some(128000),
            ZhipuModel::GLM4Flash => Some(128000),
            ZhipuModel::GLM3Turbo => Some(16000),
            ZhipuModel::GLM4V => Some(8192),
        }
    }
}
