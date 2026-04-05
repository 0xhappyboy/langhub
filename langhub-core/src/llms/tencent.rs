/// Tencent Hunyuan
use crate::types::*;

use super::{ChatMessage, LLM, LLMOptions, LLMResult};
use serde_json::json;
use std::future::Future;
use std::pin::Pin;

#[derive(Debug, Clone)]
pub enum TencentModel {
    HunyuanPro,  // Hunyuan Pro
    HunyuanStd,  // Hunyuan Standard
    HunyuanLite, // Hunyuan Lite
}

impl TencentModel {
    fn as_str(&self) -> &'static str {
        match self {
            TencentModel::HunyuanPro => "hunyuan-pro",
            TencentModel::HunyuanStd => "hunyuan-std",
            TencentModel::HunyuanLite => "hunyuan-lite",
        }
    }
}

impl From<TencentModel> for String {
    fn from(model: TencentModel) -> Self {
        model.as_str().to_string()
    }
}

pub struct TencentHunyuan {
    secret_id: String,
    secret_key: String,
    model: TencentModel,
    region: String,
    client: reqwest::Client,
    default_options: LLMOptions,
}

impl TencentHunyuan {
    pub fn new(secret_id: String, secret_key: String) -> Self {
        Self {
            secret_id,
            secret_key,
            model: TencentModel::HunyuanPro,
            region: "ap-guangzhou".to_string(),
            client: reqwest::Client::new(),
            default_options: LLMOptions::default(),
        }
    }

    pub fn with_model(mut self, model: TencentModel) -> Self {
        self.model = model;
        self
    }

    pub fn hunyuan_pro(self) -> Self {
        self.with_model(TencentModel::HunyuanPro)
    }

    pub fn hunyuan_std(self) -> Self {
        self.with_model(TencentModel::HunyuanStd)
    }

    pub fn hunyuan_lite(self) -> Self {
        self.with_model(TencentModel::HunyuanLite)
    }

    pub fn with_region(mut self, region: &str) -> Self {
        self.region = region.to_string();
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

    async fn chat_completion(
        &self,
        messages: &[ChatMessage],
        options: &LLMOptions,
    ) -> Result<String> {
        let model_name: String = self.model.clone().into();

        let mut request_body = json!({
            "Model": model_name,
            "Messages": messages.iter().map(|m| json!({
                "Role": if m.role == "assistant" { "assistant" } else { "user" },
                "Content": m.content,
            })).collect::<Vec<_>>(),
        });

        if let Some(temp) = options.temperature.or(self.default_options.temperature) {
            request_body["Temperature"] = json!(temp);
        }
        if let Some(max_tokens) = options.max_tokens.or(self.default_options.max_tokens) {
            request_body["TopP"] = json!(max_tokens);
        }
        if let Some(top_p) = options.top_p.or(self.default_options.top_p) {
            request_body["TopP"] = json!(top_p);
        }

        // Note: This is a simplified implementation
        // In production, you'd need to properly sign the request with Tencent Cloud API signatures
        let url = format!(
            "https://hunyuan.tencentcloudapi.com/?Action=ChatCompletions&Version=2023-09-01&Region={}",
            self.region
        );

        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| LangHubError::LLMError(format!("Tencent request error: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LangHubError::LLMError(format!(
                "Tencent API error ({}): {}",
                status, error_text
            )));
        }

        let json: serde_json::Value = response
            .json()
            .await
            .map_err(|e| LangHubError::LLMError(format!("JSON parse error: {}", e)))?;

        let text = json["Response"]["Choices"][0]["Message"]["Content"]
            .as_str()
            .ok_or_else(|| {
                LangHubError::ParseError("Missing 'Content' field in response".to_string())
            })?
            .to_string();

        Ok(text)
    }
}

impl LLM for TencentHunyuan {
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
            TencentModel::HunyuanPro => "hunyuan-pro",
            TencentModel::HunyuanStd => "hunyuan-std",
            TencentModel::HunyuanLite => "hunyuan-lite",
        }
    }

    fn get_provider_name(&self) -> &str {
        match self.model {
            TencentModel::HunyuanPro => "Tencent-Hunyuan-Pro",
            TencentModel::HunyuanStd => "Tencent-Hunyuan-Standard",
            TencentModel::HunyuanLite => "Tencent-Hunyuan-Lite",
        }
    }

    fn get_provider_enum(&self) -> ModelProvider {
        ModelProvider::Tencent
    }

    fn supports_function_calling(&self) -> bool {
        true
    }

    fn supports_json_mode(&self) -> bool {
        true
    }

    fn max_context_length(&self) -> Option<usize> {
        match self.model {
            TencentModel::HunyuanPro => Some(32768),
            TencentModel::HunyuanStd => Some(16384),
            TencentModel::HunyuanLite => Some(8192),
        }
    }
}
