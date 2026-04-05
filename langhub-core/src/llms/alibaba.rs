/// Alibaba Qianwen
use crate::types::*;

use super::{ChatMessage, LLM, LLMOptions, LLMResult};
use serde_json::json;
use std::future::Future;
use std::pin::Pin;

#[derive(Debug, Clone)]
pub enum AlibabaModel {
    QwenTurbo,      // Qwen-Turbo
    QwenPlus,       // Qwen-Plus
    QwenMax,        // Qwen-Max
    QwenMaxLong,    // Qwen-Max-LongContext
    Qwen72B,        // Qwen-72B-Chat
    Qwen14B,        // Qwen-14B-Chat
    Qwen7B,         // Qwen-7B-Chat
    QwenVL,         // Qwen-VL
}

impl AlibabaModel {
    fn as_str(&self) -> &'static str {
        match self {
            AlibabaModel::QwenTurbo => "qwen-turbo",
            AlibabaModel::QwenPlus => "qwen-plus",
            AlibabaModel::QwenMax => "qwen-max",
            AlibabaModel::QwenMaxLong => "qwen-max-longcontext",
            AlibabaModel::Qwen72B => "qwen-72b-chat",
            AlibabaModel::Qwen14B => "qwen-14b-chat",
            AlibabaModel::Qwen7B => "qwen-7b-chat",
            AlibabaModel::QwenVL => "qwen-vl-plus",
        }
    }
}

impl From<AlibabaModel> for String {
    fn from(model: AlibabaModel) -> Self {
        model.as_str().to_string()
    }
}

pub struct AlibabaTongyi {
    api_key: String,
    model: AlibabaModel,
    base_url: String,
    client: reqwest::Client,
    default_options: LLMOptions,
}

impl AlibabaTongyi {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            model: AlibabaModel::QwenPlus,
            base_url: "https://dashscope.aliyuncs.com/api/v1".to_string(),
            client: reqwest::Client::new(),
            default_options: LLMOptions::default(),
        }
    }

    pub fn with_model(mut self, model: AlibabaModel) -> Self {
        self.model = model;
        self
    }

    pub fn qwen_turbo(self) -> Self {
        self.with_model(AlibabaModel::QwenTurbo)
    }

    pub fn qwen_plus(self) -> Self {
        self.with_model(AlibabaModel::QwenPlus)
    }

    pub fn qwen_max(self) -> Self {
        self.with_model(AlibabaModel::QwenMax)
    }

    pub fn qwen_max_long(self) -> Self {
        self.with_model(AlibabaModel::QwenMaxLong)
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

        let mut input = json!({
            "messages": messages.iter().map(|m| json!({
                "role": m.role,
                "content": m.content,
            })).collect::<Vec<_>>(),
        });

        let mut parameters = json!({});

        if let Some(temp) = options.temperature.or(self.default_options.temperature) {
            parameters["temperature"] = json!(temp);
        }
        if let Some(max_tokens) = options.max_tokens.or(self.default_options.max_tokens) {
            parameters["max_tokens"] = json!(max_tokens);
        }
        if let Some(top_p) = options.top_p.or(self.default_options.top_p) {
            parameters["top_p"] = json!(top_p);
        }
        if let Some(top_k) = options.top_k.or(self.default_options.top_k) {
            parameters["top_k"] = json!(top_k);
        }

        let request_body = json!({
            "model": model_name,
            "input": input,
            "parameters": parameters,
        });

        let response = self
            .client
            .post(format!("{}/services/aigc/text-generation/generation", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| LangHubError::LLMError(format!("Alibaba request error: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LangHubError::LLMError(format!(
                "Alibaba API error ({}): {}",
                status, error_text
            )));
        }

        let json: serde_json::Value = response
            .json()
            .await
            .map_err(|e| LangHubError::LLMError(format!("JSON parse error: {}", e)))?;

        let text = json["output"]["text"]
            .as_str()
            .ok_or_else(|| {
                LangHubError::ParseError("Missing 'text' field in response".to_string())
            })?
            .to_string();

        Ok(text)
    }
}

impl LLM for AlibabaTongyi {
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
            AlibabaModel::QwenTurbo => "qwen-turbo",
            AlibabaModel::QwenPlus => "qwen-plus",
            AlibabaModel::QwenMax => "qwen-max",
            AlibabaModel::QwenMaxLong => "qwen-max-longcontext",
            AlibabaModel::Qwen72B => "qwen-72b",
            AlibabaModel::Qwen14B => "qwen-14b",
            AlibabaModel::Qwen7B => "qwen-7b",
            AlibabaModel::QwenVL => "qwen-vl",
        }
    }

    fn get_provider_name(&self) -> &str {
        match self.model {
            AlibabaModel::QwenTurbo => "Alibaba-Qwen-Turbo",
            AlibabaModel::QwenPlus => "Alibaba-Qwen-Plus",
            AlibabaModel::QwenMax => "Alibaba-Qwen-Max",
            AlibabaModel::QwenMaxLong => "Alibaba-Qwen-Max-Long",
            AlibabaModel::Qwen72B => "Alibaba-Qwen-72B",
            AlibabaModel::Qwen14B => "Alibaba-Qwen-14B",
            AlibabaModel::Qwen7B => "Alibaba-Qwen-7B",
            AlibabaModel::QwenVL => "Alibaba-Qwen-VL",
        }
    }

    fn get_provider_enum(&self) -> ModelProvider {
        ModelProvider::Alibaba
    }

    fn supports_function_calling(&self) -> bool {
        true
    }

    fn supports_json_mode(&self) -> bool {
        true
    }

    fn max_context_length(&self) -> Option<usize> {
        match self.model {
            AlibabaModel::QwenMaxLong => Some(1000000),
            AlibabaModel::QwenMax => Some(32768),
            AlibabaModel::QwenPlus => Some(32768),
            AlibabaModel::QwenTurbo => Some(8192),
            AlibabaModel::Qwen72B => Some(32768),
            _ => Some(8192),
        }
    }
}