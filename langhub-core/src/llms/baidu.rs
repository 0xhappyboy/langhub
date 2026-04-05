/// Baidu ERNIE
use crate::types::*;

use super::{ChatMessage, LLM, LLMOptions, LLMResult};
use serde_json::json;
use std::future::Future;
use std::pin::Pin;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone)]
pub enum BaiduModel {
    Ernie4_0,      // ERNIE 4.0
    Ernie3_5,      // ERNIE 3.5
    ErnieSpeed,    // ERNIE Speed
    ErnieLite,     // ERNIE Lite
    ErnieTiny,     // ERNIE Tiny
}

impl BaiduModel {
    fn as_str(&self) -> &'static str {
        match self {
            BaiduModel::Ernie4_0 => "completions_pro",
            BaiduModel::Ernie3_5 => "completions",
            BaiduModel::ErnieSpeed => "ernie_speed",
            BaiduModel::ErnieLite => "ernie_lite",
            BaiduModel::ErnieTiny => "ernie_tiny",
        }
    }
}

impl From<BaiduModel> for String {
    fn from(model: BaiduModel) -> Self {
        model.as_str().to_string()
    }
}

pub struct BaiduWenxin {
    api_key: String,
    secret_key: String,
    model: BaiduModel,
    access_token: Option<String>,
    token_expiry: Option<u64>,
    client: reqwest::Client,
    default_options: LLMOptions,
}

impl BaiduWenxin {
    pub fn new(api_key: String, secret_key: String) -> Self {
        Self {
            api_key,
            secret_key,
            model: BaiduModel::Ernie4_0,
            access_token: None,
            token_expiry: None,
            client: reqwest::Client::new(),
            default_options: LLMOptions::default(),
        }
    }

    pub fn with_model(mut self, model: BaiduModel) -> Self {
        self.model = model;
        self
    }

    pub fn ernie4_0(self) -> Self {
        self.with_model(BaiduModel::Ernie4_0)
    }

    pub fn ernie3_5(self) -> Self {
        self.with_model(BaiduModel::Ernie3_5)
    }

    pub fn ernie_speed(self) -> Self {
        self.with_model(BaiduModel::ErnieSpeed)
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

    async fn get_access_token(&mut self) -> Result<String> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        if let (Some(token), Some(expiry)) = (&self.access_token, self.token_expiry) {
            if now < expiry - 300 {
                return Ok(token.clone());
            }
        }

        let url = format!(
            "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={}&client_secret={}",
            self.api_key, self.secret_key
        );

        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .send()
            .await
            .map_err(|e| LangHubError::LLMError(format!("Baidu token error: {}", e)))?;

        let json: serde_json::Value = response
            .json()
            .await
            .map_err(|e| LangHubError::LLMError(format!("JSON parse error: {}", e)))?;

        let token = json["access_token"]
            .as_str()
            .ok_or_else(|| LangHubError::LLMError("Failed to get access token".to_string()))?
            .to_string();

        let expires_in = json["expires_in"].as_u64().unwrap_or(2592000);
        self.access_token = Some(token.clone());
        self.token_expiry = Some(now + expires_in);

        Ok(token)
    }

    async fn chat_completion(
        &mut self,
        messages: &[ChatMessage],
        options: &LLMOptions,
    ) -> Result<String> {
        let model_name: String = self.model.clone().into();
        let access_token = self.get_access_token().await?;

        let mut messages_json: Vec<serde_json::Value> = messages
            .iter()
            .map(|m| {
                let role = if m.role == "assistant" { "assistant" } else { "user" };
                json!({
                    "role": role,
                    "content": m.content,
                })
            })
            .collect();

        // Add system message if present
        if let Some(system_msg) = messages.iter().find(|m| m.role == "system") {
            messages_json.insert(0, json!({
                "role": "system",
                "content": system_msg.content,
            }));
        }

        let mut request_body = json!({
            "messages": messages_json,
        });

        if let Some(temp) = options.temperature.or(self.default_options.temperature) {
            request_body["temperature"] = json!(temp);
        }
        if let Some(max_tokens) = options.max_tokens.or(self.default_options.max_tokens) {
            request_body["max_output_tokens"] = json!(max_tokens);
        }
        if let Some(top_p) = options.top_p.or(self.default_options.top_p) {
            request_body["top_p"] = json!(top_p);
        }

        let url = format!(
            "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/{}/?access_token={}",
            model_name, access_token
        );

        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| LangHubError::LLMError(format!("Baidu request error: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LangHubError::LLMError(format!(
                "Baidu API error ({}): {}",
                status, error_text
            )));
        }

        let json: serde_json::Value = response
            .json()
            .await
            .map_err(|e| LangHubError::LLMError(format!("JSON parse error: {}", e)))?;

        if let Some(error_code) = json["error_code"].as_i64() {
            return Err(LangHubError::LLMError(format!(
                "Baidu error {}: {}",
                error_code,
                json["error_msg"].as_str().unwrap_or("Unknown error")
            )));
        }

        let text = json["result"]
            .as_str()
            .ok_or_else(|| {
                LangHubError::ParseError("Missing 'result' field in response".to_string())
            })?
            .to_string();

        Ok(text)
    }
}

impl LLM for BaiduWenxin {
    fn generate(
        &self,
        prompt: &str,
    ) -> Pin<Box<dyn Future<Output = Result<LLMResult>> + Send + '_>> {
        let prompt = prompt.to_string();
        let options = self.default_options.clone();
        let mut self_clone = self.clone();
        Box::pin(async move {
            let messages = vec![ChatMessage::user(&prompt)];
            let text = self_clone.chat_completion(&messages, &options).await?;
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
        let mut self_clone = self.clone();
        Box::pin(async move {
            let messages = vec![ChatMessage::user(&prompt)];
            let text = self_clone.chat_completion(&messages, &options).await?;
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
        let mut self_clone = self.clone();
        Box::pin(async move {
            let text = self_clone
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
            BaiduModel::Ernie4_0 => "ernie-4.0",
            BaiduModel::Ernie3_5 => "ernie-3.5",
            BaiduModel::ErnieSpeed => "ernie-speed",
            BaiduModel::ErnieLite => "ernie-lite",
            BaiduModel::ErnieTiny => "ernie-tiny",
        }
    }

    fn get_provider_name(&self) -> &str {
        match self.model {
            BaiduModel::Ernie4_0 => "Baidu-ERNIE-4.0",
            BaiduModel::Ernie3_5 => "Baidu-ERNIE-3.5",
            BaiduModel::ErnieSpeed => "Baidu-ERNIE-Speed",
            BaiduModel::ErnieLite => "Baidu-ERNIE-Lite",
            BaiduModel::ErnieTiny => "Baidu-ERNIE-Tiny",
        }
    }

    fn get_provider_enum(&self) -> ModelProvider {
        ModelProvider::Baidu
    }

    fn supports_function_calling(&self) -> bool {
        true
    }

    fn supports_json_mode(&self) -> bool {
        true
    }

    fn max_context_length(&self) -> Option<usize> {
        match self.model {
            BaiduModel::Ernie4_0 => Some(128000),
            BaiduModel::Ernie3_5 => Some(16000),
            BaiduModel::ErnieSpeed => Some(112000),
            _ => Some(8000),
        }
    }
}

impl Clone for BaiduWenxin {
    fn clone(&self) -> Self {
        Self {
            api_key: self.api_key.clone(),
            secret_key: self.secret_key.clone(),
            model: self.model.clone(),
            access_token: None,
            token_expiry: None,
            client: reqwest::Client::new(),
            default_options: self.default_options.clone(),
        }
    }
}