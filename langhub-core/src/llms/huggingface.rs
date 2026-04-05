/// HuggingFace 
use crate::types::*;

use super::{ChatMessage, LLM, LLMOptions, LLMResult};
use serde_json::json;
use std::future::Future;
use std::pin::Pin;

#[derive(Debug, Clone)]
pub enum HuggingFaceModel {
    Llama2_7b,
    Llama2_13b,
    Llama2_70b,
    Llama3_8b,
    Llama3_70b,
    Mistral7b,
    Mixtral8x7b,
    Falcon7b,
    Falcon40b,
    Custom(String),
}

impl HuggingFaceModel {
    fn as_str(&self) -> String {
        match self {
            HuggingFaceModel::Llama2_7b => "meta-llama/Llama-2-7b-chat-hf".to_string(),
            HuggingFaceModel::Llama2_13b => "meta-llama/Llama-2-13b-chat-hf".to_string(),
            HuggingFaceModel::Llama2_70b => "meta-llama/Llama-2-70b-chat-hf".to_string(),
            HuggingFaceModel::Llama3_8b => "meta-llama/Meta-Llama-3-8B-Instruct".to_string(),
            HuggingFaceModel::Llama3_70b => "meta-llama/Meta-Llama-3-70B-Instruct".to_string(),
            HuggingFaceModel::Mistral7b => "mistralai/Mistral-7B-Instruct-v0.2".to_string(),
            HuggingFaceModel::Mixtral8x7b => "mistralai/Mixtral-8x7B-Instruct-v0.1".to_string(),
            HuggingFaceModel::Falcon7b => "tiiuae/falcon-7b-instruct".to_string(),
            HuggingFaceModel::Falcon40b => "tiiuae/falcon-40b-instruct".to_string(),
            HuggingFaceModel::Custom(name) => name.clone(),
        }
    }
}

impl From<HuggingFaceModel> for String {
    fn from(model: HuggingFaceModel) -> Self {
        model.as_str()
    }
}

pub struct HuggingFace {
    api_key: String,
    model: HuggingFaceModel,
    base_url: String,
    client: reqwest::Client,
    default_options: LLMOptions,
}

impl HuggingFace {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            model: HuggingFaceModel::Llama3_8b,
            base_url: "https://api-inference.huggingface.co/models".to_string(),
            client: reqwest::Client::new(),
            default_options: LLMOptions::default(),
        }
    }

    pub fn with_model(mut self, model: HuggingFaceModel) -> Self {
        self.model = model;
        self
    }

    pub fn llama3_8b(self) -> Self {
        self.with_model(HuggingFaceModel::Llama3_8b)
    }

    pub fn llama3_70b(self) -> Self {
        self.with_model(HuggingFaceModel::Llama3_70b)
    }

    pub fn mistral_7b(self) -> Self {
        self.with_model(HuggingFaceModel::Mistral7b)
    }

    pub fn mixtral_8x7b(self) -> Self {
        self.with_model(HuggingFaceModel::Mixtral8x7b)
    }

    pub fn with_custom_model(mut self, model_name: &str) -> Self {
        self.model = HuggingFaceModel::Custom(model_name.to_string());
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

        let mut request_body = json!({
            "inputs": prompt,
            "parameters": {},
        });

        let parameters = &mut request_body["parameters"];
        
        if let Some(temp) = options.temperature.or(self.default_options.temperature) {
            parameters["temperature"] = json!(temp);
        }
        if let Some(max_tokens) = options.max_tokens.or(self.default_options.max_tokens) {
            parameters["max_new_tokens"] = json!(max_tokens);
        }
        if let Some(top_p) = options.top_p.or(self.default_options.top_p) {
            parameters["top_p"] = json!(top_p);
        }
        if let Some(top_k) = options.top_k.or(self.default_options.top_k) {
            parameters["top_k"] = json!(top_k);
        }

        let url = format!("{}/{}", self.base_url, model_name);

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| LangHubError::LLMError(format!("HuggingFace request error: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LangHubError::LLMError(format!(
                "HuggingFace API error ({}): {}",
                status, error_text
            )));
        }

        let json: serde_json::Value = response
            .json()
            .await
            .map_err(|e| LangHubError::LLMError(format!("JSON parse error: {}", e)))?;

        let text = if json.is_array() {
            json[0]["generated_text"]
                .as_str()
                .ok_or_else(|| {
                    LangHubError::ParseError("Missing 'generated_text' field in response".to_string())
                })?
                .to_string()
        } else {
            json["generated_text"]
                .as_str()
                .ok_or_else(|| {
                    LangHubError::ParseError("Missing 'generated_text' field in response".to_string())
                })?
                .to_string()
        };

        // Remove the prompt from the response
        let response_text = text.replace(&prompt, "").trim().to_string();

        Ok(response_text)
    }
}

impl LLM for HuggingFace {
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
            HuggingFaceModel::Llama2_7b => "llama2-7b",
            HuggingFaceModel::Llama2_13b => "llama2-13b",
            HuggingFaceModel::Llama2_70b => "llama2-70b",
            HuggingFaceModel::Llama3_8b => "llama3-8b",
            HuggingFaceModel::Llama3_70b => "llama3-70b",
            HuggingFaceModel::Mistral7b => "mistral-7b",
            HuggingFaceModel::Mixtral8x7b => "mixtral-8x7b",
            HuggingFaceModel::Falcon7b => "falcon-7b",
            HuggingFaceModel::Falcon40b => "falcon-40b",
            HuggingFaceModel::Custom(name) => name,
        }
    }

    fn get_provider_name(&self) -> &str {
        match &self.model {
            HuggingFaceModel::Llama2_7b => "HuggingFace-Llama2-7B",
            HuggingFaceModel::Llama2_13b => "HuggingFace-Llama2-13B",
            HuggingFaceModel::Llama2_70b => "HuggingFace-Llama2-70B",
            HuggingFaceModel::Llama3_8b => "HuggingFace-Llama3-8B",
            HuggingFaceModel::Llama3_70b => "HuggingFace-Llama3-70B",
            HuggingFaceModel::Mistral7b => "HuggingFace-Mistral-7B",
            HuggingFaceModel::Mixtral8x7b => "HuggingFace-Mixtral-8x7B",
            HuggingFaceModel::Falcon7b => "HuggingFace-Falcon-7B",
            HuggingFaceModel::Falcon40b => "HuggingFace-Falcon-40B",
            HuggingFaceModel::Custom(name) => &format!("HuggingFace-{}", name),
        }
    }

    fn get_provider_enum(&self) -> ModelProvider {
        ModelProvider::HuggingFace
    }

    fn supports_function_calling(&self) -> bool {
        false
    }

    fn supports_json_mode(&self) -> bool {
        false
    }

    fn max_context_length(&self) -> Option<usize> {
        match self.model {
            HuggingFaceModel::Llama3_8b | HuggingFaceModel::Llama3_70b => Some(8192),
            HuggingFaceModel::Mixtral8x7b => Some(32768),
            HuggingFaceModel::Mistral7b => Some(8192),
            _ => Some(4096),
        }
    }
}