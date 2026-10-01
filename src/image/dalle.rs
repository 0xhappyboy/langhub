use crate::image::{ImageLLM, ImageLLMOptions, ImageLLMResult, ImageTask, ImageTaskStatus};
use crate::types::{LangHubError, Result};
use serde_json::json;
use std::future::Future;
use std::pin::Pin;
#[derive(Debug, Clone)]
pub enum DallEModel {
    DallE3,
    DallE2,
    Custom(String),
}
impl DallEModel {
    fn as_str(&self) -> String {
        match self {
            DallEModel::DallE3 => "dall-e-3".to_string(),
            DallEModel::DallE2 => "dall-e-2".to_string(),
            DallEModel::Custom(name) => name.clone(),
        }
    }
}
impl From<DallEModel> for String {
    fn from(model: DallEModel) -> Self {
        model.as_str()
    }
}
/// OpenAI DALL·E image generation client.
#[derive(Clone)]
pub struct DallE {
    api_key: String,
    model: DallEModel,
    base_url: String,
    client: reqwest::Client,
    default_options: ImageLLMOptions,
}
impl DallE {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            model: DallEModel::DallE3,
            base_url: "https://api.openai.com/v1".to_string(),
            client: reqwest::Client::new(),
            default_options: ImageLLMOptions::default(),
        }
    }
    pub fn with_model(mut self, model: DallEModel) -> Self {
        self.model = model;
        self
    }
    pub fn dalle3(self) -> Self {
        self.with_model(DallEModel::DallE3)
    }
    pub fn dalle2(self) -> Self {
        self.with_model(DallEModel::DallE2)
    }
    pub fn with_base_url(mut self, base_url: &str) -> Self {
        self.base_url = base_url.to_string();
        self
    }
    pub fn with_options(mut self, options: ImageLLMOptions) -> Self {
        self.default_options = options;
        self
    }
    /// Resolve the effective model id: caller override wins over the
    /// configured default.
    fn resolve_model(&self, model_override: Option<&str>) -> String {
        match model_override {
            Some(m) => m.to_string(),
            None => self.model.clone().into(),
        }
    }
    fn build_request_body(
        &self,
        prompt: &str,
        options: &ImageLLMOptions,
        model_override: Option<&str>,
    ) -> serde_json::Value {
        let model_name: String = self.resolve_model(model_override);
        let mut body = json!({
            "model": model_name,
            "prompt": prompt,
        });
        if let Some(n) = options.n.or(self.default_options.n) {
            body["n"] = json!(n);
        }
        if let Some(resolution) = options
            .resolution
            .as_ref()
            .or(self.default_options.resolution.as_ref())
        {
            body["size"] = json!(resolution);
        }
        if let Some(quality) = options
            .quality
            .as_ref()
            .or(self.default_options.quality.as_ref())
        {
            body["quality"] = json!(quality);
        }
        if let Some(style) = options
            .style
            .as_ref()
            .or(self.default_options.style.as_ref())
        {
            body["style"] = json!(style);
        }
        if let Some(format) = options
            .output_format
            .as_ref()
            .or(self.default_options.output_format.as_ref())
        {
            body["response_format"] = json!(format);
        }
        body
    }
    async fn submit_request(
        &self,
        prompt: &str,
        options: &ImageLLMOptions,
        model_override: Option<&str>,
    ) -> Result<serde_json::Value> {
        let body = self.build_request_body(prompt, options, model_override);
        let response = self
            .client
            .post(format!("{}/images/generations", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| LangHubError::LLMError(format!("DALL·E request error: {}", e)))?;
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LangHubError::LLMError(format!(
                "DALL·E API error ({}): {}",
                status, error_text
            )));
        }
        response
            .json()
            .await
            .map_err(|e| LangHubError::LLMError(format!("DALL·E JSON parse error: {}", e)))
    }
    fn raw_to_result(raw: &serde_json::Value) -> ImageLLMResult {
        let mut image_urls = Vec::new();
        let mut image_base64 = Vec::new();
        if let Some(data) = raw["data"].as_array() {
            for item in data {
                if let Some(url) = item["url"].as_str() {
                    image_urls.push(url.to_string());
                }
                if let Some(b64) = item["b64_json"].as_str() {
                    image_base64.push(b64.to_string());
                }
            }
        }
        ImageLLMResult {
            image_urls,
            image_base64: if image_base64.is_empty() {
                None
            } else {
                Some(image_base64)
            },
            file_paths: None,
            resolution: None,
            raw_response: raw.clone(),
        }
    }
}
impl ImageLLM for DallE {
    fn generate(
        &self,
        prompt: &str,
        model: Option<&str>,
    ) -> Pin<Box<dyn Future<Output = Result<ImageLLMResult>> + Send + '_>> {
        let prompt = prompt.to_string();
        let options = self.default_options.clone();
        let model_owned: Option<String> = model.map(|m| m.to_string());
        Box::pin(async move {
            let raw = self
                .submit_request(&prompt, &options, model_owned.as_deref())
                .await?;
            Ok(Self::raw_to_result(&raw))
        })
    }
    fn generate_with_options(
        &self,
        prompt: &str,
        options: ImageLLMOptions,
        model: Option<&str>,
    ) -> Pin<Box<dyn Future<Output = Result<ImageLLMResult>> + Send + '_>> {
        let prompt = prompt.to_string();
        let model_owned: Option<String> = model.map(|m| m.to_string());
        Box::pin(async move {
            let raw = self
                .submit_request(&prompt, &options, model_owned.as_deref())
                .await?;
            Ok(Self::raw_to_result(&raw))
        })
    }
    fn submit_task(
        &self,
        prompt: &str,
        options: ImageLLMOptions,
        model: Option<&str>,
    ) -> Pin<Box<dyn Future<Output = Result<ImageTask>> + Send + '_>> {
        let prompt = prompt.to_string();
        let model_owned: Option<String> = model.map(|m| m.to_string());
        Box::pin(async move {
            let raw = self
                .submit_request(&prompt, &options, model_owned.as_deref())
                .await?;
            let task_id = raw["id"].as_str().unwrap_or("dalle-sync-task").to_string();
            Ok(ImageTask {
                task_id,
                status: ImageTaskStatus::Succeeded,
                result: Some(Self::raw_to_result(&raw)),
                error: None,
            })
        })
    }
    fn poll_task(
        &self,
        task_id: &str,
    ) -> Pin<Box<dyn Future<Output = Result<ImageTask>> + Send + '_>> {
        let task_id = task_id.to_string();
        Box::pin(async move {
            Ok(ImageTask {
                task_id,
                status: ImageTaskStatus::Succeeded,
                result: None,
                error: None,
            })
        })
    }
    fn get_model_name(&self) -> String {
        self.model.as_str()
    }
    fn get_provider_name(&self) -> String {
        "OpenAI-DALL-E".to_string()
    }
    fn supports_reference_images(&self) -> bool {
        false
    }
    fn supports_negative_prompt(&self) -> bool {
        false
    }
}
