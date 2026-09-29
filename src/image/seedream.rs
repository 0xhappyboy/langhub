use crate::image::{ImageLLM, ImageLLMOptions, ImageLLMResult, ImageTask, ImageTaskStatus};
use crate::types::{LangHubError, Result};
use serde_json::json;
use std::future::Future;
use std::pin::Pin;
#[derive(Debug, Clone)]
pub enum SeedreamModel {
    Seedream30,
    Seedream45,
    Seedream50Lite,
    Custom(String),
}
impl SeedreamModel {
    fn as_str(&self) -> String {
        match self {
            SeedreamModel::Seedream30 => "seedream-3-0".to_string(),
            SeedreamModel::Seedream45 => "doubao-seedream-4-5".to_string(),
            SeedreamModel::Seedream50Lite => "doubao-seedream-5-0-lite".to_string(),
            SeedreamModel::Custom(name) => name.clone(),
        }
    }
}
impl From<SeedreamModel> for String {
    fn from(model: SeedreamModel) -> Self {
        model.as_str()
    }
}
#[derive(Clone)]
pub struct Seedream {
    api_key: String,
    model: SeedreamModel,
    base_url: String,
    client: reqwest::Client,
    default_options: ImageLLMOptions,
}
impl Seedream {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            model: SeedreamModel::Seedream30,
            base_url: "https://ark.cn-beijing.volces.com/api/v3".to_string(),
            client: reqwest::Client::new(),
            default_options: ImageLLMOptions::default(),
        }
    }
    pub fn with_model(mut self, model: SeedreamModel) -> Self {
        self.model = model;
        self
    }
    pub fn seedream30(self) -> Self {
        self.with_model(SeedreamModel::Seedream30)
    }
    pub fn seedream45(self) -> Self {
        self.with_model(SeedreamModel::Seedream45)
    }
    pub fn seedream50_lite(self) -> Self {
        self.with_model(SeedreamModel::Seedream50Lite)
    }
    pub fn with_base_url(mut self, base_url: &str) -> Self {
        self.base_url = base_url.to_string();
        self
    }
    pub fn with_options(mut self, options: ImageLLMOptions) -> Self {
        self.default_options = options;
        self
    }
    fn build_request_body(&self, prompt: &str, options: &ImageLLMOptions) -> serde_json::Value {
        let model_name: String = self.model.clone().into();
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
        if let Some(ratio) = options
            .aspect_ratio
            .as_ref()
            .or(self.default_options.aspect_ratio.as_ref())
        {
            body["aspect_ratio"] = json!(ratio);
        }
        if let Some(seed) = options.seed.or(self.default_options.seed) {
            body["seed"] = json!(seed);
        }
        if let Some(images) = options
            .reference_images
            .as_ref()
            .or(self.default_options.reference_images.as_ref())
        {
            body["image"] = json!(images);
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
    ) -> Result<serde_json::Value> {
        let body = self.build_request_body(prompt, options);
        let response = self
            .client
            .post(format!("{}/images/generations", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| LangHubError::LLMError(format!("Seedream request error: {}", e)))?;
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LangHubError::LLMError(format!(
                "Seedream API error ({}): {}",
                status, error_text
            )));
        }
        response
            .json()
            .await
            .map_err(|e| LangHubError::LLMError(format!("Seedream JSON parse error: {}", e)))
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
        let resolution = raw["size"].as_str().map(|s| s.to_string());
        ImageLLMResult {
            image_urls,
            image_base64: if image_base64.is_empty() {
                None
            } else {
                Some(image_base64)
            },
            file_paths: None,
            resolution,
            raw_response: raw.clone(),
        }
    }
}
impl ImageLLM for Seedream {
    fn generate(
        &self,
        prompt: &str,
    ) -> Pin<Box<dyn Future<Output = Result<ImageLLMResult>> + Send + '_>> {
        let prompt = prompt.to_string();
        let options = self.default_options.clone();
        Box::pin(async move {
            let raw = self.submit_request(&prompt, &options).await?;
            Ok(Self::raw_to_result(&raw))
        })
    }
    fn generate_with_options(
        &self,
        prompt: &str,
        options: ImageLLMOptions,
    ) -> Pin<Box<dyn Future<Output = Result<ImageLLMResult>> + Send + '_>> {
        let prompt = prompt.to_string();
        Box::pin(async move {
            let raw = self.submit_request(&prompt, &options).await?;
            Ok(Self::raw_to_result(&raw))
        })
    }
    fn submit_task(
        &self,
        prompt: &str,
        options: ImageLLMOptions,
    ) -> Pin<Box<dyn Future<Output = Result<ImageTask>> + Send + '_>> {
        let prompt = prompt.to_string();
        Box::pin(async move {
            let raw = self.submit_request(&prompt, &options).await?;
            let task_id = raw["id"]
                .as_str()
                .unwrap_or("seedream-sync-task")
                .to_string();
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
            let url = format!("{}/images/generations/{}", self.base_url, task_id);
            let response = self
                .client
                .get(&url)
                .header("Authorization", format!("Bearer {}", self.api_key))
                .send()
                .await
                .map_err(|e| LangHubError::LLMError(format!("Seedream poll error: {}", e)))?;
            let raw: serde_json::Value = response
                .json()
                .await
                .map_err(|e| LangHubError::LLMError(format!("Seedream JSON parse error: {}", e)))?;
            let status_str = raw["status"].as_str().unwrap_or("succeeded");
            let (status, result, error) = match status_str {
                "succeeded" | "completed" => (
                    ImageTaskStatus::Succeeded,
                    Some(Self::raw_to_result(&raw)),
                    None,
                ),
                "failed" => (
                    ImageTaskStatus::Failed,
                    None,
                    Some(
                        raw["error"]["message"]
                            .as_str()
                            .unwrap_or("Seedream task failed")
                            .to_string(),
                    ),
                ),
                "processing" => (ImageTaskStatus::Processing, None, None),
                _ => (ImageTaskStatus::Pending, None, None),
            };
            Ok(ImageTask {
                task_id,
                status,
                result,
                error,
            })
        })
    }
    fn get_model_name(&self) -> String {
        self.model.as_str()
    }
    fn get_provider_name(&self) -> String {
        "ByteDance-Seedream".to_string()
    }
    fn supports_reference_images(&self) -> bool {
        true
    }
    fn supports_negative_prompt(&self) -> bool {
        true
    }
}
