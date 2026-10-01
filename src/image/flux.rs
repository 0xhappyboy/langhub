use crate::image::{ImageLLM, ImageLLMOptions, ImageLLMResult, ImageTask, ImageTaskStatus};
use crate::types::{LangHubError, Result};
use serde_json::json;
use std::future::Future;
use std::pin::Pin;
#[derive(Debug, Clone)]
pub enum FluxImageModel {
    Flux2Max,
    Flux2Pro,
    Flux2Klein,
    Flux2Dev,
    Custom(String),
}
impl FluxImageModel {
    fn as_str(&self) -> String {
        match self {
            FluxImageModel::Flux2Max => "flux-2-max".to_string(),
            FluxImageModel::Flux2Pro => "flux-2-pro".to_string(),
            FluxImageModel::Flux2Klein => "flux-2-klein".to_string(),
            FluxImageModel::Flux2Dev => "flux-2-dev".to_string(),
            FluxImageModel::Custom(name) => name.clone(),
        }
    }
}
impl From<FluxImageModel> for String {
    fn from(model: FluxImageModel) -> Self {
        model.as_str()
    }
}
#[derive(Clone)]
pub struct FluxImage {
    api_key: String,
    model: FluxImageModel,
    base_url: String,
    client: reqwest::Client,
    default_options: ImageLLMOptions,
}
impl FluxImage {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            model: FluxImageModel::Flux2Pro,
            base_url: "https://api.bfl.ai/v1".to_string(),
            client: reqwest::Client::new(),
            default_options: ImageLLMOptions::default(),
        }
    }
    pub fn with_model(mut self, model: FluxImageModel) -> Self {
        self.model = model;
        self
    }
    pub fn flux2_max(self) -> Self {
        self.with_model(FluxImageModel::Flux2Max)
    }
    pub fn flux2_pro(self) -> Self {
        self.with_model(FluxImageModel::Flux2Pro)
    }
    pub fn flux2_klein(self) -> Self {
        self.with_model(FluxImageModel::Flux2Klein)
    }
    pub fn flux2_dev(self) -> Self {
        self.with_model(FluxImageModel::Flux2Dev)
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
    fn build_request_body(&self, prompt: &str, options: &ImageLLMOptions) -> serde_json::Value {
        let mut body = json!({
            "prompt": prompt,
        });
        if let Some(n) = options.n.or(self.default_options.n) {
            body["num_images"] = json!(n);
        }
        if let Some(resolution) = options
            .resolution
            .as_ref()
            .or(self.default_options.resolution.as_ref())
        {
            body["width"] = json!(resolution);
            body["height"] = json!(resolution);
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
            body["input_image"] = json!(images);
        }
        if let Some(format) = options
            .output_format
            .as_ref()
            .or(self.default_options.output_format.as_ref())
        {
            body["output_format"] = json!(format);
        }
        body
    }
    async fn submit_request(
        &self,
        prompt: &str,
        options: &ImageLLMOptions,
        model_override: Option<&str>,
    ) -> Result<serde_json::Value> {
        let body = self.build_request_body(prompt, options);
        let model_name: String = self.resolve_model(model_override);
        let response = self
            .client
            .post(format!("{}/{}", self.base_url, model_name))
            .header("x-key", &self.api_key)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| LangHubError::LLMError(format!("FLUX request error: {}", e)))?;
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LangHubError::LLMError(format!(
                "FLUX API error ({}): {}",
                status, error_text
            )));
        }
        response
            .json()
            .await
            .map_err(|e| LangHubError::LLMError(format!("FLUX JSON parse error: {}", e)))
    }
    fn raw_to_result(raw: &serde_json::Value) -> ImageLLMResult {
        let mut image_urls = Vec::new();
        if let Some(url) = raw["result"]["sample"].as_str() {
            image_urls.push(url.to_string());
        }
        if let Some(samples) = raw["result"]["samples"].as_array() {
            for item in samples {
                if let Some(url) = item["url"].as_str() {
                    image_urls.push(url.to_string());
                }
            }
        }
        ImageLLMResult {
            image_urls,
            image_base64: None,
            file_paths: None,
            resolution: None,
            raw_response: raw.clone(),
        }
    }
    async fn poll_until_done(&self, task_id: &str) -> Result<ImageLLMResult> {
        let url = format!("{}/get_result?id={}", self.base_url, task_id);
        for _ in 0..180 {
            let response = self
                .client
                .get(&url)
                .header("x-key", &self.api_key)
                .send()
                .await
                .map_err(|e| LangHubError::LLMError(format!("FLUX poll error: {}", e)))?;
            let raw: serde_json::Value = response
                .json()
                .await
                .map_err(|e| LangHubError::LLMError(format!("FLUX JSON parse error: {}", e)))?;
            let status = raw["status"].as_str().unwrap_or("");
            match status {
                "Ready" => return Ok(Self::raw_to_result(&raw)),
                "Error" | "Failed" => {
                    let error = raw["error"]
                        .as_str()
                        .unwrap_or("FLUX task failed")
                        .to_string();
                    return Err(LangHubError::LLMError(error));
                }
                _ => {
                    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
                }
            }
        }
        Err(LangHubError::LLMError(
            "FLUX task polling timeout".to_string(),
        ))
    }
}
impl ImageLLM for FluxImage {
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
            let task_id = raw["id"]
                .as_str()
                .ok_or_else(|| LangHubError::ParseError("Missing task id".to_string()))?
                .to_string();
            self.poll_until_done(&task_id).await
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
            let task_id = raw["id"]
                .as_str()
                .ok_or_else(|| LangHubError::ParseError("Missing task id".to_string()))?
                .to_string();
            self.poll_until_done(&task_id).await
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
            let task_id = raw["id"]
                .as_str()
                .ok_or_else(|| LangHubError::ParseError("Missing task id".to_string()))?
                .to_string();
            Ok(ImageTask {
                task_id,
                status: ImageTaskStatus::Pending,
                result: None,
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
            let url = format!("{}/get_result?id={}", self.base_url, task_id);
            let response = self
                .client
                .get(&url)
                .header("x-key", &self.api_key)
                .send()
                .await
                .map_err(|e| LangHubError::LLMError(format!("FLUX poll error: {}", e)))?;
            let raw: serde_json::Value = response
                .json()
                .await
                .map_err(|e| LangHubError::LLMError(format!("FLUX JSON parse error: {}", e)))?;
            let status_str = raw["status"].as_str().unwrap_or("");
            let (status, result, error) = match status_str {
                "Ready" => (
                    ImageTaskStatus::Succeeded,
                    Some(Self::raw_to_result(&raw)),
                    None,
                ),
                "Error" | "Failed" => (
                    ImageTaskStatus::Failed,
                    None,
                    Some(
                        raw["error"]
                            .as_str()
                            .unwrap_or("FLUX task failed")
                            .to_string(),
                    ),
                ),
                "Processing" => (ImageTaskStatus::Processing, None, None),
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
        "BlackForestLabs-FLUX".to_string()
    }
    fn supports_reference_images(&self) -> bool {
        true
    }
    fn supports_negative_prompt(&self) -> bool {
        true
    }
}
