use crate::image::{ImageLLM, ImageLLMOptions, ImageLLMResult, ImageTask, ImageTaskStatus};
use crate::types::{LangHubError, LangHubResult};
use serde_json::json;
use std::future::Future;
use std::pin::Pin;
#[derive(Debug, Clone)]
pub enum StabilityImageModel {
    StableImageUltra,
    Sd35Large,
    Sd35LargeTurbo,
    Sd35Medium,
    Sd35Flash,
    Custom(String),
}
impl StabilityImageModel {
    fn as_str(&self) -> String {
        match self {
            StabilityImageModel::StableImageUltra => "stable-image-ultra".to_string(),
            StabilityImageModel::Sd35Large => "sd3.5-large".to_string(),
            StabilityImageModel::Sd35LargeTurbo => "sd3.5-large-turbo".to_string(),
            StabilityImageModel::Sd35Medium => "sd3.5-medium".to_string(),
            StabilityImageModel::Sd35Flash => "sd3.5-flash".to_string(),
            StabilityImageModel::Custom(name) => name.clone(),
        }
    }
}
impl From<StabilityImageModel> for String {
    fn from(model: StabilityImageModel) -> Self {
        model.as_str()
    }
}
#[derive(Clone)]
pub struct StabilityImage {
    api_key: String,
    model: StabilityImageModel,
    base_url: String,
    client: reqwest::Client,
    default_options: ImageLLMOptions,
}
impl StabilityImage {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            model: StabilityImageModel::StableImageUltra,
            base_url: "https://api.stability.ai/v2beta".to_string(),
            client: reqwest::Client::new(),
            default_options: ImageLLMOptions::default(),
        }
    }
    pub fn with_model(mut self, model: StabilityImageModel) -> Self {
        self.model = model;
        self
    }
    pub fn stable_image_ultra(self) -> Self {
        self.with_model(StabilityImageModel::StableImageUltra)
    }
    pub fn sd35_large(self) -> Self {
        self.with_model(StabilityImageModel::Sd35Large)
    }
    pub fn sd35_large_turbo(self) -> Self {
        self.with_model(StabilityImageModel::Sd35LargeTurbo)
    }
    pub fn sd35_medium(self) -> Self {
        self.with_model(StabilityImageModel::Sd35Medium)
    }
    pub fn sd35_flash(self) -> Self {
        self.with_model(StabilityImageModel::Sd35Flash)
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
            "output_format": options
                .output_format
                .as_ref()
                .or(self.default_options.output_format.as_ref())
                .map(|s| s.as_str())
                .unwrap_or("png"),
        });
        if let Some(neg) = options
            .negative_prompt
            .as_ref()
            .or(self.default_options.negative_prompt.as_ref())
        {
            body["negative_prompt"] = json!(neg);
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
        body
    }
    async fn submit_request(
        &self,
        prompt: &str,
        options: &ImageLLMOptions,
        model_override: Option<&str>,
    ) -> LangHubResult<serde_json::Value> {
        let body = self.build_request_body(prompt, options);
        let model_name: String = self.resolve_model(model_override);
        // Determine the endpoint from the effective model id so that a
        // caller-supplied override also influences routing.
        let url = if model_name == "stable-image-ultra" {
            format!("{}/stable-image/generate/ultra", self.base_url)
        } else if model_name.starts_with("sd3.5-") {
            format!("{}/stable-image/generate/sd3", self.base_url)
        } else {
            format!("{}/stable-image/generate/{}", self.base_url, model_name)
        };
        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Accept", "application/json")
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| LangHubError::LLMError(format!("Stability request error: {}", e)))?;
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LangHubError::LLMError(format!(
                "Stability API error ({}): {}",
                status, error_text
            )));
        }
        response
            .json()
            .await
            .map_err(|e| LangHubError::LLMError(format!("Stability JSON parse error: {}", e)))
    }
    fn raw_to_result(raw: &serde_json::Value) -> ImageLLMResult {
        let mut image_base64 = Vec::new();
        if let Some(b64) = raw["image"].as_str() {
            image_base64.push(b64.to_string());
        }
        if let Some(images) = raw["images"].as_array() {
            for item in images {
                if let Some(b64) = item.as_str() {
                    image_base64.push(b64.to_string());
                }
            }
        }
        ImageLLMResult {
            image_urls: Vec::new(),
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
impl ImageLLM for StabilityImage {
    fn generate(
        &self,
        prompt: &str,
        model: Option<&str>,
    ) -> Pin<Box<dyn Future<Output = LangHubResult<ImageLLMResult>> + Send + '_>> {
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
    ) -> Pin<Box<dyn Future<Output = LangHubResult<ImageLLMResult>> + Send + '_>> {
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
    ) -> Pin<Box<dyn Future<Output = LangHubResult<ImageTask>> + Send + '_>> {
        let prompt = prompt.to_string();
        let model_owned: Option<String> = model.map(|m| m.to_string());
        Box::pin(async move {
            let raw = self
                .submit_request(&prompt, &options, model_owned.as_deref())
                .await?;
            let task_id = raw["id"]
                .as_str()
                .unwrap_or("stability-sync-task")
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
    ) -> Pin<Box<dyn Future<Output = LangHubResult<ImageTask>> + Send + '_>> {
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
        "Stability-AI".to_string()
    }
    fn supports_reference_images(&self) -> bool {
        true
    }
    fn supports_negative_prompt(&self) -> bool {
        true
    }
}
