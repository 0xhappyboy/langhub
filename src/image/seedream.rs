use crate::image::{ImageLLM, ImageLLMOptions, ImageLLMResult, ImageTask, ImageTaskStatus};
use crate::types::{LangHubError, Result};
use serde_json::json;
use std::future::Future;
use std::pin::Pin;
/// ByteDance Seedream text-to-image model enum.
///
/// All model ids follow the official Volcengine Ark naming scheme:
/// - `doubao-seedream-3-0-t2i-*`   (Seedream 3.0, text-to-image only)
/// - `doubao-seedream-4-0-*`       (Seedream 4.0, text-to-image + image-to-image)
/// - `doubao-seedream-4-5`         (Seedream 4.5, text-to-image + image-to-image)
/// - `doubao-seedream-5-0-*`       (Seedream 5.0 family)
///
/// `Custom` allows passing a raw Ark endpoint id (e.g. `ep-2024xxxx-xxxxx`)
/// or any future model id without a code change.
#[derive(Debug, Clone)]
pub enum SeedreamModel {
    /// Seedream 3.0 (text-to-image).
    Seedream30,
    /// Seedream 4.0 (text-to-image + image-to-image).
    Seedream40,
    /// Seedream 4.5 (text-to-image + image-to-image).
    Seedream45,
    /// Seedream 5.0 Lite.
    Seedream50Lite,
    /// Seedream 5.0 Flash.
    Seedream50Flash,
    /// Seedream 5.0 Pro.
    Seedream50Pro,
    /// Caller-supplied model id (raw Ark endpoint id or a future model).
    Custom(String),
}
impl SeedreamModel {
    /// Returns the official Ark model id string.
    pub fn as_str(&self) -> String {
        match self {
            SeedreamModel::Seedream30 => "doubao-seedream-3-0-t2i-250415".to_string(),
            SeedreamModel::Seedream40 => "doubao-seedream-4-0-20260415".to_string(),
            SeedreamModel::Seedream45 => "doubao-seedream-4-5-251128".to_string(),
            SeedreamModel::Seedream50Lite => "doubao-seedream-5-0-260128".to_string(),
            SeedreamModel::Seedream50Flash => "doubao-seedream-5-0-flash-260915".to_string(),
            SeedreamModel::Seedream50Pro => "doubao-seedream-5-0-pro-260628".to_string(),
            SeedreamModel::Custom(name) => name.clone(),
        }
    }
    /// Whether this model supports image-to-image (reference images).
    pub fn supports_reference_images(&self) -> bool {
        match self {
            SeedreamModel::Seedream30 => false,
            SeedreamModel::Seedream40
            | SeedreamModel::Seedream45
            | SeedreamModel::Seedream50Lite
            | SeedreamModel::Seedream50Flash
            | SeedreamModel::Seedream50Pro => true,
            SeedreamModel::Custom(_) => true,
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
    /// Create a new Seedream client. Default model is Seedream 4.5, which is
    /// the currently recommended general-purpose text-to-image model.
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            model: SeedreamModel::Seedream45,
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
    pub fn seedream40(self) -> Self {
        self.with_model(SeedreamModel::Seedream40)
    }
    pub fn seedream45(self) -> Self {
        self.with_model(SeedreamModel::Seedream45)
    }
    pub fn seedream50_lite(self) -> Self {
        self.with_model(SeedreamModel::Seedream50Lite)
    }
    pub fn seedream50_flash(self) -> Self {
        self.with_model(SeedreamModel::Seedream50Flash)
    }
    pub fn seedream50_pro(self) -> Self {
        self.with_model(SeedreamModel::Seedream50Pro)
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
        model_override: Option<&str>,
    ) -> Result<serde_json::Value> {
        let body = self.build_request_body(prompt, options, model_override);
        let body = self.build_request_body(prompt, options, model_override);
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
        self.model.supports_reference_images()
    }
    fn supports_negative_prompt(&self) -> bool {
        true
    }
}
