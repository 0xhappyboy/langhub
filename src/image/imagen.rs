use crate::image::{ImageLLM, ImageLLMOptions, ImageLLMResult, ImageTask, ImageTaskStatus};
use crate::types::{LangHubError, Result};
use serde_json::json;
use std::future::Future;
use std::pin::Pin;
#[derive(Debug, Clone)]
pub enum ImagenModel {
    Imagen40,
    Imagen40Ultra,
    Custom(String),
}
impl ImagenModel {
    fn as_str(&self) -> String {
        match self {
            ImagenModel::Imagen40 => "imagen-4.0-generate-001".to_string(),
            ImagenModel::Imagen40Ultra => "imagen-4.0-ultra-generate-001".to_string(),
            ImagenModel::Custom(name) => name.clone(),
        }
    }
}
impl From<ImagenModel> for String {
    fn from(model: ImagenModel) -> Self {
        model.as_str()
    }
}
#[derive(Clone)]
pub struct Imagen {
    api_key: String,
    model: ImagenModel,
    base_url: String,
    client: reqwest::Client,
    default_options: ImageLLMOptions,
}
impl Imagen {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            model: ImagenModel::Imagen40,
            base_url: "https://generativelanguage.googleapis.com/v1beta".to_string(),
            client: reqwest::Client::new(),
            default_options: ImageLLMOptions::default(),
        }
    }
    pub fn with_model(mut self, model: ImagenModel) -> Self {
        self.model = model;
        self
    }
    pub fn imagen40(self) -> Self {
        self.with_model(ImagenModel::Imagen40)
    }
    pub fn imagen40_ultra(self) -> Self {
        self.with_model(ImagenModel::Imagen40Ultra)
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
            "instances": [
                { "prompt": prompt }
            ],
        });
        let mut parameters = json!({});
        if let Some(n) = options.n.or(self.default_options.n) {
            parameters["sampleCount"] = json!(n);
        }
        if let Some(ratio) = options
            .aspect_ratio
            .as_ref()
            .or(self.default_options.aspect_ratio.as_ref())
        {
            parameters["aspectRatio"] = json!(ratio);
        }
        if let Some(neg) = options
            .negative_prompt
            .as_ref()
            .or(self.default_options.negative_prompt.as_ref())
        {
            parameters["negativePrompt"] = json!(neg);
        }
        if let Some(format) = options
            .output_format
            .as_ref()
            .or(self.default_options.output_format.as_ref())
        {
            parameters["outputOptions"] = json!({ "mimeType": format });
        }
        body["parameters"] = parameters;
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
        let url = format!(
            "{}/models/{}:predict?key={}",
            self.base_url, model_name, self.api_key
        );
        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| LangHubError::LLMError(format!("Imagen request error: {}", e)))?;
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LangHubError::LLMError(format!(
                "Imagen API error ({}): {}",
                status, error_text
            )));
        }
        response
            .json()
            .await
            .map_err(|e| LangHubError::LLMError(format!("Imagen JSON parse error: {}", e)))
    }
    fn raw_to_result(raw: &serde_json::Value) -> ImageLLMResult {
        let mut image_base64 = Vec::new();
        if let Some(predictions) = raw["predictions"].as_array() {
            for item in predictions {
                if let Some(b64) = item["bytesBase64Encoded"].as_str() {
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
impl ImageLLM for Imagen {
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
            let task_id = raw["id"].as_str().unwrap_or("imagen-sync-task").to_string();
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
        "Google-Imagen".to_string()
    }
    fn supports_reference_images(&self) -> bool {
        true
    }
    fn supports_negative_prompt(&self) -> bool {
        true
    }
}
