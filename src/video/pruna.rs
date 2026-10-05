use crate::types::{LangHubError, LangHubResult};
use crate::video::{VideoLLM, VideoLLMOptions, VideoLLMResult, VideoTask, VideoTaskStatus};
use serde_json::json;
use std::future::Future;
use std::pin::Pin;
#[derive(Debug, Clone)]
pub enum PrunaVideoModel {
    PVideo2Pro,
    Custom(String),
}
impl PrunaVideoModel {
    fn as_str(&self) -> String {
        match self {
            PrunaVideoModel::PVideo2Pro => "p-video-2-pro".to_string(),
            PrunaVideoModel::Custom(name) => name.clone(),
        }
    }
}
impl From<PrunaVideoModel> for String {
    fn from(model: PrunaVideoModel) -> Self {
        model.as_str()
    }
}
#[derive(Clone)]
pub struct PrunaVideo {
    api_key: String,
    model: PrunaVideoModel,
    base_url: String,
    client: reqwest::Client,
    default_options: VideoLLMOptions,
}
impl PrunaVideo {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            model: PrunaVideoModel::PVideo2Pro,
            base_url: "https://api.pruna.ai/v1".to_string(),
            client: reqwest::Client::new(),
            default_options: VideoLLMOptions::default(),
        }
    }
    pub fn with_model(mut self, model: PrunaVideoModel) -> Self {
        self.model = model;
        self
    }
    pub fn p_video_2_pro(self) -> Self {
        self.with_model(PrunaVideoModel::PVideo2Pro)
    }
    pub fn with_base_url(mut self, base_url: &str) -> Self {
        self.base_url = base_url.to_string();
        self
    }
    pub fn with_options(mut self, options: VideoLLMOptions) -> Self {
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
        options: &VideoLLMOptions,
        model_override: Option<&str>,
    ) -> serde_json::Value {
        let model_name: String = self.resolve_model(model_override);
        let mut body = json!({
            "model": model_name,
            "prompt": prompt,
        });
        if let Some(resolution) = options
            .resolution
            .as_ref()
            .or(self.default_options.resolution.as_ref())
        {
            body["resolution"] = json!(resolution);
        }
        if let Some(duration) = options.duration.or(self.default_options.duration) {
            body["duration"] = json!(duration);
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
            body["reference_images"] = json!(images);
        }
        body
    }
    async fn submit_request(
        &self,
        prompt: &str,
        options: &VideoLLMOptions,
        model_override: Option<&str>,
    ) -> LangHubResult<serde_json::Value> {
        let body = self.build_request_body(prompt, options, model_override);
        let response = self
            .client
            .post(format!("{}/video/generations", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| LangHubError::LLMError(format!("Pruna request error: {}", e)))?;
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LangHubError::LLMError(format!(
                "Pruna API error ({}): {}",
                status, error_text
            )));
        }
        response
            .json()
            .await
            .map_err(|e| LangHubError::LLMError(format!("Pruna JSON parse error: {}", e)))
    }
    fn raw_to_result(raw: &serde_json::Value) -> VideoLLMResult {
        let video_url = raw["video_url"].as_str().map(|s| s.to_string());
        let duration = raw["duration"].as_f64().map(|v| v as f32);
        let resolution = raw["resolution"].as_str().map(|s| s.to_string());
        VideoLLMResult {
            video_url,
            video_base64: None,
            file_path: None,
            duration_seconds: duration,
            resolution,
            raw_response: raw.clone(),
        }
    }
}
impl VideoLLM for PrunaVideo {
    fn generate(
        &self,
        prompt: &str,
        model: Option<&str>,
    ) -> Pin<Box<dyn Future<Output = LangHubResult<VideoLLMResult>> + Send + '_>> {
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
        options: VideoLLMOptions,
        model: Option<&str>,
    ) -> Pin<Box<dyn Future<Output = LangHubResult<VideoLLMResult>> + Send + '_>> {
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
        options: VideoLLMOptions,
        model: Option<&str>,
    ) -> Pin<Box<dyn Future<Output = LangHubResult<VideoTask>> + Send + '_>> {
        let prompt = prompt.to_string();
        let model_owned: Option<String> = model.map(|m| m.to_string());
        Box::pin(async move {
            let raw = self
                .submit_request(&prompt, &options, model_owned.as_deref())
                .await?;
            let task_id = raw["id"].as_str().unwrap_or("pruna-sync-task").to_string();
            Ok(VideoTask {
                task_id,
                status: VideoTaskStatus::Succeeded,
                result: Some(Self::raw_to_result(&raw)),
                error: None,
            })
        })
    }
    fn poll_task(
        &self,
        task_id: &str,
    ) -> Pin<Box<dyn Future<Output = LangHubResult<VideoTask>> + Send + '_>> {
        let task_id = task_id.to_string();
        Box::pin(async move {
            let url = format!("{}/video/generations/{}", self.base_url, task_id);
            let response = self
                .client
                .get(&url)
                .header("Authorization", format!("Bearer {}", self.api_key))
                .send()
                .await
                .map_err(|e| LangHubError::LLMError(format!("Pruna poll error: {}", e)))?;
            let raw: serde_json::Value = response
                .json()
                .await
                .map_err(|e| LangHubError::LLMError(format!("Pruna JSON parse error: {}", e)))?;
            let status_str = raw["status"].as_str().unwrap_or("succeeded");
            let (status, result, error) = match status_str {
                "succeeded" | "completed" => (
                    VideoTaskStatus::Succeeded,
                    Some(Self::raw_to_result(&raw)),
                    None,
                ),
                "failed" => (
                    VideoTaskStatus::Failed,
                    None,
                    Some(
                        raw["error"]
                            .as_str()
                            .unwrap_or("Pruna task failed")
                            .to_string(),
                    ),
                ),
                "processing" => (VideoTaskStatus::Processing, None, None),
                _ => (VideoTaskStatus::Pending, None, None),
            };
            Ok(VideoTask {
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
        "Pruna-P-Video".to_string()
    }
    fn max_duration(&self) -> Option<f32> {
        Some(10.0)
    }
    fn supports_audio(&self) -> bool {
        false
    }
    fn supports_reference_images(&self) -> bool {
        true
    }
    fn supports_reference_videos(&self) -> bool {
        false
    }
}
