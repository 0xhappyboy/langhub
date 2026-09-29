use crate::types::{LangHubError, Result};
use crate::video::{VideoLLM, VideoLLMOptions, VideoLLMResult, VideoTask, VideoTaskStatus};
use serde_json::json;
use std::future::Future;
use std::pin::Pin;
/// Runway model variants.
#[derive(Debug, Clone)]
pub enum RunwayVideoModel {
    /// gen4.5 (balanced everyday video generation)
    Gen45,
    /// aleph2.0 (precise video editing)
    Aleph20,
    /// Custom model name
    Custom(String),
}
impl RunwayVideoModel {
    /// Returns the API model identifier.
    fn as_str(&self) -> String {
        match self {
            RunwayVideoModel::Gen45 => "gen4.5".to_string(),
            RunwayVideoModel::Aleph20 => "aleph2.0".to_string(),
            RunwayVideoModel::Custom(name) => name.clone(),
        }
    }
}
impl From<RunwayVideoModel> for String {
    fn from(model: RunwayVideoModel) -> Self {
        model.as_str()
    }
}
/// Runway video generation client.
#[derive(Clone)]
pub struct RunwayVideo {
    api_key: String,
    model: RunwayVideoModel,
    base_url: String,
    client: reqwest::Client,
    default_options: VideoLLMOptions,
}
impl RunwayVideo {
    /// Creates a new Runway client with the given API key.
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            model: RunwayVideoModel::Gen45,
            base_url: "https://api.dev.runwayml.com/v1".to_string(),
            client: reqwest::Client::new(),
            default_options: VideoLLMOptions::default(),
        }
    }
    /// Sets the model variant.
    pub fn with_model(mut self, model: RunwayVideoModel) -> Self {
        self.model = model;
        self
    }
    /// Uses the Gen-4.5 model.
    pub fn gen45(self) -> Self {
        self.with_model(RunwayVideoModel::Gen45)
    }
    /// Uses the Aleph 2.0 model.
    pub fn aleph20(self) -> Self {
        self.with_model(RunwayVideoModel::Aleph20)
    }
    /// Sets a custom base URL.
    pub fn with_base_url(mut self, base_url: &str) -> Self {
        self.base_url = base_url.to_string();
        self
    }
    /// Sets default generation options.
    pub fn with_options(mut self, options: VideoLLMOptions) -> Self {
        self.default_options = options;
        self
    }
    /// Builds the JSON request body for the Runway text-to-video API.
    fn build_request_body(&self, prompt: &str, options: &VideoLLMOptions) -> serde_json::Value {
        let model_name: String = self.model.clone().into();
        let mut body = json!({
            "model": model_name,
            "promptText": prompt,
        });
        // Duration in seconds.
        if let Some(duration) = options.duration.or(self.default_options.duration) {
            body["duration"] = json!(duration as i32);
        }
        // Ratio (Runway uses "ratio" field, e.g. "1280:720").
        if let Some(ratio) = options
            .aspect_ratio
            .as_ref()
            .or(self.default_options.aspect_ratio.as_ref())
        {
            body["ratio"] = json!(ratio);
        }
        // Seed.
        if let Some(seed) = options.seed.or(self.default_options.seed) {
            body["seed"] = json!(seed);
        }
        // Reference image (image-to-video).
        if let Some(images) = options
            .reference_images
            .as_ref()
            .or(self.default_options.reference_images.as_ref())
        {
            if let Some(first) = images.first() {
                body["promptImage"] = json!(first);
            }
        }
        body
    }
    /// Submits the async generation request and returns the raw response.
    async fn submit_request(
        &self,
        prompt: &str,
        options: &VideoLLMOptions,
    ) -> Result<serde_json::Value> {
        let body = self.build_request_body(prompt, options);
        let response = self
            .client
            .post(format!("{}/text_to_video", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("X-Runway-Version", "2024-11-06")
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| LangHubError::LLMError(format!("Runway request error: {}", e)))?;
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LangHubError::LLMError(format!(
                "Runway API error ({}): {}",
                status, error_text
            )));
        }
        response
            .json()
            .await
            .map_err(|e| LangHubError::LLMError(format!("Runway JSON parse error: {}", e)))
    }
    /// Polls the task until it succeeds or fails.
    async fn poll_until_done(&self, task_id: &str) -> Result<VideoLLMResult> {
        let url = format!("{}/tasks/{}", self.base_url, task_id);
        for _ in 0..180 {
            let response = self
                .client
                .get(&url)
                .header("Authorization", format!("Bearer {}", self.api_key))
                .header("X-Runway-Version", "2024-11-06")
                .send()
                .await
                .map_err(|e| LangHubError::LLMError(format!("Runway poll error: {}", e)))?;
            let raw: serde_json::Value = response
                .json()
                .await
                .map_err(|e| LangHubError::LLMError(format!("Runway JSON parse error: {}", e)))?;
            let status = raw["status"].as_str().unwrap_or("");
            match status {
                "SUCCEEDED" => {
                    let video_url = raw["output"][0].as_str().map(|s| s.to_string());
                    return Ok(VideoLLMResult {
                        video_url,
                        video_base64: None,
                        file_path: None,
                        duration_seconds: None,
                        resolution: None,
                        raw_response: raw,
                    });
                }
                "FAILED" => {
                    let error = raw["failure"]
                        .as_str()
                        .unwrap_or("Runway task failed")
                        .to_string();
                    return Err(LangHubError::LLMError(error));
                }
                _ => {
                    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
                }
            }
        }
        Err(LangHubError::LLMError(
            "Runway task polling timeout".to_string(),
        ))
    }
}
impl VideoLLM for RunwayVideo {
    fn generate(
        &self,
        prompt: &str,
    ) -> Pin<Box<dyn Future<Output = Result<VideoLLMResult>> + Send + '_>> {
        let prompt = prompt.to_string();
        let options = self.default_options.clone();
        Box::pin(async move {
            let raw = self.submit_request(&prompt, &options).await?;
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
        options: VideoLLMOptions,
    ) -> Pin<Box<dyn Future<Output = Result<VideoLLMResult>> + Send + '_>> {
        let prompt = prompt.to_string();
        Box::pin(async move {
            let raw = self.submit_request(&prompt, &options).await?;
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
        options: VideoLLMOptions,
    ) -> Pin<Box<dyn Future<Output = Result<VideoTask>> + Send + '_>> {
        let prompt = prompt.to_string();
        Box::pin(async move {
            let raw = self.submit_request(&prompt, &options).await?;
            let task_id = raw["id"]
                .as_str()
                .ok_or_else(|| LangHubError::ParseError("Missing task id".to_string()))?
                .to_string();
            Ok(VideoTask {
                task_id,
                status: VideoTaskStatus::Pending,
                result: None,
                error: None,
            })
        })
    }
    fn poll_task(
        &self,
        task_id: &str,
    ) -> Pin<Box<dyn Future<Output = Result<VideoTask>> + Send + '_>> {
        let task_id = task_id.to_string();
        Box::pin(async move {
            let url = format!("{}/tasks/{}", self.base_url, task_id);
            let response = self
                .client
                .get(&url)
                .header("Authorization", format!("Bearer {}", self.api_key))
                .header("X-Runway-Version", "2024-11-06")
                .send()
                .await
                .map_err(|e| LangHubError::LLMError(format!("Runway poll error: {}", e)))?;
            let raw: serde_json::Value = response
                .json()
                .await
                .map_err(|e| LangHubError::LLMError(format!("Runway JSON parse error: {}", e)))?;
            let status_str = raw["status"].as_str().unwrap_or("");
            let (status, result, error) = match status_str {
                "SUCCEEDED" => {
                    let video_url = raw["output"][0].as_str().map(|s| s.to_string());
                    (
                        VideoTaskStatus::Succeeded,
                        Some(VideoLLMResult {
                            video_url,
                            video_base64: None,
                            file_path: None,
                            duration_seconds: None,
                            resolution: None,
                            raw_response: raw.clone(),
                        }),
                        None,
                    )
                }
                "FAILED" => (
                    VideoTaskStatus::Failed,
                    None,
                    Some(
                        raw["failure"]
                            .as_str()
                            .unwrap_or("Runway task failed")
                            .to_string(),
                    ),
                ),
                "RUNNING" => (VideoTaskStatus::Processing, None, None),
                "PENDING" => (VideoTaskStatus::Pending, None, None),
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
        "Runway".to_string()
    }
    fn max_duration(&self) -> Option<f32> {
        match self.model {
            RunwayVideoModel::Aleph20 => Some(30.0),
            _ => Some(10.0),
        }
    }
    fn supports_audio(&self) -> bool {
        false
    }
    fn supports_reference_images(&self) -> bool {
        true
    }
    fn supports_reference_videos(&self) -> bool {
        true
    }
}
