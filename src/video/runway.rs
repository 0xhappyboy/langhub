use crate::types::{LangHubError, LangHubResult};
use crate::video::{VideoLLM, VideoLLMOptions, VideoLLMResult, VideoTask, VideoTaskStatus};
use serde_json::json;
use std::future::Future;
use std::pin::Pin;
#[derive(Debug, Clone)]
pub enum RunwayVideoModel {
    Gen45,
    Aleph20,
    Custom(String),
}
impl RunwayVideoModel {
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
#[derive(Clone)]
pub struct RunwayVideo {
    api_key: String,
    model: RunwayVideoModel,
    base_url: String,
    client: reqwest::Client,
    default_options: VideoLLMOptions,
}
impl RunwayVideo {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            model: RunwayVideoModel::Gen45,
            base_url: "https://api.dev.runwayml.com/v1".to_string(),
            client: reqwest::Client::new(),
            default_options: VideoLLMOptions::default(),
        }
    }
    pub fn with_model(mut self, model: RunwayVideoModel) -> Self {
        self.model = model;
        self
    }
    pub fn gen45(self) -> Self {
        self.with_model(RunwayVideoModel::Gen45)
    }
    pub fn aleph20(self) -> Self {
        self.with_model(RunwayVideoModel::Aleph20)
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
    async fn submit_request(
        &self,
        prompt: &str,
        options: &VideoLLMOptions,
        model_override: Option<&str>,
    ) -> LangHubResult<serde_json::Value> {
        let body = self.build_request_body(prompt, options, model_override);
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
    async fn poll_until_done(&self, task_id: &str) -> LangHubResult<VideoLLMResult> {
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
        model: Option<&str>,
    ) -> Pin<Box<dyn Future<Output = LangHubResult<VideoLLMResult>> + Send + '_>> {
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
        options: VideoLLMOptions,
        model: Option<&str>,
    ) -> Pin<Box<dyn Future<Output = LangHubResult<VideoLLMResult>> + Send + '_>> {
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
        options: VideoLLMOptions,
        model: Option<&str>,
    ) -> Pin<Box<dyn Future<Output = LangHubResult<VideoTask>> + Send + '_>> {
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
    ) -> Pin<Box<dyn Future<Output = LangHubResult<VideoTask>> + Send + '_>> {
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
