use crate::types::{LangHubError, Result};
use crate::video::{VideoLLM, VideoLLMOptions, VideoLLMResult, VideoTask, VideoTaskStatus};
use serde_json::json;
use std::future::Future;
use std::pin::Pin;
/// MiniMax H3 model variants.
#[derive(Debug, Clone)]
pub enum MiniMaxH3Model {
    /// MiniMax-Hailuo-H3 (native audio)
    H3,
    /// Custom model name
    Custom(String),
}
impl MiniMaxH3Model {
    /// Returns the API model identifier.
    fn as_str(&self) -> String {
        match self {
            MiniMaxH3Model::H3 => "MiniMax-Hailuo-H3".to_string(),
            MiniMaxH3Model::Custom(name) => name.clone(),
        }
    }
}
impl From<MiniMaxH3Model> for String {
    fn from(model: MiniMaxH3Model) -> Self {
        model.as_str()
    }
}
/// MiniMax H3 video generation client.
#[derive(Clone)]
pub struct MiniMaxH3 {
    api_key: String,
    group_id: Option<String>,
    model: MiniMaxH3Model,
    base_url: String,
    client: reqwest::Client,
    default_options: VideoLLMOptions,
}
impl MiniMaxH3 {
    /// Creates a new MiniMax H3 client with the given API key.
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            group_id: None,
            model: MiniMaxH3Model::H3,
            base_url: "https://api.minimax.chat/v1".to_string(),
            client: reqwest::Client::new(),
            default_options: VideoLLMOptions::default(),
        }
    }
    /// Sets the group ID (required by some MiniMax endpoints).
    pub fn with_group_id(mut self, group_id: String) -> Self {
        self.group_id = Some(group_id);
        self
    }
    /// Sets the model variant.
    pub fn with_model(mut self, model: MiniMaxH3Model) -> Self {
        self.model = model;
        self
    }
    /// Uses the H3 model.
    pub fn h3(self) -> Self {
        self.with_model(MiniMaxH3Model::H3)
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
    /// Builds the JSON request body for the MiniMax video generation API.
    fn build_request_body(&self, prompt: &str, options: &VideoLLMOptions) -> serde_json::Value {
        let model_name: String = self.model.clone().into();
        let mut body = json!({
            "model": model_name,
            "prompt": prompt,
        });
        // Resolution.
        if let Some(resolution) = options
            .resolution
            .as_ref()
            .or(self.default_options.resolution.as_ref())
        {
            body["resolution"] = json!(resolution);
        }
        // Duration.
        if let Some(duration) = options.duration.or(self.default_options.duration) {
            body["duration"] = json!(duration as i32);
        }
        // Reference image (image-to-video).
        if let Some(images) = options
            .reference_images
            .as_ref()
            .or(self.default_options.reference_images.as_ref())
        {
            if let Some(first) = images.first() {
                body["first_frame_image"] = json!(first);
            }
        }
        // Native audio toggle.
        if let Some(audio) = options
            .generate_audio
            .or(self.default_options.generate_audio)
        {
            body["audio"] = json!(audio);
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
        let url = if let Some(group) = &self.group_id {
            format!("{}/video_generation?GroupId={}", self.base_url, group)
        } else {
            format!("{}/video_generation", self.base_url)
        };
        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| LangHubError::LLMError(format!("MiniMax H3 request error: {}", e)))?;
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LangHubError::LLMError(format!(
                "MiniMax H3 API error ({}): {}",
                status, error_text
            )));
        }
        response
            .json()
            .await
            .map_err(|e| LangHubError::LLMError(format!("MiniMax H3 JSON parse error: {}", e)))
    }
    /// Polls the task until it succeeds or fails.
    async fn poll_until_done(&self, task_id: &str) -> Result<VideoLLMResult> {
        let url = format!(
            "{}/query/video_generation?task_id={}",
            self.base_url, task_id
        );
        for _ in 0..180 {
            let response = self
                .client
                .get(&url)
                .header("Authorization", format!("Bearer {}", self.api_key))
                .send()
                .await
                .map_err(|e| LangHubError::LLMError(format!("MiniMax H3 poll error: {}", e)))?;
            let raw: serde_json::Value = response.json().await.map_err(|e| {
                LangHubError::LLMError(format!("MiniMax H3 JSON parse error: {}", e))
            })?;
            let status = raw["status"].as_str().unwrap_or("");
            match status {
                "Success" => {
                    let file_id = raw["file_id"].as_str().map(|s| s.to_string());
                    return Ok(VideoLLMResult {
                        video_url: file_id,
                        video_base64: None,
                        file_path: None,
                        duration_seconds: None,
                        resolution: None,
                        raw_response: raw,
                    });
                }
                "Fail" => {
                    let error = raw["base_resp"]["status_msg"]
                        .as_str()
                        .unwrap_or("MiniMax H3 task failed")
                        .to_string();
                    return Err(LangHubError::LLMError(error));
                }
                _ => {
                    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
                }
            }
        }
        Err(LangHubError::LLMError(
            "MiniMax H3 task polling timeout".to_string(),
        ))
    }
}
impl VideoLLM for MiniMaxH3 {
    fn generate(
        &self,
        prompt: &str,
    ) -> Pin<Box<dyn Future<Output = Result<VideoLLMResult>> + Send + '_>> {
        let prompt = prompt.to_string();
        let options = self.default_options.clone();
        Box::pin(async move {
            let raw = self.submit_request(&prompt, &options).await?;
            let task_id = raw["task_id"]
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
            let task_id = raw["task_id"]
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
            let task_id = raw["task_id"]
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
            let url = format!(
                "{}/query/video_generation?task_id={}",
                self.base_url, task_id
            );
            let response = self
                .client
                .get(&url)
                .header("Authorization", format!("Bearer {}", self.api_key))
                .send()
                .await
                .map_err(|e| LangHubError::LLMError(format!("MiniMax H3 poll error: {}", e)))?;
            let raw: serde_json::Value = response.json().await.map_err(|e| {
                LangHubError::LLMError(format!("MiniMax H3 JSON parse error: {}", e))
            })?;
            let status_str = raw["status"].as_str().unwrap_or("");
            let (status, result, error) = match status_str {
                "Success" => {
                    let file_id = raw["file_id"].as_str().map(|s| s.to_string());
                    (
                        VideoTaskStatus::Succeeded,
                        Some(VideoLLMResult {
                            video_url: file_id,
                            video_base64: None,
                            file_path: None,
                            duration_seconds: None,
                            resolution: None,
                            raw_response: raw.clone(),
                        }),
                        None,
                    )
                }
                "Fail" => (
                    VideoTaskStatus::Failed,
                    None,
                    Some(
                        raw["base_resp"]["status_msg"]
                            .as_str()
                            .unwrap_or("MiniMax H3 task failed")
                            .to_string(),
                    ),
                ),
                "Processing" | "Preparing" => (VideoTaskStatus::Processing, None, None),
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
        "MiniMax-H3".to_string()
    }
    fn max_duration(&self) -> Option<f32> {
        Some(10.0)
    }
    fn supports_audio(&self) -> bool {
        true
    }
    fn supports_reference_images(&self) -> bool {
        true
    }
    fn supports_reference_videos(&self) -> bool {
        false
    }
}
