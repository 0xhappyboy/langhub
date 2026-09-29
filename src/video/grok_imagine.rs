use crate::types::{LangHubError, Result};
use crate::video::{VideoLLM, VideoLLMOptions, VideoLLMResult, VideoTask, VideoTaskStatus};
use serde_json::json;
use std::future::Future;
use std::pin::Pin;
#[derive(Debug, Clone)]
pub enum GrokImagineModel {
    GrokImagine15,
    GrokImagine10,
    Custom(String),
}
impl GrokImagineModel {
    fn as_str(&self) -> String {
        match self {
            GrokImagineModel::GrokImagine15 => "grok-imagine-video-1.5".to_string(),
            GrokImagineModel::GrokImagine10 => "grok-imagine-video-1.0".to_string(),
            GrokImagineModel::Custom(name) => name.clone(),
        }
    }
}
impl From<GrokImagineModel> for String {
    fn from(model: GrokImagineModel) -> Self {
        model.as_str()
    }
}
#[derive(Clone)]
pub struct GrokImagine {
    api_key: String,
    model: GrokImagineModel,
    base_url: String,
    client: reqwest::Client,
    default_options: VideoLLMOptions,
}
impl GrokImagine {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            model: GrokImagineModel::GrokImagine15,
            base_url: "https://api.x.ai/v1".to_string(),
            client: reqwest::Client::new(),
            default_options: VideoLLMOptions::default(),
        }
    }
    pub fn with_model(mut self, model: GrokImagineModel) -> Self {
        self.model = model;
        self
    }
    pub fn grok_imagine15(self) -> Self {
        self.with_model(GrokImagineModel::GrokImagine15)
    }
    pub fn grok_imagine10(self) -> Self {
        self.with_model(GrokImagineModel::GrokImagine10)
    }
    pub fn with_base_url(mut self, base_url: &str) -> Self {
        self.base_url = base_url.to_string();
        self
    }
    pub fn with_options(mut self, options: VideoLLMOptions) -> Self {
        self.default_options = options;
        self
    }
    fn build_request_body(&self, prompt: &str, options: &VideoLLMOptions) -> serde_json::Value {
        let model_name: String = self.model.clone().into();
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
            body["duration"] = json!(duration as i32);
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
            if let Some(first) = images.first() {
                body["image_url"] = json!(first);
            }
        }
        body
    }
    async fn submit_request(
        &self,
        prompt: &str,
        options: &VideoLLMOptions,
    ) -> Result<serde_json::Value> {
        let body = self.build_request_body(prompt, options);
        let response = self
            .client
            .post(format!("{}/video/generations", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| LangHubError::LLMError(format!("Grok Imagine request error: {}", e)))?;
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LangHubError::LLMError(format!(
                "Grok Imagine API error ({}): {}",
                status, error_text
            )));
        }
        response
            .json()
            .await
            .map_err(|e| LangHubError::LLMError(format!("Grok Imagine JSON parse error: {}", e)))
    }
    async fn poll_until_done(&self, task_id: &str) -> Result<VideoLLMResult> {
        let url = format!("{}/video/generations/{}", self.base_url, task_id);
        for _ in 0..180 {
            let response = self
                .client
                .get(&url)
                .header("Authorization", format!("Bearer {}", self.api_key))
                .send()
                .await
                .map_err(|e| LangHubError::LLMError(format!("Grok Imagine poll error: {}", e)))?;
            let raw: serde_json::Value = response.json().await.map_err(|e| {
                LangHubError::LLMError(format!("Grok Imagine JSON parse error: {}", e))
            })?;
            let status = raw["status"].as_str().unwrap_or("");
            match status {
                "succeeded" => {
                    let video_url = raw["video_url"].as_str().map(|s| s.to_string());
                    return Ok(VideoLLMResult {
                        video_url,
                        video_base64: None,
                        file_path: None,
                        duration_seconds: None,
                        resolution: None,
                        raw_response: raw,
                    });
                }
                "failed" => {
                    let error = raw["error"]["message"]
                        .as_str()
                        .unwrap_or("Grok Imagine task failed")
                        .to_string();
                    return Err(LangHubError::LLMError(error));
                }
                _ => {
                    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
                }
            }
        }
        Err(LangHubError::LLMError(
            "Grok Imagine task polling timeout".to_string(),
        ))
    }
}
impl VideoLLM for GrokImagine {
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
            let url = format!("{}/video/generations/{}", self.base_url, task_id);
            let response = self
                .client
                .get(&url)
                .header("Authorization", format!("Bearer {}", self.api_key))
                .send()
                .await
                .map_err(|e| LangHubError::LLMError(format!("Grok Imagine poll error: {}", e)))?;
            let raw: serde_json::Value = response.json().await.map_err(|e| {
                LangHubError::LLMError(format!("Grok Imagine JSON parse error: {}", e))
            })?;
            let status_str = raw["status"].as_str().unwrap_or("");
            let (status, result, error) = match status_str {
                "succeeded" => {
                    let video_url = raw["video_url"].as_str().map(|s| s.to_string());
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
                "failed" => (
                    VideoTaskStatus::Failed,
                    None,
                    Some(
                        raw["error"]["message"]
                            .as_str()
                            .unwrap_or("Grok Imagine task failed")
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
        "xAI-Grok-Imagine".to_string()
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
        true
    }
}
