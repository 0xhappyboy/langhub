use crate::types::{LangHubError, Result};
use crate::video::{VideoLLM, VideoLLMOptions, VideoLLMResult, VideoTask, VideoTaskStatus};
use serde_json::json;
use std::future::Future;
use std::pin::Pin;
/// LTX model variants.
#[derive(Debug, Clone)]
pub enum LtxVideoModel {
    /// ltx-2.3-fast
    Ltx23Fast,
    /// ltx-2.3-pro
    Ltx23Pro,
    /// Custom model name
    Custom(String),
}
impl LtxVideoModel {
    /// Returns the API model identifier.
    fn as_str(&self) -> String {
        match self {
            LtxVideoModel::Ltx23Fast => "ltx-2.3-fast".to_string(),
            LtxVideoModel::Ltx23Pro => "ltx-2.3-pro".to_string(),
            LtxVideoModel::Custom(name) => name.clone(),
        }
    }
}
impl From<LtxVideoModel> for String {
    fn from(model: LtxVideoModel) -> Self {
        model.as_str()
    }
}
/// LTX video generation client.
#[derive(Clone)]
pub struct LtxVideo {
    api_key: String,
    model: LtxVideoModel,
    base_url: String,
    client: reqwest::Client,
    default_options: VideoLLMOptions,
}
impl LtxVideo {
    /// Creates a new LTX client with the given API key.
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            model: LtxVideoModel::Ltx23Fast,
            base_url: "https://api.ltx.video/v1".to_string(),
            client: reqwest::Client::new(),
            default_options: VideoLLMOptions::default(),
        }
    }
    /// Sets the model variant.
    pub fn with_model(mut self, model: LtxVideoModel) -> Self {
        self.model = model;
        self
    }
    /// Uses the LTX-2.3 Fast model.
    pub fn ltx23_fast(self) -> Self {
        self.with_model(LtxVideoModel::Ltx23Fast)
    }
    /// Uses the LTX-2.3 Pro model.
    pub fn ltx23_pro(self) -> Self {
        self.with_model(LtxVideoModel::Ltx23Pro)
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
    /// Builds the JSON request body for the LTX video generation API.
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
        if let Some(audio) = options
            .generate_audio
            .or(self.default_options.generate_audio)
        {
            body["generate_audio"] = json!(audio);
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
    /// Submits the async generation request and returns the raw response.
    async fn submit_request(
        &self,
        prompt: &str,
        options: &VideoLLMOptions,
    ) -> Result<serde_json::Value> {
        let body = self.build_request_body(prompt, options);
        let response = self
            .client
            .post(format!("{}/generate", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| LangHubError::LLMError(format!("LTX request error: {}", e)))?;
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LangHubError::LLMError(format!(
                "LTX API error ({}): {}",
                status, error_text
            )));
        }
        response
            .json()
            .await
            .map_err(|e| LangHubError::LLMError(format!("LTX JSON parse error: {}", e)))
    }
    /// Polls the task until it succeeds or fails.
    async fn poll_until_done(&self, task_id: &str) -> Result<VideoLLMResult> {
        let url = format!("{}/jobs/{}", self.base_url, task_id);
        for _ in 0..180 {
            let response = self
                .client
                .get(&url)
                .header("Authorization", format!("Bearer {}", self.api_key))
                .send()
                .await
                .map_err(|e| LangHubError::LLMError(format!("LTX poll error: {}", e)))?;
            let raw: serde_json::Value = response
                .json()
                .await
                .map_err(|e| LangHubError::LLMError(format!("LTX JSON parse error: {}", e)))?;
            let status = raw["status"].as_str().unwrap_or("");
            match status {
                "completed" => {
                    let video_url = raw["output"]["video_url"].as_str().map(|s| s.to_string());
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
                    let error = raw["error"]
                        .as_str()
                        .unwrap_or("LTX task failed")
                        .to_string();
                    return Err(LangHubError::LLMError(error));
                }
                _ => {
                    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
                }
            }
        }
        Err(LangHubError::LLMError(
            "LTX task polling timeout".to_string(),
        ))
    }
}
impl VideoLLM for LtxVideo {
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
            let url = format!("{}/jobs/{}", self.base_url, task_id);
            let response = self
                .client
                .get(&url)
                .header("Authorization", format!("Bearer {}", self.api_key))
                .send()
                .await
                .map_err(|e| LangHubError::LLMError(format!("LTX poll error: {}", e)))?;
            let raw: serde_json::Value = response
                .json()
                .await
                .map_err(|e| LangHubError::LLMError(format!("LTX JSON parse error: {}", e)))?;
            let status_str = raw["status"].as_str().unwrap_or("");
            let (status, result, error) = match status_str {
                "completed" => {
                    let video_url = raw["output"]["video_url"].as_str().map(|s| s.to_string());
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
                        raw["error"]
                            .as_str()
                            .unwrap_or("LTX task failed")
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
        "Lightricks-LTX".to_string()
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
        true
    }
}
