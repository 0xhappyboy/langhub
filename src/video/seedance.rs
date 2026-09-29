use crate::types::{LangHubError, Result};
use crate::video::{VideoLLM, VideoLLMOptions, VideoLLMResult, VideoTask, VideoTaskStatus};
use serde_json::json;
use std::future::Future;
use std::pin::Pin;
#[derive(Debug, Clone)]
pub enum SeedanceModel {
    Seedance25,
    Seedance20,
    Seedance20Fast,
    Seedance20Mini,
    Custom(String),
}
impl SeedanceModel {
    fn as_str(&self) -> String {
        match self {
            SeedanceModel::Seedance25 => "doubao-seedance-2-5".to_string(),
            SeedanceModel::Seedance20 => "doubao-seedance-2-0".to_string(),
            SeedanceModel::Seedance20Fast => "doubao-seedance-2-0-fast".to_string(),
            SeedanceModel::Seedance20Mini => "doubao-seedance-2-0-mini".to_string(),
            SeedanceModel::Custom(name) => name.clone(),
        }
    }
}
impl From<SeedanceModel> for String {
    fn from(model: SeedanceModel) -> Self {
        model.as_str()
    }
}
#[derive(Clone)]
pub struct Seedance {
    api_key: String,
    model: SeedanceModel,
    base_url: String,
    client: reqwest::Client,
    default_options: VideoLLMOptions,
}
impl Seedance {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            model: SeedanceModel::Seedance20,
            base_url: "https://ark.cn-beijing.volces.com/api/v3".to_string(),
            client: reqwest::Client::new(),
            default_options: VideoLLMOptions::default(),
        }
    }
    pub fn with_model(mut self, model: SeedanceModel) -> Self {
        self.model = model;
        self
    }
    pub fn seedance25(self) -> Self {
        self.with_model(SeedanceModel::Seedance25)
    }
    pub fn seedance20(self) -> Self {
        self.with_model(SeedanceModel::Seedance20)
    }
    pub fn seedance20_fast(self) -> Self {
        self.with_model(SeedanceModel::Seedance20Fast)
    }
    pub fn seedance20_mini(self) -> Self {
        self.with_model(SeedanceModel::Seedance20Mini)
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
            "content": [
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        });
        // Attach reference images if provided.
        if let Some(images) = options
            .reference_images
            .as_ref()
            .or(self.default_options.reference_images.as_ref())
        {
            if let Some(content_arr) = body["content"].as_array_mut() {
                for url in images {
                    content_arr.push(json!({
                        "type": "image_url",
                        "image_url": { "url": url }
                    }));
                }
            }
        }
        // Attach reference videos if provided.
        if let Some(videos) = options
            .reference_videos
            .as_ref()
            .or(self.default_options.reference_videos.as_ref())
        {
            if let Some(content_arr) = body["content"].as_array_mut() {
                for url in videos {
                    content_arr.push(json!({
                        "type": "video_url",
                        "video_url": { "url": url }
                    }));
                }
            }
        }
        // Optional parameters.
        if let Some(duration) = options.duration.or(self.default_options.duration) {
            body["duration"] = json!(duration);
        }
        if let Some(resolution) = options
            .resolution
            .as_ref()
            .or(self.default_options.resolution.as_ref())
        {
            body["resolution"] = json!(resolution);
        }
        if let Some(ratio) = options
            .aspect_ratio
            .as_ref()
            .or(self.default_options.aspect_ratio.as_ref())
        {
            body["ratio"] = json!(ratio);
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
        if let Some(fps) = options.fps.or(self.default_options.fps) {
            body["fps"] = json!(fps);
        }
        if let Some(neg) = options
            .negative_prompt
            .as_ref()
            .or(self.default_options.negative_prompt.as_ref())
        {
            body["negative_prompt"] = json!(neg);
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
            .post(format!("{}/contents/generations/tasks", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| LangHubError::LLMError(format!("Seedance request error: {}", e)))?;
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LangHubError::LLMError(format!(
                "Seedance API error ({}): {}",
                status, error_text
            )));
        }
        response
            .json()
            .await
            .map_err(|e| LangHubError::LLMError(format!("Seedance JSON parse error: {}", e)))
    }
    async fn poll_until_done(&self, task_id: &str) -> Result<VideoLLMResult> {
        let url = format!("{}/contents/generations/tasks/{}", self.base_url, task_id);
        for _ in 0..120 {
            let response = self
                .client
                .get(&url)
                .header("Authorization", format!("Bearer {}", self.api_key))
                .send()
                .await
                .map_err(|e| LangHubError::LLMError(format!("Seedance poll error: {}", e)))?;
            let raw: serde_json::Value = response
                .json()
                .await
                .map_err(|e| LangHubError::LLMError(format!("Seedance JSON parse error: {}", e)))?;
            let status = raw["status"].as_str().unwrap_or("");
            match status {
                "succeeded" => {
                    let video_url = raw["content"]["video_url"].as_str().map(|s| s.to_string());
                    let duration = raw["duration"].as_f64().map(|v| v as f32);
                    let resolution = raw["resolution"].as_str().map(|s| s.to_string());
                    return Ok(VideoLLMResult {
                        video_url,
                        video_base64: None,
                        file_path: None,
                        duration_seconds: duration,
                        resolution,
                        raw_response: raw,
                    });
                }
                "failed" => {
                    let error = raw["error"]["message"]
                        .as_str()
                        .unwrap_or("Seedance task failed")
                        .to_string();
                    return Err(LangHubError::LLMError(error));
                }
                _ => {
                    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
                }
            }
        }
        Err(LangHubError::LLMError(
            "Seedance task polling timeout".to_string(),
        ))
    }
}
impl VideoLLM for Seedance {
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
            let url = format!("{}/contents/generations/tasks/{}", self.base_url, task_id);
            let response = self
                .client
                .get(&url)
                .header("Authorization", format!("Bearer {}", self.api_key))
                .send()
                .await
                .map_err(|e| LangHubError::LLMError(format!("Seedance poll error: {}", e)))?;
            let raw: serde_json::Value = response
                .json()
                .await
                .map_err(|e| LangHubError::LLMError(format!("Seedance JSON parse error: {}", e)))?;
            let status_str = raw["status"].as_str().unwrap_or("");
            let (status, result, error) = match status_str {
                "succeeded" => {
                    let video_url = raw["content"]["video_url"].as_str().map(|s| s.to_string());
                    let duration = raw["duration"].as_f64().map(|v| v as f32);
                    let resolution = raw["resolution"].as_str().map(|s| s.to_string());
                    (
                        VideoTaskStatus::Succeeded,
                        Some(VideoLLMResult {
                            video_url,
                            video_base64: None,
                            file_path: None,
                            duration_seconds: duration,
                            resolution,
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
                            .unwrap_or("Seedance task failed")
                            .to_string(),
                    ),
                ),
                "processing" | "running" => (VideoTaskStatus::Processing, None, None),
                "cancelled" => (VideoTaskStatus::Cancelled, None, None),
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
        "ByteDance-Seedance".to_string()
    }
    fn max_duration(&self) -> Option<f32> {
        match self.model {
            SeedanceModel::Seedance25 => Some(30.0),
            _ => Some(12.0),
        }
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
