use crate::types::{LangHubError, LangHubResult};
use crate::video::{VideoLLM, VideoLLMOptions, VideoLLMResult, VideoTask, VideoTaskStatus};
use serde_json::json;
use std::future::Future;
use std::pin::Pin;
/// ByteDance Seedance video generation model enum
#[derive(Debug, Clone)]
pub enum SeedanceModel {
    Seedance10Pro,
    Seedance10ProFast,
    Seedance15Pro,
    Seedance20,
    Seedance20Fast,
    Seedance20Mini,
    Seedance25,
    Custom(String),
}
impl SeedanceModel {
    /// Return official ark model id string
    fn as_str(&self) -> String {
        match self {
            SeedanceModel::Seedance10Pro => "doubao-seedance-1-0-pro-250528".to_string(),
            SeedanceModel::Seedance10ProFast => "doubao-seedance-1-0-pro-fast-251015".to_string(),
            SeedanceModel::Seedance15Pro => "doubao-seedance-1-5-pro-251215".to_string(),
            SeedanceModel::Seedance20 => "doubao-seedance-2-0-260128".to_string(),
            SeedanceModel::Seedance20Fast => "doubao-seedance-2-0-fast-260128".to_string(),
            SeedanceModel::Seedance20Mini => "doubao-seedance-2-0-mini-260615".to_string(),
            SeedanceModel::Seedance25 => "doubao-seedance-2-5".to_string(),
            SeedanceModel::Custom(name) => name.clone(),
        }
    }
    /// Check whether this model supports native audio generation
    fn support_audio(&self) -> bool {
        match self {
            SeedanceModel::Seedance10Pro | SeedanceModel::Seedance10ProFast => false,
            SeedanceModel::Seedance15Pro
            | SeedanceModel::Seedance20
            | SeedanceModel::Seedance20Fast
            | SeedanceModel::Seedance20Mini
            | SeedanceModel::Seedance25 => true,
            SeedanceModel::Custom(_) => true,
        }
    }
    /// Get max allowed video duration for current model
    fn max_duration(&self) -> f32 {
        match self {
            SeedanceModel::Seedance10Pro | SeedanceModel::Seedance10ProFast => 12.0,
            SeedanceModel::Seedance15Pro
            | SeedanceModel::Seedance20
            | SeedanceModel::Seedance20Fast
            | SeedanceModel::Seedance20Mini => 15.0,
            SeedanceModel::Seedance25 => 30.0,
            SeedanceModel::Custom(_) => 15.0,
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
    /// Create new Seedance client, default model is Seedance20
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            model: SeedanceModel::Seedance20,
            base_url: "https://ark.cn-beijing.volces.com/api/v3".to_string(),
            client: reqwest::Client::new(),
            default_options: VideoLLMOptions::default(),
        }
    }
    /// Set model by SeedanceModel enum
    pub fn with_model(mut self, model: SeedanceModel) -> Self {
        self.model = model;
        self
    }
    /// Builder shortcut: Seedance 1.0 Pro
    pub fn seedance10_pro(self) -> Self {
        self.with_model(SeedanceModel::Seedance10Pro)
    }
    /// Builder shortcut: Seedance 1.0 Pro Fast
    pub fn seedance10_pro_fast(self) -> Self {
        self.with_model(SeedanceModel::Seedance10ProFast)
    }
    /// Builder shortcut: Seedance 1.5 Pro
    pub fn seedance15_pro(self) -> Self {
        self.with_model(SeedanceModel::Seedance15Pro)
    }
    /// Builder shortcut: Seedance 2.0
    pub fn seedance20(self) -> Self {
        self.with_model(SeedanceModel::Seedance20)
    }
    /// Builder shortcut: Seedance 2.0 Fast
    pub fn seedance20_fast(self) -> Self {
        self.with_model(SeedanceModel::Seedance20Fast)
    }
    /// Builder shortcut: Seedance 2.0 Mini
    pub fn seedance20_mini(self) -> Self {
        self.with_model(SeedanceModel::Seedance20Mini)
    }
    /// Builder shortcut: Seedance 2.5
    pub fn seedance25(self) -> Self {
        self.with_model(SeedanceModel::Seedance25)
    }
    /// Override api base url
    pub fn with_base_url(mut self, base_url: &str) -> Self {
        self.base_url = base_url.to_string();
        self
    }
    /// Set default video generation options
    pub fn with_options(mut self, options: VideoLLMOptions) -> Self {
        self.default_options = options;
        self
    }
    /// Build request json payload, automatically strip unsupported fields
    fn build_request_body(
        &self,
        prompt: &str,
        options: &VideoLLMOptions,
        model_override: Option<&str>,
    ) -> serde_json::Value {
        let model_name: String = match model_override {
            Some(m) => m.to_string(),
            None => self.model.as_str(),
        };
        let mut body = json!({
            "model": model_name,
            "content": [
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        });
        // Attach reference images
        if let Some(images) = options.reference_images.as_ref() {
            if let Some(content_arr) = body["content"].as_array_mut() {
                for url in images {
                    content_arr.push(json!({
                        "type": "image_url",
                        "image_url": {"url": url}
                    }));
                }
            }
        }
        // Attach reference videos
        if let Some(videos) = options.reference_videos.as_ref() {
            if let Some(content_arr) = body["content"].as_array_mut() {
                for url in videos {
                    content_arr.push(json!({
                        "type": "video_url",
                        "video_url": {"url": url}
                    }));
                }
            }
        }
        // Generation parameters
        let mut params = serde_json::Map::new();
        if let Some(duration) = options.duration {
            params.insert("duration".to_string(), json!(duration));
        }
        if let Some(resolution) = &options.resolution {
            params.insert("resolution".to_string(), json!(resolution));
        }
        if let Some(ratio) = &options.aspect_ratio {
            params.insert("ratio".to_string(), json!(ratio));
        }
        if let Some(seed) = options.seed {
            params.insert("seed".to_string(), json!(seed));
        }
        // Only add generate_audio if model supports audio
        if let Some(audio) = options.generate_audio {
            if self.model.support_audio() {
                params.insert("generate_audio".to_string(), json!(audio));
            }
        }
        if let Some(fps) = options.fps {
            params.insert("fps".to_string(), json!(fps));
        }
        if let Some(neg_prompt) = &options.negative_prompt {
            params.insert("negative_prompt".to_string(), json!(neg_prompt));
        }
        if !params.is_empty() {
            body["parameters"] = serde_json::Value::Object(params);
        }
        body
    }
    /// Submit generate request, return raw response json
    async fn submit_request(
        &self,
        prompt: &str,
        options: &VideoLLMOptions,
        model_override: Option<&str>,
    ) -> LangHubResult<serde_json::Value> {
        let body = self.build_request_body(prompt, options, model_override);
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
            let status = response.status().as_u16();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LangHubError::LLMError(format!(
                "Seedance API error (HTTP {}): {}",
                status, error_text
            )));
        }
        let json_val = response
            .json::<serde_json::Value>()
            .await
            .map_err(|e| LangHubError::LLMError(format!("Seedance parse json error: {}", e)))?;
        Ok(json_val)
    }
    /// Extract the produced video url from a raw Ark response.
    fn extract_video_url(raw: &serde_json::Value) -> Option<String> {
        raw["content"]["video_url"]
            .as_str()
            .or_else(|| raw["output"][0]["video_url"].as_str())
            .or_else(|| raw["output"]["video_url"].as_str())
            .or_else(|| raw["video_url"].as_str())
            .or_else(|| raw["data"]["video_url"].as_str())
            .or_else(|| raw["data"]["content"]["video_url"].as_str())
            .map(|s| s.to_string())
    }
    /// Extract duration (seconds) from a raw Ark response.
    fn extract_duration(raw: &serde_json::Value) -> Option<f32> {
        raw["content"]["duration"]
            .as_f64()
            .or_else(|| raw["output"][0]["duration"].as_f64())
            .or_else(|| raw["output"]["duration"].as_f64())
            .or_else(|| raw["duration"].as_f64())
            .map(|v| v as f32)
    }
    /// Extract resolution string from a raw Ark response.
    fn extract_resolution(raw: &serde_json::Value) -> Option<String> {
        raw["content"]["resolution"]
            .as_str()
            .or_else(|| raw["output"][0]["resolution"].as_str())
            .or_else(|| raw["output"]["resolution"].as_str())
            .or_else(|| raw["resolution"].as_str())
            .map(|s| s.to_string())
    }
    /// Poll task status until finished, parse video result
    async fn poll_until_done(&self, task_id: &str) -> LangHubResult<VideoLLMResult> {
        let url = format!("{}/contents/generations/tasks/{}", self.base_url, task_id);
        // Max poll 120 times, sleep 2s each
        for _ in 0..120 {
            let resp = self
                .client
                .get(&url)
                .header("Authorization", format!("Bearer {}", self.api_key))
                .send()
                .await
                .map_err(|e| LangHubError::LLMError(format!("Poll request error: {}", e)))?;
            if !resp.status().is_success() {
                let status = resp.status().as_u16();
                let err_txt = resp.text().await.unwrap_or_default();
                return Err(LangHubError::LLMError(format!(
                    "Poll api error HTTP {}: {}",
                    status, err_txt
                )));
            }
            let raw: serde_json::Value = resp
                .json()
                .await
                .map_err(|e| LangHubError::LLMError(format!("Poll json parse error: {}", e)))?;
            let status = raw["status"].as_str().unwrap_or("");
            match status {
                "success" | "succeeded" => {
                    let video_url = Self::extract_video_url(&raw);
                    let duration = Self::extract_duration(&raw);
                    let resolution = Self::extract_resolution(&raw);
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
                    let msg = raw["error"]["message"]
                        .as_str()
                        .unwrap_or("Unknown seedance task failure")
                        .to_string();
                    return Err(LangHubError::LLMError(format!(
                        "Seedance task failed: {}",
                        msg
                    )));
                }
                _ => {
                    // pending / running / queued / anything else -> keep waiting
                    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
                }
            }
        }
        Err(LangHubError::LLMError(
            "Seedance polling timeout after max retry".to_string(),
        ))
    }
}
impl VideoLLM for Seedance {
    /// Generate video with default options
    fn generate(
        &self,
        prompt: &str,
        model: Option<&str>,
    ) -> Pin<Box<dyn Future<Output = LangHubResult<VideoLLMResult>> + Send + '_>> {
        let prompt = prompt.to_string();
        let options = self.default_options.clone();
        let model_override = model.map(|s| s.to_string());
        Box::pin(async move {
            let res = self
                .submit_request(&prompt, &options, model_override.as_deref())
                .await?;
            let task_id = res["id"]
                .as_str()
                .ok_or_else(|| LangHubError::ParseError("Missing task id".to_string()))?;
            self.poll_until_done(task_id).await
        })
    }
    /// Generate video with custom VideoLLMOptions
    fn generate_with_options(
        &self,
        prompt: &str,
        options: VideoLLMOptions,
        model: Option<&str>,
    ) -> Pin<Box<dyn Future<Output = LangHubResult<VideoLLMResult>> + Send + '_>> {
        let prompt = prompt.to_string();
        let model_override = model.map(|s| s.to_string());
        Box::pin(async move {
            let res = self
                .submit_request(&prompt, &options, model_override.as_deref())
                .await?;
            let task_id = res["id"]
                .as_str()
                .ok_or_else(|| LangHubError::ParseError("Missing task id".to_string()))?;
            self.poll_until_done(task_id).await
        })
    }
    /// Submit async task without waiting, return task handle
    fn submit_task(
        &self,
        prompt: &str,
        options: VideoLLMOptions,
        model: Option<&str>,
    ) -> Pin<Box<dyn Future<Output = LangHubResult<VideoTask>> + Send + '_>> {
        let prompt = prompt.to_string();
        let model_override = model.map(|s| s.to_string());
        Box::pin(async move {
            let res = self
                .submit_request(&prompt, &options, model_override.as_deref())
                .await?;
            let task_id = res["id"]
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
    /// Poll task by task_id
    fn poll_task(
        &self,
        task_id: &str,
    ) -> Pin<Box<dyn Future<Output = LangHubResult<VideoTask>> + Send + '_>> {
        let tid = task_id.to_string();
        Box::pin(async move {
            let url = format!("{}/contents/generations/tasks/{}", self.base_url, tid);
            let resp = self
                .client
                .get(&url)
                .header("Authorization", format!("Bearer {}", self.api_key))
                .send()
                .await
                .map_err(|e| LangHubError::LLMError(format!("Poll task req error: {}", e)))?;
            if !resp.status().is_success() {
                let status = resp.status().as_u16();
                let err_txt = resp.text().await.unwrap_or_default();
                return Err(LangHubError::LLMError(format!(
                    "Poll HTTP {}: {}",
                    status, err_txt
                )));
            }
            let raw: serde_json::Value = resp
                .json()
                .await
                .map_err(|e| LangHubError::LLMError(format!("Poll json parse error: {}", e)))?;
            let status_str = raw["status"].as_str().unwrap_or("");
            // Map the provider status string to our unified task status.
            let task_status = match status_str {
                "success" | "succeeded" => VideoTaskStatus::Succeeded,
                "failed" => VideoTaskStatus::Failed,
                "cancelled" | "canceled" => VideoTaskStatus::Cancelled,
                // "queued" / "running" / anything else -> still in progress
                _ => VideoTaskStatus::Processing,
            };
            let mut result: Option<VideoLLMResult> = None;
            let mut error_msg: Option<String> = None;
            if task_status == VideoTaskStatus::Succeeded {
                let video_url = Seedance::extract_video_url(&raw);
                let duration = Seedance::extract_duration(&raw);
                let resolution = Seedance::extract_resolution(&raw);
                result = Some(VideoLLMResult {
                    video_url,
                    video_base64: None,
                    file_path: None,
                    duration_seconds: duration,
                    resolution,
                    raw_response: raw,
                });
            } else if task_status == VideoTaskStatus::Failed {
                error_msg = raw["error"]["message"].as_str().map(|s| s.to_string());
            }
            Ok(VideoTask {
                task_id: tid,
                status: task_status,
                result,
                error: error_msg,
            })
        })
    }
    /// Get current configured model name
    fn get_model_name(&self) -> String {
        self.model.as_str()
    }
    /// Get provider name
    fn get_provider_name(&self) -> String {
        "ByteDance-Seedance".to_string()
    }
    /// Get max duration limit of selected model
    fn max_duration(&self) -> Option<f32> {
        Some(self.model.max_duration())
    }
    /// Check if selected model supports native audio generation
    fn supports_audio(&self) -> bool {
        self.model.support_audio()
    }
    /// All Seedance models support reference image
    fn supports_reference_images(&self) -> bool {
        true
    }
    /// All Seedance models support reference video
    fn supports_reference_videos(&self) -> bool {
        true
    }
}
