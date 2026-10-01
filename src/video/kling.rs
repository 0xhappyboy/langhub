use crate::types::{LangHubError, Result};
use crate::video::{VideoLLM, VideoLLMOptions, VideoLLMResult, VideoTask, VideoTaskStatus};
use serde_json::json;
use std::future::Future;
use std::pin::Pin;
#[derive(Debug, Clone)]
pub enum KlingVideoModel {
    Kling30,
    Kling30Omni,
    Kling40,
    Custom(String),
}
impl KlingVideoModel {
    fn as_str(&self) -> String {
        match self {
            KlingVideoModel::Kling30 => "kling-v3".to_string(),
            KlingVideoModel::Kling30Omni => "kling-v3-omni".to_string(),
            KlingVideoModel::Kling40 => "kling-v4".to_string(),
            KlingVideoModel::Custom(name) => name.clone(),
        }
    }
}
impl From<KlingVideoModel> for String {
    fn from(model: KlingVideoModel) -> Self {
        model.as_str()
    }
}
#[derive(Clone)]
pub struct KlingVideo {
    api_key: String,
    secret_key: Option<String>,
    model: KlingVideoModel,
    base_url: String,
    client: reqwest::Client,
    default_options: VideoLLMOptions,
}
impl KlingVideo {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            secret_key: None,
            model: KlingVideoModel::Kling30,
            base_url: "https://api.klingai.com".to_string(),
            client: reqwest::Client::new(),
            default_options: VideoLLMOptions::default(),
        }
    }
    pub fn with_secret_key(mut self, secret_key: String) -> Self {
        self.secret_key = Some(secret_key);
        self
    }
    pub fn with_model(mut self, model: KlingVideoModel) -> Self {
        self.model = model;
        self
    }
    pub fn kling30(self) -> Self {
        self.with_model(KlingVideoModel::Kling30)
    }
    pub fn kling30_omni(self) -> Self {
        self.with_model(KlingVideoModel::Kling30Omni)
    }
    pub fn kling40(self) -> Self {
        self.with_model(KlingVideoModel::Kling40)
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
            "model_name": model_name,
            "prompt": prompt,
        });
        // Duration: Kling expects integer seconds as string.
        if let Some(duration) = options.duration.or(self.default_options.duration) {
            body["duration"] = json!(format!("{}", duration as i32));
        }
        // Aspect ratio.
        if let Some(ratio) = options
            .aspect_ratio
            .as_ref()
            .or(self.default_options.aspect_ratio.as_ref())
        {
            body["aspect_ratio"] = json!(ratio);
        }
        // Mode / resolution (std / pro).
        if let Some(resolution) = options
            .resolution
            .as_ref()
            .or(self.default_options.resolution.as_ref())
        {
            body["mode"] = json!(resolution);
        }
        // Reference image (image-to-video).
        if let Some(images) = options
            .reference_images
            .as_ref()
            .or(self.default_options.reference_images.as_ref())
        {
            if let Some(first) = images.first() {
                body["image"] = json!(first);
            }
        }
        // Negative prompt.
        if let Some(neg) = options
            .negative_prompt
            .as_ref()
            .or(self.default_options.negative_prompt.as_ref())
        {
            body["negative_prompt"] = json!(neg);
        }
        // CFG scale / seed if provided.
        if let Some(seed) = options.seed.or(self.default_options.seed) {
            body["seed"] = json!(seed);
        }
        body
    }
    async fn submit_request(
        &self,
        prompt: &str,
        options: &VideoLLMOptions,
        model_override: Option<&str>,
    ) -> Result<serde_json::Value> {
        let body = self.build_request_body(prompt, options, model_override);
        let response = self
            .client
            .post(format!("{}/v1/videos/text2video", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| LangHubError::LLMError(format!("Kling request error: {}", e)))?;
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LangHubError::LLMError(format!(
                "Kling API error ({}): {}",
                status, error_text
            )));
        }
        response
            .json()
            .await
            .map_err(|e| LangHubError::LLMError(format!("Kling JSON parse error: {}", e)))
    }
    async fn poll_until_done(&self, task_id: &str) -> Result<VideoLLMResult> {
        let url = format!("{}/v1/videos/text2video/{}", self.base_url, task_id);
        for _ in 0..180 {
            let response = self
                .client
                .get(&url)
                .header("Authorization", format!("Bearer {}", self.api_key))
                .send()
                .await
                .map_err(|e| LangHubError::LLMError(format!("Kling poll error: {}", e)))?;
            let raw: serde_json::Value = response
                .json()
                .await
                .map_err(|e| LangHubError::LLMError(format!("Kling JSON parse error: {}", e)))?;
            let status = raw["data"]["task_status"].as_str().unwrap_or("");
            match status {
                "succeed" => {
                    let video_url = raw["data"]["task_result"]["videos"][0]["url"]
                        .as_str()
                        .map(|s| s.to_string());
                    let duration = raw["data"]["task_result"]["videos"][0]["duration"]
                        .as_str()
                        .and_then(|s| s.parse::<f32>().ok());
                    return Ok(VideoLLMResult {
                        video_url,
                        video_base64: None,
                        file_path: None,
                        duration_seconds: duration,
                        resolution: None,
                        raw_response: raw,
                    });
                }
                "failed" => {
                    let error = raw["data"]["task_status_msg"]
                        .as_str()
                        .unwrap_or("Kling task failed")
                        .to_string();
                    return Err(LangHubError::LLMError(error));
                }
                _ => {
                    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
                }
            }
        }
        Err(LangHubError::LLMError(
            "Kling task polling timeout".to_string(),
        ))
    }
}
impl VideoLLM for KlingVideo {
    fn generate(
        &self,
        prompt: &str,
        model: Option<&str>,
    ) -> Pin<Box<dyn Future<Output = Result<VideoLLMResult>> + Send + '_>> {
        let prompt = prompt.to_string();
        let options = self.default_options.clone();
        let model_owned: Option<String> = model.map(|m| m.to_string());
        Box::pin(async move {
            let raw = self
                .submit_request(&prompt, &options, model_owned.as_deref())
                .await?;
            let task_id = raw["data"]["task_id"]
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
    ) -> Pin<Box<dyn Future<Output = Result<VideoLLMResult>> + Send + '_>> {
        let prompt = prompt.to_string();
        let model_owned: Option<String> = model.map(|m| m.to_string());
        Box::pin(async move {
            let raw = self
                .submit_request(&prompt, &options, model_owned.as_deref())
                .await?;
            let task_id = raw["data"]["task_id"]
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
    ) -> Pin<Box<dyn Future<Output = Result<VideoTask>> + Send + '_>> {
        let prompt = prompt.to_string();
        let model_owned: Option<String> = model.map(|m| m.to_string());
        Box::pin(async move {
            let raw = self
                .submit_request(&prompt, &options, model_owned.as_deref())
                .await?;
            let task_id = raw["data"]["task_id"]
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
            let url = format!("{}/v1/videos/text2video/{}", self.base_url, task_id);
            let response = self
                .client
                .get(&url)
                .header("Authorization", format!("Bearer {}", self.api_key))
                .send()
                .await
                .map_err(|e| LangHubError::LLMError(format!("Kling poll error: {}", e)))?;
            let raw: serde_json::Value = response
                .json()
                .await
                .map_err(|e| LangHubError::LLMError(format!("Kling JSON parse error: {}", e)))?;
            let status_str = raw["data"]["task_status"].as_str().unwrap_or("");
            let (status, result, error) = match status_str {
                "succeed" => {
                    let video_url = raw["data"]["task_result"]["videos"][0]["url"]
                        .as_str()
                        .map(|s| s.to_string());
                    let duration = raw["data"]["task_result"]["videos"][0]["duration"]
                        .as_str()
                        .and_then(|s| s.parse::<f32>().ok());
                    (
                        VideoTaskStatus::Succeeded,
                        Some(VideoLLMResult {
                            video_url,
                            video_base64: None,
                            file_path: None,
                            duration_seconds: duration,
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
                        raw["data"]["task_status_msg"]
                            .as_str()
                            .unwrap_or("Kling task failed")
                            .to_string(),
                    ),
                ),
                "processing" => (VideoTaskStatus::Processing, None, None),
                "submitted" => (VideoTaskStatus::Pending, None, None),
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
        "Kuaishou-Kling".to_string()
    }
    fn max_duration(&self) -> Option<f32> {
        match self.model {
            KlingVideoModel::Kling40 => Some(30.0),
            _ => Some(15.0),
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
