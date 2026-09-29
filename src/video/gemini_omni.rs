use crate::types::{LangHubError, Result};
use crate::video::{VideoLLM, VideoLLMOptions, VideoLLMResult, VideoTask, VideoTaskStatus};
use serde_json::json;
use std::future::Future;
use std::pin::Pin;
#[derive(Debug, Clone)]
pub enum GeminiOmniFlashModel {
    GeminiOmniFlash11,
    Custom(String),
}
impl GeminiOmniFlashModel {
    fn as_str(&self) -> String {
        match self {
            GeminiOmniFlashModel::GeminiOmniFlash11 => "gemini-omni-flash-1.1".to_string(),
            GeminiOmniFlashModel::Custom(name) => name.clone(),
        }
    }
}
impl From<GeminiOmniFlashModel> for String {
    fn from(model: GeminiOmniFlashModel) -> Self {
        model.as_str()
    }
}
#[derive(Clone)]
pub struct GeminiOmniFlash {
    api_key: String,
    model: GeminiOmniFlashModel,
    base_url: String,
    client: reqwest::Client,
    default_options: VideoLLMOptions,
}
impl GeminiOmniFlash {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            model: GeminiOmniFlashModel::GeminiOmniFlash11,
            base_url: "https://generativelanguage.googleapis.com/v1beta".to_string(),
            client: reqwest::Client::new(),
            default_options: VideoLLMOptions::default(),
        }
    }
    pub fn with_model(mut self, model: GeminiOmniFlashModel) -> Self {
        self.model = model;
        self
    }
    pub fn gemini_omni_flash11(self) -> Self {
        self.with_model(GeminiOmniFlashModel::GeminiOmniFlash11)
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
        let mut instance = json!({
            "prompt": prompt
        });
        // Reference image (image-to-video).
        if let Some(images) = options
            .reference_images
            .as_ref()
            .or(self.default_options.reference_images.as_ref())
        {
            if let Some(first) = images.first() {
                instance["image"] = json!({
                    "imageUri": first
                });
            }
        }
        // Parameters.
        let mut parameters = json!({});
        if let Some(duration) = options.duration.or(self.default_options.duration) {
            parameters["durationSeconds"] = json!(duration as i32);
        }
        if let Some(ratio) = options
            .aspect_ratio
            .as_ref()
            .or(self.default_options.aspect_ratio.as_ref())
        {
            parameters["aspectRatio"] = json!(ratio);
        }
        if let Some(resolution) = options
            .resolution
            .as_ref()
            .or(self.default_options.resolution.as_ref())
        {
            parameters["resolution"] = json!(resolution);
        }
        if let Some(audio) = options
            .generate_audio
            .or(self.default_options.generate_audio)
        {
            parameters["generateAudio"] = json!(audio);
        }
        json!({
            "instances": [instance],
            "parameters": parameters,
        })
    }
    async fn submit_request(
        &self,
        prompt: &str,
        options: &VideoLLMOptions,
    ) -> Result<serde_json::Value> {
        let body = self.build_request_body(prompt, options);
        let model_name: String = self.model.clone().into();
        let url = format!(
            "{}/models/{}:predictLongRunning?key={}",
            self.base_url, model_name, self.api_key
        );
        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                LangHubError::LLMError(format!("Gemini Omni Flash request error: {}", e))
            })?;
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LangHubError::LLMError(format!(
                "Gemini Omni Flash API error ({}): {}",
                status, error_text
            )));
        }
        response.json().await.map_err(|e| {
            LangHubError::LLMError(format!("Gemini Omni Flash JSON parse error: {}", e))
        })
    }
    async fn poll_until_done(&self, operation_name: &str) -> Result<VideoLLMResult> {
        let url = format!("{}/{}?key={}", self.base_url, operation_name, self.api_key);
        for _ in 0..180 {
            let response = self.client.get(&url).send().await.map_err(|e| {
                LangHubError::LLMError(format!("Gemini Omni Flash poll error: {}", e))
            })?;
            let raw: serde_json::Value = response.json().await.map_err(|e| {
                LangHubError::LLMError(format!("Gemini Omni Flash JSON parse error: {}", e))
            })?;
            let done = raw["done"].as_bool().unwrap_or(false);
            if done {
                if let Some(err) = raw.get("error") {
                    let msg = err["message"]
                        .as_str()
                        .unwrap_or("Gemini Omni Flash operation failed")
                        .to_string();
                    return Err(LangHubError::LLMError(msg));
                }
                let video_uri =
                    raw["response"]["generateVideoResponse"]["generatedSamples"][0]["video"]["uri"]
                        .as_str()
                        .map(|s| s.to_string());
                return Ok(VideoLLMResult {
                    video_url: video_uri,
                    video_base64: None,
                    file_path: None,
                    duration_seconds: None,
                    resolution: None,
                    raw_response: raw,
                });
            }
            tokio::time::sleep(tokio::time::Duration::from_secs(3)).await;
        }
        Err(LangHubError::LLMError(
            "Gemini Omni Flash operation polling timeout".to_string(),
        ))
    }
}
impl VideoLLM for GeminiOmniFlash {
    fn generate(
        &self,
        prompt: &str,
    ) -> Pin<Box<dyn Future<Output = Result<VideoLLMResult>> + Send + '_>> {
        let prompt = prompt.to_string();
        let options = self.default_options.clone();
        Box::pin(async move {
            let raw = self.submit_request(&prompt, &options).await?;
            let op_name = raw["name"]
                .as_str()
                .ok_or_else(|| LangHubError::ParseError("Missing operation name".to_string()))?
                .to_string();
            self.poll_until_done(&op_name).await
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
            let op_name = raw["name"]
                .as_str()
                .ok_or_else(|| LangHubError::ParseError("Missing operation name".to_string()))?
                .to_string();
            self.poll_until_done(&op_name).await
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
            let op_name = raw["name"]
                .as_str()
                .ok_or_else(|| LangHubError::ParseError("Missing operation name".to_string()))?
                .to_string();
            Ok(VideoTask {
                task_id: op_name,
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
            let url = format!("{}/{}?key={}", self.base_url, task_id, self.api_key);
            let response = self.client.get(&url).send().await.map_err(|e| {
                LangHubError::LLMError(format!("Gemini Omni Flash poll error: {}", e))
            })?;
            let raw: serde_json::Value = response.json().await.map_err(|e| {
                LangHubError::LLMError(format!("Gemini Omni Flash JSON parse error: {}", e))
            })?;
            let done = raw["done"].as_bool().unwrap_or(false);
            if done {
                if let Some(err) = raw.get("error") {
                    let msg = err["message"]
                        .as_str()
                        .unwrap_or("Gemini Omni Flash operation failed")
                        .to_string();
                    return Ok(VideoTask {
                        task_id,
                        status: VideoTaskStatus::Failed,
                        result: None,
                        error: Some(msg),
                    });
                }
                let video_uri =
                    raw["response"]["generateVideoResponse"]["generatedSamples"][0]["video"]["uri"]
                        .as_str()
                        .map(|s| s.to_string());
                return Ok(VideoTask {
                    task_id,
                    status: VideoTaskStatus::Succeeded,
                    result: Some(VideoLLMResult {
                        video_url: video_uri,
                        video_base64: None,
                        file_path: None,
                        duration_seconds: None,
                        resolution: None,
                        raw_response: raw,
                    }),
                    error: None,
                });
            }
            Ok(VideoTask {
                task_id,
                status: VideoTaskStatus::Processing,
                result: None,
                error: None,
            })
        })
    }
    fn get_model_name(&self) -> String {
        self.model.as_str()
    }
    fn get_provider_name(&self) -> String {
        "Google-Gemini-Omni-Flash".to_string()
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
