use crate::audio::{AudioLLM, AudioLLMOptions, AudioLLMResult, AudioTask, AudioTaskStatus};
use crate::types::{LangHubError, Result};
use serde_json::json;
use std::future::Future;
use std::pin::Pin;
#[derive(Debug, Clone)]
pub enum StableAudioModel {
    StableAudio25,
    Custom(String),
}
impl StableAudioModel {
    fn as_str(&self) -> String {
        match self {
            StableAudioModel::StableAudio25 => "stable-audio-2-5".to_string(),
            StableAudioModel::Custom(name) => name.clone(),
        }
    }
}
impl From<StableAudioModel> for String {
    fn from(model: StableAudioModel) -> Self {
        model.as_str()
    }
}
#[derive(Clone)]
pub struct StableAudio {
    api_key: String,
    model: StableAudioModel,
    base_url: String,
    client: reqwest::Client,
    default_options: AudioLLMOptions,
}
impl StableAudio {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            model: StableAudioModel::StableAudio25,
            base_url: "https://api.stability.ai/v2beta".to_string(),
            client: reqwest::Client::new(),
            default_options: AudioLLMOptions::default(),
        }
    }
    pub fn with_model(mut self, model: StableAudioModel) -> Self {
        self.model = model;
        self
    }
    pub fn stable_audio_25(self) -> Self {
        self.with_model(StableAudioModel::StableAudio25)
    }
    pub fn with_base_url(mut self, base_url: &str) -> Self {
        self.base_url = base_url.to_string();
        self
    }
    pub fn with_options(mut self, options: AudioLLMOptions) -> Self {
        self.default_options = options;
        self
    }
    fn build_request_body(&self, prompt: &str, options: &AudioLLMOptions) -> serde_json::Value {
        let text = options
            .text
            .as_ref()
            .or(self.default_options.text.as_ref())
            .cloned()
            .unwrap_or_else(|| prompt.to_string());
        let mut body = json!({
            "prompt": text,
        });
        if let Some(duration) = options
            .duration_seconds
            .or(self.default_options.duration_seconds)
        {
            body["duration"] = json!(duration);
        }
        if let Some(format) = options
            .format
            .as_ref()
            .or(self.default_options.format.as_ref())
        {
            body["output_format"] = json!(format);
        }
        body
    }
    async fn submit_request(
        &self,
        prompt: &str,
        options: &AudioLLMOptions,
    ) -> Result<serde_json::Value> {
        let body = self.build_request_body(prompt, options);
        let response = self
            .client
            .post(format!("{}/audio/generate", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| LangHubError::LLMError(format!("Stable Audio request error: {}", e)))?;
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LangHubError::LLMError(format!(
                "Stable Audio API error ({}): {}",
                status, error_text
            )));
        }
        let bytes = response
            .bytes()
            .await
            .map_err(|e| LangHubError::LLMError(format!("Stable Audio read bytes error: {}", e)))?;
        use base64::Engine;
        let engine = base64::engine::general_purpose::STANDARD;
        let audio_base64 = engine.encode(&bytes);
        Ok(json!({
            "audio_data": audio_base64,
            "format": options
                .format
                .as_ref()
                .or(self.default_options.format.as_ref())
                .cloned()
                .unwrap_or_else(|| "mp3".to_string()),
        }))
    }
    fn raw_to_result(raw: &serde_json::Value) -> AudioLLMResult {
        let audio_base64 = raw["audio_data"].as_str().map(|s| s.to_string());
        let format = raw["format"].as_str().map(|s| s.to_string());
        AudioLLMResult {
            audio_url: None,
            audio_base64,
            file_path: None,
            duration_seconds: None,
            format,
            raw_response: raw.clone(),
        }
    }
}
impl AudioLLM for StableAudio {
    fn generate(
        &self,
        prompt: &str,
    ) -> Pin<Box<dyn Future<Output = Result<AudioLLMResult>> + Send + '_>> {
        let prompt = prompt.to_string();
        let options = self.default_options.clone();
        Box::pin(async move {
            let raw = self.submit_request(&prompt, &options).await?;
            Ok(Self::raw_to_result(&raw))
        })
    }
    fn generate_with_options(
        &self,
        prompt: &str,
        options: AudioLLMOptions,
    ) -> Pin<Box<dyn Future<Output = Result<AudioLLMResult>> + Send + '_>> {
        let prompt = prompt.to_string();
        Box::pin(async move {
            let raw = self.submit_request(&prompt, &options).await?;
            Ok(Self::raw_to_result(&raw))
        })
    }
    fn submit_task(
        &self,
        prompt: &str,
        options: AudioLLMOptions,
    ) -> Pin<Box<dyn Future<Output = Result<AudioTask>> + Send + '_>> {
        let prompt = prompt.to_string();
        Box::pin(async move {
            let raw = self.submit_request(&prompt, &options).await?;
            let task_id = "stable-audio-sync-task".to_string();
            Ok(AudioTask {
                task_id,
                status: AudioTaskStatus::Succeeded,
                result: Some(Self::raw_to_result(&raw)),
                error: None,
            })
        })
    }
    fn poll_task(
        &self,
        task_id: &str,
    ) -> Pin<Box<dyn Future<Output = Result<AudioTask>> + Send + '_>> {
        let task_id = task_id.to_string();
        Box::pin(async move {
            Ok(AudioTask {
                task_id,
                status: AudioTaskStatus::Succeeded,
                result: None,
                error: None,
            })
        })
    }
    fn get_model_name(&self) -> String {
        self.model.as_str()
    }
    fn get_provider_name(&self) -> String {
        "StabilityAI-StableAudio".to_string()
    }
    fn supports_instrumental(&self) -> bool {
        true
    }
    fn max_duration(&self) -> Option<f32> {
        Some(180.0)
    }
}
