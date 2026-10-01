use crate::audio::{AudioLLM, AudioLLMOptions, AudioLLMResult, AudioTask, AudioTaskStatus};
use crate::types::{LangHubError, Result};
use serde_json::json;
use std::future::Future;
use std::pin::Pin;
#[derive(Debug, Clone)]
pub enum SeedAudioModel {
    SeedAudio10,
    Custom(String),
}
impl SeedAudioModel {
    fn as_str(&self) -> String {
        match self {
            SeedAudioModel::SeedAudio10 => "seed-audio-1-0".to_string(),
            SeedAudioModel::Custom(name) => name.clone(),
        }
    }
}
impl From<SeedAudioModel> for String {
    fn from(model: SeedAudioModel) -> Self {
        model.as_str()
    }
}
#[derive(Clone)]
pub struct SeedAudio {
    api_key: String,
    model: SeedAudioModel,
    base_url: String,
    client: reqwest::Client,
    default_options: AudioLLMOptions,
}
impl SeedAudio {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            model: SeedAudioModel::SeedAudio10,
            base_url: "https://ark.cn-beijing.volces.com/api/v3".to_string(),
            client: reqwest::Client::new(),
            default_options: AudioLLMOptions::default(),
        }
    }
    pub fn with_model(mut self, model: SeedAudioModel) -> Self {
        self.model = model;
        self
    }
    pub fn seed_audio_10(self) -> Self {
        self.with_model(SeedAudioModel::SeedAudio10)
    }
    pub fn with_base_url(mut self, base_url: &str) -> Self {
        self.base_url = base_url.to_string();
        self
    }
    pub fn with_options(mut self, options: AudioLLMOptions) -> Self {
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
        options: &AudioLLMOptions,
        model_override: Option<&str>,
    ) -> serde_json::Value {
        let model_name: String = self.resolve_model(model_override);
        let text = options
            .text
            .as_ref()
            .or(self.default_options.text.as_ref())
            .cloned()
            .unwrap_or_else(|| prompt.to_string());
        let mut body = json!({
            "model": model_name,
            "prompt": text,
        });
        if let Some(format) = options
            .format
            .as_ref()
            .or(self.default_options.format.as_ref())
        {
            body["format"] = json!(format);
        }
        if let Some(duration) = options
            .duration_seconds
            .or(self.default_options.duration_seconds)
        {
            body["duration"] = json!(duration);
        }
        if let Some(rate) = options.sample_rate.or(self.default_options.sample_rate) {
            body["sample_rate"] = json!(rate);
        }
        body
    }
    async fn submit_request(
        &self,
        prompt: &str,
        options: &AudioLLMOptions,
        model_override: Option<&str>,
    ) -> Result<serde_json::Value> {
        let body = self.build_request_body(prompt, options, model_override);
        let response = self
            .client
            .post(format!("{}/audio/generations", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| LangHubError::LLMError(format!("Seed Audio request error: {}", e)))?;
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LangHubError::LLMError(format!(
                "Seed Audio API error ({}): {}",
                status, error_text
            )));
        }
        response
            .json()
            .await
            .map_err(|e| LangHubError::LLMError(format!("Seed Audio JSON parse error: {}", e)))
    }
    fn raw_to_result(raw: &serde_json::Value) -> AudioLLMResult {
        let audio_url = raw["audio_url"].as_str().map(|s| s.to_string());
        let audio_base64 = raw["audio_data"].as_str().map(|s| s.to_string());
        let duration = raw["duration"].as_f64().map(|v| v as f32);
        let format = raw["format"].as_str().map(|s| s.to_string());
        AudioLLMResult {
            audio_url,
            audio_base64,
            file_path: None,
            duration_seconds: duration,
            format,
            raw_response: raw.clone(),
        }
    }
}
impl AudioLLM for SeedAudio {
    fn generate(
        &self,
        prompt: &str,
        model: Option<&str>,
    ) -> Pin<Box<dyn Future<Output = Result<AudioLLMResult>> + Send + '_>> {
        let prompt = prompt.to_string();
        let options = self.default_options.clone();
        let model_owned: Option<String> = model.map(|m| m.to_string());
        Box::pin(async move {
            let raw = self
                .submit_request(&prompt, &options, model_owned.as_deref())
                .await?;
            Ok(Self::raw_to_result(&raw))
        })
    }
    fn generate_with_options(
        &self,
        prompt: &str,
        options: AudioLLMOptions,
        model: Option<&str>,
    ) -> Pin<Box<dyn Future<Output = Result<AudioLLMResult>> + Send + '_>> {
        let prompt = prompt.to_string();
        let model_owned: Option<String> = model.map(|m| m.to_string());
        Box::pin(async move {
            let raw = self
                .submit_request(&prompt, &options, model_owned.as_deref())
                .await?;
            Ok(Self::raw_to_result(&raw))
        })
    }
    fn submit_task(
        &self,
        prompt: &str,
        options: AudioLLMOptions,
        model: Option<&str>,
    ) -> Pin<Box<dyn Future<Output = Result<AudioTask>> + Send + '_>> {
        let prompt = prompt.to_string();
        let model_owned: Option<String> = model.map(|m| m.to_string());
        Box::pin(async move {
            let raw = self
                .submit_request(&prompt, &options, model_owned.as_deref())
                .await?;
            let task_id = raw["id"]
                .as_str()
                .unwrap_or("seed-audio-sync-task")
                .to_string();
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
        "ByteDance-Seed-Audio".to_string()
    }
    fn max_duration(&self) -> Option<f32> {
        Some(120.0)
    }
}
