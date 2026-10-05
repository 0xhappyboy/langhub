use crate::audio::{AudioLLM, AudioLLMOptions, AudioLLMResult, AudioTask, AudioTaskStatus};
use crate::types::{LangHubError, LangHubResult};
use serde_json::json;
use std::future::Future;
use std::pin::Pin;
#[derive(Debug, Clone)]
pub enum StepAudioModel {
    StepAudio3Tts,
    StepAudio3Gen,
    StepAudio3Music,
    Custom(String),
}
impl StepAudioModel {
    fn as_str(&self) -> String {
        match self {
            StepAudioModel::StepAudio3Tts => "stepaudio3-tts".to_string(),
            StepAudioModel::StepAudio3Gen => "stepaudio3-gen".to_string(),
            StepAudioModel::StepAudio3Music => "stepaudio3-music".to_string(),
            StepAudioModel::Custom(name) => name.clone(),
        }
    }
}
impl From<StepAudioModel> for String {
    fn from(model: StepAudioModel) -> Self {
        model.as_str()
    }
}
#[derive(Clone)]
pub struct StepAudio {
    api_key: String,
    model: StepAudioModel,
    base_url: String,
    client: reqwest::Client,
    default_options: AudioLLMOptions,
}
impl StepAudio {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            model: StepAudioModel::StepAudio3Tts,
            base_url: "https://api.stepfun.com/v1".to_string(),
            client: reqwest::Client::new(),
            default_options: AudioLLMOptions::default(),
        }
    }
    pub fn with_model(mut self, model: StepAudioModel) -> Self {
        self.model = model;
        self
    }
    pub fn stepaudio3_tts(self) -> Self {
        self.with_model(StepAudioModel::StepAudio3Tts)
    }
    pub fn stepaudio3_gen(self) -> Self {
        self.with_model(StepAudioModel::StepAudio3Gen)
    }
    pub fn stepaudio3_music(self) -> Self {
        self.with_model(StepAudioModel::StepAudio3Music)
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
            "input": text,
        });
        if let Some(voice) = options
            .voice
            .as_ref()
            .or(self.default_options.voice.as_ref())
        {
            body["voice"] = json!(voice);
        }
        if let Some(emotion) = options
            .emotion
            .as_ref()
            .or(self.default_options.emotion.as_ref())
        {
            body["emotion"] = json!(emotion);
        }
        if let Some(format) = options
            .format
            .as_ref()
            .or(self.default_options.format.as_ref())
        {
            body["response_format"] = json!(format);
        }
        if let Some(duration) = options
            .duration_seconds
            .or(self.default_options.duration_seconds)
        {
            body["duration"] = json!(duration);
        }
        if let Some(lyrics) = options
            .lyrics
            .as_ref()
            .or(self.default_options.lyrics.as_ref())
        {
            body["lyrics"] = json!(lyrics);
        }
        if let Some(instrumental) = options.instrumental.or(self.default_options.instrumental) {
            body["instrumental"] = json!(instrumental);
        }
        body
    }
    async fn submit_request(
        &self,
        prompt: &str,
        options: &AudioLLMOptions,
        model_override: Option<&str>,
    ) -> LangHubResult<serde_json::Value> {
        let body = self.build_request_body(prompt, options, model_override);
        let response = self
            .client
            .post(format!("{}/audio/generations", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| LangHubError::LLMError(format!("StepAudio request error: {}", e)))?;
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LangHubError::LLMError(format!(
                "StepAudio API error ({}): {}",
                status, error_text
            )));
        }
        response
            .json()
            .await
            .map_err(|e| LangHubError::LLMError(format!("StepAudio JSON parse error: {}", e)))
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
impl AudioLLM for StepAudio {
    fn generate(
        &self,
        prompt: &str,
        model: Option<&str>,
    ) -> Pin<Box<dyn Future<Output = LangHubResult<AudioLLMResult>> + Send + '_>> {
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
    ) -> Pin<Box<dyn Future<Output = LangHubResult<AudioLLMResult>> + Send + '_>> {
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
    ) -> Pin<Box<dyn Future<Output = LangHubResult<AudioTask>> + Send + '_>> {
        let prompt = prompt.to_string();
        let model_owned: Option<String> = model.map(|m| m.to_string());
        Box::pin(async move {
            let raw = self
                .submit_request(&prompt, &options, model_owned.as_deref())
                .await?;
            let task_id = raw["id"]
                .as_str()
                .unwrap_or("stepaudio-sync-task")
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
    ) -> Pin<Box<dyn Future<Output = LangHubResult<AudioTask>> + Send + '_>> {
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
        "StepFun-StepAudio".to_string()
    }
    fn supports_voice(&self) -> bool {
        true
    }
    fn supports_emotion(&self) -> bool {
        true
    }
    fn supports_lyrics(&self) -> bool {
        true
    }
    fn supports_instrumental(&self) -> bool {
        true
    }
    fn max_duration(&self) -> Option<f32> {
        Some(180.0)
    }
}
