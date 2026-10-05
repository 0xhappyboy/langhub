use crate::audio::{AudioLLM, AudioLLMOptions, AudioLLMResult, AudioTask, AudioTaskStatus};
use crate::types::{LangHubError, LangHubResult};
use serde_json::json;
use std::future::Future;
use std::pin::Pin;
#[derive(Debug, Clone)]
pub enum ElevenLabsModel {
    ElevenV4,
    Custom(String),
}
impl ElevenLabsModel {
    fn as_str(&self) -> String {
        match self {
            ElevenLabsModel::ElevenV4 => "eleven-v4".to_string(),
            ElevenLabsModel::Custom(name) => name.clone(),
        }
    }
}
impl From<ElevenLabsModel> for String {
    fn from(model: ElevenLabsModel) -> Self {
        model.as_str()
    }
}
#[derive(Clone)]
pub struct ElevenLabs {
    api_key: String,
    model: ElevenLabsModel,
    base_url: String,
    client: reqwest::Client,
    default_options: AudioLLMOptions,
}
impl ElevenLabs {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            model: ElevenLabsModel::ElevenV4,
            base_url: "https://api.elevenlabs.io/v1".to_string(),
            client: reqwest::Client::new(),
            default_options: AudioLLMOptions::default(),
        }
    }
    pub fn with_model(mut self, model: ElevenLabsModel) -> Self {
        self.model = model;
        self
    }
    pub fn eleven_v4(self) -> Self {
        self.with_model(ElevenLabsModel::ElevenV4)
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
            "text": text,
            "model_id": model_name,
        });
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
        model_override: Option<&str>,
    ) -> LangHubResult<serde_json::Value> {
        let body = self.build_request_body(prompt, options, model_override);
        let voice = options
            .voice
            .as_ref()
            .or(self.default_options.voice.as_ref())
            .cloned()
            .unwrap_or_else(|| "21m00Tcm4TlvDq8ikWAM".to_string());
        let response = self
            .client
            .post(format!("{}/text-to-speech/{}", self.base_url, voice))
            .header("xi-api-key", &self.api_key)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| LangHubError::LLMError(format!("ElevenLabs request error: {}", e)))?;
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LangHubError::LLMError(format!(
                "ElevenLabs API error ({}): {}",
                status, error_text
            )));
        }
        let bytes = response
            .bytes()
            .await
            .map_err(|e| LangHubError::LLMError(format!("ElevenLabs read bytes error: {}", e)))?;
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
impl AudioLLM for ElevenLabs {
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
            let task_id = "elevenlabs-sync-task".to_string();
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
        "ElevenLabs".to_string()
    }
    fn supports_voice(&self) -> bool {
        true
    }
    fn supports_emotion(&self) -> bool {
        true
    }
}
