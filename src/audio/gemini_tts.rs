use crate::audio::{AudioLLM, AudioLLMOptions, AudioLLMResult, AudioTask, AudioTaskStatus};
use crate::types::{LangHubError, Result};
use serde_json::json;
use std::future::Future;
use std::pin::Pin;
#[derive(Debug, Clone)]
pub enum GeminiTtsModel {
    Gemini38FlashTts,
    Custom(String),
}
impl GeminiTtsModel {
    fn as_str(&self) -> String {
        match self {
            GeminiTtsModel::Gemini38FlashTts => "gemini-3.8-flash-tts".to_string(),
            GeminiTtsModel::Custom(name) => name.clone(),
        }
    }
}
impl From<GeminiTtsModel> for String {
    fn from(model: GeminiTtsModel) -> Self {
        model.as_str()
    }
}
#[derive(Clone)]
pub struct GeminiTts {
    api_key: String,
    model: GeminiTtsModel,
    base_url: String,
    client: reqwest::Client,
    default_options: AudioLLMOptions,
}
impl GeminiTts {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            model: GeminiTtsModel::Gemini38FlashTts,
            base_url: "https://generativelanguage.googleapis.com/v1beta".to_string(),
            client: reqwest::Client::new(),
            default_options: AudioLLMOptions::default(),
        }
    }
    pub fn with_model(mut self, model: GeminiTtsModel) -> Self {
        self.model = model;
        self
    }
    pub fn gemini_38_flash_tts(self) -> Self {
        self.with_model(GeminiTtsModel::Gemini38FlashTts)
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
    fn build_request_body(&self, prompt: &str, options: &AudioLLMOptions) -> serde_json::Value {
        let text = options
            .text
            .as_ref()
            .or(self.default_options.text.as_ref())
            .cloned()
            .unwrap_or_else(|| prompt.to_string());
        let mut generation_config = json!({
            "responseModalities": ["AUDIO"],
        });
        if let Some(voice) = options
            .voice
            .as_ref()
            .or(self.default_options.voice.as_ref())
        {
            generation_config["speechConfig"] = json!({
                "voiceConfig": {
                    "prebuiltVoiceConfig": {
                        "voiceName": voice
                    }
                }
            });
        }
        if let Some(language) = options
            .language
            .as_ref()
            .or(self.default_options.language.as_ref())
        {
            generation_config["speechConfig"]["languageCode"] = json!(language);
        }
        json!({
            "contents": [{
                "parts": [{ "text": text }]
            }],
            "generationConfig": generation_config,
        })
    }
    async fn submit_request(
        &self,
        prompt: &str,
        options: &AudioLLMOptions,
        model_override: Option<&str>,
    ) -> Result<serde_json::Value> {
        let body = self.build_request_body(prompt, options);
        let model_name: String = self.resolve_model(model_override);
        let url = format!(
            "{}/models/{}:generateContent?key={}",
            self.base_url, model_name, self.api_key
        );
        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| LangHubError::LLMError(format!("Gemini TTS request error: {}", e)))?;
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LangHubError::LLMError(format!(
                "Gemini TTS API error ({}): {}",
                status, error_text
            )));
        }
        response
            .json()
            .await
            .map_err(|e| LangHubError::LLMError(format!("Gemini TTS JSON parse error: {}", e)))
    }
    fn raw_to_result(raw: &serde_json::Value) -> AudioLLMResult {
        let audio_base64 = raw["candidates"][0]["content"]["parts"][0]["inlineData"]["data"]
            .as_str()
            .map(|s| s.to_string());
        AudioLLMResult {
            audio_url: None,
            audio_base64,
            file_path: None,
            duration_seconds: None,
            format: Some("wav".to_string()),
            raw_response: raw.clone(),
        }
    }
}
impl AudioLLM for GeminiTts {
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
            let task_id = raw["candidates"][0]["content"]["parts"][0]["inlineData"]["mimeType"]
                .as_str()
                .unwrap_or("gemini-tts-sync-task")
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
        "Google-Gemini-TTS".to_string()
    }
    fn supports_voice(&self) -> bool {
        true
    }
    fn supports_emotion(&self) -> bool {
        true
    }
}
