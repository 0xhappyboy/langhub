use crate::audio::{AudioLLM, AudioLLMOptions, AudioLLMResult, AudioTask, AudioTaskStatus};
use crate::types::{LangHubError, Result};
use serde_json::json;
use std::future::Future;
use std::pin::Pin;
#[derive(Debug, Clone)]
pub enum QwenTtsModel {
    QwenAudio31Tts,
    QwenAudio31TtsNext,
    Custom(String),
}
impl QwenTtsModel {
    fn as_str(&self) -> String {
        match self {
            QwenTtsModel::QwenAudio31Tts => "qwen-audio-3.1-tts".to_string(),
            QwenTtsModel::QwenAudio31TtsNext => "qwen-audio-3.1-tts-next".to_string(),
            QwenTtsModel::Custom(name) => name.clone(),
        }
    }
}
impl From<QwenTtsModel> for String {
    fn from(model: QwenTtsModel) -> Self {
        model.as_str()
    }
}
#[derive(Clone)]
pub struct QwenTts {
    api_key: String,
    model: QwenTtsModel,
    base_url: String,
    client: reqwest::Client,
    default_options: AudioLLMOptions,
}
impl QwenTts {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            model: QwenTtsModel::QwenAudio31Tts,
            base_url: "https://dashscope.aliyuncs.com/api/v1".to_string(),
            client: reqwest::Client::new(),
            default_options: AudioLLMOptions::default(),
        }
    }
    pub fn with_model(mut self, model: QwenTtsModel) -> Self {
        self.model = model;
        self
    }
    pub fn qwen_audio_31_tts(self) -> Self {
        self.with_model(QwenTtsModel::QwenAudio31Tts)
    }
    pub fn qwen_audio_31_tts_next(self) -> Self {
        self.with_model(QwenTtsModel::QwenAudio31TtsNext)
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
        let mut input = json!({
            "text": text,
        });
        if let Some(voice) = options
            .voice
            .as_ref()
            .or(self.default_options.voice.as_ref())
        {
            input["voice"] = json!(voice);
        }
        if let Some(language) = options
            .language
            .as_ref()
            .or(self.default_options.language.as_ref())
        {
            input["language_type"] = json!(language);
        }
        let mut parameters = json!({});
        if let Some(speed) = options.speed.or(self.default_options.speed) {
            parameters["rate"] = json!(speed);
        }
        if let Some(emotion) = options
            .emotion
            .as_ref()
            .or(self.default_options.emotion.as_ref())
        {
            parameters["emotion"] = json!(emotion);
        }
        if let Some(format) = options
            .format
            .as_ref()
            .or(self.default_options.format.as_ref())
        {
            parameters["format"] = json!(format);
        }
        if let Some(rate) = options.sample_rate.or(self.default_options.sample_rate) {
            parameters["sample_rate"] = json!(rate);
        }
        json!({
            "model": model_name,
            "input": input,
            "parameters": parameters,
        })
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
            .post(format!(
                "{}/services/aigc/multimodal-generation/generation",
                self.base_url
            ))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| LangHubError::LLMError(format!("Qwen TTS request error: {}", e)))?;
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LangHubError::LLMError(format!(
                "Qwen TTS API error ({}): {}",
                status, error_text
            )));
        }
        response
            .json()
            .await
            .map_err(|e| LangHubError::LLMError(format!("Qwen TTS JSON parse error: {}", e)))
    }
    fn raw_to_result(raw: &serde_json::Value) -> AudioLLMResult {
        let audio_url = raw["output"]["audio_url"].as_str().map(|s| s.to_string());
        let audio_base64 = raw["output"]["audio_data"].as_str().map(|s| s.to_string());
        let duration = raw["output"]["duration"].as_f64().map(|v| v as f32);
        let format = raw["output"]["format"].as_str().map(|s| s.to_string());
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
impl AudioLLM for QwenTts {
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
            let task_id = raw["output"]["task_id"]
                .as_str()
                .unwrap_or("qwen-tts-sync-task")
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
        "Alibaba-Qwen-TTS".to_string()
    }
    fn supports_voice(&self) -> bool {
        true
    }
    fn supports_emotion(&self) -> bool {
        true
    }
    fn max_duration(&self) -> Option<f32> {
        None
    }
}
