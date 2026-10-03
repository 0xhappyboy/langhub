use crate::audio::{AudioLLM, AudioLLMOptions, AudioLLMResult, AudioTask, AudioTaskStatus};
use crate::types::{LangHubError, Result};
use serde_json::json;
use std::future::Future;
use std::pin::Pin;
/// Seed Audio model variants.
#[derive(Debug, Clone)]
pub enum SeedAudioModel {
    SeedAudio10,
    Custom(String),
}
impl SeedAudioModel {
    fn as_str(&self) -> String {
        match self {
            SeedAudioModel::SeedAudio10 => "seed-audio-1.0".to_string(),
            SeedAudioModel::Custom(name) => name.clone(),
        }
    }
}
impl From<SeedAudioModel> for String {
    fn from(model: SeedAudioModel) -> Self {
        model.as_str()
    }
}
/// Seed Audio (Doubao Audio Generation 1.0) provider.
///
/// Talks to the ByteDance OpenSpeech audio generation endpoint:
///   POST https://openspeech.bytedance.com/api/v3/tts/create
///
/// Auth: single `X-Api-Key` header (new console). Do NOT use
/// `Authorization: Bearer`; that belongs to the Ark protocol and will
/// be rejected with a 401 "The API key format is incorrect".
///
/// Request body (per official docs):
/// {
///   "model": "seed-audio-1.0",
///   "text_prompt": "...",
///   "references": [{ "speaker": "..." }],
///   "audio_config": {
///     "format": "mp3",
///     "sample_rate": 48000,
///     "pitch_rate": 0,
///     "speech_rate": 0,
///     "loudness_rate": 0
///   },
///   "watermark": {}
/// }
///
/// Response body:
/// {
///   "audio": "<base64>",
///   "duration": 82.8,
///   "original_duration": 82.8,
///   "url": "<audio_url>"
/// }
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
            base_url: "https://openspeech.bytedance.com/api/v3".to_string(),
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
    /// Resolve the effective model id: caller override wins over the configured default.
    fn resolve_model(&self, model_override: Option<&str>) -> String {
        match model_override {
            Some(m) => m.to_string(),
            None => self.model.clone().into(),
        }
    }
    /// Build the OpenSpeech-compatible request body exactly as the official cURL example does.
    ///
    /// Fields:
    /// - `model`: model id, e.g. `seed-audio-1.0`
    /// - `text_prompt`: natural-language prompt / text to synthesize
    /// - `references`: optional list of reference assets. The official
    ///   example always sends a `[{ "speaker": "<speaker_id>" }]` entry
    ///   when a speaker is provided.
    /// - `audio_config`: output audio settings
    /// - `watermark`: watermark config
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
        let format = options
            .format
            .as_ref()
            .or(self.default_options.format.as_ref())
            .cloned()
            .unwrap_or_else(|| "mp3".to_string());
        let sample_rate = options
            .sample_rate
            .or(self.default_options.sample_rate)
            .unwrap_or(48000);
        let speech_rate = options.speed.or(self.default_options.speed).unwrap_or(0.0) as i64;
        let audio_config = json!({
            "format": format,
            "sample_rate": sample_rate,
            "pitch_rate": 0,
            "speech_rate": speech_rate,
            "loudness_rate": 0,
        });
        let mut body = json!({
            "model": model_name,
            "text_prompt": text,
            "audio_config": audio_config,
            // Empty watermark object per the official example.
            "watermark": {},
        });
        if let Some(voice) = options
            .voice
            .as_ref()
            .or(self.default_options.voice.as_ref())
        {
            body["references"] = json!([{ "speaker": voice }]);
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
        let mut body = body;
        if let Some(refs) = body.get("references").and_then(|v| v.as_array()) {
            let all_empty = refs.iter().all(|r| {
                r.get("speaker")
                    .and_then(|s| s.as_str())
                    .map(|s| s.trim().is_empty())
                    .unwrap_or(true)
            });
            if all_empty {
                if let Some(obj) = body.as_object_mut() {
                    obj.remove("references");
                }
            }
        }
        let url = format!("{}/tts/create", self.base_url);
        let response = self
            .client
            .post(&url)
            .header("X-Api-Key", &self.api_key)
            .header("Content-Type", "application/json")
            // Optional client-side request id for cross-system tracing.
            .header("X-Api-Request-Id", uuid_like_id())
            .json(&body)
            .send()
            .await
            .map_err(|e| LangHubError::LLMError(format!("Seed Audio request error: {}", e)))?;
        let log_id = response
            .headers()
            .get("X-Tt-Logid")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string());
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LangHubError::LLMError(format!(
                "Seed Audio API error ({}): {}",
                status, error_text
            )));
        }
        let mut parsed: serde_json::Value = response
            .json()
            .await
            .map_err(|e| LangHubError::LLMError(format!("Seed Audio JSON parse error: {}", e)))?;
        // Surface the response header log id into the JSON so downstream
        // consumers (task metadata, polling) can use it as the task id.
        if let Some(id) = log_id {
            if let Some(obj) = parsed.as_object_mut() {
                obj.insert("_log_id".to_string(), serde_json::Value::String(id));
            }
        }
        Ok(parsed)
    }
    /// Map the OpenSpeech response into the crate's unified result type.
    fn raw_to_result(raw: &serde_json::Value) -> AudioLLMResult {
        let audio_url = raw["url"].as_str().map(|s| s.to_string());
        let audio_base64 = raw["audio"].as_str().map(|s| s.to_string());
        let duration = raw["duration"]
            .as_f64()
            .or_else(|| raw["original_duration"].as_f64())
            .map(|v| v as f32);
        // The response does not echo the requested format, so leave it as
        // `None` and let the caller fall back to the request-side value.
        AudioLLMResult {
            audio_url,
            audio_base64,
            file_path: None,
            duration_seconds: duration,
            format: None,
            raw_response: raw.clone(),
        }
    }
}
/// Generate a lightweight unique request id without pulling in a uuid crate at this layer. Format is `langhub-<millis>-<counter>`.
fn uuid_like_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    format!("langhub-{}-{}", ms, n)
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
            let task_id = raw["_log_id"]
                .as_str()
                .or_else(|| raw["log_id"].as_str())
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
