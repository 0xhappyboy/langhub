//! Audio generation providers module.
mod elevenlabs;
mod gemini_tts;
mod lyria;
mod qwen_tts;
mod seed_audio;
mod stable_audio;
mod step_audio;
mod suno;
use crate::types::Result;
pub use elevenlabs::{ElevenLabs, ElevenLabsModel};
pub use gemini_tts::{GeminiTts, GeminiTtsModel};
pub use lyria::{Lyria, LyriaModel};
pub use qwen_tts::{QwenTts, QwenTtsModel};
pub use seed_audio::{SeedAudio, SeedAudioModel};
use serde::{Deserialize, Serialize};
pub use stable_audio::{StableAudio, StableAudioModel};
use std::future::Future;
use std::pin::Pin;
pub use step_audio::{StepAudio, StepAudioModel};
pub use suno::{Suno, SunoModel};
/// Usage information from an audio generation API response.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AudioUsage {
    /// Number of seconds of audio billed.
    pub billed_seconds: f32,
    /// Number of characters billed (TTS providers such as ElevenLabs).
    pub billed_characters: Option<u64>,
    /// Estimated cost in USD.
    pub estimated_cost_usd: Option<f32>,
}
/// Complete audio generation API response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioLLMResult {
    /// URL of the generated audio (may expire).
    pub audio_url: Option<String>,
    /// Base64-encoded audio data, if returned inline.
    pub audio_base64: Option<String>,
    /// Local file path, if the audio was downloaded.
    pub file_path: Option<String>,
    /// Duration of the generated audio in seconds.
    pub duration_seconds: Option<f32>,
    /// Audio format, e.g. "mp3", "wav", "opus".
    pub format: Option<String>,
    /// Complete raw response from the provider API.
    pub raw_response: serde_json::Value,
}
impl AudioLLMResult {
    /// Extracts usage information from the raw response.
    pub fn extract_usage(&self) -> Option<AudioUsage> {
        extract_audio_usage_from_raw(&self.raw_response)
    }
}
/// Extracts usage information from various provider response formats.
pub fn extract_audio_usage_from_raw(raw: &serde_json::Value) -> Option<AudioUsage> {
    // Generic format with an explicit cost field.
    if let Some(cost) = raw.get("cost").and_then(|v| v.as_f64()) {
        return Some(AudioUsage {
            billed_seconds: raw.get("duration").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
            billed_characters: raw.get("characters").and_then(|v| v.as_u64()),
            estimated_cost_usd: Some(cost as f32),
        });
    }
    // Character-based format (TTS providers).
    if let Some(chars) = raw.get("characters").and_then(|v| v.as_u64()) {
        return Some(AudioUsage {
            billed_seconds: 0.0,
            billed_characters: Some(chars),
            estimated_cost_usd: None,
        });
    }
    // Token / seconds based format.
    if let Some(usage) = raw.get("usage") {
        let seconds = usage.get("seconds").and_then(|v| v.as_f64());
        let chars = usage.get("characters").and_then(|v| v.as_u64());
        if seconds.is_some() || chars.is_some() {
            return Some(AudioUsage {
                billed_seconds: seconds.unwrap_or(0.0) as f32,
                billed_characters: chars,
                estimated_cost_usd: None,
            });
        }
    }
    None
}
/// Unified options for audio generation.
///
/// # TTS
/// - `text`: the text to read.
/// - `voice`: voice identifier.
/// - `speed`: speaking rate multiplier.
/// - `emotion`: emotion tag (e.g. "happy", "sad").
/// - `language`: BCP-47 language tag.
/// - `format`: output audio format.
///
/// # Audio generation (soundscape)
/// - `text`: the scene / dialogue prompt.
/// - `format`: output audio format.
/// - `duration_seconds`: target duration.
///
/// # Music generation
/// - `text`: style / mood prompt.
/// - `lyrics`: optional lyrics.
/// - `instrumental`: whether to produce an instrumental track.
/// - `duration_seconds`: target duration.
#[derive(Debug, Clone, Default)]
pub struct AudioLLMOptions {
    /// The primary text input. For TTS this is the text to read; for audio
    /// generation this is the scene / dialogue prompt; for music generation
    /// this is the style / mood prompt.
    pub text: Option<String>,
    /// Voice identifier (TTS only).
    pub voice: Option<String>,
    /// Speaking rate multiplier (TTS only).
    pub speed: Option<f32>,
    /// Emotion tag (TTS only), e.g. "happy", "sad", "neutral".
    pub emotion: Option<String>,
    /// BCP-47 language tag (TTS only), e.g. "zh-CN", "en-US".
    pub language: Option<String>,
    /// Output audio format, e.g. "mp3", "wav", "opus".
    pub format: Option<String>,
    /// Target duration in seconds.
    pub duration_seconds: Option<f32>,
    /// Optional lyrics (music generation only).
    pub lyrics: Option<String>,
    /// Whether to produce an instrumental track (music generation only).
    pub instrumental: Option<bool>,
    /// Sample rate in Hz.
    pub sample_rate: Option<u32>,
}
/// Audio generation task status.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AudioTaskStatus {
    Pending,
    Processing,
    Succeeded,
    Failed,
    Cancelled,
}
/// Audio generation task handle for asynchronous providers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioTask {
    pub task_id: String,
    pub status: AudioTaskStatus,
    pub result: Option<AudioLLMResult>,
    pub error: Option<String>,
}
/// AudioLLM trait - unified interface for all audio providers.
pub trait AudioLLM: Send + Sync {
    /// Generates audio from a text prompt.
    fn generate(
        &self,
        prompt: &str,
    ) -> Pin<Box<dyn Future<Output = Result<AudioLLMResult>> + Send + '_>>;
    /// Generates audio with options.
    fn generate_with_options(
        &self,
        prompt: &str,
        options: AudioLLMOptions,
    ) -> Pin<Box<dyn Future<Output = Result<AudioLLMResult>> + Send + '_>>;
    /// Submits an asynchronous generation task and returns a task handle.
    fn submit_task(
        &self,
        prompt: &str,
        options: AudioLLMOptions,
    ) -> Pin<Box<dyn Future<Output = Result<AudioTask>> + Send + '_>>;
    /// Polls an asynchronous task by its ID.
    fn poll_task(
        &self,
        task_id: &str,
    ) -> Pin<Box<dyn Future<Output = Result<AudioTask>> + Send + '_>>;
    /// Returns the model name.
    fn get_model_name(&self) -> String;
    /// Returns the provider name.
    fn get_provider_name(&self) -> String;
    /// Whether the provider supports voice selection (TTS only).
    fn supports_voice(&self) -> bool {
        false
    }
    /// Whether the provider supports emotion control (TTS only).
    fn supports_emotion(&self) -> bool {
        false
    }
    /// Whether the provider supports lyrics input (music only).
    fn supports_lyrics(&self) -> bool {
        false
    }
    /// Whether the provider supports instrumental mode (music only).
    fn supports_instrumental(&self) -> bool {
        false
    }
    /// Returns the maximum supported duration in seconds, if any.
    fn max_duration(&self) -> Option<f32> {
        None
    }
}
/// Family of an audio model, used to group providers in the UI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AudioFamily {
    /// Text-to-speech: read text out loud.
    Tts,
    /// Audio generation (soundscape): dialogue + SFX + ambience.
    AudioGen,
    /// Music generation: music / songs.
    Music,
}
impl std::fmt::Display for AudioFamily {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AudioFamily::Tts => write!(f, "TTS"),
            AudioFamily::AudioGen => write!(f, "AudioGen"),
            AudioFamily::Music => write!(f, "Music"),
        }
    }
}
/// Audio model provider enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AudioModelProvider {
    QwenTts,
    SeedAudio,
    StepAudio,
    GeminiTts,
    ElevenLabs,
    Lyria,
    Suno,
    StableAudio,
}
impl std::fmt::Display for AudioModelProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AudioModelProvider::QwenTts => write!(f, "QwenTts"),
            AudioModelProvider::SeedAudio => write!(f, "SeedAudio"),
            AudioModelProvider::StepAudio => write!(f, "StepAudio"),
            AudioModelProvider::GeminiTts => write!(f, "GeminiTts"),
            AudioModelProvider::ElevenLabs => write!(f, "ElevenLabs"),
            AudioModelProvider::Lyria => write!(f, "Lyria"),
            AudioModelProvider::Suno => write!(f, "Suno"),
            AudioModelProvider::StableAudio => write!(f, "StableAudio"),
        }
    }
}
impl AudioModelProvider {
    /// Returns all supported audio model providers.
    pub fn all() -> Vec<AudioModelProvider> {
        vec![
            AudioModelProvider::QwenTts,
            AudioModelProvider::SeedAudio,
            AudioModelProvider::StepAudio,
            AudioModelProvider::GeminiTts,
            AudioModelProvider::ElevenLabs,
            AudioModelProvider::Lyria,
            AudioModelProvider::Suno,
            AudioModelProvider::StableAudio,
        ]
    }
    /// Returns the vendor of this audio model provider.
    pub fn vendor(&self) -> crate::types::AudioVendor {
        match self {
            AudioModelProvider::QwenTts => crate::types::AudioVendor::Alibaba,
            AudioModelProvider::SeedAudio => crate::types::AudioVendor::ByteDance,
            AudioModelProvider::StepAudio => crate::types::AudioVendor::StepFun,
            AudioModelProvider::GeminiTts => crate::types::AudioVendor::Google,
            AudioModelProvider::ElevenLabs => crate::types::AudioVendor::ElevenLabs,
            AudioModelProvider::Lyria => crate::types::AudioVendor::Google,
            AudioModelProvider::Suno => crate::types::AudioVendor::Suno,
            AudioModelProvider::StableAudio => crate::types::AudioVendor::StabilityAI,
        }
    }
    /// Returns the family (TTS / AudioGen / Music) of this provider.
    pub fn family(&self) -> AudioFamily {
        match self {
            AudioModelProvider::QwenTts => AudioFamily::Tts,
            AudioModelProvider::SeedAudio => AudioFamily::AudioGen,
            AudioModelProvider::StepAudio => AudioFamily::AudioGen,
            AudioModelProvider::GeminiTts => AudioFamily::Tts,
            AudioModelProvider::ElevenLabs => AudioFamily::Tts,
            AudioModelProvider::Lyria => AudioFamily::Music,
            AudioModelProvider::Suno => AudioFamily::Music,
            AudioModelProvider::StableAudio => AudioFamily::Music,
        }
    }
    /// Returns a short human-readable description of this provider's特色.
    pub fn description(&self) -> &'static str {
        match self {
            AudioModelProvider::QwenTts => {
                "Alibaba Qwen-Audio TTS: flagship text-to-speech with emotion, dialect and multi-language control"
            }
            AudioModelProvider::SeedAudio => {
                "ByteDance Seed Audio: one prompt generates a full audio track with dialogue, SFX and ambience"
            }
            AudioModelProvider::StepAudio => {
                "StepFun StepAudio3Gen: natural-language orchestration of dialogue, SFX, ambience and music"
            }
            AudioModelProvider::GeminiTts => {
                "Google Gemini TTS: expressive multi-speaker text-to-speech with native audio grounding"
            }
            AudioModelProvider::ElevenLabs => {
                "ElevenLabs Eleven v4: top-ranked TTS with the most natural voice cloning and emotion"
            }
            AudioModelProvider::Lyria => {
                "Google Lyria 3: music generation with vocals, timed lyrics and full song structure"
            }
            AudioModelProvider::Suno => {
                "Suno v6: music generation with natural-language editing of song sections"
            }
            AudioModelProvider::StableAudio => {
                "Stability AI Stable Audio 2.5: fast music and sound-effect generation from text prompts"
            }
        }
    }
    pub fn description_zh(&self) -> &'static str {
        match self {
            AudioModelProvider::QwenTts => {
                "阿里云 Qwen-Audio TTS：旗舰文本转语音，支持情绪、方言和多语种控制"
            }
            AudioModelProvider::SeedAudio => {
                "字节跳动 Seed Audio：一条提示词生成含对白、音效、环境声的完整音轨"
            }
            AudioModelProvider::StepAudio => {
                "阶跃星辰 StepAudio3Gen：用自然语言编排对白、音效、环境声和音乐"
            }
            AudioModelProvider::GeminiTts => {
                "Google Gemini TTS：富有表现力的多说话人文本转语音，原生音频接地"
            }
            AudioModelProvider::ElevenLabs => {
                "ElevenLabs Eleven v4：排名第一的 TTS，最自然的语音克隆和情绪表达"
            }
            AudioModelProvider::Lyria => {
                "Google Lyria 3：音乐生成，支持人声、定时歌词和完整歌曲结构"
            }
            AudioModelProvider::Suno => "Suno v6：音乐生成，支持用自然语言编辑歌曲段落",
            AudioModelProvider::StableAudio => {
                "Stability AI Stable Audio 2.5：快速从文本提示生成音乐和音效"
            }
        }
    }
    pub fn supports_voice(&self) -> bool {
        match self {
            AudioModelProvider::QwenTts => true,
            AudioModelProvider::SeedAudio => false,
            AudioModelProvider::StepAudio => true,
            AudioModelProvider::GeminiTts => true,
            AudioModelProvider::ElevenLabs => true,
            AudioModelProvider::Lyria => false,
            AudioModelProvider::Suno => false,
            AudioModelProvider::StableAudio => false,
        }
    }
    pub fn supports_emotion(&self) -> bool {
        match self {
            AudioModelProvider::QwenTts => true,
            AudioModelProvider::SeedAudio => false,
            AudioModelProvider::StepAudio => true,
            AudioModelProvider::GeminiTts => true,
            AudioModelProvider::ElevenLabs => true,
            AudioModelProvider::Lyria => false,
            AudioModelProvider::Suno => false,
            AudioModelProvider::StableAudio => false,
        }
    }
    pub fn supports_lyrics(&self) -> bool {
        match self {
            AudioModelProvider::QwenTts => false,
            AudioModelProvider::SeedAudio => false,
            AudioModelProvider::StepAudio => false,
            AudioModelProvider::GeminiTts => false,
            AudioModelProvider::ElevenLabs => false,
            AudioModelProvider::Lyria => true,
            AudioModelProvider::Suno => true,
            AudioModelProvider::StableAudio => false,
        }
    }
    pub fn supports_instrumental(&self) -> bool {
        match self {
            AudioModelProvider::QwenTts => false,
            AudioModelProvider::SeedAudio => false,
            AudioModelProvider::StepAudio => false,
            AudioModelProvider::GeminiTts => false,
            AudioModelProvider::ElevenLabs => false,
            AudioModelProvider::Lyria => true,
            AudioModelProvider::Suno => true,
            AudioModelProvider::StableAudio => true,
        }
    }
    /// Returns all concrete models under this provider, as `(model_id, display_name, is_recommended)` tuples.
    pub fn models(&self) -> Vec<(String, String, bool)> {
        match self {
            AudioModelProvider::QwenTts => vec![
                (
                    "qwen-audio-3.1-tts".to_string(),
                    "Qwen-Audio 3.1 TTS".to_string(),
                    true,
                ),
                (
                    "qwen-audio-3.1-tts-next".to_string(),
                    "Qwen-Audio 3.1 TTS-Next".to_string(),
                    false,
                ),
            ],
            AudioModelProvider::SeedAudio => vec![(
                "seed-audio-1-0".to_string(),
                "Seed Audio 1.0".to_string(),
                true,
            )],
            AudioModelProvider::StepAudio => vec![
                (
                    "stepaudio3-tts".to_string(),
                    "StepAudio3 TTS".to_string(),
                    true,
                ),
                (
                    "stepaudio3-gen".to_string(),
                    "StepAudio3 Gen".to_string(),
                    false,
                ),
                (
                    "stepaudio3-music".to_string(),
                    "StepAudio3 Music".to_string(),
                    false,
                ),
            ],
            AudioModelProvider::GeminiTts => vec![(
                "gemini-3.8-flash-tts".to_string(),
                "Gemini 3.8 Flash TTS".to_string(),
                true,
            )],
            AudioModelProvider::ElevenLabs => {
                vec![("eleven-v4".to_string(), "Eleven v4".to_string(), true)]
            }
            AudioModelProvider::Lyria => vec![
                ("lyria-3-pro".to_string(), "Lyria 3 Pro".to_string(), true),
                ("lyria-3".to_string(), "Lyria 3".to_string(), false),
            ],
            AudioModelProvider::Suno => vec![
                ("suno-v6".to_string(), "Suno v6".to_string(), true),
                (
                    "suno-v6-wild".to_string(),
                    "Suno v6 Wild".to_string(),
                    false,
                ),
                (
                    "suno-v6-mini".to_string(),
                    "Suno v6 Mini".to_string(),
                    false,
                ),
            ],
            AudioModelProvider::StableAudio => vec![(
                "stable-audio-2-5".to_string(),
                "Stable Audio 2.5".to_string(),
                true,
            )],
        }
    }
}
