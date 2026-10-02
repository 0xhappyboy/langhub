//! Video generation providers module.
mod gemini_omni;
mod grok_imagine;
mod happyhorse;
mod kling;
mod ltx;
mod minimax_h3;
mod pruna;
mod runway;
mod seedance;
mod veo;
mod wan;
use crate::types::Result;
pub use gemini_omni::{GeminiOmniFlash, GeminiOmniFlashModel};
pub use grok_imagine::{GrokImagine, GrokImagineModel};
pub use happyhorse::{HappyHorse, HappyHorseModel};
pub use kling::{KlingVideo, KlingVideoModel};
pub use ltx::{LtxVideo, LtxVideoModel};
pub use minimax_h3::{MiniMaxH3, MiniMaxH3Model};
pub use pruna::{PrunaVideo, PrunaVideoModel};
pub use runway::{RunwayVideo, RunwayVideoModel};
pub use seedance::{Seedance, SeedanceModel};
use serde::{Deserialize, Serialize};
use std::future::Future;
use std::pin::Pin;
pub use veo::{Veo, VeoModel};
pub use wan::{WanVideo, WanVideoModel};
/// Usage information from a video generation API response.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VideoUsage {
    pub billed_seconds: f32,
    pub billed_tokens: Option<u64>,
    pub estimated_cost_usd: Option<f32>,
}
/// Complete video generation API response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoLLMResult {
    pub video_url: Option<String>,
    pub video_base64: Option<String>,
    pub file_path: Option<String>,
    pub duration_seconds: Option<f32>,
    pub resolution: Option<String>,
    pub raw_response: serde_json::Value,
}
impl VideoLLMResult {
    pub fn extract_usage(&self) -> Option<VideoUsage> {
        extract_video_usage_from_raw(&self.raw_response)
    }
}
/// Extracts usage information from various provider response formats.
pub fn extract_video_usage_from_raw(raw: &serde_json::Value) -> Option<VideoUsage> {
    // Runway / generic format with an explicit cost field.
    if let Some(cost) = raw.get("cost").and_then(|v| v.as_f64()) {
        return Some(VideoUsage {
            billed_seconds: raw.get("duration").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
            billed_tokens: None,
            estimated_cost_usd: Some(cost as f32),
        });
    }
    // ByteDance / token-based format.
    if let Some(usage) = raw.get("usage") {
        let tokens = usage
            .get("total_tokens")
            .and_then(|v| v.as_u64())
            .or_else(|| usage.get("completion_tokens").and_then(|v| v.as_u64()));
        if let Some(t) = tokens {
            return Some(VideoUsage {
                billed_seconds: 0.0,
                billed_tokens: Some(t),
                estimated_cost_usd: None,
            });
        }
    }
    // Google Veo format.
    if let Some(usage) = raw.get("usageMetadata") {
        if let Some(t) = usage.get("totalTokenCount").and_then(|v| v.as_u64()) {
            return Some(VideoUsage {
                billed_seconds: 0.0,
                billed_tokens: Some(t),
                estimated_cost_usd: None,
            });
        }
    }
    None
}
/// Unified options for video generation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VideoLLMOptions {
    pub duration: Option<f32>,
    pub resolution: Option<String>,
    pub aspect_ratio: Option<String>,
    pub seed: Option<u64>,
    pub generate_audio: Option<bool>,
    pub reference_images: Option<Vec<String>>,
    pub reference_videos: Option<Vec<String>>,
    pub negative_prompt: Option<String>,
    pub fps: Option<u32>,
}
/// Video generation task status.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VideoTaskStatus {
    Pending,
    Processing,
    Succeeded,
    Failed,
    Cancelled,
}
/// Video generation task handle for asynchronous providers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoTask {
    pub task_id: String,
    pub status: VideoTaskStatus,
    pub result: Option<VideoLLMResult>,
    pub error: Option<String>,
}
/// VideoLLM trait - unified interface for all text-to-video providers.
pub trait VideoLLM: Send + Sync {
    /// Generate a video from a prompt using the configured default model.
    ///
    /// `model` - Optional model id override. When `None`, the configured
    /// default model is used.
    fn generate(
        &self,
        prompt: &str,
        model: Option<&str>,
    ) -> Pin<Box<dyn Future<Output = Result<VideoLLMResult>> + Send + '_>>;
    /// Generate a video from a prompt with explicit options.
    ///
    /// `model` - Optional model id override. When `None`, the configured
    /// default model is used.
    fn generate_with_options(
        &self,
        prompt: &str,
        options: VideoLLMOptions,
        model: Option<&str>,
    ) -> Pin<Box<dyn Future<Output = Result<VideoLLMResult>> + Send + '_>>;
    /// Submit an asynchronous generation task.
    ///
    /// `model` - Optional model id override. When `None`, the configured
    /// default model is used.
    fn submit_task(
        &self,
        prompt: &str,
        options: VideoLLMOptions,
        model: Option<&str>,
    ) -> Pin<Box<dyn Future<Output = Result<VideoTask>> + Send + '_>>;
    /// Poll an asynchronous generation task by id.
    fn poll_task(
        &self,
        task_id: &str,
    ) -> Pin<Box<dyn Future<Output = Result<VideoTask>> + Send + '_>>;
    /// Get the configured default model name.
    fn get_model_name(&self) -> String;
    /// Get provider name.
    fn get_provider_name(&self) -> String;
    fn max_duration(&self) -> Option<f32> {
        None
    }
    fn supports_audio(&self) -> bool {
        false
    }
    fn supports_reference_images(&self) -> bool {
        false
    }
    fn supports_reference_videos(&self) -> bool {
        false
    }
}
/// Video model provider enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VideoModelProvider {
    Seedance,
    Wan,
    Kling,
    Veo,
    Runway,
    MiniMaxH3,
    HappyHorse,
    Ltx,
    GrokImagine,
    Pruna,
    GeminiOmniFlash,
}
impl std::fmt::Display for VideoModelProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VideoModelProvider::Seedance => write!(f, "Seedance"),
            VideoModelProvider::Wan => write!(f, "Wan"),
            VideoModelProvider::Kling => write!(f, "Kling"),
            VideoModelProvider::Veo => write!(f, "Veo"),
            VideoModelProvider::Runway => write!(f, "Runway"),
            VideoModelProvider::MiniMaxH3 => write!(f, "MiniMaxH3"),
            VideoModelProvider::HappyHorse => write!(f, "HappyHorse"),
            VideoModelProvider::Ltx => write!(f, "Ltx"),
            VideoModelProvider::GrokImagine => write!(f, "GrokImagine"),
            VideoModelProvider::Pruna => write!(f, "Pruna"),
            VideoModelProvider::GeminiOmniFlash => write!(f, "GeminiOmniFlash"),
        }
    }
}
impl VideoModelProvider {
    pub fn all() -> Vec<VideoModelProvider> {
        vec![
            VideoModelProvider::Seedance,
            VideoModelProvider::Wan,
            VideoModelProvider::Kling,
            VideoModelProvider::Veo,
            VideoModelProvider::Runway,
            VideoModelProvider::MiniMaxH3,
            VideoModelProvider::HappyHorse,
            VideoModelProvider::Ltx,
            VideoModelProvider::GrokImagine,
            VideoModelProvider::Pruna,
            VideoModelProvider::GeminiOmniFlash,
        ]
    }
    pub fn vendor(&self) -> crate::types::VideoVendor {
        match self {
            VideoModelProvider::Seedance => crate::types::VideoVendor::ByteDance,
            VideoModelProvider::Wan => crate::types::VideoVendor::Alibaba,
            VideoModelProvider::Kling => crate::types::VideoVendor::Kuaishou,
            VideoModelProvider::Veo => crate::types::VideoVendor::Google,
            VideoModelProvider::Runway => crate::types::VideoVendor::Runway,
            VideoModelProvider::MiniMaxH3 => crate::types::VideoVendor::MiniMax,
            VideoModelProvider::HappyHorse => crate::types::VideoVendor::Alibaba,
            VideoModelProvider::Ltx => crate::types::VideoVendor::Lightricks,
            VideoModelProvider::GrokImagine => crate::types::VideoVendor::Xai,
            VideoModelProvider::Pruna => crate::types::VideoVendor::Pruna,
            VideoModelProvider::GeminiOmniFlash => crate::types::VideoVendor::Google,
        }
    }
    /// Returns a short human-readable description of this provider (English).
    pub fn description(&self) -> &'static str {
        match self {
            VideoModelProvider::Seedance => {
                "ByteDance Seedance: cinematic text-to-video with strong motion, native audio and multi-shot consistency"
            }
            VideoModelProvider::Wan => {
                "Alibaba Wan: Tongyi Wanxiang video generation with rich style control and reference-to-video support"
            }
            VideoModelProvider::Kling => {
                "Kuaishou Kling: high-fidelity text-to-video with native audio, image and video references"
            }
            VideoModelProvider::Veo => {
                "Google Veo: state-of-the-art text-to-video with native audio, cinematic quality and long duration"
            }
            VideoModelProvider::Runway => {
                "Runway: Gen-4.5 and Aleph video models for creative filmmaking and video-to-video editing"
            }
            VideoModelProvider::MiniMaxH3 => {
                "MiniMax Hailuo H3: expressive text-to-video with native audio and image-to-video support"
            }
            VideoModelProvider::HappyHorse => {
                "Alibaba HappyHorse: text-to-video with reference image support and fast generation"
            }
            VideoModelProvider::Ltx => {
                "Lightricks LTX: efficient text-to-video with audio, fast and pro tiers for different quality needs"
            }
            VideoModelProvider::GrokImagine => {
                "xAI Grok Imagine: creative text-to-video with reference image and video support"
            }
            VideoModelProvider::Pruna => {
                "Pruna P-Video: optimized text-to-video for fast, cost-effective generation"
            }
            VideoModelProvider::GeminiOmniFlash => {
                "Google Gemini Omni Flash: multimodal text-to-video with native audio and image/video references"
            }
        }
    }
    /// Returns a short human-readable description of this provider (Chinese).
    pub fn description_zh(&self) -> &'static str {
        match self {
            VideoModelProvider::Seedance => {
                "字节跳动 Seedance：电影级文生视频，动态强，原生音频，多镜头一致性好"
            }
            VideoModelProvider::Wan => "阿里云通义万相：文生视频，风格控制丰富，支持参考图生视频",
            VideoModelProvider::Kling => "快手可灵：高保真文生视频，原生音频，支持图片和视频参考",
            VideoModelProvider::Veo => "Google Veo：顶尖文生视频，原生音频，电影级画质，时长更长",
            VideoModelProvider::Runway => {
                "Runway：Gen-4.5 与 Aleph 视频模型，适合创意影视制作和视频到视频编辑"
            }
            VideoModelProvider::MiniMaxH3 => {
                "MiniMax 海螺 H3：表现力强的文生视频，原生音频，支持图生视频"
            }
            VideoModelProvider::HappyHorse => "阿里云 HappyHorse：文生视频，支持参考图，生成速度快",
            VideoModelProvider::Ltx => {
                "Lightricks LTX：高效文生视频，支持音频，提供 Fast 和 Pro 两档画质"
            }
            VideoModelProvider::GrokImagine => {
                "xAI Grok Imagine：创意文生视频，支持参考图和参考视频"
            }
            VideoModelProvider::Pruna => "Pruna P-Video：优化型文生视频，生成快、成本低",
            VideoModelProvider::GeminiOmniFlash => {
                "Google Gemini Omni Flash：多模态文生视频，原生音频，支持图片和视频参考"
            }
        }
    }
    pub fn supports_audio(&self) -> bool {
        match self {
            VideoModelProvider::Seedance => true,
            VideoModelProvider::Wan => true,
            VideoModelProvider::Kling => true,
            VideoModelProvider::Veo => true,
            VideoModelProvider::Runway => false,
            VideoModelProvider::MiniMaxH3 => true,
            VideoModelProvider::HappyHorse => false,
            VideoModelProvider::Ltx => true,
            VideoModelProvider::GrokImagine => false,
            VideoModelProvider::Pruna => false,
            VideoModelProvider::GeminiOmniFlash => true,
        }
    }
    pub fn supports_reference_images(&self) -> bool {
        match self {
            VideoModelProvider::Seedance => true,
            VideoModelProvider::Wan => true,
            VideoModelProvider::Kling => true,
            VideoModelProvider::Veo => true,
            VideoModelProvider::Runway => true,
            VideoModelProvider::MiniMaxH3 => true,
            VideoModelProvider::HappyHorse => true,
            VideoModelProvider::Ltx => true,
            VideoModelProvider::GrokImagine => true,
            VideoModelProvider::Pruna => true,
            VideoModelProvider::GeminiOmniFlash => true,
        }
    }
    pub fn supports_reference_videos(&self) -> bool {
        match self {
            VideoModelProvider::Seedance => true,
            VideoModelProvider::Wan => true,
            VideoModelProvider::Kling => true,
            VideoModelProvider::Veo => true,
            VideoModelProvider::Runway => true,
            VideoModelProvider::MiniMaxH3 => false,
            VideoModelProvider::HappyHorse => false,
            VideoModelProvider::Ltx => true,
            VideoModelProvider::GrokImagine => true,
            VideoModelProvider::Pruna => false,
            VideoModelProvider::GeminiOmniFlash => true,
        }
    }
    /// Returns all concrete models under this provider, as `(model_id, display_name, is_recommended)` tuples.
    pub fn models(&self) -> Vec<(String, String, bool)> {
        match self {
            VideoModelProvider::Seedance => vec![
                (
                    "doubao-seedance-1-0-pro-250528".to_string(),
                    "Seedance 1.0 Pro".to_string(),
                    false,
                ),
                (
                    "doubao-seedance-1-0-pro-fast-251015".to_string(),
                    "Seedance 1.0 Pro Fast".to_string(),
                    true,
                ),
                (
                    "doubao-seedance-1-5-pro-251215".to_string(),
                    "Seedance 1.5 Pro".to_string(),
                    false,
                ),
                (
                    "doubao-seedance-2-0-260128".to_string(),
                    "Seedance 2.0".to_string(),
                    false,
                ),
                (
                    "doubao-seedance-2-0-fast-260128".to_string(),
                    "Seedance 2.0 Fast".to_string(),
                    false,
                ),
                (
                    "doubao-seedance-2-0-mini-260615".to_string(),
                    "Seedance 2.0 Mini".to_string(),
                    false,
                ),
                (
                    "doubao-seedance-2-5".to_string(),
                    "Seedance 2.5".to_string(),
                    false,
                ),
            ],
            VideoModelProvider::Wan => vec![
                ("wan3.0-video".to_string(), "Wan 3.0".to_string(), true),
                (
                    "wan3.0-video-prime".to_string(),
                    "Wan 3.0 Prime".to_string(),
                    false,
                ),
            ],
            VideoModelProvider::Kling => vec![
                ("kling-v3".to_string(), "Kling 3.0".to_string(), true),
                (
                    "kling-v3-omni".to_string(),
                    "Kling 3.0 Omni".to_string(),
                    false,
                ),
                ("kling-v4".to_string(), "Kling 4.0".to_string(), false),
            ],
            VideoModelProvider::Veo => vec![
                (
                    "veo-3.1-generate-preview".to_string(),
                    "Veo 3.1".to_string(),
                    true,
                ),
                (
                    "veo-3.1-fast-generate-preview".to_string(),
                    "Veo 3.1 Fast".to_string(),
                    false,
                ),
                (
                    "veo-3.1-lite-generate-preview".to_string(),
                    "Veo 3.1 Lite".to_string(),
                    false,
                ),
            ],
            VideoModelProvider::Runway => vec![
                ("gen4.5".to_string(), "Runway Gen-4.5".to_string(), true),
                (
                    "aleph2.0".to_string(),
                    "Runway Aleph 2.0".to_string(),
                    false,
                ),
            ],
            VideoModelProvider::MiniMaxH3 => vec![(
                "MiniMax-Hailuo-H3".to_string(),
                "MiniMax H3".to_string(),
                true,
            )],
            VideoModelProvider::HappyHorse => vec![(
                "happyhorse-1.0".to_string(),
                "HappyHorse 1.0".to_string(),
                true,
            )],
            VideoModelProvider::Ltx => vec![
                ("ltx-2.3-fast".to_string(), "LTX-2.3 Fast".to_string(), true),
                ("ltx-2.3-pro".to_string(), "LTX-2.3 Pro".to_string(), false),
            ],
            VideoModelProvider::GrokImagine => vec![
                (
                    "grok-imagine-video-1.5".to_string(),
                    "Grok Imagine 1.5".to_string(),
                    true,
                ),
                (
                    "grok-imagine-video-1.0".to_string(),
                    "Grok Imagine 1.0".to_string(),
                    false,
                ),
            ],
            VideoModelProvider::Pruna => vec![(
                "p-video-2-pro".to_string(),
                "Pruna P-Video 2 Pro".to_string(),
                true,
            )],
            VideoModelProvider::GeminiOmniFlash => vec![(
                "gemini-omni-flash-1.1".to_string(),
                "Gemini Omni Flash 1.1".to_string(),
                true,
            )],
        }
    }
    /// Probes this provider with a real authenticated read-only request.
    pub async fn probe(&self, api_key: &str, base_url: Option<&str>) -> Result<()> {
        let key = api_key.to_string();
        if key.is_empty() {
            return Err(crate::types::LangHubError::LLMError(
                "video key empty".to_string(),
            ));
        }
        let base = match base_url {
            Some(b) if !b.trim().is_empty() => b.trim().trim_end_matches('/').to_string(),
            _ => self.default_base_url().trim_end_matches('/').to_string(),
        };
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .user_agent("langhub-healthcheck/1.0")
            .build()
            .map_err(|e| crate::types::LangHubError::LLMError(format!("build client: {}", e)))?;
        let req = match self {
            VideoModelProvider::Runway => client
                .get(format!("{}/tasks", base))
                .header("Authorization", format!("Bearer {}", key))
                .header("X-Runway-Version", "2024-11-06"),
            VideoModelProvider::Kling => client
                .get(format!("{}/v1/videos/text2video", base))
                .header("Authorization", format!("Bearer {}", key)),
            VideoModelProvider::MiniMaxH3 => client
                .get(format!("{}/query/video_generation", base))
                .header("Authorization", format!("Bearer {}", key)),
            VideoModelProvider::Ltx => client
                .get(format!("{}/jobs", base))
                .header("Authorization", format!("Bearer {}", key)),
            VideoModelProvider::Pruna => client
                .get(format!("{}/video/generations", base))
                .header("Authorization", format!("Bearer {}", key)),
            VideoModelProvider::Veo | VideoModelProvider::GeminiOmniFlash => {
                client.get(format!("{}/models?key={}", base, key))
            }
            _ => client
                .get(format!("{}/models", base))
                .header("Authorization", format!("Bearer {}", key)),
        };
        let response = req
            .send()
            .await
            .map_err(|e| crate::types::LangHubError::LLMError(format!("probe request: {}", e)))?;
        let status = response.status().as_u16();
        if status == 401 || status == 403 {
            return Err(crate::types::LangHubError::LLMError(format!(
                "auth rejected HTTP {}",
                status
            )));
        }
        if status >= 500 {
            return Err(crate::types::LangHubError::LLMError(format!(
                "server error HTTP {}",
                status
            )));
        }
        Ok(())
    }
    /// Returns the default base URL for this provider.
    fn default_base_url(&self) -> &'static str {
        match self {
            VideoModelProvider::Seedance => "https://ark.cn-beijing.volces.com/api/v3",
            VideoModelProvider::Wan => "https://dashscope.aliyuncs.com/api/v1",
            VideoModelProvider::Kling => "https://api.klingai.com",
            VideoModelProvider::Veo => "https://generativelanguage.googleapis.com/v1beta",
            VideoModelProvider::Runway => "https://api.dev.runwayml.com/v1",
            VideoModelProvider::MiniMaxH3 => "https://api.minimax.chat/v1",
            VideoModelProvider::HappyHorse => "https://dashscope.aliyuncs.com/api/v1",
            VideoModelProvider::Ltx => "https://api.ltx.video/v1",
            VideoModelProvider::GrokImagine => "https://api.x.ai/v1",
            VideoModelProvider::Pruna => "https://api.pruna.ai/v1",
            VideoModelProvider::GeminiOmniFlash => {
                "https://generativelanguage.googleapis.com/v1beta"
            }
        }
    }
}
