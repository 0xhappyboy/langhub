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
    /// Number of seconds billed.
    pub billed_seconds: f32,
    /// Number of tokens billed (for token-based providers such as Seedance).
    pub billed_tokens: Option<u64>,
    /// Estimated cost in USD.
    pub estimated_cost_usd: Option<f32>,
}
/// Complete video generation API response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoLLMResult {
    /// URL of the generated video (may expire).
    pub video_url: Option<String>,
    /// Base64-encoded video data, if returned inline.
    pub video_base64: Option<String>,
    /// Local file path, if the video was downloaded.
    pub file_path: Option<String>,
    /// Duration of the generated video in seconds.
    pub duration_seconds: Option<f32>,
    /// Resolution of the generated video, e.g. "1280x720".
    pub resolution: Option<String>,
    /// Complete raw response from the provider API.
    pub raw_response: serde_json::Value,
}
impl VideoLLMResult {
    /// Extracts usage information from the raw response.
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
#[derive(Debug, Clone, Default)]
pub struct VideoLLMOptions {
    /// Video duration in seconds.
    pub duration: Option<f32>,
    /// Resolution string, e.g. "720p", "1080p".
    pub resolution: Option<String>,
    /// Aspect ratio, e.g. "16:9", "9:16", "1:1".
    pub aspect_ratio: Option<String>,
    /// Random seed for reproducibility.
    pub seed: Option<u64>,
    /// Whether to generate native audio.
    pub generate_audio: Option<bool>,
    /// Reference image URLs for image-to-video.
    pub reference_images: Option<Vec<String>>,
    /// Reference video URLs for video-to-video.
    pub reference_videos: Option<Vec<String>>,
    /// Negative prompt.
    pub negative_prompt: Option<String>,
    /// Frame rate.
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
    /// Generates a video from a text prompt.
    fn generate(
        &self,
        prompt: &str,
    ) -> Pin<Box<dyn Future<Output = Result<VideoLLMResult>> + Send + '_>>;
    /// Generates a video with options.
    fn generate_with_options(
        &self,
        prompt: &str,
        options: VideoLLMOptions,
    ) -> Pin<Box<dyn Future<Output = Result<VideoLLMResult>> + Send + '_>>;
    /// Submits an asynchronous generation task and returns a task handle.
    fn submit_task(
        &self,
        prompt: &str,
        options: VideoLLMOptions,
    ) -> Pin<Box<dyn Future<Output = Result<VideoTask>> + Send + '_>>;
    /// Polls an asynchronous task by its ID.
    fn poll_task(
        &self,
        task_id: &str,
    ) -> Pin<Box<dyn Future<Output = Result<VideoTask>> + Send + '_>>;
    /// Returns the model name.
    fn get_model_name(&self) -> String;
    /// Returns the provider name.
    fn get_provider_name(&self) -> String;
    /// Returns the maximum supported duration in seconds.
    fn max_duration(&self) -> Option<f32> {
        None
    }
    /// Whether the provider supports native audio generation.
    fn supports_audio(&self) -> bool {
        false
    }
    /// Whether the provider supports reference images.
    fn supports_reference_images(&self) -> bool {
        false
    }
    /// Whether the provider supports reference videos.
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
    /// Returns all supported video model providers.
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
    /// Returns the vendor of this video model provider.
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
    /// Whether the provider supports native audio generation.
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
    /// Whether the provider supports reference images.
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
    /// Whether the provider supports reference videos.
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
}
