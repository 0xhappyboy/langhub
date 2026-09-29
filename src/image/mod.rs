//! Image generation providers module.
mod dalle;
mod flux;
mod imagen;
mod seedream;
mod stability;
mod wan_image;
use crate::types::Result;
pub use dalle::{DallE, DallEModel};
pub use flux::{FluxImage, FluxImageModel};
pub use imagen::{Imagen, ImagenModel};
pub use seedream::{Seedream, SeedreamModel};
use serde::{Deserialize, Serialize};
pub use stability::{StabilityImage, StabilityImageModel};
use std::future::Future;
use std::pin::Pin;
pub use wan_image::{WanImage, WanImageModel};
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ImageUsage {
    pub billed_images: u32,
    pub billed_megapixels: Option<f32>,
    pub estimated_cost_usd: Option<f32>,
}
/// Complete image generation API response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageLLMResult {
    /// URLs of the generated images (may expire).
    pub image_urls: Vec<String>,
    /// Base64-encoded image data, if returned inline.
    pub image_base64: Option<Vec<String>>,
    /// Local file paths, if the images were downloaded.
    pub file_paths: Option<Vec<String>>,
    /// Resolution of the generated images, e.g. "1024x1024".
    pub resolution: Option<String>,
    /// Complete raw response from the provider API.
    pub raw_response: serde_json::Value,
}
impl ImageLLMResult {
    pub fn extract_usage(&self) -> Option<ImageUsage> {
        extract_image_usage_from_raw(&self.raw_response)
    }
}
pub fn extract_image_usage_from_raw(raw: &serde_json::Value) -> Option<ImageUsage> {
    if let Some(cost) = raw.get("cost").and_then(|v| v.as_f64()) {
        return Some(ImageUsage {
            billed_images: raw.get("image_count").and_then(|v| v.as_u64()).unwrap_or(1) as u32,
            billed_megapixels: raw
                .get("megapixels")
                .and_then(|v| v.as_f64())
                .map(|v| v as f32),
            estimated_cost_usd: Some(cost as f32),
        });
    }
    // Token-based format.
    if let Some(usage) = raw.get("usage") {
        if let Some(images) = usage.get("images").and_then(|v| v.as_u64()) {
            return Some(ImageUsage {
                billed_images: images as u32,
                billed_megapixels: None,
                estimated_cost_usd: None,
            });
        }
    }
    None
}
/// Unified options for image generation.
#[derive(Debug, Clone, Default)]
pub struct ImageLLMOptions {
    /// Number of images to generate.
    pub n: Option<u32>,
    /// Resolution string, e.g. "1024x1024".
    pub resolution: Option<String>,
    /// Aspect ratio, e.g. "16:9", "1:1".
    pub aspect_ratio: Option<String>,
    /// Random seed for reproducibility.
    pub seed: Option<u64>,
    /// Negative prompt.
    pub negative_prompt: Option<String>,
    /// Reference image URLs for image-to-image.
    pub reference_images: Option<Vec<String>>,
    /// Output format, e.g. "png", "jpeg", "webp".
    pub output_format: Option<String>,
    /// Quality hint, e.g. "standard", "hd".
    pub quality: Option<String>,
    /// Style hint, e.g. "vivid", "natural".
    pub style: Option<String>,
}
/// Image generation task status.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImageTaskStatus {
    Pending,
    Processing,
    Succeeded,
    Failed,
    Cancelled,
}
/// Image generation task handle for asynchronous providers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageTask {
    pub task_id: String,
    pub status: ImageTaskStatus,
    pub result: Option<ImageLLMResult>,
    pub error: Option<String>,
}
/// ImageLLM trait - unified interface for all text-to-image providers.
pub trait ImageLLM: Send + Sync {
    fn generate(
        &self,
        prompt: &str,
    ) -> Pin<Box<dyn Future<Output = Result<ImageLLMResult>> + Send + '_>>;
    fn generate_with_options(
        &self,
        prompt: &str,
        options: ImageLLMOptions,
    ) -> Pin<Box<dyn Future<Output = Result<ImageLLMResult>> + Send + '_>>;
    fn submit_task(
        &self,
        prompt: &str,
        options: ImageLLMOptions,
    ) -> Pin<Box<dyn Future<Output = Result<ImageTask>> + Send + '_>>;
    fn poll_task(
        &self,
        task_id: &str,
    ) -> Pin<Box<dyn Future<Output = Result<ImageTask>> + Send + '_>>;
    fn get_model_name(&self) -> String;
    fn get_provider_name(&self) -> String;
    fn supports_reference_images(&self) -> bool {
        false
    }
    fn supports_negative_prompt(&self) -> bool {
        false
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ImageModelProvider {
    Seedream,
    WanImage,
    StabilityImage,
    Flux,
    Imagen,
    DallE,
}
impl std::fmt::Display for ImageModelProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ImageModelProvider::Seedream => write!(f, "Seedream"),
            ImageModelProvider::WanImage => write!(f, "WanImage"),
            ImageModelProvider::StabilityImage => write!(f, "StabilityImage"),
            ImageModelProvider::Flux => write!(f, "Flux"),
            ImageModelProvider::Imagen => write!(f, "Imagen"),
            ImageModelProvider::DallE => write!(f, "DallE"),
        }
    }
}
impl ImageModelProvider {
    pub fn all() -> Vec<ImageModelProvider> {
        vec![
            ImageModelProvider::Seedream,
            ImageModelProvider::WanImage,
            ImageModelProvider::StabilityImage,
            ImageModelProvider::Flux,
            ImageModelProvider::Imagen,
            ImageModelProvider::DallE,
        ]
    }
    pub fn vendor(&self) -> crate::types::ImageVendor {
        match self {
            ImageModelProvider::Seedream => crate::types::ImageVendor::ByteDance,
            ImageModelProvider::WanImage => crate::types::ImageVendor::Alibaba,
            ImageModelProvider::StabilityImage => crate::types::ImageVendor::StabilityAI,
            ImageModelProvider::Flux => crate::types::ImageVendor::BlackForestLabs,
            ImageModelProvider::Imagen => crate::types::ImageVendor::Google,
            ImageModelProvider::DallE => crate::types::ImageVendor::OpenAI,
        }
    }
    pub fn supports_reference_images(&self) -> bool {
        match self {
            ImageModelProvider::Seedream => true,
            ImageModelProvider::WanImage => true,
            ImageModelProvider::StabilityImage => true,
            ImageModelProvider::Flux => true,
            ImageModelProvider::Imagen => true,
            ImageModelProvider::DallE => false,
        }
    }
    pub fn supports_negative_prompt(&self) -> bool {
        match self {
            ImageModelProvider::Seedream => true,
            ImageModelProvider::WanImage => true,
            ImageModelProvider::StabilityImage => true,
            ImageModelProvider::Flux => true,
            ImageModelProvider::Imagen => true,
            ImageModelProvider::DallE => false,
        }
    }
}
