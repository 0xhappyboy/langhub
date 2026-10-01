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
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
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
    /// Generate images from a prompt using the configured default model.
    ///
    /// `model` - Optional model id override. When `None`, the configured
    /// default model is used.
    fn generate(
        &self,
        prompt: &str,
        model: Option<&str>,
    ) -> Pin<Box<dyn Future<Output = Result<ImageLLMResult>> + Send + '_>>;
    /// Generate images with explicit options.
    ///
    /// `model` - Optional model id override. When `None`, the configured
    /// default model is used.
    fn generate_with_options(
        &self,
        prompt: &str,
        options: ImageLLMOptions,
        model: Option<&str>,
    ) -> Pin<Box<dyn Future<Output = Result<ImageLLMResult>> + Send + '_>>;
    /// Submit an asynchronous generation task.
    ///
    /// `model` - Optional model id override. When `None`, the configured
    /// default model is used.
    fn submit_task(
        &self,
        prompt: &str,
        options: ImageLLMOptions,
        model: Option<&str>,
    ) -> Pin<Box<dyn Future<Output = Result<ImageTask>> + Send + '_>>;
    /// Poll an asynchronous generation task by id.
    fn poll_task(
        &self,
        task_id: &str,
    ) -> Pin<Box<dyn Future<Output = Result<ImageTask>> + Send + '_>>;
    /// Get the configured default model name.
    fn get_model_name(&self) -> String;
    /// Get provider name.
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
    /// Returns a short human-readable description of this provider (English).
    pub fn description(&self) -> &'static str {
        match self {
            ImageModelProvider::Seedream => {
                "ByteDance Seedream: high-quality text-to-image with strong prompt adherence and Chinese scene understanding"
            }
            ImageModelProvider::WanImage => {
                "Alibaba Wan Image: Tongyi Wanxiang text-to-image with reference image support and rich style control"
            }
            ImageModelProvider::StabilityImage => {
                "Stability AI: Stable Image Ultra and Stable Diffusion 3.5 family for photorealistic and artistic output"
            }
            ImageModelProvider::Flux => {
                "Black Forest Labs FLUX: state-of-the-art text-to-image with fine detail and typography rendering"
            }
            ImageModelProvider::Imagen => {
                "Google Imagen: photorealistic text-to-image with strong prompt fidelity and negative prompt support"
            }
            ImageModelProvider::DallE => {
                "OpenAI DALL·E: creative text-to-image with strong instruction following and vivid styles"
            }
        }
    }
    /// Returns a short human-readable description of this provider (Chinese).
    pub fn description_zh(&self) -> &'static str {
        match self {
            ImageModelProvider::Seedream => {
                "字节跳动 Seedream：高质量文生图，提示词遵循度强，中文场景理解好"
            }
            ImageModelProvider::WanImage => "阿里云通义万相：文生图，支持参考图和丰富的风格控制",
            ImageModelProvider::StabilityImage => {
                "Stability AI：Stable Image Ultra 与 Stable Diffusion 3.5 系列，写实与艺术风格兼备"
            }
            ImageModelProvider::Flux => {
                "Black Forest Labs FLUX：顶尖文生图，细节与文字排版表现出色"
            }
            ImageModelProvider::Imagen => {
                "Google Imagen：写实文生图，提示词还原度高，支持负面提示词"
            }
            ImageModelProvider::DallE => "OpenAI DALL·E：创意文生图，指令遵循强，风格鲜明",
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
    /// Returns all concrete models under this provider, as `(model_id, display_name, is_recommended)` tuples.
    pub fn models(&self) -> Vec<(String, String, bool)> {
        match self {
            ImageModelProvider::Seedream => vec![
                ("seedream-3-0".to_string(), "Seedream 3.0".to_string(), true),
                (
                    "doubao-seedream-4-5".to_string(),
                    "Doubao-Seedream-4.5".to_string(),
                    false,
                ),
                (
                    "doubao-seedream-5-0-lite".to_string(),
                    "Doubao-Seedream-5.0-lite".to_string(),
                    false,
                ),
            ],
            ImageModelProvider::WanImage => vec![
                (
                    "wan2.5-t2i-preview".to_string(),
                    "Wan 2.5 T2I Preview".to_string(),
                    true,
                ),
                (
                    "wanx2.1-t2i-turbo".to_string(),
                    "Wan 2.1 T2I Turbo".to_string(),
                    false,
                ),
            ],
            ImageModelProvider::StabilityImage => vec![
                (
                    "stable-image-ultra".to_string(),
                    "Stable Image Ultra".to_string(),
                    true,
                ),
                (
                    "sd3.5-large".to_string(),
                    "Stable Diffusion 3.5 Large".to_string(),
                    false,
                ),
                (
                    "sd3.5-large-turbo".to_string(),
                    "Stable Diffusion 3.5 Large Turbo".to_string(),
                    false,
                ),
                (
                    "sd3.5-medium".to_string(),
                    "Stable Diffusion 3.5 Medium".to_string(),
                    false,
                ),
                (
                    "sd3.5-flash".to_string(),
                    "Stable Diffusion 3.5 Flash".to_string(),
                    false,
                ),
            ],
            ImageModelProvider::Flux => vec![
                ("flux-2-max".to_string(), "FLUX.2 [max]".to_string(), true),
                ("flux-2-pro".to_string(), "FLUX.2 [pro]".to_string(), false),
                (
                    "flux-2-klein".to_string(),
                    "FLUX.2 [klein]".to_string(),
                    false,
                ),
                ("flux-2-dev".to_string(), "FLUX.2 [dev]".to_string(), false),
            ],
            ImageModelProvider::Imagen => vec![
                (
                    "imagen-4.0-generate-001".to_string(),
                    "Imagen 4.0".to_string(),
                    true,
                ),
                (
                    "imagen-4.0-ultra-generate-001".to_string(),
                    "Imagen 4 Ultra".to_string(),
                    false,
                ),
            ],
            ImageModelProvider::DallE => vec![
                ("dall-e-3".to_string(), "DALL·E 3".to_string(), true),
                ("dall-e-2".to_string(), "DALL·E 2".to_string(), false),
            ],
        }
    }
    /// Probes this provider with a real authenticated read-only request.
    pub async fn probe(&self, api_key: &str, base_url: Option<&str>) -> Result<()> {
        let key = api_key.to_string();
        if key.is_empty() {
            return Err(crate::types::LangHubError::LLMError(
                "image key empty".to_string(),
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
            ImageModelProvider::Flux => client
                .get(format!("{}/get_result", base))
                .header("x-key", key),
            ImageModelProvider::StabilityImage => client
                .get(format!("{}/user/account", base))
                .header("Authorization", format!("Bearer {}", key)),
            ImageModelProvider::Imagen => client.get(format!("{}/models?key={}", base, key)),
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
            ImageModelProvider::Seedream => "https://ark.cn-beijing.volces.com/api/v3",
            ImageModelProvider::WanImage => "https://dashscope.aliyuncs.com/api/v1",
            ImageModelProvider::StabilityImage => "https://api.stability.ai/v1",
            ImageModelProvider::Flux => "https://api.bfl.ai/v1",
            ImageModelProvider::Imagen => "https://generativelanguage.googleapis.com/v1beta",
            ImageModelProvider::DallE => "https://api.openai.com/v1",
        }
    }
}
