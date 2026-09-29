#[cfg(test)]
mod tests {
    use super::*;
    use langhub::types::VideoVendor;
use langhub::video::VideoModelProvider;
use langhub::{LLMClient, LLMConfig, types::ModelProvider};
    #[tokio::test]
    async fn test_openai_client_creation() {
        let config = LLMConfig::new().openai("test-api-key".to_string());
        let client = LLMClient::new_with_config(ModelProvider::OpenAI, &config);
        assert!(client.is_ok());
    }
    #[tokio::test]
    async fn test_deepseek_client_creation() {
        let config = LLMConfig::new().deepseek("test-api-key".to_string());
        let client = LLMClient::new_with_config(ModelProvider::DeepSeek, &config);
        assert!(client.is_ok());
    }
    #[tokio::test]
    async fn test_anthropic_client_creation() {
        let config = LLMConfig::new().anthropic("test-api-key".to_string());
        let client = LLMClient::new_with_config(ModelProvider::Anthropic, &config);
        assert!(client.is_ok());
    }
    #[tokio::test]
    async fn test_google_client_creation() {
        let config = LLMConfig::new().google("test-api-key".to_string());
        let client = LLMClient::new_with_config(ModelProvider::Google, &config);
        assert!(client.is_ok());
    }
    #[tokio::test]
    async fn test_missing_api_key_error() {
        let config = LLMConfig::new();
        let client = LLMClient::new_with_config(ModelProvider::OpenAI, &config);
        assert!(client.is_err());
    }
    // Video LLM client tests
    use langhub::{VideoLLMClient, VideoLLMConfig};
    /// Verifies that a Seedance client can be created from a config with an API key.
    #[tokio::test]
    async fn test_seedance_client_creation() {
        let config = VideoLLMConfig::new().seedance("test-api-key".to_string());
        let client = VideoLLMClient::new_with_config(VideoModelProvider::Seedance, &config);
        assert!(client.is_ok());
    }
    /// Verifies that a Wan client can be created from a config with an API key.
    #[tokio::test]
    async fn test_wan_client_creation() {
        let config = VideoLLMConfig::new().wan("test-api-key".to_string());
        let client = VideoLLMClient::new_with_config(VideoModelProvider::Wan, &config);
        assert!(client.is_ok());
    }
    /// Verifies that a Kling client can be created with both API key and secret key.
    #[tokio::test]
    async fn test_kling_client_creation() {
        let config =
            VideoLLMConfig::new().kling("test-api-key".to_string(), "test-secret-key".to_string());
        let client = VideoLLMClient::new_with_config(VideoModelProvider::Kling, &config);
        assert!(client.is_ok());
    }
    /// Verifies that a Veo client can be created from a config with an API key.
    #[tokio::test]
    async fn test_veo_client_creation() {
        let config = VideoLLMConfig::new().veo("test-api-key".to_string());
        let client = VideoLLMClient::new_with_config(VideoModelProvider::Veo, &config);
        assert!(client.is_ok());
    }
    /// Verifies that a Runway client can be created from a config with an API key.
    #[tokio::test]
    async fn test_runway_client_creation() {
        let config = VideoLLMConfig::new().runway("test-api-key".to_string());
        let client = VideoLLMClient::new_with_config(VideoModelProvider::Runway, &config);
        assert!(client.is_ok());
    }
    /// Verifies that a MiniMax H3 client can be created with API key and group ID.
    #[tokio::test]
    async fn test_minimax_h3_client_creation() {
        let config = VideoLLMConfig::new()
            .minimax_h3("test-api-key".to_string(), "test-group-id".to_string());
        let client = VideoLLMClient::new_with_config(VideoModelProvider::MiniMaxH3, &config);
        assert!(client.is_ok());
    }
    /// Verifies that a HappyHorse client can be created from a config with an API key.
    #[tokio::test]
    async fn test_happyhorse_client_creation() {
        let config = VideoLLMConfig::new().happyhorse("test-api-key".to_string());
        let client = VideoLLMClient::new_with_config(VideoModelProvider::HappyHorse, &config);
        assert!(client.is_ok());
    }
    /// Verifies that an LTX client can be created from a config with an API key.
    #[tokio::test]
    async fn test_ltx_client_creation() {
        let config = VideoLLMConfig::new().ltx("test-api-key".to_string());
        let client = VideoLLMClient::new_with_config(VideoModelProvider::Ltx, &config);
        assert!(client.is_ok());
    }
    /// Verifies that a Grok Imagine client can be created from a config with an API key.
    #[tokio::test]
    async fn test_grok_imagine_client_creation() {
        let config = VideoLLMConfig::new().grok("test-api-key".to_string());
        let client = VideoLLMClient::new_with_config(VideoModelProvider::GrokImagine, &config);
        assert!(client.is_ok());
    }
    /// Verifies that a Pruna client can be created from a config with an API key.
    #[tokio::test]
    async fn test_pruna_client_creation() {
        let config = VideoLLMConfig::new().pruna("test-api-key".to_string());
        let client = VideoLLMClient::new_with_config(VideoModelProvider::Pruna, &config);
        assert!(client.is_ok());
    }
    /// Verifies that a Gemini Omni Flash client can be created from a config with an API key.
    #[tokio::test]
    async fn test_gemini_omni_flash_client_creation() {
        let config = VideoLLMConfig::new().gemini("test-api-key".to_string());
        let client = VideoLLMClient::new_with_config(VideoModelProvider::GeminiOmniFlash, &config);
        assert!(client.is_ok());
    }
    /// Verifies that creating a video client without an API key returns an error.
    #[tokio::test]
    async fn test_video_missing_api_key_error() {
        let config = VideoLLMConfig::new();
        let client = VideoLLMClient::new_with_config(VideoModelProvider::Seedance, &config);
        assert!(client.is_err());
    }
    /// Verifies that `new_with_key` creates a Seedance client successfully.
    #[tokio::test]
    async fn test_video_new_with_key_seedance() {
        let client = VideoLLMClient::new_with_key(
            VideoModelProvider::Seedance,
            Some("test-api-key".to_string()),
            None,
        );
        assert!(client.is_ok());
    }
    /// Verifies that `new_with_key` returns an error when the API key is missing.
    #[tokio::test]
    async fn test_video_new_with_key_missing() {
        let client = VideoLLMClient::new_with_key(VideoModelProvider::Wan, None, None);
        assert!(client.is_err());
    }
    /// Verifies that `get_provider_enum` returns the correct provider.
    #[tokio::test]
    async fn test_video_get_provider_enum() {
        let config = VideoLLMConfig::new().seedance("test-api-key".to_string());
        let client =
            VideoLLMClient::new_with_config(VideoModelProvider::Seedance, &config).unwrap();
        assert_eq!(client.get_provider_enum(), VideoModelProvider::Seedance);
    }
    /// Verifies that `get_vendor` returns the correct vendor for each provider.
    #[tokio::test]
    async fn test_video_get_vendor() {
        let config = VideoLLMConfig::new()
            .seedance("test-api-key".to_string())
            .wan("test-api-key".to_string())
            .kling("test-api-key".to_string(), "test-secret".to_string())
            .veo("test-api-key".to_string())
            .runway("test-api-key".to_string())
            .minimax_h3("test-api-key".to_string(), "test-group".to_string())
            .happyhorse("test-api-key".to_string())
            .ltx("test-api-key".to_string())
            .grok("test-api-key".to_string())
            .pruna("test-api-key".to_string())
            .gemini("test-api-key".to_string());
        let seedance =
            VideoLLMClient::new_with_config(VideoModelProvider::Seedance, &config).unwrap();
        assert_eq!(seedance.get_vendor(), VideoVendor::ByteDance);
        let wan = VideoLLMClient::new_with_config(VideoModelProvider::Wan, &config).unwrap();
        assert_eq!(wan.get_vendor(), VideoVendor::Alibaba);
        let kling = VideoLLMClient::new_with_config(VideoModelProvider::Kling, &config).unwrap();
        assert_eq!(kling.get_vendor(), VideoVendor::Kuaishou);
        let veo = VideoLLMClient::new_with_config(VideoModelProvider::Veo, &config).unwrap();
        assert_eq!(veo.get_vendor(), VideoVendor::Google);
        let runway = VideoLLMClient::new_with_config(VideoModelProvider::Runway, &config).unwrap();
        assert_eq!(runway.get_vendor(), VideoVendor::Runway);
        let minimax =
            VideoLLMClient::new_with_config(VideoModelProvider::MiniMaxH3, &config).unwrap();
        assert_eq!(minimax.get_vendor(), VideoVendor::MiniMax);
        let happyhorse =
            VideoLLMClient::new_with_config(VideoModelProvider::HappyHorse, &config).unwrap();
        assert_eq!(happyhorse.get_vendor(), VideoVendor::Alibaba);
        let ltx = VideoLLMClient::new_with_config(VideoModelProvider::Ltx, &config).unwrap();
        assert_eq!(ltx.get_vendor(), VideoVendor::Lightricks);
        let grok =
            VideoLLMClient::new_with_config(VideoModelProvider::GrokImagine, &config).unwrap();
        assert_eq!(grok.get_vendor(), VideoVendor::Xai);
        let pruna = VideoLLMClient::new_with_config(VideoModelProvider::Pruna, &config).unwrap();
        assert_eq!(pruna.get_vendor(), VideoVendor::Pruna);
        let gemini =
            VideoLLMClient::new_with_config(VideoModelProvider::GeminiOmniFlash, &config).unwrap();
        assert_eq!(gemini.get_vendor(), VideoVendor::Google);
    }
    /// Verifies `VideoModelProvider::all()` returns every supported provider.
    #[tokio::test]
    async fn test_video_model_provider_all() {
        let all = VideoModelProvider::all();
        assert_eq!(all.len(), 11);
        assert!(all.contains(&VideoModelProvider::Seedance));
        assert!(all.contains(&VideoModelProvider::Wan));
        assert!(all.contains(&VideoModelProvider::Kling));
        assert!(all.contains(&VideoModelProvider::Veo));
        assert!(all.contains(&VideoModelProvider::Runway));
        assert!(all.contains(&VideoModelProvider::MiniMaxH3));
        assert!(all.contains(&VideoModelProvider::HappyHorse));
        assert!(all.contains(&VideoModelProvider::Ltx));
        assert!(all.contains(&VideoModelProvider::GrokImagine));
        assert!(all.contains(&VideoModelProvider::Pruna));
        assert!(all.contains(&VideoModelProvider::GeminiOmniFlash));
    }
    /// Verifies `VideoVendor::all()` returns every supported vendor.
    #[tokio::test]
    async fn test_video_vendor_all() {
        let all = VideoVendor::all();
        assert_eq!(all.len(), 10);
    }
    /// Verifies capability flags for a few representative providers.
    #[tokio::test]
    async fn test_video_capability_flags() {
        assert!(VideoModelProvider::Seedance.supports_audio());
        assert!(VideoModelProvider::Seedance.supports_reference_images());
        assert!(VideoModelProvider::Seedance.supports_reference_videos());
        assert!(!VideoModelProvider::Runway.supports_audio());
        assert!(VideoModelProvider::Runway.supports_reference_images());
        assert!(VideoModelProvider::Runway.supports_reference_videos());
        assert!(!VideoModelProvider::HappyHorse.supports_audio());
        assert!(VideoModelProvider::HappyHorse.supports_reference_images());
        assert!(!VideoModelProvider::HappyHorse.supports_reference_videos());
    }
}
