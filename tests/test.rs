#[cfg(test)]
mod tests {
    use langhub::{
        LLMClient,
        types::{ChatMessage, ModelProvider},
    };

    use super::*;

    #[tokio::test]
    async fn test_openai_client_creation() {
        if std::env::var("OPENAI_API_KEY").is_ok() {
            let client = LLMClient::new(ModelProvider::OpenAI);
            assert!(client.is_ok());
        } else {
            println!("not env key:OPENAI_API_KEY");
        }
    }

    #[tokio::test]
    async fn test_deepseek_client_creation() {
        if std::env::var("DEEPSEEK_API_KEY").is_ok() {
            let client = LLMClient::new(ModelProvider::DeepSeek);
            assert!(client.is_ok());
        } else {
            println!("not env key:DEEPSEEK_API_KEY");
        }
    }
}
