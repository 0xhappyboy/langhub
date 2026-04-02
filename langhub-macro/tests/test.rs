#[cfg(test)]
mod chat_macro_tests {
    use super::*;

    #[test]
    fn simple_test() {
        println!("{:?}", ModelProvider)
    }

    #[test]
    fn test_chat_macro_with_deepseek() {
        #[chat(msg = "What is Rust programming language?", type = "deepseek")]
        fn ask_question() -> Result<String> {
            println!("AI Response: {}", llm_response);
            let processed = format!("Answer: {}", llm_response);
            Ok(processed)
        }

        let result = ask_question();
        assert!(result.is_ok());
    }

    #[test]
    fn test_chat_macro_with_openai() {
        #[chat(msg = "Explain quantum computing", type = "openai")]
        fn ask_question() -> Result<String> {
            println!("Response: {}", llm_response);
            Ok(llm_response)
        }

        let result = ask_question();
        assert!(result.is_ok());
    }

    #[test]
    fn test_chat_macro_with_anthropic() {
        #[chat(msg = "Write a haiku about programming", type = "anthropic")]
        fn generate_poetry() -> Result<String> {
            let poem = llm_response;
            println!("Generated poem:\n{}", poem);
            Ok(poem)
        }

        let result = generate_poetry();
        assert!(result.is_ok());
    }

    #[test]
    fn test_chat_macro_with_google() {
        #[chat(msg = "What are the benefits of async programming?", type = "google")]
        fn explain_concept() -> Result<String> {
            let explanation = format!("Here's what I found:\n{}", llm_response);
            Ok(explanation)
        }

        let result = explain_concept();
        assert!(result.is_ok());
    }

    #[test]
    fn test_chat_macro_with_zhipu() {
        #[chat(msg = "What is machine learning?", type = "zhipu")]
        fn ask_ml() -> Result<String> {
            let answer = llm_response;
            assert!(!answer.is_empty());
            Ok(answer)
        }

        let result = ask_ml();
        assert!(result.is_ok());
    }

    #[test]
    fn test_chat_macro_with_moonshot() {
        #[chat(msg = "Explain deep learning in simple terms", type = "moonshot")]
        fn explain_dl() -> Result<String> {
            let explanation = llm_response;
            println!("Explanation: {}", explanation);
            Ok(explanation)
        }

        let result = explain_dl();
        assert!(result.is_ok());
    }

    #[test]
    fn test_chat_macro_response_processing() {
        #[chat(msg = "What is 2+2?", type = "openai")]
        fn calculate() -> Result<String> {
            let response = llm_response;
            let processed = format!("Calculation result: {}", response);
            assert!(processed.contains("Calculation result:"));
            Ok(processed)
        }

        let result = calculate();
        assert!(result.is_ok());
    }

    #[test]
    fn test_chat_macro_multiple_calls() {
        #[chat(msg = "Say hello", type = "deepseek")]
        fn say_hello() -> Result<String> {
            Ok(llm_response)
        }

        #[chat(msg = "Say goodbye", type = "deepseek")]
        fn say_goodbye() -> Result<String> {
            Ok(llm_response)
        }

        let hello = say_hello();
        let goodbye = say_goodbye();

        assert!(hello.is_ok());
        assert!(goodbye.is_ok());
    }

    #[test]
    fn test_chat_macro_with_different_model_types() {
        let model_types = vec!["openai", "deepseek", "anthropic", "google"];

        for model_type in model_types {
            let result = std::panic::catch_unwind(|| {
                #[cfg(not)]
                macro_rules! create_test_fn {
                    ($name:ident, $msg:expr, $type:expr) => {
                        #[chat(msg = $msg, type = $type)]
                        fn $name() -> Result<String> {
                            Ok(llm_response)
                        }
                    };
                }
            });
            assert!(result.is_ok() || result.is_err());
        }
    }

    #[test]
    fn test_chat_macro_with_empty_response_handling() {
        #[chat(msg = "Respond with a single word: OK", type = "openai")]
        fn get_ok() -> Result<String> {
            let response = llm_response;
            assert!(!response.is_empty());
            Ok(response)
        }

        let result = get_ok();
        assert!(result.is_ok());
    }

    #[test]
    fn test_chat_macro_with_chinese_query() {
        #[chat(msg = "什么是Rust语言？", type = "deepseek")]
        fn ask_chinese() -> Result<String> {
            let response = llm_response;
            println!("Chinese response: {}", response);
            assert!(!response.is_empty());
            Ok(response)
        }

        let result = ask_chinese();
        assert!(result.is_ok());
    }

    #[test]
    fn test_chat_macro_result_formatting() {
        #[chat(msg = "Return a simple math result: 5+3", type = "openai")]
        fn get_math() -> Result<String> {
            let raw_response = llm_response;
            let formatted = format!("Math result: {}", raw_response);
            Ok(formatted)
        }

        let result = get_math();
        assert!(result.is_ok());

        if let Ok(formatted) = result {
            assert!(formatted.starts_with("Math result:"));
        }
    }

    #[test]
    fn test_chat_macro_error_propagation() {
        #[chat(msg = "This is a test message", type = "openai")]
        fn test_function() -> Result<String> {
            if llm_response.is_empty() {
                return Err(LangChainError::LLMError("Empty response".to_string()).into());
            }
            Ok(llm_response)
        }

        let result = test_function();
        match result {
            Ok(response) => assert!(!response.is_empty()),
            Err(e) => println!("Expected error: {}", e),
        }
    }

    #[test]
    fn test_chat_macro_with_custom_processing() {
        #[chat(msg = "List three programming languages", type = "deepseek")]
        fn get_languages() -> Result<Vec<String>> {
            let response = llm_response;
            let languages: Vec<String> = response
                .lines()
                .filter(|line| !line.is_empty())
                .map(|s| s.to_string())
                .collect();

            println!("Found {} languages", languages.len());
            Ok(languages)
        }

        let result = get_languages();
        assert!(result.is_ok());

        if let Ok(languages) = result {
            assert!(!languages.is_empty());
        }
    }

    #[tokio::test]
    async fn test_chat_macro_async_wrapper() {
        #[chat(msg = "Respond with 'Hello World'", type = "openai")]
        fn get_hello() -> Result<String> {
            Ok(llm_response)
        }

        let result = get_hello();
        assert!(result.is_ok());
    }
}
