<h1 align="center">
    langhub
</h1>
<h4 align="center">
An LLM application development framework based on Rust.
</h4>
<p align="center">
  <a href="https://github.com/0xhappyboy/langhub/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache2.0-d1d1f6.svg?style=flat&labelColor=1C2C2E&color=BEC5C9&logo=googledocs&label=license&logoColor=BEC5C9" alt="License"></a>
    <a href="https://crates.io/crates/langhub">
<img src="https://img.shields.io/badge/crates-langhub-20B2AA.svg?style=flat&labelColor=0F1F2D&color=FFD700&logo=rust&logoColor=FFD700">
</a>
</p>
<p align="center">
<a href="./README_zh-CN.md">简体中文</a> | <a href="./README.md">English</a>
</p>

# Basic Usage

## Generate a welcome message with OpenAI

```rust
use langhub_macros::chat;

#[chat(msg = "Write a welcome message for a developer tool", type = "openai")]
fn generate_welcome() -> Result<String, Box<dyn std::error::Error>> {
println!("Generated: {}", llm_response);
Ok(llm_response)
}

```

## Reply to user using DeepSeek

```rust
use langhub_macros::chat;

#[chat(msg = format!("Reply to the user in a friendly tone: {}", user_input), type = "deepseek")]
fn reply_to_user(user_input: String) -> Result<String, Box<dyn std::error::Error>> {
    println!("AI reply: {}", llm_response);
    Ok(llm_response)
}
```

## Analyze sentiment with Anthropic

```rust
use langhub_macros::chat;

#[chat(msg = format!("Analyze the sentiment (positive/negative/neutral) of this text:\n{}", text), type = "anthropic")]
fn analyze_sentiment(text: String) -> Result<String, Box<dyn std::error::Error>> {
    match llm_response {
        Ok(sentiment) => {
            println!("Sentiment analysis result: {}", sentiment);
            Ok(sentiment)
        }
        Err(e) => {
            eprintln!("Analysis failed: {}", e);
            Err(e.into())
        }
    }
}
```
