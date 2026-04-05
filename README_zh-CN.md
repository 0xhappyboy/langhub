<h1 align="center">
    langhub
</h1>
<h4 align="center">
一个基于Rust开发的LLM应用开发框架.
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

# 基本用法

## 使用 OpenAI 生成欢迎语

```rust
use langhub_macros::chat;

#[chat(msg = "为开发者工具写一句欢迎词", type = "openai")]
fn generate_welcome() -> Result<String, Box<dyn std::error::Error>> {
    println!("生成结果：{}", llm_response);
    Ok(llm_response)
}

```

## 使用 DeepSeek 回复用户

```rust
use langhub_macros::chat;

#[chat(msg = format!("请用友好的语气回复用户：{}", user_input), type = "deepseek")]
fn reply_to_user(user_input: String) -> Result<String, Box<dyn std::error::Error>> {
    println!("AI 回复：{}", llm_response);
    Ok(llm_response)
}
```

## 使用 Anthropic 分析文本情感

```rust
use langhub_macros::chat;

#[chat(msg = format!("分析以下文本的情感（积极/消极/中性）：\n{}", text), type = "anthropic")]
fn analyze_sentiment(text: String) -> Result<String, Box<dyn std::error::Error>> {
    match llm_response {
        Ok(sentiment) => {
            println!("情感分析结果：{}", sentiment);
            Ok(sentiment)
        }
        Err(e) => {
            eprintln!("分析失败：{}", e);
            Err(e.into())
        }
    }
}
```
