<h1 align="center">langhub</h1>
<h4 align="center">一个基于Rust开发的LLM统一抽象层框架.</h4>

<p align="center">
  <a href="https://github.com/0xhappyboy/langhub/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache2.0-d1d1f6.svg?style=flat&labelColor=1C2C2E&color=BEC5C9&logo=googledocs&label=license&logoColor=BEC5C9" alt="License"></a>
  <a href="https://crates.io/crates/langhub"><img src="https://img.shields.io/badge/crates-langhub-20B2AA.svg?style=flat&labelColor=0F1F2D&color=FFD700&logo=rust&logoColor=FFD700"></a>
  <a href="https://crates.io/crates/langhub"><img src="https://img.shields.io/crates/d/langhub?style=flat&labelColor=0F1F2D&color=20B2AA&logo=rust&logoColor=white&label=downloads" alt="Crates.io Downloads"></a>
</p>

<p align="center">
<a href="./README_zh-CN.md">简体中文</a> | <a href="./README.md">English</a>
</p>

---

## 简介

**langhub** 一个基于Rust开发的LLM统一抽象层框架, 用**统一的接口**接入多家 AI 厂商，覆盖四大模态：

| 模态        | Trait      | 客户端           | 说明                         |
| ----------- | ---------- | ---------------- | ---------------------------- |
| 对话 / 文本 | `LLM`      | `ChatLLMClient`  | 文本生成、多轮对话、工具调用 |
| 图像生成    | `ImageLLM` | `ImageLLMClient` | 文生图、参考图、负面提示词   |
| 视频生成    | `VideoLLM` | `VideoLLMClient` | 文生视频、图生视频、原生音频 |
| 音频生成    | `AudioLLM` | `AudioLLMClient` | TTS、音效、音乐生成          |

每个模态都提供统一的 `generate` / `generate_with_options` / `submit_task` / `poll_task` 接口，切换厂商只需换一个 `Provider` 枚举值。

## 安装

```bash
cargo add langhub
```

---

## 支持的厂商与模型

### 一、对话 / 文本模型（Chat / Text）

| 厂商           | Provider                         | 代表模型                                                                                                                       |
| -------------- | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| OpenAI         | `ChatModelProvider::OpenAI`      | GPT-4o、GPT-4o mini、GPT-4 Turbo、GPT-4、GPT-3.5 Turbo、o1、o1-mini、o1-preview、o3-mini                                       |
| Anthropic      | `ChatModelProvider::Anthropic`   | Claude Sonnet 4、Claude Opus 4、Claude 3.7 Sonnet、Claude 3.5 Sonnet / Haiku、Claude 3 Opus / Sonnet / Haiku、Claude 2.1 / 2.0 |
| Google         | `ChatModelProvider::Google`      | Gemini 2.5 Pro / Flash、Gemini 2.0 Flash / Flash Lite、Gemini 1.5 Pro / Flash、Gemini 1.0 Pro                                  |
| DeepSeek       | `ChatModelProvider::DeepSeek`    | DeepSeek Chat、DeepSeek Reasoner、DeepSeek Coder                                                                               |
| Cohere         | `ChatModelProvider::Cohere`      | Command A、Command R+、Command R、Command R7B、Command Light、Command Nightly                                                  |
| HuggingFace    | `ChatModelProvider::HuggingFace` | Llama 3 8B / 70B、Llama 2 系列、Mistral 7B、Mixtral 8x7B、Qwen 2.5 72B、Falcon 7B / 40B                                        |
| Azure OpenAI   | `ChatModelProvider::Azure`       | GPT-4o、GPT-4o mini、GPT-4 Turbo、GPT-4、GPT-3.5 Turbo                                                                         |
| Mistral AI     | `ChatModelProvider::Mistral`     | Mistral Large / Medium / Small、Pixtral Large、Codestral、Mixtral 8x22B、Mistral 7B                                            |
| Groq           | `ChatModelProvider::Groq`        | Llama 3.3 70B、Llama 3.1 8B、Mixtral 8x7B、Gemma 2 9B、DeepSeek R1 Distill、Llama 3 8B / 70B                                   |
| Together.ai    | `ChatModelProvider::Together`    | Llama 3.3 70B Turbo、Qwen 2.5 72B Turbo、Llama 3 / 2 系列、Mixtral 8x7B、Mistral 7B、Gemma 7B、DeepSeek R1                     |
| Replicate      | `ChatModelProvider::Replicate`   | Llama 3 8B / 70B、Mixtral 8x7B、Mistral 7B                                                                                     |
| Fireworks AI   | `ChatModelProvider::Fireworks`   | Llama 3 8B / 70B、Mixtral 8x7B、Mistral 7B                                                                                     |
| Perplexity     | `ChatModelProvider::Perplexity`  | Sonar、Sonar Pro、Sonar Reasoning / Pro、Sonar Deep Research                                                                   |
| 百度文心       | `ChatModelProvider::Baidu`       | ERNIE 4.0、ERNIE 3.5、ERNIE Speed、ERNIE Lite、ERNIE Tiny                                                                      |
| 阿里云通义千问 | `ChatModelProvider::Alibaba`     | Qwen Max / Plus / Turbo、Qwen Max Long Context、Qwen 72B / 14B / 7B Chat、Qwen VL Plus                                         |
| 腾讯混元       | `ChatModelProvider::Tencent`     | Hunyuan Pro、Hunyuan Standard、Hunyuan Lite                                                                                    |
| 智谱 AI        | `ChatModelProvider::Zhipu`       | GLM-4 Plus、GLM-4、GLM-4 Air、GLM-4 Flash、GLM-3 Turbo、GLM-4V                                                                 |
| MiniMax        | `ChatModelProvider::MiniMax`     | abab6.5 Chat、abab6.5s Chat、abab5.5 Chat、abab5.5s Chat                                                                       |
| 月之暗面 Kimi  | `ChatModelProvider::Moonshot`    | Moonshot v1 128K / 32K / 8K、Kimi K2                                                                                           |
| 百川智能       | `ChatModelProvider::Baichuan`    | Baichuan 4、Baichuan 3 Turbo / 3、Baichuan 2 Turbo / 2                                                                         |
| 零一万物       | `ChatModelProvider::Yi`          | Yi Large、Yi 34B Chat、Yi 34B Chat 200K、Yi 9B Chat、Yi 6B Chat                                                                |
| 自定义         | `ChatModelProvider::Custom`      | 任意 OpenAI 兼容端点（可配置 API Base URL）                                                                                    |

### 二、图像生成模型（Image）

| 厂商              | Provider                             | 代表模型                                                        |
| ----------------- | ------------------------------------ | --------------------------------------------------------------- |
| 字节跳动 Seedream | `ImageModelProvider::Seedream`       | Seedream 3.0、4.0、4.5、5.0 Lite / Flash / Pro                  |
| 阿里云通义万相    | `ImageModelProvider::WanImage`       | Wan 2.5 T2I Preview、Wan 2.1 T2I Turbo                          |
| Stability AI      | `ImageModelProvider::StabilityImage` | Stable Image Ultra、SD 3.5 Large / Large Turbo / Medium / Flash |
| Black Forest Labs | `ImageModelProvider::Flux`           | FLUX.2 [max]、[pro]、[klein]、[dev]                             |
| Google Imagen     | `ImageModelProvider::Imagen`         | Imagen 4.0、Imagen 4 Ultra                                      |
| OpenAI DALL·E     | `ImageModelProvider::DallE`          | DALL·E 3、DALL·E 2                                              |

### 三、视频生成模型（Video）

| 厂商               | Provider                              | 代表模型                                                     |
| ------------------ | ------------------------------------- | ------------------------------------------------------------ |
| 字节跳动 Seedance  | `VideoModelProvider::Seedance`        | Seedance 1.0 Pro / Pro Fast、1.5 Pro、2.0 / Fast / Mini、2.5 |
| 阿里云通义万相     | `VideoModelProvider::Wan`             | Wan 3.0、Wan 3.0 Prime                                       |
| 快手可灵           | `VideoModelProvider::Kling`           | Kling 3.0、Kling 3.0 Omni、Kling 4.0                         |
| Google Veo         | `VideoModelProvider::Veo`             | Veo 3.1、Veo 3.1 Fast、Veo 3.1 Lite                          |
| Runway             | `VideoModelProvider::Runway`          | Runway Gen-4.5、Runway Aleph 2.0                             |
| MiniMax 海螺       | `VideoModelProvider::MiniMaxH3`       | MiniMax Hailuo H3                                            |
| 阿里云 HappyHorse  | `VideoModelProvider::HappyHorse`      | HappyHorse 1.0                                               |
| Lightricks LTX     | `VideoModelProvider::Ltx`             | LTX-2.3 Fast、LTX-2.3 Pro                                    |
| xAI Grok           | `VideoModelProvider::GrokImagine`     | Grok Imagine 1.5 / 1.0                                       |
| Pruna              | `VideoModelProvider::Pruna`           | Pruna P-Video 2 Pro                                          |
| Google Gemini Omni | `VideoModelProvider::GeminiOmniFlash` | Gemini Omni Flash 1.1                                        |

### 四、音频生成模型（Audio）

| 厂商         | Provider                          | 类别              | 代表模型                                         |
| ------------ | --------------------------------- | ----------------- | ------------------------------------------------ |
| 阿里云       | `AudioModelProvider::QwenTts`     | TTS               | Qwen-Audio 3.1 TTS、Qwen-Audio 3.1 TTS-Next      |
| 字节跳动     | `AudioModelProvider::SeedAudio`   | 音效              | Seed Audio 1.0                                   |
| 阶跃星辰     | `AudioModelProvider::StepAudio`   | TTS / 音效 / 音乐 | StepAudio3 TTS、StepAudio3 Gen、StepAudio3 Music |
| Google       | `AudioModelProvider::GeminiTts`   | TTS               | Gemini 3.8 Flash TTS                             |
| ElevenLabs   | `AudioModelProvider::ElevenLabs`  | TTS               | Eleven v4                                        |
| Google       | `AudioModelProvider::Lyria`       | 音乐              | Lyria 3 Pro、Lyria 3                             |
| Suno         | `AudioModelProvider::Suno`        | 音乐              | Suno v6、Suno v6 Wild、Suno v6 Mini              |
| Stability AI | `AudioModelProvider::StableAudio` | 音乐              | Stable Audio 2.5                                 |

---

## 快速开始

```rust
use langhub::{ChatLLMClient, ChatLLMConfig, ChatModelProvider};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = ChatLLMConfig::new().openai("sk-xxx".to_string());
    let client = ChatLLMClient::new_with_config(ChatModelProvider::OpenAI, &config)?;

    let response = client.generate("Hello, world!").await?;
    println!("{}", response.text);

    Ok(())
}
```

切换厂商只需更改 `Provider` 枚举：

```rust
// 换成 DeepSeek
let config = ChatLLMConfig::new().deepseek("your-key".to_string());
let client = ChatLLMClient::new_with_config(ChatModelProvider::DeepSeek, &config)?;
```

### 视频生成示例

```rust
use langhub::{VideoLLMClient, VideoLLMConfig, VideoModelProvider};

let config = VideoLLMConfig::new().seedance("your-ark-api-key".to_string());
let client = VideoLLMClient::new_with_config(VideoModelProvider::Seedance, &config)?;
let result = client.generate("A cat walking on the beach", None).await?;
```

### 图像生成示例

```rust
use langhub::{ImageLLMClient, ImageLLMConfig, ImageModelProvider};

let config = ImageLLMConfig::new().seedream("your-ark-api-key".to_string());
let client = ImageLLMClient::new_with_config(ImageModelProvider::Seedream, &config)?;
let result = client.generate("A cat sitting on a windowsill", None).await?;
```

### 音频生成示例

```rust
use langhub::{AudioLLMClient, AudioLLMConfig, AudioModelProvider};

let config = AudioLLMConfig::new().qwen_tts("your-dashscope-api-key".to_string());
let client = AudioLLMClient::new_with_config(AudioModelProvider::QwenTts, &config)?;
let result = client.generate("Hello, world!", None).await?;
```

---

## License

Apache-2.0
