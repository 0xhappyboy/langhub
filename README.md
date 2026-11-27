<h1 align="center">langhub</h1>
<h4 align="center">A Rust framework providing unified access to multimodal AI models (text, image, video, audio) across multiple vendors.</h4>

<p align="center">
  <a href="https://github.com/0xhappyboy/langhub/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache2.0-d1d1f6.svg?style=flat&labelColor=1C2C2E&color=BEC5C9&logo=googledocs&label=license&logoColor=BEC5C9" alt="License"></a>
  <a href="https://crates.io/crates/langhub"><img src="https://img.shields.io/badge/crates-langhub-20B2AA.svg?style=flat&labelColor=0F1F2D&color=FFD700&logo=rust&logoColor=FFD700"></a>
  <a href="https://crates.io/crates/langhub"><img src="https://img.shields.io/crates/d/langhub?style=flat&labelColor=0F1F2D&color=20B2AA&logo=rust&logoColor=white&label=downloads" alt="Crates.io Downloads"></a>
</p>

<p align="center">
<a href="./README_zh-CN.md">简体中文</a> | <a href="./README.md">English</a>
</p>

---

## Introduction

**langhub** is a Rust framework providing unified access to multimodal AI models (text, image, video, audio) across multiple vendors. It exposes a single unified interface covering four modalities:

| Modality    | Trait      | Client           | Description                                       |
| ----------- | ---------- | ---------------- | ------------------------------------------------- |
| Chat / Text | `LLM`      | `ChatLLMClient`  | Text generation, multi-turn chat, tool calling    |
| Image       | `ImageLLM` | `ImageLLMClient` | Text-to-image, reference images, negative prompts |
| Video       | `VideoLLM` | `VideoLLMClient` | Text-to-video, image-to-video, native audio       |
| Audio       | `AudioLLM` | `AudioLLMClient` | TTS, sound effects, music generation              |

Every modality exposes a unified set of `generate` / `generate_with_options` / `submit_task` / `poll_task` methods. Switching vendors is as simple as changing one `Provider` enum value.

## Install

```rust
cargo add langhub
```

---

## Supported Vendors and Models

### 1. Chat / Text Models

| Vendor          | Provider                         | Representative Models                                                                                                          |
| --------------- | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| OpenAI          | `ChatModelProvider::OpenAI`      | GPT-4o, GPT-4o mini, GPT-4 Turbo, GPT-4, GPT-3.5 Turbo, o1, o1-mini, o1-preview, o3-mini                                       |
| Anthropic       | `ChatModelProvider::Anthropic`   | Claude Sonnet 4, Claude Opus 4, Claude 3.7 Sonnet, Claude 3.5 Sonnet / Haiku, Claude 3 Opus / Sonnet / Haiku, Claude 2.1 / 2.0 |
| Google          | `ChatModelProvider::Google`      | Gemini 2.5 Pro / Flash, Gemini 2.0 Flash / Flash Lite, Gemini 1.5 Pro / Flash, Gemini 1.0 Pro                                  |
| DeepSeek        | `ChatModelProvider::DeepSeek`    | DeepSeek Chat, DeepSeek Reasoner, DeepSeek Coder                                                                               |
| Cohere          | `ChatModelProvider::Cohere`      | Command A, Command R+, Command R, Command R7B, Command Light, Command Nightly                                                  |
| HuggingFace     | `ChatModelProvider::HuggingFace` | Llama 3 8B / 70B, Llama 2 series, Mistral 7B, Mixtral 8x7B, Qwen 2.5 72B, Falcon 7B / 40B                                      |
| Azure OpenAI    | `ChatModelProvider::Azure`       | GPT-4o, GPT-4o mini, GPT-4 Turbo, GPT-4, GPT-3.5 Turbo                                                                         |
| Mistral AI      | `ChatModelProvider::Mistral`     | Mistral Large / Medium / Small, Pixtral Large, Codestral, Mixtral 8x22B, Mistral 7B                                            |
| Groq            | `ChatModelProvider::Groq`        | Llama 3.3 70B, Llama 3.1 8B, Mixtral 8x7B, Gemma 2 9B, DeepSeek R1 Distill, Llama 3 8B / 70B                                   |
| Together.ai     | `ChatModelProvider::Together`    | Llama 3.3 70B Turbo, Qwen 2.5 72B Turbo, Llama 3 / 2 series, Mixtral 8x7B, Mistral 7B, Gemma 7B, DeepSeek R1                   |
| Replicate       | `ChatModelProvider::Replicate`   | Llama 3 8B / 70B, Mixtral 8x7B, Mistral 7B                                                                                     |
| Fireworks AI    | `ChatModelProvider::Fireworks`   | Llama 3 8B / 70B, Mixtral 8x7B, Mistral 7B                                                                                     |
| Perplexity      | `ChatModelProvider::Perplexity`  | Sonar, Sonar Pro, Sonar Reasoning / Pro, Sonar Deep Research                                                                   |
| Baidu ERNIE     | `ChatModelProvider::Baidu`       | ERNIE 4.0, ERNIE 3.5, ERNIE Speed, ERNIE Lite, ERNIE Tiny                                                                      |
| Alibaba Qwen    | `ChatModelProvider::Alibaba`     | Qwen Max / Plus / Turbo, Qwen Max Long Context, Qwen 72B / 14B / 7B Chat, Qwen VL Plus                                         |
| Tencent Hunyuan | `ChatModelProvider::Tencent`     | Hunyuan Pro, Hunyuan Standard, Hunyuan Lite                                                                                    |
| Zhipu AI        | `ChatModelProvider::Zhipu`       | GLM-4 Plus, GLM-4, GLM-4 Air, GLM-4 Flash, GLM-3 Turbo, GLM-4V                                                                 |
| MiniMax         | `ChatModelProvider::MiniMax`     | abab6.5 Chat, abab6.5s Chat, abab5.5 Chat, abab5.5s Chat                                                                       |
| Moonshot Kimi   | `ChatModelProvider::Moonshot`    | Moonshot v1 128K / 32K / 8K, Kimi K2                                                                                           |
| Baichuan        | `ChatModelProvider::Baichuan`    | Baichuan 4, Baichuan 3 Turbo / 3, Baichuan 2 Turbo / 2                                                                         |
| 01.AI Yi        | `ChatModelProvider::Yi`          | Yi Large, Yi 34B Chat, Yi 34B Chat 200K, Yi 9B Chat, Yi 6B Chat                                                                |
| Custom          | `ChatModelProvider::Custom`      | Any OpenAI-compatible endpoint (configurable API Base URL)                                                                     |

### 2. Image Generation Models

| Vendor             | Provider                             | Representative Models                                           |
| ------------------ | ------------------------------------ | --------------------------------------------------------------- |
| ByteDance Seedream | `ImageModelProvider::Seedream`       | Seedream 3.0, 4.0, 4.5, 5.0 Lite / Flash / Pro                  |
| Alibaba Wan        | `ImageModelProvider::WanImage`       | Wan 2.5 T2I Preview, Wan 2.1 T2I Turbo                          |
| Stability AI       | `ImageModelProvider::StabilityImage` | Stable Image Ultra, SD 3.5 Large / Large Turbo / Medium / Flash |
| Black Forest Labs  | `ImageModelProvider::Flux`           | FLUX.2 [max], [pro], [klein], [dev]                             |
| Google Imagen      | `ImageModelProvider::Imagen`         | Imagen 4.0, Imagen 4 Ultra                                      |
| OpenAI DALL·E      | `ImageModelProvider::DallE`          | DALL·E 3, DALL·E 2                                              |

### 3. Video Generation Models

| Vendor             | Provider                              | Representative Models                                                    |
| ------------------ | ------------------------------------- | ------------------------------------------------------------------------ |
| ByteDance Seedance | `VideoModelProvider::Seedance`        | Seedance 1.0 Pro / Pro Fast, 1.5 Pro, 2.0 / Fast / Mini, 2.5 / 2.5 Flash |
| Alibaba Wan        | `VideoModelProvider::Wan`             | Wan 3.0, Wan 3.0 Prime                                                   |
| Kuaishou Kling     | `VideoModelProvider::Kling`           | Kling 3.0, Kling 3.0 Omni, Kling 4.0                                     |
| Google Veo         | `VideoModelProvider::Veo`             | Veo 3.1, Veo 3.1 Fast, Veo 3.1 Lite                                      |
| Runway             | `VideoModelProvider::Runway`          | Runway Gen-4.5, Runway Aleph 2.0                                         |
| MiniMax Hailuo     | `VideoModelProvider::MiniMaxH3`       | MiniMax Hailuo H3                                                        |
| Alibaba HappyHorse | `VideoModelProvider::HappyHorse`      | HappyHorse 1.0                                                           |
| Lightricks LTX     | `VideoModelProvider::Ltx`             | LTX-2.3 Fast, LTX-2.3 Pro                                                |
| xAI Grok           | `VideoModelProvider::GrokImagine`     | Grok Imagine 1.5 / 1.0                                                   |
| Pruna              | `VideoModelProvider::Pruna`           | Pruna P-Video 2 Pro                                                      |
| Google Gemini Omni | `VideoModelProvider::GeminiOmniFlash` | Gemini Omni Flash 1.1                                                    |

### 4. Audio Generation Models

| Vendor        | Provider                          | Category          | Representative Models                            |
| ------------- | --------------------------------- | ----------------- | ------------------------------------------------ |
| Alibaba Cloud | `AudioModelProvider::QwenTts`     | TTS               | Qwen-Audio 3.1 TTS, Qwen-Audio 3.1 TTS-Next      |
| ByteDance     | `AudioModelProvider::SeedAudio`   | Sound effects     | Seed Audio 1.0                                   |
| StepFun       | `AudioModelProvider::StepAudio`   | TTS / SFX / Music | StepAudio3 TTS, StepAudio3 Gen, StepAudio3 Music |
| Google        | `AudioModelProvider::GeminiTts`   | TTS               | Gemini 3.8 Flash TTS                             |
| ElevenLabs    | `AudioModelProvider::ElevenLabs`  | TTS               | Eleven v4                                        |
| Google        | `AudioModelProvider::Lyria`       | Music             | Lyria 3 Pro, Lyria 3                             |
| Suno          | `AudioModelProvider::Suno`        | Music             | Suno v6, Suno v6 Wild, Suno v6 Mini              |
| Stability AI  | `AudioModelProvider::StableAudio` | Music             | Stable Audio 2.5                                 |

---

## Quick Start

### Chat Example

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

Switching vendors is as simple as changing the `Provider` enum:

```rust
// Switch to DeepSeek
let config = ChatLLMConfig::new().deepseek("your-key".to_string());
let client = ChatLLMClient::new_with_config(ChatModelProvider::DeepSeek, &config)?;
```

### Video Generation Example

```rust
use langhub::{VideoLLMClient, VideoLLMConfig, VideoModelProvider};

let config = VideoLLMConfig::new().seedance("your-ark-api-key".to_string());
let client = VideoLLMClient::new_with_config(VideoModelProvider::Seedance, &config)?;
let result = client.generate("A cat walking on the beach", None).await?;
```

**Seedance 2.0 with first/last frame:**

```rust
use langhub::{VideoLLMClient, VideoLLMConfig, VideoModelProvider, VideoLLMOptions};

let config = VideoLLMConfig::new().seedance("your-ark-api-key".to_string());
let client = VideoLLMClient::new_with_config(VideoModelProvider::Seedance, &config)?;

let mut options = VideoLLMOptions::default();
options.first_frame = Some("https://example.com/start.jpg".to_string());
options.last_frame = Some("https://example.com/end.jpg".to_string());
options.duration = Some(5.0);

let result = client.generate_with_options(
    "A cinematic transition from day to night",
    options,
    Some("doubao-seedance-2-0-260128")
).await?;
```

**Seedance 2.5 with content array and roles:**

```rust
use langhub::{VideoLLMClient, VideoLLMConfig, VideoModelProvider, VideoLLMOptions};

let config = VideoLLMConfig::new().seedance("your-ark-api-key".to_string());
let client = VideoLLMClient::new_with_config(VideoModelProvider::Seedance, &config)?;

let mut options = VideoLLMOptions::default();
options.reference_images = Some(vec![
    "https://example.com/ref1.jpg".to_string(),
    "https://example.com/ref2.jpg".to_string(),
]);
options.duration = Some(10.0);
options.return_last_frame = Some(true);

let result = client.generate_with_options(
    "A character walking through a scene",
    options,
    Some("doubao-seedance-2-5-260628")
).await?;
```

### Image Generation Example

```rust
use langhub::{ImageLLMClient, ImageLLMConfig, ImageModelProvider};

let config = ImageLLMConfig::new().seedream("your-ark-api-key".to_string());
let client = ImageLLMClient::new_with_config(ImageModelProvider::Seedream, &config)?;
let result = client.generate("A cat sitting on a windowsill", None).await?;
```

**Image with reference and negative prompt:**

```rust
use langhub::{ImageLLMClient, ImageLLMConfig, ImageModelProvider, ImageLLMOptions};

let config = ImageLLMConfig::new().flux("your-bfl-api-key".to_string());
let client = ImageLLMClient::new_with_config(ImageModelProvider::Flux, &config)?;

let mut options = ImageLLMOptions::default();
options.reference_images = Some(vec!["https://example.com/style-ref.jpg".to_string()]);
options.negative_prompt = Some("blurry, low quality".to_string());
options.aspect_ratio = Some("16:9".to_string());

let result = client.generate_with_options(
    "A futuristic cityscape at sunset",
    options,
    Some("flux-2-pro")
).await?;
```

### Audio Generation Example

```rust
use langhub::{AudioLLMClient, AudioLLMConfig, AudioModelProvider};

let config = AudioLLMConfig::new().qwen_tts("your-dashscope-api-key".to_string());
let client = AudioLLMClient::new_with_config(AudioModelProvider::QwenTts, &config)?;
let result = client.generate("Hello, world!", None).await?;
```

**TTS with voice and emotion:**

```rust
use langhub::{AudioLLMClient, AudioLLMConfig, AudioModelProvider, AudioLLMOptions};

let config = AudioLLMConfig::new().qwen_tts("your-dashscope-api-key".to_string());
let client = AudioLLMClient::new_with_config(AudioModelProvider::QwenTts, &config)?;

let mut options = AudioLLMOptions::default();
options.voice = Some("Cherry".to_string());
options.emotion = Some("happy".to_string());
options.speed = Some(1.2);

let result = client.generate_with_options(
    "Hello, this is a test!",
    options,
    Some("qwen-audio-3.1-tts")
).await?;
```

**Music generation with lyrics:**

```rust
use langhub::{AudioLLMClient, AudioLLMConfig, AudioModelProvider, AudioLLMOptions};

let config = AudioLLMConfig::new().suno("your-suno-api-key".to_string());
let client = AudioLLMClient::new_with_config(AudioModelProvider::Suno, &config)?;

let mut options = AudioLLMOptions::default();
options.lyrics = Some("Verse 1: ...\nChorus: ...".to_string());
options.instrumental = Some(false);

let result = client.generate_with_options(
    "A pop song about summer",
    options,
    Some("suno-v6")
).await?;
```

---

## License

Apache-2.0
