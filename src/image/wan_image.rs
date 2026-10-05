use crate::image::{ImageLLM, ImageLLMOptions, ImageLLMResult, ImageTask, ImageTaskStatus};
use crate::types::{LangHubError, LangHubResult};
use serde_json::json;
use std::future::Future;
use std::pin::Pin;
#[derive(Debug, Clone)]
pub enum WanImageModel {
    Wan25,
    Wan21Turbo,
    Custom(String),
}
impl WanImageModel {
    fn as_str(&self) -> String {
        match self {
            WanImageModel::Wan25 => "wan2.5-t2i-preview".to_string(),
            WanImageModel::Wan21Turbo => "wanx2.1-t2i-turbo".to_string(),
            WanImageModel::Custom(name) => name.clone(),
        }
    }
}
impl From<WanImageModel> for String {
    fn from(model: WanImageModel) -> Self {
        model.as_str()
    }
}
#[derive(Clone)]
pub struct WanImage {
    api_key: String,
    model: WanImageModel,
    base_url: String,
    client: reqwest::Client,
    default_options: ImageLLMOptions,
}
impl WanImage {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            model: WanImageModel::Wan25,
            base_url: "https://dashscope.aliyuncs.com/api/v1".to_string(),
            client: reqwest::Client::new(),
            default_options: ImageLLMOptions::default(),
        }
    }
    pub fn with_model(mut self, model: WanImageModel) -> Self {
        self.model = model;
        self
    }
    pub fn wan25(self) -> Self {
        self.with_model(WanImageModel::Wan25)
    }
    pub fn wan21_turbo(self) -> Self {
        self.with_model(WanImageModel::Wan21Turbo)
    }
    pub fn with_base_url(mut self, base_url: &str) -> Self {
        self.base_url = base_url.to_string();
        self
    }
    pub fn with_options(mut self, options: ImageLLMOptions) -> Self {
        self.default_options = options;
        self
    }
    /// Resolve the effective model id: caller override wins over the
    /// configured default.
    fn resolve_model(&self, model_override: Option<&str>) -> String {
        match model_override {
            Some(m) => m.to_string(),
            None => self.model.clone().into(),
        }
    }
    fn build_request_body(
        &self,
        prompt: &str,
        options: &ImageLLMOptions,
        model_override: Option<&str>,
    ) -> serde_json::Value {
        let model_name: String = self.resolve_model(model_override);
        let mut input = json!({
            "prompt": prompt
        });
        if let Some(neg) = options
            .negative_prompt
            .as_ref()
            .or(self.default_options.negative_prompt.as_ref())
        {
            input["negative_prompt"] = json!(neg);
        }
        if let Some(images) = options
            .reference_images
            .as_ref()
            .or(self.default_options.reference_images.as_ref())
        {
            input["ref_images_url"] = json!(images);
        }
        let mut parameters = json!({});
        if let Some(n) = options.n.or(self.default_options.n) {
            parameters["n"] = json!(n);
        }
        if let Some(resolution) = options
            .resolution
            .as_ref()
            .or(self.default_options.resolution.as_ref())
        {
            parameters["size"] = json!(resolution);
        }
        if let Some(seed) = options.seed.or(self.default_options.seed) {
            parameters["seed"] = json!(seed);
        }
        json!({
            "model": model_name,
            "input": input,
            "parameters": parameters,
        })
    }
    async fn submit_request(
        &self,
        prompt: &str,
        options: &ImageLLMOptions,
        model_override: Option<&str>,
    ) -> LangHubResult<serde_json::Value> {
        let body = self.build_request_body(prompt, options, model_override);
        let response = self
            .client
            .post(format!(
                "{}/services/aigc/text2image/image-synthesis",
                self.base_url
            ))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .header("X-DashScope-Async", "enable")
            .json(&body)
            .send()
            .await
            .map_err(|e| LangHubError::LLMError(format!("Wan image request error: {}", e)))?;
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LangHubError::LLMError(format!(
                "Wan image API error ({}): {}",
                status, error_text
            )));
        }
        response
            .json()
            .await
            .map_err(|e| LangHubError::LLMError(format!("Wan image JSON parse error: {}", e)))
    }
    fn raw_to_result(raw: &serde_json::Value) -> ImageLLMResult {
        let mut image_urls = Vec::new();
        if let Some(results) = raw["output"]["results"].as_array() {
            for item in results {
                if let Some(url) = item["url"].as_str() {
                    image_urls.push(url.to_string());
                }
            }
        }
        if let Some(url) = raw["output"]["image_url"].as_str() {
            image_urls.push(url.to_string());
        }
        let resolution = raw["output"]["size"].as_str().map(|s| s.to_string());
        ImageLLMResult {
            image_urls,
            image_base64: None,
            file_paths: None,
            resolution,
            raw_response: raw.clone(),
        }
    }
    async fn poll_until_done(&self, task_id: &str) -> LangHubResult<ImageLLMResult> {
        let url = format!("{}/tasks/{}", self.base_url, task_id);
        for _ in 0..180 {
            let response = self
                .client
                .get(&url)
                .header("Authorization", format!("Bearer {}", self.api_key))
                .send()
                .await
                .map_err(|e| LangHubError::LLMError(format!("Wan image poll error: {}", e)))?;
            let raw: serde_json::Value = response.json().await.map_err(|e| {
                LangHubError::LLMError(format!("Wan image JSON parse error: {}", e))
            })?;
            let status = raw["output"]["task_status"].as_str().unwrap_or("");
            match status {
                "SUCCEEDED" => return Ok(Self::raw_to_result(&raw)),
                "FAILED" => {
                    let error = raw["output"]["message"]
                        .as_str()
                        .unwrap_or("Wan image task failed")
                        .to_string();
                    return Err(LangHubError::LLMError(error));
                }
                _ => {
                    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
                }
            }
        }
        Err(LangHubError::LLMError(
            "Wan image task polling timeout".to_string(),
        ))
    }
}
impl ImageLLM for WanImage {
    fn generate(
        &self,
        prompt: &str,
        model: Option<&str>,
    ) -> Pin<Box<dyn Future<Output = LangHubResult<ImageLLMResult>> + Send + '_>> {
        let prompt = prompt.to_string();
        let options = self.default_options.clone();
        let model_owned: Option<String> = model.map(|m| m.to_string());
        Box::pin(async move {
            let raw = self
                .submit_request(&prompt, &options, model_owned.as_deref())
                .await?;
            let task_id = raw["output"]["task_id"]
                .as_str()
                .ok_or_else(|| LangHubError::ParseError("Missing task id".to_string()))?
                .to_string();
            self.poll_until_done(&task_id).await
        })
    }
    fn generate_with_options(
        &self,
        prompt: &str,
        options: ImageLLMOptions,
        model: Option<&str>,
    ) -> Pin<Box<dyn Future<Output = LangHubResult<ImageLLMResult>> + Send + '_>> {
        let prompt = prompt.to_string();
        let model_owned: Option<String> = model.map(|m| m.to_string());
        Box::pin(async move {
            let raw = self
                .submit_request(&prompt, &options, model_owned.as_deref())
                .await?;
            let task_id = raw["output"]["task_id"]
                .as_str()
                .ok_or_else(|| LangHubError::ParseError("Missing task id".to_string()))?
                .to_string();
            self.poll_until_done(&task_id).await
        })
    }
    fn submit_task(
        &self,
        prompt: &str,
        options: ImageLLMOptions,
        model: Option<&str>,
    ) -> Pin<Box<dyn Future<Output = LangHubResult<ImageTask>> + Send + '_>> {
        let prompt = prompt.to_string();
        let model_owned: Option<String> = model.map(|m| m.to_string());
        Box::pin(async move {
            let raw = self
                .submit_request(&prompt, &options, model_owned.as_deref())
                .await?;
            let task_id = raw["output"]["task_id"]
                .as_str()
                .ok_or_else(|| LangHubError::ParseError("Missing task id".to_string()))?
                .to_string();
            Ok(ImageTask {
                task_id,
                status: ImageTaskStatus::Pending,
                result: None,
                error: None,
            })
        })
    }
    fn poll_task(
        &self,
        task_id: &str,
    ) -> Pin<Box<dyn Future<Output = LangHubResult<ImageTask>> + Send + '_>> {
        let task_id = task_id.to_string();
        Box::pin(async move {
            let url = format!("{}/tasks/{}", self.base_url, task_id);
            let response = self
                .client
                .get(&url)
                .header("Authorization", format!("Bearer {}", self.api_key))
                .send()
                .await
                .map_err(|e| LangHubError::LLMError(format!("Wan image poll error: {}", e)))?;
            let raw: serde_json::Value = response.json().await.map_err(|e| {
                LangHubError::LLMError(format!("Wan image JSON parse error: {}", e))
            })?;
            let status_str = raw["output"]["task_status"].as_str().unwrap_or("");
            let (status, result, error) = match status_str {
                "SUCCEEDED" => (
                    ImageTaskStatus::Succeeded,
                    Some(Self::raw_to_result(&raw)),
                    None,
                ),
                "FAILED" => (
                    ImageTaskStatus::Failed,
                    None,
                    Some(
                        raw["output"]["message"]
                            .as_str()
                            .unwrap_or("Wan image task failed")
                            .to_string(),
                    ),
                ),
                "RUNNING" => (ImageTaskStatus::Processing, None, None),
                _ => (ImageTaskStatus::Pending, None, None),
            };
            Ok(ImageTask {
                task_id,
                status,
                result,
                error,
            })
        })
    }
    fn get_model_name(&self) -> String {
        self.model.as_str()
    }
    fn get_provider_name(&self) -> String {
        "Alibaba-Wan-Image".to_string()
    }
    fn supports_reference_images(&self) -> bool {
        true
    }
    fn supports_negative_prompt(&self) -> bool {
        true
    }
}
