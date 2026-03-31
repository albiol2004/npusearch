use crate::config::ApiConfig;
use serde_json::json;
use std::time::Duration;

pub struct EmbeddingClient {
    pub http: reqwest::blocking::Client,
    endpoint: String,
    api_path: String,
    model: String,
    llm_model: String,
}

pub struct EmbeddingInfo {
    pub dimensions: u32,
    pub model_id: String,
}

impl EmbeddingClient {
    pub fn new(config: &ApiConfig) -> Self {
        let http = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(config.timeout))
            .build()
            .expect("failed to build HTTP client");

        Self {
            http,
            endpoint: config.endpoint.clone(),
            api_path: config.api_path.clone(),
            model: config.model.clone(),
            llm_model: config.llm_model.clone(),
        }
    }

    /// Create a new client sharing the same HTTP client (for worker threads).
    pub fn clone_for_worker(&self, config: &ApiConfig) -> Self {
        Self {
            http: self.http.clone(),
            endpoint: config.endpoint.clone(),
            api_path: config.api_path.clone(),
            model: config.model.clone(),
            llm_model: config.llm_model.clone(),
        }
    }

    /// Generate an embedding for the given text.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let url = format!("{}{}/embeddings", self.endpoint, self.api_path);

        let body = json!({
            "input": text,
            "model": self.model,
        });

        let response = self
            .http
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().unwrap_or_default();
            return Err(format!("Embedding API request failed ({}): {}", status, text).into());
        }

        let data: serde_json::Value = response.json()?;

        let embedding = data["data"]
            .get(0)
            .and_then(|d| d["embedding"].as_array())
            .ok_or("unexpected API response: missing data[0].embedding")?;

        let floats: Vec<f32> = embedding
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0) as f32)
            .collect();

        Ok(floats)
    }

    /// Send a chat completion request (used for --refine).
    pub fn chat_complete(
        &self,
        system: &str,
        user: &str,
        max_tokens: u32,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let url = format!("{}{}/chat/completions", self.endpoint, self.api_path);

        let body = json!({
            "model": self.llm_model,
            "messages": [
                { "role": "system", "content": system },
                { "role": "user", "content": user },
            ],
            "temperature": 0.3,
            "max_tokens": max_tokens,
        });

        let response = self
            .http
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().unwrap_or_default();
            return Err(format!("Chat API request failed ({}): {}", status, text).into());
        }

        let data: serde_json::Value = response.json()?;

        let content = data["choices"]
            .get(0)
            .and_then(|c| c["message"]["content"].as_str())
            .ok_or("unexpected API response: missing choices[0].message.content")?;

        Ok(content.to_string())
    }

    /// Check API connectivity and return embedding dimensions.
    pub fn health_check(&self) -> Result<EmbeddingInfo, Box<dyn std::error::Error>> {
        // Check /models endpoint
        let url = format!("{}{}/models", self.endpoint, self.api_path);
        let response = self.http.get(&url).send()?;

        if !response.status().is_success() {
            return Err(format!("API returned status {}", response.status()).into());
        }

        // Use the configured model name — /models may list multiple models
        // and the first one isn't necessarily the embedding model
        let model_id = self.model.clone();

        // Do a test embedding to get dimensions
        let test_embedding = self.embed("test")?;
        let dimensions = test_embedding.len() as u32;

        Ok(EmbeddingInfo {
            dimensions,
            model_id,
        })
    }
}
