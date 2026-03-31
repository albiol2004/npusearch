use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct Config {
    pub api: ApiConfig,
    pub index: IndexConfig,
    pub search: SearchConfig,
}

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct ApiConfig {
    pub endpoint: String,
    pub api_path: String,
    pub model: String,
    pub llm_model: String,
    pub timeout: u64,
}

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct IndexConfig {
    pub root: String,
    pub workers: usize,
    pub max_file_size: u64,
    pub content_preview_bytes: usize,
    pub extra_skip_dirs: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct SearchConfig {
    pub top_n: usize,
    pub threshold: f32,
    pub refine: bool,
    pub auto_update_hours: u64,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            endpoint: "http://localhost:8000".to_string(),
            api_path: "/api/v1".to_string(),
            model: "embed-gemma-300m-FLM".to_string(),
            llm_model: "default".to_string(),
            timeout: 30,
        }
    }
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            root: "~".to_string(),
            workers: 4,
            max_file_size: 1_048_576,
            content_preview_bytes: 500,
            extra_skip_dirs: Vec::new(),
        }
    }
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            top_n: 10,
            threshold: 0.3,
            refine: false,
            auto_update_hours: 24,
        }
    }
}

impl IndexConfig {
    /// Expand the root path, replacing "~" with the home directory.
    pub fn root_path(&self) -> PathBuf {
        if self.root == "~" || self.root.starts_with("~/") {
            if let Some(home) = dirs::home_dir() {
                if self.root == "~" {
                    return home;
                }
                return home.join(&self.root[2..]);
            }
        }
        PathBuf::from(&self.root)
    }
}

impl Config {
    pub fn config_path() -> PathBuf {
        let mut path = dirs::config_dir().unwrap_or_else(|| PathBuf::from("~/.config"));
        path.push("npusearch");
        path.push("config.toml");
        path
    }

    pub fn load() -> Self {
        let path = Self::config_path();
        if path.exists() {
            match std::fs::read_to_string(&path) {
                Ok(contents) => match toml::from_str(&contents) {
                    Ok(config) => return config,
                    Err(e) => {
                        eprintln!("Warning: failed to parse config at {}: {}", path.display(), e);
                    }
                },
                Err(e) => {
                    eprintln!("Warning: failed to read config at {}: {}", path.display(), e);
                }
            }
        }
        Self::default()
    }

    pub fn display(&self) {
        println!("npusearch configuration");
        println!("=======================");
        println!("Config file: {}", Self::config_path().display());
        println!();
        println!("[api]");
        println!("  endpoint   = \"{}\"", self.api.endpoint);
        println!("  api_path   = \"{}\"", self.api.api_path);
        println!("  model      = \"{}\"", self.api.model);
        println!("  llm_model  = \"{}\"", self.api.llm_model);
        println!("  timeout    = {}s", self.api.timeout);
        println!();
        println!("[index]");
        println!("  root                 = \"{}\"", self.index.root);
        println!("  workers              = {}", self.index.workers);
        println!("  max_file_size        = {} bytes", self.index.max_file_size);
        println!("  content_preview_bytes = {}", self.index.content_preview_bytes);
        println!("  extra_skip_dirs      = {:?}", self.index.extra_skip_dirs);
        println!();
        println!("[search]");
        println!("  top_n              = {}", self.search.top_n);
        println!("  threshold          = {}", self.search.threshold);
        println!("  refine             = {}", self.search.refine);
        println!("  auto_update_hours  = {}", self.search.auto_update_hours);
    }
}
