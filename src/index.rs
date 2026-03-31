use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Serialize, Deserialize)]
pub struct Index {
    pub version: u32,
    pub dimensions: u32,
    pub model_id: String,
    pub created_at: u64,
    pub updated_at: u64,
    pub entries: HashMap<PathBuf, FileEntry>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FileEntry {
    pub mtime: u64,
    pub embedding: Vec<f32>,
}

impl Index {
    pub fn new(dimensions: u32, model_id: String) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            version: 1,
            dimensions,
            model_id,
            created_at: now,
            updated_at: now,
            entries: HashMap::new(),
        }
    }

    /// Path to the index file: ~/.local/share/npusearch/index.bin
    pub fn data_path() -> PathBuf {
        let mut path = dirs::data_local_dir().unwrap_or_else(|| {
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("~"))
                .join(".local/share")
        });
        path.push("npusearch");
        path.push("index.bin");
        path
    }

    /// Load the index from disk.
    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let bytes = std::fs::read(path)?;
        let (index, _): (Self, _) =
            bincode::serde::decode_from_slice(&bytes, bincode::config::standard())?;
        Ok(index)
    }

    /// Save the index to disk atomically (write to .tmp, then rename).
    pub fn save(&mut self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        self.updated_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let tmp_path = path.with_extension("bin.tmp");
        let bytes = bincode::serde::encode_to_vec(&*self, bincode::config::standard())?;
        std::fs::write(&tmp_path, &bytes)?;
        std::fs::rename(&tmp_path, path)?;

        Ok(())
    }
}
