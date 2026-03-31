use crate::index::Index;
use std::path::{Path, PathBuf};

pub struct SearchResult {
    pub path: PathBuf,
    pub score: f32,
}

/// Search the index for files most similar to the query embedding.
/// Uses dot product (embeddings are pre-normalized).
pub fn search(
    index: &Index,
    query_embedding: &[f32],
    top_n: usize,
    threshold: f32,
) -> Vec<SearchResult> {
    let mut results: Vec<SearchResult> = index
        .entries
        .iter()
        .filter_map(|(path, entry)| {
            let score = dot_product(query_embedding, &entry.embedding);
            if score >= threshold {
                Some(SearchResult {
                    path: path.clone(),
                    score,
                })
            } else {
                None
            }
        })
        .collect();

    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(top_n);
    results
}

fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Format a path for display, replacing the home directory with ~.
pub fn format_path(path: &Path) -> String {
    if let Some(home) = dirs::home_dir() {
        if let Ok(stripped) = path.strip_prefix(&home) {
            return format!("~/{}", stripped.display());
        }
    }
    path.display().to_string()
}
