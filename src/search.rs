use crate::index::Index;
use std::path::{Path, PathBuf};

/// Weight for semantic score vs keyword score.
const SEMANTIC_WEIGHT: f32 = 0.5;
const KEYWORD_WEIGHT: f32 = 0.5;

pub struct SearchResult {
    pub path: PathBuf,
    pub score: f32,
}

/// Hybrid search: combines semantic similarity with keyword matching on paths.
pub fn search_hybrid(
    index: &Index,
    query_embedding: &[f32],
    query: &str,
    top_n: usize,
    threshold: f32,
) -> Vec<SearchResult> {
    let query_tokens: Vec<String> = tokenize_query(query);

    let mut results: Vec<SearchResult> = index
        .entries
        .iter()
        .filter_map(|(path, entry)| {
            let semantic = dot_product(query_embedding, &entry.embedding);
            let keyword = keyword_score(path, &query_tokens);
            let combined = SEMANTIC_WEIGHT * semantic + KEYWORD_WEIGHT * keyword;

            // Pass threshold on either score — a perfect keyword match should
            // surface even if semantic score is low, and vice versa
            if semantic >= threshold || keyword > 0.0 {
                Some(SearchResult {
                    path: path.clone(),
                    score: combined,
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

/// Tokenize a query into lowercase keywords, splitting on whitespace and punctuation.
fn tokenize_query(query: &str) -> Vec<String> {
    query
        .to_lowercase()
        .split(|c: char| c.is_whitespace() || c == '/' || c == '\\' || c == '-' || c == '_')
        .filter(|s| !s.is_empty() && s.len() >= 2)
        .map(String::from)
        .collect()
}

/// Score a file path against query tokens.
/// Returns 0.0..1.0 based on fraction of query tokens found in path components.
fn keyword_score(path: &Path, query_tokens: &[String]) -> f32 {
    if query_tokens.is_empty() {
        return 0.0;
    }

    let path_str = path.to_string_lossy().to_lowercase();

    let mut matched = 0;
    for token in query_tokens {
        if path_str.contains(token.as_str()) {
            matched += 1;
        }
    }

    matched as f32 / query_tokens.len() as f32
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
