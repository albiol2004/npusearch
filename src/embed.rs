use crate::client::EmbeddingClient;
use crate::config::{ApiConfig, IndexConfig};
use crate::index::{FileEntry, Index};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::Instant;

/// How often to save a checkpoint (every N files).
const CHECKPOINT_INTERVAL: usize = 500;

/// Prepare the text to embed for a given file.
/// Format: "{last 3 path components} | {first N chars of content}"
pub fn prepare_text(path: &Path, content_preview_bytes: usize) -> String {
    // Take last 3 path components, space-separated
    let components: Vec<&str> = path
        .components()
        .filter_map(|c| c.as_os_str().to_str())
        .collect();
    let path_part: String = if components.len() <= 3 {
        components.join(" ")
    } else {
        components[components.len() - 3..].join(" ")
    };

    // Try to read content preview
    if is_binary(path) {
        return path_part;
    }

    match std::fs::read(path) {
        Ok(bytes) => {
            let preview_len = content_preview_bytes.min(bytes.len());
            let text = String::from_utf8_lossy(&bytes[..preview_len]);
            // Truncate at last word boundary
            let text = text.trim();
            if text.is_empty() {
                path_part
            } else {
                format!("{} | {}", path_part, text)
            }
        }
        Err(_) => path_part,
    }
}

/// Check if a file is binary by looking for null bytes in the first 512 bytes.
pub fn is_binary(path: &Path) -> bool {
    match std::fs::read(path) {
        Ok(bytes) => {
            let check_len = 512.min(bytes.len());
            bytes[..check_len].contains(&0)
        }
        Err(_) => true, // If we can't read it, treat as binary
    }
}

/// L2-normalize a vector in place.
fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Embed files using worker threads with progress reporting.
/// Writes results directly into the index and checkpoints every 500 files.
pub fn embed_files(
    client: &EmbeddingClient,
    api_config: &ApiConfig,
    files: Vec<(PathBuf, u64)>,
    config: &IndexConfig,
    index: &mut Index,
) {
    let total = files.len();
    if total == 0 {
        return;
    }

    let workers = config.workers.max(1);
    let content_preview_bytes = config.content_preview_bytes;
    let data_path = Index::data_path();

    // Channel for sending work items to workers
    let (work_tx, work_rx) = mpsc::channel::<(PathBuf, u64)>();
    // Channel for receiving results from workers
    let (result_tx, result_rx) = mpsc::channel::<(PathBuf, FileEntry)>();

    // Wrap receiver in Arc<Mutex> so multiple workers can pull from it
    let work_rx = std::sync::Arc::new(std::sync::Mutex::new(work_rx));

    // Spawn worker threads
    let mut handles = Vec::new();
    for _ in 0..workers {
        let work_rx = work_rx.clone();
        let result_tx = result_tx.clone();
        let worker_client = client.clone_for_worker(api_config);

        let handle = std::thread::spawn(move || {
            loop {
                let item = {
                    let rx = work_rx.lock().unwrap();
                    rx.recv()
                };

                match item {
                    Ok((path, mtime)) => {
                        let text = prepare_text(&path, content_preview_bytes);
                        match worker_client.embed(&text) {
                            Ok(mut embedding) => {
                                l2_normalize(&mut embedding);
                                let entry = FileEntry { mtime, embedding };
                                let _ = result_tx.send((path, entry));
                            }
                            Err(e) => {
                                eprintln!("\nWarning: failed to embed {}: {}", path.display(), e);
                            }
                        }
                    }
                    Err(_) => break, // Channel closed, no more work
                }
            }
        });
        handles.push(handle);
    }

    // Drop the extra result_tx so result_rx closes when workers finish
    drop(result_tx);

    // Send all work items
    let sender = std::thread::spawn(move || {
        for item in files {
            let _ = work_tx.send(item);
        }
        // work_tx drops here, closing the channel
    });

    // Collect results with progress reporting + periodic checkpointing
    let start = Instant::now();
    let mut count = 0usize;
    let mut since_checkpoint = 0usize;

    for (path, entry) in &result_rx {
        index.entries.insert(path, entry);
        count += 1;
        since_checkpoint += 1;

        // Progress display
        let elapsed = start.elapsed().as_secs_f64();
        let rate = count as f64 / elapsed;
        let remaining = if rate > 0.0 {
            ((total - count) as f64 / rate) as u64
        } else {
            0
        };
        let mins = remaining / 60;
        let secs = remaining % 60;
        eprint!(
            "\rIndexing: {}/{} files [{}%] (~{}m{}s remaining)   ",
            count,
            total,
            (count * 100) / total,
            mins,
            secs,
        );
        let _ = std::io::stderr().flush();

        // Checkpoint
        if since_checkpoint >= CHECKPOINT_INTERVAL {
            since_checkpoint = 0;
            if let Err(e) = index.save(&data_path) {
                eprintln!("\nWarning: checkpoint save failed: {}", e);
            }
        }
    }
    eprintln!();

    // Wait for all threads to finish
    sender.join().unwrap();
    for handle in handles {
        let _ = handle.join();
    }
}
