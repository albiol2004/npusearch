use crate::config::IndexConfig;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

/// Binary file extensions to skip.
const BINARY_EXTENSIONS: &[&str] = &[
    "o", "so", "pyc", "class", "jar", "zip", "tar", "gz", "png", "jpg", "jpeg", "gif", "bmp",
    "ico", "webp", "mp3", "mp4", "avi", "mkv", "flac", "wav", "pdf", "doc", "docx", "xls",
    "xlsx", "ppt", "pptx", "woff", "woff2", "ttf", "exe", "dll", "bin", "dat", "db", "sqlite",
    "iso", "img", "dmg", "deb", "rpm", "snap", "flatpak", "AppImage",
];

/// Directories to always skip.
const SKIP_DIRS: &[&str] = &[
    ".git",
    ".cache",
    "node_modules",
    "target",
    ".local",
    ".var",
    ".mozilla",
    ".thunderbird",
    "__pycache__",
    ".mypy_cache",
    "venv",
    ".venv",
    "dist",
    "build",
    // Heavy app-data dirs inside .config
    "chromium",
    "google-chrome",
    "Code",
    "Code - OSS",
    "discord",
    "Slack",
    "spotify",
    "BraveSoftware",
    "session",
    "sessions",
    "Cache",
    "CacheStorage",
    "GPUCache",
    "DawnGraphiteCache",
    "DawnWebGPUCache",
    "Service Worker",
    "blob_storage",
    "WebStorage",
    "IndexedDB",
    // IDE / editor state
    "Cursor",
    "VSCodium",
    "Code - Insiders",
    "CherryStudio",
    "LM Studio",
    // App data
    "obsidian",
    "calibre",
    "Ryujinx",
    "libreoffice",
    // Claude Code internal state
    "file-history",
    "shell-snapshots",
];

/// Hidden directories that should NOT be skipped.
const ALLOWED_HIDDEN_DIRS: &[&str] = &[".ssh", ".config"];

/// Walk the file tree and return (path, mtime) pairs for indexable files.
pub fn walk_files(root: &Path, config: &IndexConfig) -> Vec<(PathBuf, u64)> {
    let mut results = Vec::new();

    let walker = WalkDir::new(root)
        .follow_links(false)
        .into_iter();

    for entry in walker.filter_entry(|e| !should_skip_dir(e, config)) {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };

        // Skip directories themselves (we only want files)
        if entry.file_type().is_dir() {
            continue;
        }

        // Skip symlinks
        if entry.file_type().is_symlink() {
            continue;
        }

        let path = entry.path();

        // Skip files with binary extensions
        if has_binary_extension(path) {
            continue;
        }

        // Skip files over max_file_size
        if let Ok(metadata) = entry.metadata() {
            if metadata.len() > config.max_file_size {
                continue;
            }

            let mtime = metadata
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs())
                .unwrap_or(0);

            results.push((path.to_path_buf(), mtime));
        }
    }

    results
}

fn should_skip_dir(entry: &walkdir::DirEntry, config: &IndexConfig) -> bool {
    // Only apply skip logic to directories
    if !entry.file_type().is_dir() {
        return false;
    }

    let name = entry.file_name().to_string_lossy();

    // Check built-in skip list
    if SKIP_DIRS.iter().any(|&d| name == d) {
        return true;
    }

    // Check extra skip dirs from config
    if config.extra_skip_dirs.iter().any(|d| name.as_ref() == d) {
        return true;
    }

    // Skip hidden directories (starting with .) except allowed ones
    if name.starts_with('.') && !ALLOWED_HIDDEN_DIRS.iter().any(|&a| name == a) {
        // Don't skip the root directory even if it starts with .
        if entry.depth() > 0 {
            return true;
        }
    }

    false
}

fn has_binary_extension(path: &Path) -> bool {
    if let Some(ext) = path.extension() {
        let ext = ext.to_string_lossy();
        BINARY_EXTENSIONS.iter().any(|&b| ext.eq_ignore_ascii_case(b))
    } else {
        false
    }
}
