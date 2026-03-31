mod client;
mod config;
mod embed;
mod index;
mod interactive;
mod search;
mod walker;

use clap::{Parser, Subcommand};
use config::Config;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Parser)]
#[command(name = "npusearch", version, about = "Semantic file search powered by local NPU")]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,

    /// Search query (when no subcommand is given)
    #[arg(trailing_var_arg = true)]
    query: Vec<String>,
}

#[derive(Subcommand)]
enum Command {
    /// Build a full index of your files
    Index {
        /// Root directory to index
        #[arg(long)]
        root: Option<String>,
        /// Number of concurrent embedding workers
        #[arg(long)]
        workers: Option<usize>,
        /// List files that would be indexed without indexing
        #[arg(long)]
        dry_run: bool,
        /// Show per-file progress
        #[arg(long)]
        verbose: bool,
    },
    /// Incrementally update the index (changed/new/deleted files)
    Update {
        /// Root directory
        #[arg(long)]
        root: Option<String>,
        /// Number of concurrent embedding workers
        #[arg(long)]
        workers: Option<usize>,
        /// Show per-file progress
        #[arg(long)]
        verbose: bool,
    },
    /// Check API connectivity and index status
    Doctor,
    /// Show current configuration
    Config,
    /// Set up npusearch: create config and build first index
    Init,
}

fn main() {
    let cli = Cli::parse();
    let mut config = Config::load();

    match cli.command {
        Some(Command::Index {
            root,
            workers,
            dry_run,
            verbose: _,
        }) => {
            if let Some(r) = root {
                config.index.root = r;
            }
            if let Some(w) = workers {
                config.index.workers = w;
            }
            handle_index(&config, dry_run);
        }
        Some(Command::Update {
            root,
            workers,
            verbose: _,
        }) => {
            if let Some(r) = root {
                config.index.root = r;
            }
            if let Some(w) = workers {
                config.index.workers = w;
            }
            handle_update(&config);
        }
        Some(Command::Doctor) => {
            handle_doctor(&config);
        }
        Some(Command::Config) => {
            handle_config(&config);
        }
        Some(Command::Init) => {
            handle_init(&config);
        }
        None => {
            if cli.query.is_empty() {
                if let Err(e) = interactive::run(&config) {
                    eprintln!("Error: {}", e);
                    std::process::exit(1);
                }
                return;
            }

            // Check if the first word looks like a misspelled subcommand
            let first = &cli.query[0];
            let subcommands = ["index", "update", "doctor", "config", "init"];
            if let Some(suggestion) = find_similar(first, &subcommands) {
                eprintln!(
                    "Unknown subcommand '{}'. Did you mean '{}'?\n",
                    first, suggestion
                );
                eprintln!(
                    "To search, use: npusearch \"{}\"",
                    cli.query.join(" ")
                );
                std::process::exit(1);
            }

            let query = cli.query.join(" ");
            handle_search(&config, &query);
        }
    }
}

fn handle_index(config: &Config, dry_run: bool) {
    let root = config.index.root_path();
    eprintln!("Walking {}...", root.display());

    let files = walker::walk_files(&root, &config.index);
    eprintln!("Found {} files to index.", files.len());

    if dry_run {
        for (path, _) in &files {
            println!("{}", path.display());
        }
        return;
    }

    // Connect to API and get dimensions
    let api_client = client::EmbeddingClient::new(&config.api);
    let info = match api_client.health_check() {
        Ok(info) => info,
        Err(e) => {
            eprintln!("Error: cannot connect to embedding API: {}", e);
            eprintln!("Make sure lemonade-server is running and run 'npusearch doctor'.");
            std::process::exit(1);
        }
    };

    eprintln!(
        "Embedding model: {} ({}D)",
        info.model_id, info.dimensions
    );

    // Try to resume from a partial index (previous interrupted run)
    let data_path = index::Index::data_path();
    let mut idx = if let Ok(existing) = index::Index::load(&data_path) {
        let already = existing.entries.len();
        if already > 0 {
            eprintln!("Resuming from existing index ({} files already indexed).", already);
        }
        existing
    } else {
        index::Index::new(info.dimensions, info.model_id)
    };

    // Filter out files already in the index with matching mtime
    let files_to_embed: Vec<(PathBuf, u64)> = files
        .into_iter()
        .filter(|(path, mtime)| {
            match idx.entries.get(path) {
                Some(entry) if entry.mtime == *mtime => false, // already indexed
                _ => true,
            }
        })
        .collect();

    eprintln!("{} files to embed.", files_to_embed.len());

    if files_to_embed.is_empty() {
        eprintln!("Index is already complete.");
        return;
    }

    embed::embed_files(&api_client, &config.api, files_to_embed, &config.index, &mut idx);

    match idx.save(&data_path) {
        Ok(()) => {
            eprintln!(
                "Index saved: {} files -> {}",
                idx.entries.len(),
                data_path.display()
            );
        }
        Err(e) => {
            eprintln!("Error saving index: {}", e);
            std::process::exit(1);
        }
    }
}

fn handle_update(config: &Config) {
    let data_path = index::Index::data_path();
    let mut idx = match index::Index::load(&data_path) {
        Ok(idx) => idx,
        Err(e) => {
            eprintln!("Error loading index: {}", e);
            eprintln!("Run 'npusearch index' to create an index first.");
            std::process::exit(1);
        }
    };

    let root = config.index.root_path();
    eprintln!("Walking {}...", root.display());

    let files = walker::walk_files(&root, &config.index);
    let current_paths: std::collections::HashSet<PathBuf> =
        files.iter().map(|(p, _)| p.clone()).collect();

    // Find files to embed (new or changed)
    let mut to_embed: Vec<(PathBuf, u64)> = Vec::new();
    for (path, mtime) in &files {
        match idx.entries.get(path) {
            Some(entry) if entry.mtime == *mtime => {
                // Unchanged, skip
            }
            _ => {
                to_embed.push((path.clone(), *mtime));
            }
        }
    }

    // Find deleted files
    let deleted: Vec<PathBuf> = idx
        .entries
        .keys()
        .filter(|p| !current_paths.contains(*p))
        .cloned()
        .collect();

    eprintln!(
        "{} new/changed, {} deleted, {} unchanged",
        to_embed.len(),
        deleted.len(),
        files.len() - to_embed.len(),
    );

    // Remove deleted
    for path in &deleted {
        idx.entries.remove(path);
    }

    if to_embed.is_empty() {
        // Just save to update the timestamp
        if !deleted.is_empty() {
            match idx.save(&data_path) {
                Ok(()) => eprintln!("Index updated."),
                Err(e) => {
                    eprintln!("Error saving index: {}", e);
                    std::process::exit(1);
                }
            }
        } else {
            eprintln!("Index is up to date.");
            // Update the timestamp even if nothing changed
            let _ = idx.save(&data_path);
        }
        return;
    }

    // Check model consistency
    let api_client = client::EmbeddingClient::new(&config.api);
    if let Ok(info) = api_client.health_check() {
        if info.model_id != idx.model_id {
            eprintln!(
                "Warning: model changed from '{}' to '{}'. Consider running 'npusearch index' for a full re-index.",
                idx.model_id, info.model_id
            );
        }
    }

    embed::embed_files(&api_client, &config.api, to_embed, &config.index, &mut idx);

    match idx.save(&data_path) {
        Ok(()) => {
            eprintln!("Index updated: {} total files.", idx.entries.len());
        }
        Err(e) => {
            eprintln!("Error saving index: {}", e);
            std::process::exit(1);
        }
    }
}

fn handle_search(config: &Config, query: &str) {
    let data_path = index::Index::data_path();

    // Load or bail
    let idx = match index::Index::load(&data_path) {
        Ok(idx) => idx,
        Err(_) => {
            eprintln!("No index found. Run 'npusearch index' first.");
            std::process::exit(1);
        }
    };

    // Auto-update check
    if config.search.auto_update_hours > 0 {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let age_hours = (now.saturating_sub(idx.updated_at)) / 3600;
        if age_hours >= config.search.auto_update_hours {
            eprintln!(
                "Index is {}h old (threshold: {}h), running auto-update...",
                age_hours, config.search.auto_update_hours
            );
            handle_update(config);
            // Reload after update
            return handle_search_inner(config, query);
        }
    }

    handle_search_inner(config, query);
}

fn handle_search_inner(config: &Config, query: &str) {
    let data_path = index::Index::data_path();
    let idx = match index::Index::load(&data_path) {
        Ok(idx) => idx,
        Err(_) => {
            eprintln!("No index found. Run 'npusearch index' first.");
            std::process::exit(1);
        }
    };

    let api_client = client::EmbeddingClient::new(&config.api);

    // Embed the query
    let mut query_embedding = match api_client.embed(query) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Error embedding query: {}", e);
            std::process::exit(1);
        }
    };

    // L2-normalize the query embedding
    let norm: f32 = query_embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in query_embedding.iter_mut() {
            *x /= norm;
        }
    }

    let results = search::search_hybrid(
        &idx,
        &query_embedding,
        query,
        config.search.top_n,
        config.search.threshold,
    );

    if results.is_empty() {
        eprintln!("No results found.");
        return;
    }

    if config.search.refine {
        handle_refine(config, &api_client, query, &results);
    } else {
        for result in &results {
            println!(
                "  {:.3}  {}",
                result.score,
                search::format_path(&result.path)
            );
        }
    }
}

fn handle_refine(
    config: &Config,
    api_client: &client::EmbeddingClient,
    query: &str,
    results: &[search::SearchResult],
) {
    let file_list: String = results
        .iter()
        .enumerate()
        .map(|(i, r)| {
            format!(
                "{}. {} (score: {:.3})",
                i + 1,
                search::format_path(&r.path),
                r.score
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    let system = "You are a file search assistant. The user searched for files and got results ranked by semantic similarity. Re-rank these results by relevance to the query and explain briefly why each is relevant. Output the re-ranked list.";
    let user_prompt = format!(
        "Query: \"{}\"\n\nResults:\n{}\n\nRe-rank these by relevance and briefly explain each.",
        query, file_list
    );

    match api_client.chat_complete(system, &user_prompt, 1024) {
        Ok(response) => {
            println!("{}", response);
        }
        Err(e) => {
            eprintln!("Refine failed: {}. Showing raw results:", e);
            for result in results {
                println!(
                    "  {:.3}  {}",
                    result.score,
                    search::format_path(&result.path)
                );
            }
        }
    }

    // Also print the config reminder to use --refine
    let _ = config;
}

fn handle_init(config: &Config) {
    use std::io::{self, BufRead, Write};

    println!("npusearch init");
    println!("==============\n");

    // 1. Create config file
    let config_path = Config::config_path();
    print!("Config file... ");
    if config_path.exists() {
        println!("already exists at {}", config_path.display());
    } else {
        if let Some(parent) = config_path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let example = include_str!("../config/config.example.toml");
        match std::fs::write(&config_path, example) {
            Ok(()) => println!("created at {}", config_path.display()),
            Err(e) => println!("FAILED ({})", e),
        }
    }

    // 2. Check API connectivity
    println!();
    let api_client = client::EmbeddingClient::new(&config.api);
    print!("API endpoint ({})... ", config.api.endpoint);
    match api_client.health_check() {
        Ok(info) => {
            println!("OK");
            println!("  Model: {} ({}D)", info.model_id, info.dimensions);
        }
        Err(e) => {
            println!("FAILED ({})", e);
            println!("\nMake sure lemonade-server is running, then try again.");
            println!("You can edit the config at: {}", config_path.display());
            return;
        }
    }

    // 3. Check for existing index
    let data_path = index::Index::data_path();
    println!();
    if data_path.exists() {
        if let Ok(idx) = index::Index::load(&data_path) {
            println!("Index already exists: {} files indexed.", idx.entries.len());
            print!("Re-index from scratch? [y/N] ");
            io::stdout().flush().ok();
            let mut answer = String::new();
            io::stdin().lock().read_line(&mut answer).ok();
            if !answer.trim().to_lowercase().starts_with('y') {
                println!("\nSetup complete! Try: npusearch \"your search query\"");
                return;
            }
        }
    }

    // 4. Offer to build first index
    let root = config.index.root_path();
    print!("Build first index of {}? [Y/n] ", root.display());
    io::stdout().flush().ok();
    let mut answer = String::new();
    io::stdin().lock().read_line(&mut answer).ok();
    if answer.trim().is_empty() || answer.trim().to_lowercase().starts_with('y') {
        println!();
        handle_index(config, false);
    } else {
        println!("\nSkipped. Run 'npusearch index' when ready.");
    }

    println!("\nSetup complete! Try: npusearch \"your search query\"");
}

fn handle_doctor(config: &Config) {
    println!("npusearch doctor");
    println!("================");
    println!();

    // Check API
    print!("API endpoint: {} ... ", config.api.endpoint);
    let api_client = client::EmbeddingClient::new(&config.api);
    match api_client.health_check() {
        Ok(info) => {
            println!("OK");
            println!("  Model: {}", info.model_id);
            println!("  Embedding dimensions: {}", info.dimensions);
        }
        Err(e) => {
            println!("FAILED");
            println!("  Error: {}", e);
        }
    }
    println!();

    // Check index
    let data_path = index::Index::data_path();
    print!("Index: {} ... ", data_path.display());
    match index::Index::load(&data_path) {
        Ok(idx) => {
            println!("OK");
            println!("  Files: {}", idx.entries.len());
            println!("  Model: {}", idx.model_id);
            println!("  Dimensions: {}", idx.dimensions);

            // Show age
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            let age_hours = now.saturating_sub(idx.updated_at) / 3600;
            let age_mins = (now.saturating_sub(idx.updated_at) % 3600) / 60;
            println!("  Last updated: {}h {}m ago", age_hours, age_mins);

            // Show file size
            if let Ok(meta) = std::fs::metadata(&data_path) {
                let size_mb = meta.len() as f64 / (1024.0 * 1024.0);
                println!("  Index size: {:.1} MB", size_mb);
            }
        }
        Err(_) => {
            println!("NOT FOUND");
            println!("  Run 'npusearch index' to create an index.");
        }
    }
}

fn handle_config(config: &Config) {
    config.display();
}

/// Find a similar string from candidates using edit distance.
/// Returns Some if the best match is within 2 edits.
fn find_similar<'a>(input: &str, candidates: &[&'a str]) -> Option<&'a str> {
    let input_lower = input.to_lowercase();
    let mut best: Option<(&str, usize)> = None;

    for &candidate in candidates {
        if input_lower == candidate {
            return None; // Exact match, not a typo
        }
        let dist = edit_distance(&input_lower, candidate);
        if dist <= 2 && (best.is_none() || dist < best.unwrap().1) {
            best = Some((candidate, dist));
        }
    }

    best.map(|(s, _)| s)
}

#[allow(clippy::needless_range_loop)]
fn edit_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut dp = vec![vec![0usize; b.len() + 1]; a.len() + 1];

    for i in 0..=a.len() {
        dp[i][0] = i;
    }
    for j in 0..=b.len() {
        dp[0][j] = j;
    }
    for i in 1..=a.len() {
        for j in 1..=b.len() {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            dp[i][j] = (dp[i - 1][j] + 1)
                .min(dp[i][j - 1] + 1)
                .min(dp[i - 1][j - 1] + cost);
        }
    }
    dp[a.len()][b.len()]
}
