use crate::client::EmbeddingClient;
use crate::config::Config;
use crate::index::Index;
use crate::search;

use crossterm::{
    cursor,
    event::{self, Event, KeyCode, KeyEvent, KeyModifiers},
    style::{self, Attribute, Color, SetAttribute, SetForegroundColor, ResetColor},
    terminal::{self, Clear, ClearType},
};
use std::collections::HashMap;
use std::io::{self, Write};
use std::time::{Duration, Instant};

const DEBOUNCE_MS: u64 = 400;
const POLL_MS: u64 = 50;
const CACHE_MAX: usize = 10;

/// RAII guard that restores terminal state on drop.
struct TerminalGuard;

impl TerminalGuard {
    fn new() -> io::Result<Self> {
        terminal::enable_raw_mode()?;
        // Hide cursor for cleaner UI
        crossterm::execute!(io::stderr(), cursor::Hide)?;
        Ok(Self)
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = crossterm::execute!(io::stderr(), cursor::Show);
        let _ = terminal::disable_raw_mode();
    }
}

struct State {
    query: String,
    results: Vec<search::SearchResult>,
    selected: usize,
    searching: bool,
    last_keystroke: Instant,
    last_searched_query: String,
    cache: HashMap<String, Vec<f32>>,
}

impl State {
    fn new() -> Self {
        Self {
            query: String::new(),
            results: Vec::new(),
            selected: 0,
            searching: false,
            last_keystroke: Instant::now(),
            last_searched_query: String::new(),
            cache: HashMap::new(),
        }
    }

    fn cache_insert(&mut self, query: String, embedding: Vec<f32>) {
        if self.cache.len() >= CACHE_MAX {
            // Remove an arbitrary entry to make room
            if let Some(key) = self.cache.keys().next().cloned() {
                self.cache.remove(&key);
            }
        }
        self.cache.insert(query, embedding);
    }
}

pub fn run(config: &Config) -> Result<(), Box<dyn std::error::Error>> {
    // Load index
    let data_path = Index::data_path();
    let idx = match Index::load(&data_path) {
        Ok(idx) => idx,
        Err(_) => {
            eprintln!("No index found. Run 'npusearch index' first.");
            std::process::exit(1);
        }
    };

    let api_client = EmbeddingClient::new(&config.api);
    let top_n = config.search.top_n;
    let threshold = config.search.threshold;

    let _guard = TerminalGuard::new()?;
    let mut stderr = io::stderr();

    let mut state = State::new();

    // Initial render
    render(&mut stderr, &state, top_n)?;

    loop {
        // Poll for events
        if event::poll(Duration::from_millis(POLL_MS))? {
            if let Event::Key(key_event) = event::read()? {
                match handle_key(&mut state, key_event) {
                    Action::Continue => {}
                    Action::Exit => {
                        // Clear our UI before exiting
                        clear_screen(&mut stderr, top_n)?;
                        return Ok(());
                    }
                    Action::Select => {
                        if let Some(result) = state.results.get(state.selected) {
                            let path = search::format_path(&result.path);
                            // Clear our UI before printing
                            clear_screen(&mut stderr, top_n)?;
                            // Drop guard first to restore terminal
                            drop(_guard);
                            // Print selected path to stdout
                            println!("{}", path);
                            return Ok(());
                        }
                        // No result selected, just continue
                    }
                }
                render(&mut stderr, &state, top_n)?;
            }
        }

        // Check debounce: if enough time has passed since last keystroke and query changed
        if !state.query.is_empty()
            && state.query != state.last_searched_query
            && state.last_keystroke.elapsed() >= Duration::from_millis(DEBOUNCE_MS)
        {
            // Perform search
            state.searching = true;
            render(&mut stderr, &state, top_n)?;

            let embedding = if let Some(cached) = state.cache.get(&state.query) {
                cached.clone()
            } else {
                match api_client.embed(&state.query) {
                    Ok(emb) => {
                        let query_clone = state.query.clone();
                        state.cache_insert(query_clone, emb.clone());
                        emb
                    }
                    Err(_) => {
                        state.searching = false;
                        state.last_searched_query = state.query.clone();
                        state.results.clear();
                        render(&mut stderr, &state, top_n)?;
                        continue;
                    }
                }
            };

            // L2-normalize
            let mut query_embedding = embedding;
            let norm: f32 = query_embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for x in query_embedding.iter_mut() {
                    *x /= norm;
                }
            }

            state.results = search::search(&idx, &query_embedding, top_n, threshold);
            state.selected = 0;
            state.searching = false;
            state.last_searched_query = state.query.clone();
            render(&mut stderr, &state, top_n)?;
        }

        // If query was cleared, reset results
        if state.query.is_empty() && !state.last_searched_query.is_empty() {
            state.results.clear();
            state.selected = 0;
            state.last_searched_query.clear();
            render(&mut stderr, &state, top_n)?;
        }
    }
}

enum Action {
    Continue,
    Exit,
    Select,
}

fn handle_key(state: &mut State, key: KeyEvent) -> Action {
    match key.code {
        KeyCode::Esc => Action::Exit,
        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => Action::Exit,
        KeyCode::Enter => Action::Select,
        KeyCode::Backspace => {
            state.query.pop();
            state.last_keystroke = Instant::now();
            Action::Continue
        }
        KeyCode::Char(c) => {
            state.query.push(c);
            state.last_keystroke = Instant::now();
            Action::Continue
        }
        KeyCode::Up => {
            if state.selected > 0 {
                state.selected -= 1;
            }
            Action::Continue
        }
        KeyCode::Down => {
            if !state.results.is_empty() && state.selected < state.results.len() - 1 {
                state.selected += 1;
            }
            Action::Continue
        }
        _ => Action::Continue,
    }
}

fn clear_screen(stderr: &mut io::Stderr, top_n: usize) -> io::Result<()> {
    // Move to top-left and clear all lines we used
    crossterm::queue!(stderr, cursor::MoveTo(0, 0))?;
    for _ in 0..=top_n {
        crossterm::queue!(stderr, Clear(ClearType::CurrentLine), cursor::MoveDown(1))?;
    }
    crossterm::queue!(stderr, cursor::MoveTo(0, 0))?;
    stderr.flush()
}

fn render(stderr: &mut io::Stderr, state: &State, top_n: usize) -> io::Result<()> {
    crossterm::queue!(stderr, cursor::MoveTo(0, 0), Clear(ClearType::CurrentLine))?;

    // Prompt line
    crossterm::queue!(
        stderr,
        style::Print(format!("\u{1F50D} {}", state.query))
    )?;

    // Results area
    for i in 0..top_n {
        crossterm::queue!(
            stderr,
            cursor::MoveTo(0, (i + 1) as u16),
            Clear(ClearType::CurrentLine)
        )?;

        if state.searching {
            if i == 0 {
                crossterm::queue!(stderr, style::Print("  Searching..."))?;
            }
        } else if state.query.is_empty() {
            if i == 0 {
                crossterm::queue!(
                    stderr,
                    SetForegroundColor(Color::DarkGrey),
                    style::Print("  Type to search..."),
                    ResetColor
                )?;
            }
        } else if state.results.is_empty() {
            if i == 0 && !state.last_searched_query.is_empty() {
                crossterm::queue!(
                    stderr,
                    SetForegroundColor(Color::DarkGrey),
                    style::Print("  No results found."),
                    ResetColor
                )?;
            }
        } else if let Some(result) = state.results.get(i) {
            let marker = if i == state.selected { "\u{25B6}" } else { " " };
            let path = search::format_path(&result.path);

            if i == state.selected {
                crossterm::queue!(
                    stderr,
                    SetForegroundColor(Color::Cyan),
                    SetAttribute(Attribute::Bold),
                    style::Print(format!("  {} {:.3}  {}", marker, result.score, path)),
                    SetAttribute(Attribute::Reset),
                    ResetColor
                )?;
            } else {
                crossterm::queue!(
                    stderr,
                    style::Print(format!("  {} {:.3}  {}", marker, result.score, path))
                )?;
            }
        }
    }

    // Position cursor after the query text on the prompt line
    let cursor_x = 3 + state.query.len() as u16; // "🔍 " is 3 display columns (emoji + space)
    crossterm::queue!(stderr, cursor::MoveTo(cursor_x, 0))?;

    stderr.flush()
}
