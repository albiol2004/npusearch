# npusearch

Semantic file search powered by your local NPU.

Find files by describing them in natural language. npusearch uses a local embedding model running on your AMD NPU (via [lemonade-server](https://github.com/onnx/turnkeyml)) to build a searchable index of your filesystem. Search combines semantic similarity with keyword matching on file paths for best results.

## Installation

### Build from source

```bash
git clone https://github.com/albiol2004/npusearch
cd npusearch
make install    # installs to ~/.local/bin
npusearch init
```

Requires the Rust toolchain.

### Prebuilt binaries

Coming soon via GitHub Releases.

### Shell completions

```bash
# Bash
npusearch completions bash > ~/.local/share/bash-completion/completions/npusearch

# Zsh
npusearch completions zsh > ~/.local/share/zsh/site-functions/_npusearch

# Fish
npusearch completions fish > ~/.config/fish/completions/npusearch.fish
```

## Quick start

```bash
npusearch init          # create config + build first index
npusearch "query"       # search by natural language
npusearch               # interactive fzf-like mode
```

## Commands

| Command       | Description                                              |
|---------------|----------------------------------------------------------|
| `init`        | Set up npusearch: create config and build first index    |
| `index`       | Build a full index of your files                         |
| `update`      | Incrementally update the index (new/changed/deleted)     |
| `doctor`      | Check API connectivity and index status                  |
| `config`      | Show current configuration                               |
| `completions` | Generate shell completions (bash, zsh, fish)             |

Running `npusearch` with no arguments launches an interactive fuzzy-search mode. Passing a string directly (e.g. `npusearch "meeting notes from march"`) runs a one-shot search.

### Index options

```bash
npusearch index --root ~/projects    # index a specific directory
npusearch index --workers 8          # use more concurrent workers
npusearch index --dry-run            # list files without indexing
```

## How it works

1. **Indexing** -- npusearch walks your filesystem, reads each file's path and a content preview (first 500 bytes), and sends that text to your local embedding model. The resulting vectors are stored in a compact binary index at `~/.local/share/npusearch/index.bin`.

2. **Hybrid search** -- queries are embedded with the same model, then ranked using a 50/50 blend of cosine similarity (semantic) and keyword matching on file paths. This means exact filename matches surface even when semantic scores are low, and conceptually related files appear even without keyword overlap.

3. **Incremental updates** -- `npusearch update` compares file modification times against the index, only re-embeds changed or new files, and removes deleted ones. Auto-update can trigger before searches when the index is older than a configurable threshold (default: 24 hours).

4. **Checkpointing** -- during indexing, progress is saved every 500 files so interrupted runs can resume.

## Document support

npusearch can extract text content from documents before embedding:

- **PDF** -- uses `pdftotext` (first 2 pages)
- **Office/document formats** (docx, doc, pptx, odt, odp, ods, epub, rtf) -- uses `pandoc`

If these tools are not installed, documents are still indexed by filename and path.

## Configuration

Config file: `~/.config/npusearch/config.toml`

Created automatically by `npusearch init`, or copy the example:

```toml
[api]
endpoint = "http://localhost:8000"
model = "embed-gemma-300m-FLM"
timeout = 30

[index]
root = "~"
workers = 4
max_file_size = 1048576
extra_skip_dirs = []

[search]
top_n = 10
threshold = 0.3
auto_update_hours = 24
```

Run `npusearch config` to see all active settings.

## Requirements

- [lemonade-server](https://github.com/onnx/turnkeyml) (or any OpenAI-compatible embedding API) with an embedding model loaded
- Linux (x86_64 or aarch64)
- Optional: `pdftotext` (from poppler-utils) for PDF content extraction
- Optional: `pandoc` for docx/pptx/odt and other document formats

## License

MIT
