# localmind

Run any local LLM with persistent memory.

## What it is

A single CLI binary (`llm`) that turns an [Ollama](https://ollama.com)-served
model into an interactive agent with long-term memory, learnable skills, and
permissioned tools. Everything runs on your machine — no cloud, no telemetry,
no dependencies beyond Ollama.

## What it does

- Talk to any local chat / vision / embedding model Ollama has installed.
- Remember what you tell it across sessions. Facts, decisions, skills,
  embeddings — all in one SQLite file you can back up or move.
- Recall relevant prior context automatically at the start of every turn.
- Read files, PDFs, docx, xlsx, images. Describe images with vision models.
- Run shell, networking, and web tools with per-call permission prompts and
  a hard credential deny-list.

## How it works

```
you type  ─►  auto-extract facts  ─►  hybrid recall  ─►  agent loop  ─►  streamed reply
                    │                      │                 │
                    ▼                      ▼                 ▼
              SQLite memory.db       BM25 + vector ANN    tool calls
```

1. **Auto-extract.** A regex catches obvious directives (`my name is X`,
   `call me X`, `remember X`) before the model sees the turn. Stored
   regardless of whether the model would have called `store_memory`.
2. **Recall.** Hybrid search (BM25 + vector, fused) over past memories is
   run concurrently, and the top hits are injected as a system message.
   BM25 and the embed round-trip execute in parallel so recall latency is
   the slower of the two, not the sum. Trivial turns (`hi`, `ok`, `thanks`)
   skip recall entirely.
3. **Agent loop.** The model sees tool specs and may call `read_file`,
   `shell`, `web_fetch`, `store_memory`, etc. Side-effecting tools are
   gated by the permission mode and / or prompt before running.
4. **Streamed reply.** Tokens are printed as they're generated — no long
   "thinking…" pause before the wall of text drops.
5. **Persistence.** New facts, skills, and conversation summaries are
   written back to SQLite and embedded in the background.

## Features

- **Portable memory.** Single SQLite file, `llm backup` / `llm restore` copy
  it cleanly between machines. No re-teaching.
- **Hybrid recall.** BM25 + vector ANN via `sqlite-vec`, fused with
  temporal decay. Falls back to pure BM25 with one config flag.
- **Learnable skills.** Tell it _"from now on when X, do Y"_ — stored as a
  `kind="skill"` memory, surfaced automatically on matching turns.
- **Safe-by-default tools.** Workspace-confined writes, SSRF guard on
  `web_fetch` and `whois`, destructive-pattern detection on `shell`,
  credential deny-list that's always on.
- **Streaming responses.** Tokens appear in real time.
- **Fast repeat turns.** `keep_alive: 30m` stops Ollama from unloading the
  model between turns; no cold-load per message.
- **Cascade router.** Optional two-model setup: short / chatty turns run on
  a small `fast_model` (e.g. `qwen2.5-coder:3b`, ~50 tok/s), code-heavy or
  long turns route to the configured `chat_model`. `/retry-big` manually
  escalates the last turn when the router picked wrong.
- **Self-updating.** Daily background check for a newer release; `llm update`
  re-runs the installer in place.
- **Model picker.** `llm models` lists installed Ollama models and lets you
  set chat / vision / embed non-interactively or via a picker.
- **Inspectable.** `llm health` reports DB, embedder, and Ollama state.
  `/recall <q>`, `/context`, `/audit` expose what the model is actually
  seeing. JSONL audit log of every tool call.

---

## Install

One-liner (macOS arm64, Linux x86_64/arm64). Downloads the latest release
binary, verifies SHA256, installs to `~/.local/bin/llm`, installs Ollama
(headless CLI via Homebrew on macOS, official installer on Linux), starts
the server, and pulls the default chat + embed models:

```bash
curl -fsSL https://raw.githubusercontent.com/nevenkordic/localmind/main/install.sh | sh
```

**Environment overrides:**

| var                         | default                 | what                                          |
|-----------------------------|-------------------------|-----------------------------------------------|
| `LOCALMIND_INSTALL_DIR`     | `$HOME/.local/bin`      | install target (auto-added to PATH)           |
| `LOCALMIND_VERSION`         | `latest`                | pin a release tag                             |
| `LOCALMIND_CHAT_MODEL`      | `qwen2.5-coder:3b`      | chat model the installer pulls (1.9 GB, fast)  |
| `LOCALMIND_EMBED_MODEL`     | `nomic-embed-text`      | embed model the installer pulls               |
| `LOCALMIND_OLLAMA_GUI=1`    | —                       | install the full Ollama.app (macOS cask)      |
| `LOCALMIND_SKIP_OLLAMA=1`   | —                       | don't install or start Ollama                 |
| `LOCALMIND_SKIP_MODELS=1`   | —                       | don't pull models (saves ~5 GB on metered)    |

**Build from source** (Intel Mac / Windows, or if you don't want the release
binary):

```bash
git clone https://github.com/nevenkordic/localmind
cd localmind
./scripts/install.sh        # macOS / Linux
.\scripts\install.ps1       # Windows
```

## Use

```bash
llm                         # interactive REPL
llm ask "fix the failing test"
llm health                  # DB stats, Ollama reachability, recall config
llm memory search "deploy procedure"
llm memory search "deploy procedure" --bm25    # skip embedding (fast)
llm models                  # pick chat / vision / embed models
llm backup [<path>]         # copy memory DB to a file
llm restore <path>          # replace memory DB from a backup
llm update                  # grab a newer release
```

**REPL slash commands:**

```
/help    /quit    /init    /stats    /health    /audit
/config  /tools   /mode    /model
/skills  /forget <id>  /remember <fact>
/recall <query>  /context
```

## Configure

```bash
cp config/config.example.toml config/local.toml
```

Common knobs in `config/local.toml`:

| `[ollama]`           |                                                      |
|----------------------|------------------------------------------------------|
| `chat_model`         | capable model — used for code / tools / long prompts |
| `fast_model`         | optional small model for short/chatty turns (cascade)|
| `embed_model`        | for the memory index                                 |
| `num_ctx`            | per-reply token budget (default 8192)                |
| `keep_alive`         | how long Ollama holds the model in RAM (default 30m) |

| `[memory]`           |                                                      |
|----------------------|------------------------------------------------------|
| `vector_search`      | `false` = pure BM25 recall, ~10× faster              |
| `expansion_variants` | LLM query paraphrasings (default 0)                  |
| `bm25_weight` / `vector_weight` | fusion weights                            |

| `[tools]`            |                                                      |
|----------------------|------------------------------------------------------|
| `mode`               | `read-only` / `workspace-write` / `unrestricted`     |
| `workspace_root`     | confines writes to a directory tree                  |
| `deny_globs`         | extra paths to refuse                                |

| `[web]`              |                                                      |
|----------------------|------------------------------------------------------|
| `brave_api_key`      | enables `web_search`                                 |
| `block_private_addrs`| refuse fetches to RFC1918 / metadata IPs             |

Env vars override: `LOCALMIND_CHAT_MODEL`, `LOCALMIND_DB_PATH`,
`BRAVE_API_KEY`, `LOCALMIND_NO_UPDATE_CHECK`, etc.

### Permission modes

```
read-only          no writes, no shell mutations, no outbound network
workspace-write    writes confined to workspace_root; shell/web prompt
unrestricted       prompts only; no extra guard-rails
```

Switch mid-session with `/mode <ro|ww|full>`. Default in `[tools].mode`.

## Backup / move to another machine

The memory DB is a single SQLite file — everything the agent knows lives there.

```bash
llm backup                          # ~/localmind-backup-YYYYMMDD-HHMMSS.db
llm backup /path/to/somewhere.db    # explicit destination

# On the other machine:
llm restore /path/to/somewhere.db   # prompts y/N before overwriting
```

`backup` uses SQLite's `VACUUM INTO` — safe while localmind is running.
`restore` keeps your previous DB at `memory.db.bak-YYYYMMDD-HHMMSS` so repeat
restores never clobber each other's rollback points.

## Update

`localmind` checks GitHub for a newer release once every 24 hours (background,
non-blocking). When one is available, you see this at startup:

```
↑ v0.2.0 available (you have 0.1.6) — run 'llm update' to upgrade
```

Then:

```bash
llm update            # re-runs install.sh
llm update --force    # reinstall even when on latest
```

Disable with `[updates] check = false` or `LOCALMIND_NO_UPDATE_CHECK=1`.

## Uninstall

```bash
curl -fsSL https://raw.githubusercontent.com/nevenkordic/localmind/main/uninstall.sh | sh
```

Removes the binary and strips the PATH line. Your memory DB and audit log are
**kept**. Opt-in flags: `LOCALMIND_PURGE_DATA=1` wipes the memory DB;
`LOCALMIND_PURGE_MODELS=1` runs `ollama rm` on the default models. Ollama
itself is never removed.

## Where things live

```
~/Library/Application Support/com.calligoit.localmind/   macOS
~/.local/share/localmind/                                Linux
%LOCALAPPDATA%\localmind\                                Windows
  ├── memory.db       facts, skills, embeddings, KG
  ├── audit.log       JSONL log of every tool call
  └── history.txt     REPL history (mode 0600 on Unix)
```

`llm health` prints the resolved paths.

## Development

```bash
cargo test                 # unit + smoke + e2e
bash scripts/preflight.sh  # full pre-ship verification
cargo build --release      # binary at target/release/llm
```

## License

MIT — see [LICENSE](LICENSE).
