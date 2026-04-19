# localmind

Run any local LLM with persistent memory and context.

A single CLI binary that turns an [Ollama](https://ollama.com)-served model
into an interactive agent with long-term recall, learnable skills, and
permissioned tools. Everything runs on your machine — no cloud, no telemetry,
no dependencies beyond Ollama.

## What it does

- **Talk to any local model.** Pick any chat / vision / embedding model you
  have installed in Ollama (`llm models` opens a picker). Switch any time.
- **Remembers across sessions.** A single SQLite file holds every fact,
  decision, skill, and embedding. Survives restarts indefinitely.
- **Recalls automatically.** Hybrid BM25 + vector search runs at the start
  of every turn and injects relevant prior context — the model never has to
  remember to look things up.
- **Learns skills.** Tell it "from now on when X, do Y" and that procedure
  surfaces every time a future turn matches.
- **Reads files, PDFs, docx, xlsx, images.** Vision-capable models describe
  images via the same call.
- **Runs shell, networking, and web tools.** Cross-platform native
  implementations of `port_check`, `dns_lookup`, `whois`, `http_fetch`,
  `listening_ports`, plus optional Brave Search and arbitrary URL fetch.
- **Acts safely.** Per-call permission prompts on side-effecting tools,
  destructive-pattern detection on shell commands, hard credential
  deny-list always on, SSRF guard against private/loopback/metadata IPs.
- **Inspects everything.** `llm health`, `/recall <q>`, `/context`, plus a
  JSONL audit log of every tool call (large args truncated).

## Install

**One-line install** (macOS arm64, Linux x86_64/arm64) — downloads the latest
release binary, verifies its SHA256, installs to `~/.local/bin/llm`, installs
[Ollama](https://ollama.com) if missing, starts its server, and pulls the
default chat + embed models (~5 GB). After it finishes, `llm` works from any
shell:

```bash
curl -fsSL https://raw.githubusercontent.com/nevenkordic/localmind/main/install.sh | sh
```

Environment overrides:

| var                         | default                 | what it does                                    |
|-----------------------------|-------------------------|-------------------------------------------------|
| `LOCALMIND_INSTALL_DIR`     | `$HOME/.local/bin`      | install target (auto-added to PATH if missing)  |
| `LOCALMIND_VERSION`         | `latest`                | pin a release tag, e.g. `v0.1.0`                |
| `LOCALMIND_CHAT_MODEL`      | `qwen2.5-coder:7b`      | chat model the installer pulls                  |
| `LOCALMIND_EMBED_MODEL`     | `nomic-embed-text`      | embed model the installer pulls                 |
| `LOCALMIND_SKIP_OLLAMA=1`   | —                       | don't install or start Ollama                   |
| `LOCALMIND_SKIP_MODELS=1`   | —                       | don't pull models (saves ~5 GB on metered nets) |

**Build from source** (Intel Mac and Windows have to go this route; Intel Mac
release binaries are not published):

```bash
git clone https://github.com/nevenkordic/localmind
cd localmind
./scripts/install.sh        # macOS / Linux
.\scripts\install.ps1       # Windows
```

Want more capability and have the RAM? Swap in bigger models any time — they're
not installed by default because the 32B download is 20 GB and most laptops
can't run it:

```bash
ollama pull qwen2.5-coder:32b
llm models                  # interactive picker
llm models --chat qwen2.5-coder:32b --vision gemma3:27b
```

## Backup / move to another machine

Your memory DB (facts, skills, embeddings, KG) is a single SQLite file. Two
commands let you copy it around safely — no need to re-teach on a new laptop.

```bash
llm backup                          # writes ~/localmind-backup-YYYYMMDD-HHMMSS.db
llm backup /path/to/somewhere.db    # explicit destination

# On the new machine (or after `ollama pull` elsewhere):
llm restore /path/to/somewhere.db   # prompts y/N before overwriting
llm restore ...  --yes              # skip the prompt (required in non-TTY)
```

`backup` uses SQLite's `VACUUM INTO` — safe to run while localmind is in use
and produces a self-contained file (no need for WAL/shm sidecars). `restore`
keeps your previous DB at `memory.db.bak-YYYYMMDD-HHMMSS` (timestamped so
repeated restores never clobber each other's rollback points) so you can
fall back manually if a restore looks wrong. Quit any other running `llm`
processes before restoring so nothing races the file swap.

## Update

`localmind` checks GitHub for a newer release once every 24 hours (cached in
`<data_dir>/update-check.json`, background task, non-blocking). When one is
available, a single line prints at startup:

```
↑ v0.2.0 available (you have 0.1.0) — run 'llm update' to upgrade
```

Then:

```bash
llm update            # re-runs install.sh — pulls new binary, removes older ones
llm update --force    # reinstall even when already on latest
```

Turn off the check with `[updates] check = false` in `local.toml`, or with
`LOCALMIND_NO_UPDATE_CHECK=1` in the environment.

## Uninstall

One-liner — removes the `llm` binary and strips the PATH line the installer
added to your shell rc. Your memory database and audit log are **kept** so you
can reinstall later without losing context:

```bash
curl -fsSL https://raw.githubusercontent.com/nevenkordic/localmind/main/uninstall.sh | sh
```

Flags:

| var                          | what it does                                          |
|------------------------------|-------------------------------------------------------|
| `LOCALMIND_PURGE_DATA=1`     | also wipe the memory DB, audit log, REPL history      |
| `LOCALMIND_PURGE_MODELS=1`   | also `ollama rm` the default chat + embed models      |

Ollama itself is never removed (you may use it for other tools).

## Use

```bash
llm                         # interactive REPL
llm ask "fix the failing test"
llm health                  # DB stats, Ollama reachability, recall config
llm memory search "deploy procedure"
llm memory search "deploy procedure" --bm25   # skip embedding (fast)
```

### REPL slash commands

```
/help    /quit    /init    /stats    /health    /audit
/config  /tools   /mode    /model
/skills  /forget <id>  /remember <fact>
/recall <query>  /context
```

### Teaching it

`localmind` learns in two ways, both model-agnostic:

1. **Auto-extract.** `my name is X`, `call me X`, and `remember X` are
   captured by regex before the model sees the turn — they're stored even
   if the model would have skipped `store_memory`.
2. **Skills.** Tell it _"from now on when I X, do Y"_ and the model stores
   a `kind="skill"` memory. Future turns whose query matches surface the
   skill into context automatically.

### Memory primer

At the start of every turn, `localmind` runs a hybrid (BM25 + vector)
search against the user's input and injects the top hits as a system
message. The model gets relevant prior context whether or not it
remembers to call `search_memory`.

## Configure

```bash
cp config/config.example.toml config/local.toml
```

Then edit `config/local.toml`. Common knobs:

| `[ollama]`                     | what it does                              |
|--------------------------------|-------------------------------------------|
| `chat_model`                   | primary model — `llm models` to pick      |
| `embed_model`                  | for the memory index                      |
| `num_ctx`                      | per-reply token budget                    |

| `[memory]`                     | what it does                              |
|--------------------------------|-------------------------------------------|
| `vector_search`                | false = pure BM25 recall, ~10× faster     |
| `expansion_variants`           | LLM query paraphrasings (default 0)       |
| `bm25_weight` / `vector_weight`| fusion weights for hybrid search          |

| `[tools]`                      | what it does                              |
|--------------------------------|-------------------------------------------|
| `mode`                         | `read-only` / `workspace-write` / `unrestricted` |
| `workspace_root`               | confines writes to a directory tree       |
| `deny_globs`                   | extra paths to refuse                     |
| `allow_rules` / `deny_rules` / `ask_rules` | rule-based permissions        |

| `[web]`                        | what it does                              |
|--------------------------------|-------------------------------------------|
| `brave_api_key`                | enables `web_search`                      |
| `block_private_addrs`          | refuse fetches to RFC1918 / metadata IPs  |

Env vars override: `LOCALMIND_CHAT_MODEL`, `LOCALMIND_DB_PATH`, `BRAVE_API_KEY`, etc.

## Permission modes

```
read-only          no writes, no shell mutations, no outbound network
workspace-write    writes confined to workspace_root; shell/web prompt
unrestricted       prompts only; no extra guard-rails
```

Switch mid-session with `/mode <ro|ww|full>`. Set the default in
`[tools].mode` or per-invocation with `--mode <...>`.

## Where things live

```
~/.local/share/localmind/memory.db      Linux/macOS memory DB
%LOCALAPPDATA%\localmind\memory.db      Windows
~/.local/share/localmind/audit.log      JSONL audit log
~/.local/share/localmind/history.txt    REPL history (mode 0600)
config/local.toml                       Your config (gitignored)
```

`llm health` shows the resolved paths and embedder status.

## Development

```bash
cargo test                 # unit + smoke + e2e
bash scripts/preflight.sh  # full pre-ship verification
cargo build --release      # binary at target/release/llm
```

## License

MIT — see [LICENSE](LICENSE).
