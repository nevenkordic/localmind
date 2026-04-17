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

**One-line install** (macOS / Linux, x86_64 or arm64) — downloads the latest
release binary, verifies its SHA256, installs to `/usr/local/bin/llm`:

```bash
curl -fsSL https://raw.githubusercontent.com/nevenkordic/localmind/main/install.sh | sh
```

Pin a version with `LOCALMIND_VERSION=v0.1.0`, or change the install path with
`LOCALMIND_INSTALL_DIR=$HOME/.local/bin`.

**Build from source:**

```bash
git clone https://github.com/nevenkordic/localmind
cd localmind
./scripts/install.sh        # macOS / Linux
.\scripts\install.ps1       # Windows
```

You also need [Ollama](https://ollama.com/download) and at least one model:

```bash
ollama pull qwen2.5-coder:7b
ollama pull nomic-embed-text
```

Hardware too modest for the default 32B? Pick smaller models any time:

```bash
llm models                  # interactive picker
llm models --chat qwen2.5-coder:7b --vision gemma3:12b
```

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
