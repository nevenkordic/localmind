#!/usr/bin/env bash
# localmind install script (macOS / Linux).
# Does NOT escalate privileges. Does NOT register any service or scheduled task.
# All work happens under the invoking user.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

log()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!!\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31merr\033[0m %s\n' "$*" >&2; exit 1; }

# ---- 1. Rust toolchain ------------------------------------------------------
if ! command -v cargo >/dev/null; then
    log "Installing Rust via rustup (user-local, no sudo)"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env"
else
    log "cargo found: $(cargo --version)"
fi

# ---- 2. Ollama --------------------------------------------------------------
if ! command -v ollama >/dev/null; then
    warn "Ollama is not installed."
    warn "Install from https://ollama.com/download and then rerun this script."
    warn "Skipping Ollama setup; you can run 'ollama pull <model>' later."
else
    log "ollama found: $(ollama --version 2>&1 | head -1)"
    log "pulling models (large downloads — can be skipped with SKIP_MODELS=1)"
    if [[ "${SKIP_MODELS:-0}" != "1" ]]; then
        ollama pull qwen2.5-coder:32b || warn "qwen2.5-coder:32b pull failed; try a smaller tag"
        ollama pull gemma3:27b       || warn "gemma3:27b pull failed; try gemma3:12b"
        ollama pull nomic-embed-text || warn "nomic-embed-text pull failed"
    fi
fi

# ---- 3. Build ---------------------------------------------------------------
log "Building release binary (this can take a few minutes)"
cargo build --release

# ---- 4. Config --------------------------------------------------------------
mkdir -p config
if [[ ! -f config/local.toml ]]; then
    cp config/config.example.toml config/local.toml
    log "Wrote config/local.toml — edit it to set your Brave API key (optional)"
fi

# ---- 5. Hash manifest -------------------------------------------------------
log "Generating SHA256 hash manifest for your IT team"
BIN=target/release/localmind
if [[ -f "$BIN" ]]; then
    if command -v shasum >/dev/null; then
        shasum -a 256 "$BIN" > target/release/SHA256SUMS
    else
        sha256sum "$BIN" > target/release/SHA256SUMS
    fi
    cat target/release/SHA256SUMS
fi

log "Done. Run:"
echo
echo "    $(pwd)/target/release/localmind init"
echo "    $(pwd)/target/release/localmind"
echo
