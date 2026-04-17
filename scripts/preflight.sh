#!/usr/bin/env bash
# Pre-ship verification — run this before tagging or copying the binary out
# of target/release. Stops at the first failing step and prints what broke.
#
# Stages:
#   1. fmt check         — code style is consistent
#   2. clippy            — no new lints (release profile, where the optimiser
#                          can surface dead code that debug doesn't)
#   3. unit + regression — `cargo test --bin llm` (fast, no Ollama)
#   4. smoke + e2e       — `cargo test --tests` (CLI subprocess + mock Ollama)
#   5. release build     — final binary produced under target/release/llm
#   6. binary smoke      — sanity-check the actual binary that ships
#
# Run with `OFFLINE=1 bash scripts/preflight.sh` to skip any future steps that
# need network. Currently every stage is offline-safe.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

green() { printf '\033[1;32m%s\033[0m\n' "$*"; }
blue()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
red()   { printf '\033[1;31m!!\033[0m %s\n' "$*" >&2; }

trap 'red "preflight FAILED at: $BASH_COMMAND"' ERR

start_ts=$(date +%s)

blue "1/6  cargo fmt --check"
cargo fmt --all -- --check

blue "2/6  cargo clippy (release, warnings allowed)"
# We don't gate on -D warnings yet — codebase is young and lint policy hasn't
# been chosen. Surface them so they can't be missed.
cargo clippy --release --all-targets 2>&1 | tee target/preflight-clippy.log

blue "3/6  unit + regression tests (cargo test --bin llm)"
cargo test --bin llm --release

blue "4/6  smoke + e2e integration tests (cargo test --tests)"
cargo test --tests --release

blue "5/6  release build"
cargo build --release

blue "6/6  binary smoke"
BIN="target/release/llm"
test -x "$BIN" || { red "binary missing: $BIN"; exit 1; }
"$BIN" --version | grep -q "$(grep '^version' Cargo.toml | head -1 | cut -d'"' -f2)" \
    || { red "version mismatch from $BIN --version"; exit 1; }

elapsed=$(( $(date +%s) - start_ts ))
echo
green "preflight PASS  (${elapsed}s)"
echo
echo "Next steps to ship:"
echo "  sudo cp $BIN /usr/local/bin/llm"
echo "  shasum -a 256 $BIN"
