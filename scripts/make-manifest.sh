#!/usr/bin/env bash
# Produce a SHA-256 hash manifest for the release binary and the files
# that ship alongside it, so anyone can verify what they received.

set -euo pipefail
cd "$(dirname "$0")/.."

OUT="target/release/SHA256SUMS"
mkdir -p target/release

files=(
    "target/release/localmind"
    "target/release/localmind.exe"
    "migrations/001_init.sql"
    "migrations/002_vector_index.sql"
    "config/config.example.toml"
    "security-notes.txt"
    "SETUP.md"
)

: > "$OUT"
hasher="sha256sum"; command -v "$hasher" >/dev/null || hasher="shasum -a 256"
for f in "${files[@]}"; do
    [[ -f "$f" ]] && $hasher "$f" >> "$OUT"
done
cat "$OUT"
