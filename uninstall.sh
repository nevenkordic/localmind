#!/usr/bin/env sh
# Uninstaller for localmind. Removes the `llm` binary from its install
# locations and strips the PATH line the installer added to your shell rc.
#
#   curl -fsSL https://raw.githubusercontent.com/nevenkordic/localmind/main/uninstall.sh | sh
#
# By default your memory database and audit log are PRESERVED — they contain
# your facts, skills, and conversation history and are not replaceable. Opt in
# to wipe them with LOCALMIND_PURGE_DATA=1. Ollama and any pulled models are
# never touched unless you set LOCALMIND_PURGE_MODELS=1, since you probably
# use them for other things too.
#
# Environment overrides:
#   LOCALMIND_PURGE_DATA=1    also delete the memory DB, audit log, history
#   LOCALMIND_PURGE_MODELS=1  also `ollama rm` the default models this
#                             installer pulled (qwen2.5-coder:7b + nomic-embed-text)
#   LOCALMIND_CHAT_MODEL      override which chat model is removed on purge
#   LOCALMIND_EMBED_MODEL     override which embed model is removed on purge

set -eu

CHAT_MODEL="${LOCALMIND_CHAT_MODEL:-qwen2.5-coder:7b}"
EMBED_MODEL="${LOCALMIND_EMBED_MODEL:-nomic-embed-text}"

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
info() { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m %s\n' "$*" >&2; }

removed_any=0

# ---- remove binary(ies) -----------------------------------------------------
remove_binary() {
  path="$1"
  [ -e "$path" ] || [ -L "$path" ] || return 1
  if [ -w "$(dirname "$path")" ] && [ ! -L "$path" ]; then
    rm -f "$path" && info "Removed $path"
  elif [ -w "$path" ]; then
    rm -f "$path" && info "Removed $path"
  else
    info "Removing $path (sudo)"
    sudo rm -f "$path"
  fi
  removed_any=1
}

# Known locations — check each. Also respect an explicit override.
for candidate in \
  "${LOCALMIND_INSTALL_DIR:-}/llm" \
  "$HOME/.local/bin/llm" \
  "/usr/local/bin/llm" \
  "/opt/homebrew/bin/llm"
do
  # Skip blanks (LOCALMIND_INSTALL_DIR unset).
  case "$candidate" in
    /llm) continue ;;
  esac
  remove_binary "$candidate" 2>/dev/null || true
done

if [ "$removed_any" = "0" ]; then
  warn "No llm binary found in ~/.local/bin, /usr/local/bin, or /opt/homebrew/bin"
fi

# ---- strip PATH line from shell rc -----------------------------------------
# Match the comment + export pair the installer writes. We match ONLY the exact
# 2-line block the installer added, so hand-written PATH entries are untouched.
strip_rc() {
  rc="$1"
  [ -f "$rc" ] || return 0
  grep -q '# Added by localmind installer' "$rc" 2>/dev/null || return 0

  tmp=$(mktemp)
  awk '
    /^# Added by localmind installer$/ { skip = 2; next }
    skip > 0 { skip--; next }
    { print }
  ' "$rc" > "$tmp"

  # Only overwrite if it actually changed (awk with no match still copies).
  if ! cmp -s "$rc" "$tmp"; then
    cp "$rc" "$rc.localmind-bak"
    mv "$tmp" "$rc"
    info "Stripped localmind PATH line from $rc (backup at $rc.localmind-bak)"
  else
    rm -f "$tmp"
  fi
}

for rc in "$HOME/.zshrc" "$HOME/.bashrc" "$HOME/.profile" "$HOME/.config/fish/config.fish"; do
  strip_rc "$rc" || true
done

# ---- optional: purge memory DB / audit log / history ------------------------
os=$(uname -s | tr '[:upper:]' '[:lower:]')
case "$os" in
  darwin) DATA_DIR="$HOME/Library/Application Support/com.calligoit.localmind" ;;
  linux)  DATA_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/localmind" ;;
  *)      DATA_DIR="" ;;
esac

if [ "${LOCALMIND_PURGE_DATA:-0}" = "1" ]; then
  if [ -n "$DATA_DIR" ] && [ -d "$DATA_DIR" ]; then
    info "Removing data dir: $DATA_DIR"
    rm -rf "$DATA_DIR"
  else
    warn "No data dir found at $DATA_DIR — nothing to purge"
  fi
else
  if [ -n "$DATA_DIR" ] && [ -d "$DATA_DIR" ]; then
    warn "Preserving $DATA_DIR (memory DB, audit log). Pass LOCALMIND_PURGE_DATA=1 to wipe."
  fi
fi

# ---- optional: remove the default models -----------------------------------
if [ "${LOCALMIND_PURGE_MODELS:-0}" = "1" ]; then
  if command -v ollama >/dev/null 2>&1; then
    for m in "$CHAT_MODEL" "$EMBED_MODEL"; do
      if ollama list 2>/dev/null | awk 'NR>1{print $1}' | grep -Fxq "$m"; then
        info "ollama rm $m"
        ollama rm "$m" >/dev/null 2>&1 || warn "failed to remove $m"
      fi
    done
  else
    warn "ollama command not found — can't purge models"
  fi
fi

# ---- done -------------------------------------------------------------------
bold ""
bold "localmind uninstalled."
if [ "${LOCALMIND_PURGE_DATA:-0}" != "1" ] && [ -n "$DATA_DIR" ] && [ -d "$DATA_DIR" ]; then
  echo "Memory / audit files kept at: $DATA_DIR"
  echo "Re-run with LOCALMIND_PURGE_DATA=1 to remove them, or delete the dir manually."
fi
echo "Open a new terminal to refresh PATH."
