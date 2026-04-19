#!/usr/bin/env sh
# Turn-key installer for localmind. Fetches the latest release binary, verifies
# its SHA256, installs to a PATH-friendly location, installs Ollama if missing,
# starts the Ollama server, and pulls the default chat + embed models so the
# first `llm` run actually works.
#
#   curl -fsSL https://raw.githubusercontent.com/nevenkordic/localmind/main/install.sh | sh
#
# Environment overrides:
#   LOCALMIND_INSTALL_DIR   install target (default: ~/.local/bin, falls back
#                           to /usr/local/bin with sudo)
#   LOCALMIND_VERSION       pin a release tag (default: latest)
#   LOCALMIND_CHAT_MODEL    chat model to pull (default: qwen2.5-coder:7b)
#   LOCALMIND_EMBED_MODEL   embed model to pull (default: nomic-embed-text)
#   LOCALMIND_SKIP_OLLAMA=1 don't install or touch Ollama
#   LOCALMIND_SKIP_MODELS=1 don't pull default models (saves ~5 GB)

set -eu

REPO="nevenkordic/localmind"
VERSION="${LOCALMIND_VERSION:-latest}"
CHAT_MODEL="${LOCALMIND_CHAT_MODEL:-qwen2.5-coder:3b}"
EMBED_MODEL="${LOCALMIND_EMBED_MODEL:-nomic-embed-text}"

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
info() { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31merr\033[0m %s\n' "$*" >&2; exit 1; }

# ---- detect platform --------------------------------------------------------
os=$(uname -s | tr '[:upper:]' '[:lower:]')
case "$os" in
  darwin) os_target="apple-darwin" ;;
  linux)  os_target="unknown-linux-gnu" ;;
  *)      die "unsupported OS: $os (only macOS and Linux; Windows users see scripts/install.ps1)" ;;
esac

arch=$(uname -m)
case "$arch" in
  x86_64|amd64)   arch_target="x86_64" ;;
  arm64|aarch64)  arch_target="aarch64" ;;
  *)              die "unsupported arch: $arch" ;;
esac

# Intel Macs are not built (upstream macos-13 runner is unreliable).
if [ "$os" = "darwin" ] && [ "$arch_target" = "x86_64" ]; then
  die "Intel Mac binaries are not published. Build from source: git clone https://github.com/${REPO} && cd localmind && cargo build --release"
fi

target="${arch_target}-${os_target}"
asset="llm-${target}"

# ---- resolve install dir ----------------------------------------------------
# Order of preference:
#   1. $LOCALMIND_INSTALL_DIR          (explicit override)
#   2. ~/.local/bin                    (user-writable, no sudo, XDG-standard)
#   3. /usr/local/bin                  (system-wide, needs sudo)
default_install_dir() {
  if [ -n "${LOCALMIND_INSTALL_DIR:-}" ]; then
    printf '%s' "$LOCALMIND_INSTALL_DIR"
  else
    printf '%s' "$HOME/.local/bin"
  fi
}
INSTALL_DIR=$(default_install_dir)

# ---- resolve release tag ----------------------------------------------------
if [ "$VERSION" = "latest" ]; then
  info "Resolving latest release tag"
  tag=$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" \
        | grep '"tag_name"' | head -1 | cut -d'"' -f4)
  [ -n "$tag" ] || die "could not resolve latest tag (rate-limited?). Try LOCALMIND_VERSION=v0.1.0"
else
  tag="$VERSION"
fi

base_url="https://github.com/${REPO}/releases/download/${tag}"

# ---- download + verify ------------------------------------------------------
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

info "Downloading ${asset} (${tag})"
curl -fsSL "${base_url}/${asset}" -o "$tmp/llm" \
  || die "download failed. Check that ${asset} exists at ${base_url}"
curl -fsSL "${base_url}/${asset}.sha256" -o "$tmp/llm.sha256" \
  || die "checksum download failed"

info "Verifying SHA256"
expected=$(awk '{print $1}' "$tmp/llm.sha256")
if command -v sha256sum >/dev/null 2>&1; then
  actual=$(sha256sum "$tmp/llm" | awk '{print $1}')
elif command -v shasum >/dev/null 2>&1; then
  actual=$(shasum -a 256 "$tmp/llm" | awk '{print $1}')
else
  die "no sha256 utility found (need sha256sum or shasum)"
fi
[ "$expected" = "$actual" ] || die "SHA256 mismatch — refusing to install. expected=$expected actual=$actual"

chmod +x "$tmp/llm"

# ---- install ----------------------------------------------------------------
install_to() {
  dir="$1"
  # Create the directory first. If we can't create it as the current user,
  # re-try with sudo so /usr/local/bin (parent owned by root on fresh macOS
  # Apple Silicon) still works.
  if ! mkdir -p "$dir" 2>/dev/null; then
    info "Creating ${dir} (sudo)"
    sudo mkdir -p "$dir" || return 1
  fi
  if [ -w "$dir" ]; then
    mv "$tmp/llm" "$dir/llm"
  else
    info "Installing to ${dir} (sudo)"
    sudo mv "$tmp/llm" "$dir/llm"
    sudo chmod +x "$dir/llm"
  fi
}

if ! install_to "$INSTALL_DIR"; then
  # ~/.local/bin failed (somehow). Fall back to /usr/local/bin with sudo.
  if [ "$INSTALL_DIR" != "/usr/local/bin" ]; then
    warn "could not install to $INSTALL_DIR — falling back to /usr/local/bin"
    INSTALL_DIR="/usr/local/bin"
    install_to "$INSTALL_DIR" || die "install failed for both $HOME/.local/bin and /usr/local/bin"
  else
    die "install failed"
  fi
fi

# ---- remove older binaries at other locations ------------------------------
# If an older `llm` is still sitting in another PATH dir (e.g. the user
# originally installed to /usr/local/bin with sudo, and we just wrote a newer
# one to ~/.local/bin), remove the stale copy so `which llm` resolves to the
# version we just installed regardless of PATH order.
new_bin="$INSTALL_DIR/llm"
old_IFS=$IFS
IFS=:
# shellcheck disable=SC2086  # word-splitting on PATH is intentional here
for p in $PATH; do
  [ -n "$p" ] || continue
  candidate="$p/llm"
  # Skip the newly-installed binary.
  case "$candidate" in
    "$new_bin") continue ;;
  esac
  [ -f "$candidate" ] || continue
  info "Removing older llm at ${candidate}"
  if [ -w "$p" ]; then
    rm -f "$candidate" || warn "could not remove ${candidate}"
  else
    sudo rm -f "$candidate" || warn "could not remove ${candidate}"
  fi
done
IFS=$old_IFS

# ---- PATH check + shell rc wiring ------------------------------------------
# If INSTALL_DIR isn't already on PATH, append an export line to the user's
# shell rc so new shells find the binary. We touch at most one rc file and
# skip the write if the line already exists.
ensure_on_path() {
  dir="$1"
  case ":$PATH:" in
    *":$dir:"*) return 0 ;;
  esac

  rc=""
  shell_name=$(basename "${SHELL:-}")
  case "$shell_name" in
    zsh)  rc="$HOME/.zshrc" ;;
    bash) rc="$HOME/.bashrc" ;;
    fish) rc="$HOME/.config/fish/config.fish" ;;
    *)    rc="$HOME/.profile" ;;
  esac

  line="export PATH=\"$dir:\$PATH\""
  if [ "$shell_name" = "fish" ]; then
    line="set -gx PATH $dir \$PATH"
    mkdir -p "$(dirname "$rc")" 2>/dev/null || true
  fi

  if [ -f "$rc" ] && grep -Fq "$dir" "$rc" 2>/dev/null; then
    warn "$dir not on PATH but already referenced in $rc — open a new terminal or 'source $rc'"
    return 0
  fi

  info "Adding $dir to PATH in $rc"
  printf '\n# Added by localmind installer\n%s\n' "$line" >> "$rc"
  warn "PATH updated in $rc — open a new terminal or run 'source $rc' to pick it up"
}
ensure_on_path "$INSTALL_DIR"

# ---- Ollama bootstrap -------------------------------------------------------
install_ollama_linux() {
  info "Installing Ollama (official installer)"
  curl -fsSL https://ollama.com/install.sh | sh
}

install_ollama_macos() {
  if command -v brew >/dev/null 2>&1; then
    # Formula (CLI + launchd) is the default — smaller (~50 MB vs ~300 MB
    # for the cask) and no menu-bar app. Pass LOCALMIND_OLLAMA_GUI=1 to get
    # the full Ollama.app cask if you want the GUI.
    if [ "${LOCALMIND_OLLAMA_GUI:-0}" = "1" ]; then
      info "Installing Ollama via Homebrew (cask, full GUI app)"
      brew install --cask ollama 2>&1 | tail -5
    else
      info "Installing Ollama via Homebrew (formula, CLI + launchd, headless)"
      brew install ollama 2>&1 | tail -5
      # Start the launchd service so the server auto-starts on login and
      # right now. `brew services` is part of the homebrew-services tap
      # that ships with Homebrew.
      brew services start ollama >/dev/null 2>&1 || true
    fi
  else
    warn "Homebrew not found. Please install Ollama from https://ollama.com/download"
    warn "Then re-run this installer, or pass LOCALMIND_SKIP_OLLAMA=1 to skip."
    return 1
  fi
}

start_ollama_server() {
  # Already up?
  if curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    return 0
  fi
  case "$os" in
    darwin)
      # Prefer the launchd service installed by `brew install ollama`
      # (headless). Fall back to the GUI app if someone opted in via
      # LOCALMIND_OLLAMA_GUI=1, else spawn in background.
      if command -v brew >/dev/null 2>&1 && brew services list 2>/dev/null | grep -q '^ollama'; then
        info "Starting Ollama launchd service"
        brew services start ollama >/dev/null 2>&1 || true
      elif [ -d "/Applications/Ollama.app" ]; then
        info "Starting Ollama.app"
        open -a Ollama || true
      else
        info "Starting Ollama server in background (nohup)"
        nohup ollama serve >/tmp/ollama.log 2>&1 &
      fi
      ;;
    linux)
      if command -v systemctl >/dev/null 2>&1 && systemctl list-unit-files ollama.service >/dev/null 2>&1; then
        info "Starting ollama.service"
        sudo systemctl start ollama 2>/dev/null || systemctl --user start ollama 2>/dev/null || true
      else
        info "Starting Ollama server in background (nohup)"
        nohup ollama serve >/tmp/ollama.log 2>&1 &
      fi
      ;;
  esac
  # Wait up to ~30s for the server to accept connections.
  i=0
  while [ $i -lt 60 ]; do
    if curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.5
    i=$((i + 1))
  done
  return 1
}

if [ "${LOCALMIND_SKIP_OLLAMA:-0}" = "1" ]; then
  info "Skipping Ollama setup (LOCALMIND_SKIP_OLLAMA=1)"
else
  if command -v ollama >/dev/null 2>&1; then
    info "Ollama already installed ($(ollama --version 2>/dev/null | head -1))"
  else
    case "$os" in
      linux)  install_ollama_linux || die "Ollama install failed" ;;
      darwin) install_ollama_macos || die "Ollama install failed — install manually from https://ollama.com/download then re-run, or pass LOCALMIND_SKIP_OLLAMA=1" ;;
    esac
  fi

  if ! start_ollama_server; then
    warn "Ollama server did not come up within 30s — you may need to start it manually."
  else
    info "Ollama server reachable on http://127.0.0.1:11434"
  fi

  if [ "${LOCALMIND_SKIP_MODELS:-0}" = "1" ]; then
    info "Skipping model pull (LOCALMIND_SKIP_MODELS=1)"
  else
    if curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
      info "Pulling chat model: ${CHAT_MODEL}"
      ollama pull "$CHAT_MODEL" || warn "pull of $CHAT_MODEL failed — you can retry with 'ollama pull $CHAT_MODEL'"
      info "Pulling embed model: ${EMBED_MODEL}"
      ollama pull "$EMBED_MODEL" || warn "pull of $EMBED_MODEL failed — you can retry with 'ollama pull $EMBED_MODEL'"
    else
      warn "Ollama server not reachable — skipping model pulls. Start it then run: ollama pull $CHAT_MODEL && ollama pull $EMBED_MODEL"
    fi
  fi
fi

# ---- done -------------------------------------------------------------------
bold ""
bold "Installed: $("$INSTALL_DIR/llm" --version 2>/dev/null || echo "llm ($INSTALL_DIR/llm)")"
bold ""
case ":$PATH:" in
  *":$INSTALL_DIR:"*)
    echo "Run 'llm' to start. First turn may take ~30s as the chat model warms up."
    ;;
  *)
    echo "Open a new terminal (or 'source' your shell rc), then run 'llm'."
    echo "Or use it right now: $INSTALL_DIR/llm"
    ;;
esac
