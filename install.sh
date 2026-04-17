#!/usr/bin/env sh
# One-line installer for localmind. Downloads the latest release binary
# matching your OS/arch, verifies its SHA256, and installs to /usr/local/bin.
#
#   curl -fsSL https://raw.githubusercontent.com/nevenkordic/localmind/main/install.sh | sh
#
# Override the install location with $LOCALMIND_INSTALL_DIR. Pin a specific
# version with $LOCALMIND_VERSION (e.g. v0.1.0).

set -eu

REPO="nevenkordic/localmind"
INSTALL_DIR="${LOCALMIND_INSTALL_DIR:-/usr/local/bin}"
VERSION="${LOCALMIND_VERSION:-latest}"

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
info() { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31merr\033[0m %s\n' "$*" >&2; exit 1; }

# ---- detect platform --------------------------------------------------------
os=$(uname -s | tr '[:upper:]' '[:lower:]')
case "$os" in
  darwin) os_target="apple-darwin" ;;
  linux)  os_target="unknown-linux-gnu" ;;
  *)      die "unsupported OS: $os (only macOS and Linux supported by this script; Windows users see scripts/install.ps1)" ;;
esac

arch=$(uname -m)
case "$arch" in
  x86_64|amd64)   arch_target="x86_64" ;;
  arm64|aarch64)  arch_target="aarch64" ;;
  *)              die "unsupported arch: $arch" ;;
esac

target="${arch_target}-${os_target}"
asset="llm-${target}"

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
mkdir -p "$INSTALL_DIR" 2>/dev/null || true
if [ -w "$INSTALL_DIR" ]; then
  mv "$tmp/llm" "$INSTALL_DIR/llm"
else
  info "Installing to ${INSTALL_DIR} (sudo)"
  sudo mv "$tmp/llm" "$INSTALL_DIR/llm"
fi

bold ""
bold "Installed: $("$INSTALL_DIR/llm" --version)"
bold ""
echo "Next:"
echo "  1. Install Ollama (https://ollama.com) and pull a chat model"
echo "       ollama pull qwen2.5-coder:7b"
echo "       ollama pull nomic-embed-text"
echo "  2. Run 'llm init' to verify connectivity"
echo "  3. Run 'llm' for the interactive REPL, or 'llm ask \"...\"' for one-shot"
