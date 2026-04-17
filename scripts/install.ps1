# localmind install script (Windows, PowerShell).
# Does NOT require Administrator. Does NOT write to the Registry. Does NOT
# create scheduled tasks, services, or startup entries.

$ErrorActionPreference = 'Stop'

function Log($msg)  { Write-Host "==> $msg" -ForegroundColor Cyan }
function Warn($msg) { Write-Host "!!! $msg" -ForegroundColor Yellow }

Set-Location (Split-Path -Parent $PSCommandPath)
Set-Location ..

# ---- 1. Rust toolchain ------------------------------------------------------
if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    Log "Installing Rust via rustup-init (user-local)"
    $installer = Join-Path $env:TEMP "rustup-init.exe"
    Invoke-WebRequest -Uri "https://win.rustup.rs/x86_64" -OutFile $installer
    & $installer -y --profile minimal --default-toolchain stable
    Remove-Item $installer -Force
    $env:Path = "$env:USERPROFILE\.cargo\bin;$env:Path"
} else {
    Log "cargo found: $(cargo --version)"
}

# ---- 2. Ollama --------------------------------------------------------------
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Warn "Ollama is not installed. Download from https://ollama.com/download/windows"
    Warn "Re-run this script after Ollama is installed."
} else {
    Log "ollama found"
    if ($env:SKIP_MODELS -ne "1") {
        Log "pulling models (large)"
        ollama pull qwen2.5-coder:32b
        ollama pull gemma3:27b
        ollama pull nomic-embed-text
    }
}

# ---- 3. Build ---------------------------------------------------------------
Log "Building release binary"
cargo build --release

# ---- 4. Config --------------------------------------------------------------
if (-not (Test-Path "config\local.toml")) {
    Copy-Item "config\config.example.toml" "config\local.toml"
    Log "Wrote config\local.toml"
}

# ---- 5. Hash manifest -------------------------------------------------------
$bin = "target\release\localmind.exe"
if (Test-Path $bin) {
    $hash = Get-FileHash -Algorithm SHA256 $bin
    "$($hash.Hash)  $bin" | Out-File -Encoding ascii "target\release\SHA256SUMS"
    Log "SHA256: $($hash.Hash)"
}

Log "Done. Run:"
Write-Host ""
Write-Host "    .\target\release\localmind.exe init"
Write-Host "    .\target\release\localmind.exe"
Write-Host ""
