//! Configuration loader — reads TOML, overlays env vars, resolves paths.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Baked-in default config. Used when no local.toml / XDG config /
/// ./config/config.example.toml exists on disk — keeps `llm` running for
/// fresh curl-installed users whose cwd is unrelated to the repo. Also
/// seeded into the user-scope config file when the preflight prompt
/// converts a "no config at all" state into a persisted choice.
pub(crate) const EMBEDDED_DEFAULT_CONFIG: &str =
    include_str!("../config/config.example.toml");

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub ollama: OllamaConfig,
    pub memory: MemoryConfig,
    pub tools: ToolsConfig,
    pub web: WebConfig,
    pub repl: ReplConfig,
    #[serde(default)]
    pub updates: UpdatesConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaConfig {
    pub host: String,
    pub chat_model: String,
    pub vision_model: String,
    pub embed_model: String,
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
    #[serde(default = "default_ctx")]
    pub num_ctx: u32,
}
fn default_timeout() -> u64 {
    600
}
fn default_ctx() -> u32 {
    32768
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    pub db_path: String,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    #[serde(default = "default_bm25_weight")]
    pub bm25_weight: f32,
    #[serde(default = "default_vector_weight")]
    pub vector_weight: f32,
    #[serde(default = "default_half_life")]
    pub temporal_half_life_days: f32,
    #[serde(default)]
    pub auto_persist: bool,
    #[serde(default = "default_expansion")]
    pub expansion_variants: usize,
    /// When true, hybrid_search asks Ollama for an embedding of the query
    /// and runs vector ANN. When false, recall is pure BM25 (sub-millisecond,
    /// no model round-trip — ~10× faster on a fresh Ollama).
    #[serde(default = "default_vector_search")]
    pub vector_search: bool,
}
fn default_top_k() -> usize {
    12
}
fn default_bm25_weight() -> f32 {
    0.4
}
fn default_vector_weight() -> f32 {
    0.6
}
fn default_half_life() -> f32 {
    30.0
}
// Default 0 — query expansion costs an extra Ollama round-trip per recall
// for marginal quality gain. Opt in by raising this.
fn default_expansion() -> usize {
    0
}
fn default_vector_search() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolsConfig {
    #[serde(default)]
    pub workspace_root: String,
    #[serde(default)]
    pub deny_globs: Vec<String>,
    #[serde(default)]
    pub shell_allow_regex: String,
    /// Default session permission mode.
    /// One of: read-only | workspace-write (default) | unrestricted.
    #[serde(default = "default_mode")]
    pub mode: String,
    /// Permission rules. Each entry is `tool(matcher)` or a bare `tool`.
    /// Matchers support `*` for prefix/suffix/contains.
    #[serde(default)]
    pub allow_rules: Vec<String>,
    #[serde(default)]
    pub deny_rules: Vec<String>,
    /// Ask rules always prompt — even when an allow-rule or "always this
    /// session" would otherwise auto-approve. Use for sensitive ops like
    /// `web_fetch(*)` or `shell(git push*)`.
    #[serde(default)]
    pub ask_rules: Vec<String>,
}
fn default_mode() -> String {
    "workspace-write".into()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebConfig {
    #[serde(default)]
    pub brave_api_key: String,
    #[serde(default = "default_brave_endpoint")]
    pub brave_endpoint: String,
    #[serde(default = "default_web_max")]
    pub max_results: usize,
    /// Refuse outbound HTTP/WHOIS connections to private, loopback,
    /// link-local (incl. cloud-metadata 169.254.169.254), and multicast
    /// addresses. Defaults to true so prompt injection can't trick the
    /// agent into fetching internal services. Set to false on trusted
    /// networks where you legitimately need internal HTTP access.
    #[serde(default = "default_block_private")]
    pub block_private_addrs: bool,
}
fn default_brave_endpoint() -> String {
    "https://api.search.brave.com/res/v1/web/search".into()
}
fn default_web_max() -> usize {
    8
}
fn default_block_private() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplConfig {
    #[serde(default = "default_true")]
    pub show_tool_calls: bool,
    #[serde(default = "default_true")]
    pub confirm_writes: bool,
    #[serde(default = "default_true")]
    pub color: bool,
}
fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdatesConfig {
    /// Periodically check GitHub releases for a newer binary. Results are
    /// cached in <data_dir>/update-check.json for `check_interval_hours`;
    /// the HTTP call is fire-and-forget and never blocks startup.
    #[serde(default = "default_true")]
    pub check: bool,
    #[serde(default = "default_update_interval")]
    pub check_interval_hours: u64,
}
fn default_update_interval() -> u64 {
    24
}
impl Default for UpdatesConfig {
    fn default() -> Self {
        Self {
            check: true,
            check_interval_hours: 24,
        }
    }
}

impl Config {
    /// Find and load the config file. Precedence:
    ///   1. --config <path>
    ///   2. ./config/local.toml
    ///   3. $XDG_CONFIG_HOME/localmind/config.toml (or platform equivalent)
    ///   4. ./config/config.example.toml (as a last-resort default)
    /// Then env vars prefixed LOCALMIND_ override.
    /// Resolve the same path Config::load would read from, without parsing.
    /// Used by `llm models` so it can write changes back to the right file.
    pub fn source_path(explicit: Option<&Path>) -> Option<PathBuf> {
        if let Some(p) = explicit {
            return Some(PathBuf::from(p));
        }
        let local = PathBuf::from("config/local.toml");
        if local.exists() {
            return Some(local);
        }
        if let Some(d) = directories::ProjectDirs::from("com", "calligoit", "localmind") {
            let p = d.config_dir().join("config.toml");
            if p.exists() {
                return Some(p);
            }
        }
        let example = PathBuf::from("config/config.example.toml");
        if example.exists() {
            return Some(example);
        }
        None
    }

    pub fn load(explicit: Option<&Path>) -> Result<Self> {
        let path = Self::source_path(explicit);
        let raw = match path {
            Some(p) => {
                std::fs::read_to_string(&p).with_context(|| format!("reading {}", p.display()))?
            }
            // No config file anywhere on disk — fall back to the example
            // config baked into the binary. Fresh curl-installed users don't
            // have a repo checkout handy, so this makes `llm` work out of
            // the box. Any explicit config file still takes precedence.
            None => EMBEDDED_DEFAULT_CONFIG.to_string(),
        };

        let mut cfg: Config = toml::from_str(&raw).context("parsing TOML")?;
        cfg.apply_env_overrides();
        Ok(cfg)
    }

    fn apply_env_overrides(&mut self) {
        if let Ok(v) = std::env::var("LOCALMIND_OLLAMA_HOST") {
            self.ollama.host = v;
        }
        if let Ok(v) = std::env::var("LOCALMIND_CHAT_MODEL") {
            self.ollama.chat_model = v;
        }
        if let Ok(v) = std::env::var("LOCALMIND_VISION_MODEL") {
            self.ollama.vision_model = v;
        }
        if let Ok(v) = std::env::var("LOCALMIND_EMBED_MODEL") {
            self.ollama.embed_model = v;
        }
        if let Ok(v) = std::env::var("LOCALMIND_DB_PATH") {
            self.memory.db_path = v;
        }
        if let Ok(v) = std::env::var("BRAVE_API_KEY") {
            self.web.brave_api_key = v;
        }
        if let Ok(v) = std::env::var("LOCALMIND_BRAVE_API_KEY") {
            self.web.brave_api_key = v;
        }
        if let Ok(v) = std::env::var("LOCALMIND_WORKSPACE_ROOT") {
            self.tools.workspace_root = v;
        }
        if let Ok(v) = std::env::var("LOCALMIND_MODE") {
            self.tools.mode = v;
        }
    }

    pub fn pretty(&self) -> Result<String> {
        Ok(toml::to_string_pretty(self)?)
    }

    pub fn summary(&self) -> String {
        format!(
            "ollama={} chat={} embed={} db={}",
            self.ollama.host,
            self.ollama.chat_model,
            self.ollama.embed_model,
            self.memory.db_path_resolved().display()
        )
    }
}

impl MemoryConfig {
    /// Expand the %DATA% token to the platform data directory and return an
    /// absolute path. Ensures the parent directory exists.
    pub fn db_path_resolved(&self) -> PathBuf {
        let expanded = if self.db_path.contains("%DATA%") {
            let data_dir = directories::ProjectDirs::from("com", "calligoit", "localmind")
                .map(|p| p.data_dir().to_path_buf())
                .unwrap_or_else(|| PathBuf::from("./data"));
            PathBuf::from(
                self.db_path
                    .replace("%DATA%", data_dir.to_str().unwrap_or(".")),
            )
        } else {
            PathBuf::from(&self.db_path)
        };
        if let Some(parent) = expanded.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        expanded
    }
}
