//! Build a minimal `--config` TOML pointing at a temp DB + a chosen Ollama URL.
//! Used by smoke and e2e tests so they never touch the real config or DB.

use std::path::Path;

pub fn write_config(dir: &Path, ollama_url: &str) -> std::path::PathBuf {
    let cfg_path = dir.join("config.toml");
    let db_path = dir.join("memory.db");
    let body = format!(
        r#"[ollama]
host = "{ollama}"
chat_model = "test-chat"
vision_model = "test-vision"
embed_model = "test-embed"
timeout_secs = 10
num_ctx = 2048

[memory]
db_path = "{db}"
top_k = 4
bm25_weight = 0.4
vector_weight = 0.6
temporal_half_life_days = 30.0
auto_persist = false
expansion_variants = 0
vector_search = false

[tools]
workspace_root = ""
deny_globs = []
shell_allow_regex = ""
mode = "workspace-write"
allow_rules = []
deny_rules = []
ask_rules = []

[web]
brave_api_key = ""
brave_endpoint = "https://api.search.brave.com/res/v1/web/search"
max_results = 4
block_private_addrs = true

[repl]
show_tool_calls = false
confirm_writes = false
color = false
"#,
        ollama = ollama_url,
        db = db_path.display(),
    );
    std::fs::write(&cfg_path, body).expect("write test config");
    cfg_path
}
