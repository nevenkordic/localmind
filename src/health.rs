//! Health report — what's the state of localmind right now? Surfaced via
//! `llm health` (CLI subcommand) and `/health` (REPL slash command). Both
//! call the same `report()` function so the output is identical.

use crate::config::Config;
use crate::llm::ollama::OllamaClient;
use crate::memory::Store;

pub async fn report(cfg: &Config, store: &Store) -> String {
    let mut out = String::new();
    out.push_str("localmind health\n");
    out.push_str("================\n\n");

    // Memory DB
    out.push_str("memory:\n");
    out.push_str(&format!(
        "  db_path:           {}\n",
        cfg.memory.db_path_resolved().display()
    ));
    match store.stats().await {
        Ok(s) => {
            out.push_str(&format!("  memories:          {}\n", s.memory_count));
            out.push_str(&format!(
                "  with embedding:    {} / {} ({:.0}%)\n",
                s.vector_count,
                s.memory_count,
                pct(s.vector_count, s.memory_count),
            ));
            out.push_str(&format!("  kg entities:       {}\n", s.entity_count));
            out.push_str(&format!("  kg edges:          {}\n", s.edge_count));
            out.push_str(&format!("  db size:           {} bytes\n", s.db_size));
        }
        Err(e) => out.push_str(&format!("  ERROR reading stats: {e}\n")),
    }
    out.push('\n');

    // Embedder outbox
    out.push_str("embedder outbox:\n");
    match store.outbox_health().await {
        Ok(h) => {
            out.push_str(&format!("  pending:           {}\n", h.pending));
            out.push_str(&format!("  running:           {}\n", h.running));
            out.push_str(&format!("  done:              {}\n", h.done));
            out.push_str(&format!("  failed:            {}\n", h.failed));
            if let Some(err) = h.last_error {
                let truncated: String = err.chars().take(200).collect();
                out.push_str(&format!("  last_error:        {truncated}\n"));
            }
            if h.pending > 0 && h.failed == 0 {
                out.push_str("  (rows pending — embedder is catching up)\n");
            } else if h.failed > 0 {
                out.push_str("  (failed rows present — check Ollama reachability)\n");
            }
        }
        Err(e) => out.push_str(&format!("  ERROR reading outbox: {e}\n")),
    }
    out.push('\n');

    // Ollama reachability
    out.push_str("ollama:\n");
    out.push_str(&format!("  host:              {}\n", cfg.ollama.host));
    out.push_str(&format!("  chat_model:        {}\n", cfg.ollama.chat_model));
    out.push_str(&format!(
        "  embed_model:       {}\n",
        cfg.ollama.embed_model
    ));
    let client = OllamaClient::new(&cfg.ollama);
    match client.health().await {
        Ok(models) => {
            out.push_str(&format!(
                "  reachable:         yes ({} models)\n",
                models.len()
            ));
            let chat_present = model_is_installed(&models, &cfg.ollama.chat_model);
            let embed_present = model_is_installed(&models, &cfg.ollama.embed_model);
            out.push_str(&format!(
                "  chat_model loaded: {}\n",
                if chat_present {
                    "yes"
                } else {
                    "NO — run `ollama pull ...`"
                }
            ));
            out.push_str(&format!(
                "  embed loaded:      {}\n",
                if embed_present {
                    "yes"
                } else {
                    "NO — embeddings will fail"
                }
            ));
        }
        Err(e) => {
            out.push_str(&format!("  reachable:         NO ({e})\n"));
        }
    }
    out.push('\n');

    // Recall config — what affects speed
    out.push_str("recall config:\n");
    out.push_str(&format!(
        "  expansion_variants: {}  (each adds 1 Ollama round-trip per recall)\n",
        cfg.memory.expansion_variants
    ));
    out.push_str(&format!(
        "  vector_search:      {}  (false = pure BM25, ~10× faster)\n",
        cfg.memory.vector_search
    ));
    out
}

fn pct(num: i64, denom: i64) -> f64 {
    if denom <= 0 {
        return 0.0;
    }
    (num as f64) * 100.0 / (denom as f64)
}

/// True if `needle` names an installed model. Handles the `:latest` tag
/// normalisation: a config entry without an explicit tag must match an
/// installed `<name>:latest` (which is how Ollama stores the default
/// tag). Without this, health reported "embed loaded: NO" even though
/// the model was right there.
fn model_is_installed(models: &[String], needle: &str) -> bool {
    if needle.is_empty() {
        return true;
    }
    if models.iter().any(|m| m == needle) {
        return true;
    }
    if !needle.contains(':') {
        let with_latest = format!("{needle}:latest");
        return models.iter().any(|m| m == &with_latest);
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_name_matches_latest_tag() {
        let installed = vec!["nomic-embed-text:latest".to_string()];
        assert!(model_is_installed(&installed, "nomic-embed-text"));
    }

    #[test]
    fn exact_tag_match() {
        let installed = vec!["qwen2.5-coder:7b".to_string()];
        assert!(model_is_installed(&installed, "qwen2.5-coder:7b"));
    }

    #[test]
    fn missing_returns_false() {
        let installed = vec!["other:1b".to_string()];
        assert!(!model_is_installed(&installed, "qwen2.5-coder:7b"));
    }

    #[test]
    fn empty_needle_is_installed() {
        assert!(model_is_installed(&[], ""));
    }
}
