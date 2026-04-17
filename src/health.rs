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
            let chat_present = models.iter().any(|m| m == &cfg.ollama.chat_model);
            let embed_present = models.iter().any(|m| m == &cfg.ollama.embed_model);
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
