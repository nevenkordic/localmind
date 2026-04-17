//! Background embedding worker — the outbox-polling pattern from broodlink's
//! embedding-worker service. Polls embedding_outbox every 2s, picks one
//! pending row at a time, calls Ollama for an embedding, upserts the vector.

use crate::config::Config;
use crate::llm::ollama::OllamaClient as Ollama;
use crate::memory::Store;
use std::time::Duration;
use tokio::task::JoinHandle;

pub fn spawn(store: Store, cfg: Config) -> JoinHandle<()> {
    tokio::spawn(async move {
        let client = Ollama::new(&cfg.ollama);
        let embed_model = cfg.ollama.embed_model.clone();
        loop {
            match store.claim_next_embedding_job().await {
                Ok(Some((outbox_id, memory_id, text))) => {
                    let chunk = smart_chunk(&text);
                    match client.embed(&chunk).await {
                        Ok(vec) => {
                            if let Err(e) =
                                store.upsert_embedding(&memory_id, &vec, &embed_model).await
                            {
                                tracing::warn!("upsert_embedding failed: {e}");
                                let _ =
                                    store.mark_embedding_failed(outbox_id, &e.to_string()).await;
                            } else {
                                let _ = store.mark_embedding_done(outbox_id).await;
                                tracing::debug!("embedded memory {memory_id}");
                            }
                        }
                        Err(e) => {
                            tracing::warn!("embed failed for {memory_id}: {e}");
                            let _ = store.mark_embedding_failed(outbox_id, &e.to_string()).await;
                            tokio::time::sleep(Duration::from_secs(10)).await;
                        }
                    }
                }
                Ok(None) => {
                    tokio::time::sleep(Duration::from_secs(2)).await;
                }
                Err(e) => {
                    tracing::warn!("outbox poll failed: {e}");
                    tokio::time::sleep(Duration::from_secs(5)).await;
                }
            }
        }
    })
}

/// Broodlink-style "smart chunk splitting at heading/code-fence/paragraph
/// boundaries". For a single-memory text we just truncate politely — if the
/// content is huge, take the first ~2000 chars up to a clean boundary.
fn smart_chunk(text: &str) -> String {
    const MAX: usize = 6000;
    if text.len() <= MAX {
        return text.to_string();
    }
    let cut = text[..MAX]
        .rfind("\n\n")
        .or_else(|| text[..MAX].rfind('.'))
        .unwrap_or(MAX);
    text[..cut].to_string()
}
