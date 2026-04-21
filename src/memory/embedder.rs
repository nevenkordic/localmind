//! Background embedding worker — outbox-polling pattern. Polls
//! embedding_outbox every 2s, picks one pending row at a time, calls
//! Ollama for an embedding, upserts the vector.

use crate::config::Config;
use crate::llm::ollama::OllamaClient as Ollama;
use crate::memory::Store;
use std::time::Duration;
use tokio::task::JoinHandle;

pub fn spawn(store: Store, cfg: Config) -> JoinHandle<()> {
    tokio::spawn(async move {
        let client = Ollama::new(&cfg.ollama);
        let embed_model = cfg.ollama.embed_model.clone();
        let contextual = cfg.memory.contextual_embed;
        let extract_entities = cfg.memory.entity_extraction;
        loop {
            match store.claim_next_embedding_job().await {
                Ok(Some((outbox_id, memory_id, text))) => {
                    let chunk = smart_chunk(&text);
                    // Outbox rows store `title\ncontent` — split so the
                    // contextualizer and entity extractor see them
                    // distinctly. An empty title is fine; both tolerate it.
                    let (title, body) = chunk.split_once('\n').unwrap_or(("", chunk.as_str()));
                    // Contextual Retrieval: prepend a model-generated
                    // context line so the vector space picks up topic /
                    // entity signals that short notes alone don't carry.
                    // Failure (timeout, model unavailable) falls back to
                    // the raw chunk — no retry, no hard error.
                    let to_embed = if contextual {
                        match client.contextualize(title, body).await {
                            Ok(ctx) if !ctx.is_empty() => format!("{ctx}\n\n{chunk}"),
                            _ => chunk.clone(),
                        }
                    } else {
                        chunk.clone()
                    };
                    match client.embed(&to_embed).await {
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
                                // Entity extraction is post-embed because
                                // the embedding is the critical path — if
                                // the extraction call fails the memory is
                                // still searchable by vector / BM25, just
                                // without graph provenance.
                                if extract_entities {
                                    extract_and_store(&client, &store, &memory_id, title, body)
                                        .await;
                                }
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

/// Run entity extraction on `memory_id`'s content and upsert the result
/// into the knowledge graph. All errors are logged and swallowed — the
/// memory remains searchable via vector / BM25 even if the graph is
/// incomplete. Caps at 8 entities / 12 edges per memory; the LLM prompt
/// enforces the same limits but a malicious/bugged model could still
/// over-produce, so we truncate defensively.
async fn extract_and_store(
    client: &Ollama,
    store: &Store,
    memory_id: &str,
    title: &str,
    body: &str,
) {
    const MAX_ENTITIES: usize = 8;
    const MAX_EDGES: usize = 12;
    let (entities, edges) = match client.extract_entities(title, body).await {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!("extract_entities failed for {memory_id}: {e}");
            return;
        }
    };
    for e in entities.into_iter().take(MAX_ENTITIES) {
        // The entity's first-seen "summary" is the note's title — gives
        // it a human-readable label without another LLM call. Repeated
        // upserts of the same (name, type) keep the latest summary.
        let id = match store.upsert_entity(&e.name, &e.etype, title).await {
            Ok(id) => id,
            Err(err) => {
                tracing::warn!("upsert_entity({}) failed: {err}", e.name);
                continue;
            }
        };
        if let Err(err) = store.link_entity_memory(&id, memory_id).await {
            tracing::warn!("link_entity_memory failed: {err}");
        }
    }
    for edge in edges.into_iter().take(MAX_EDGES) {
        if let Err(err) = store
            .upsert_edge(
                &edge.src,
                &edge.src_type,
                &edge.dst,
                &edge.dst_type,
                &edge.relation,
            )
            .await
        {
            tracing::warn!("upsert_edge failed: {err}");
        }
    }
    tracing::debug!("extracted entities + edges for memory {memory_id}");
}

/// Smart chunk splitting at heading / code-fence / paragraph boundaries.
/// For a single-memory text we just truncate politely — if the content
/// is huge, take the first ~2000 chars up to a clean boundary.
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
