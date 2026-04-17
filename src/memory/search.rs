//! Hybrid search — broodlink's pattern. BM25 + vector, fused by max score,
//! with temporal decay. Query expansion via the LLM (N paraphrasings).

use crate::config::Config;
use crate::llm::ollama::OllamaClient as Ollama;
use crate::memory::store::{Store, StoredMemory};
use crate::util;
use anyhow::Result;
use std::collections::HashMap;

pub struct Hit {
    pub memory: StoredMemory,
    pub score: f32,
    pub bm25: Option<f32>,
    pub vector: Option<f32>,
}

pub async fn hybrid_search(
    store: &Store,
    ollama: &Ollama,
    cfg: &Config,
    query: &str,
    top_k: usize,
) -> Result<Vec<Hit>> {
    let k = top_k.max(cfg.memory.top_k);
    let fetch = k * 3; // overfetch so fusion has room to reorder

    // 1. Query expansion.
    let mut variants = vec![query.to_string()];
    if cfg.memory.expansion_variants > 0 {
        if let Ok(extra) = ollama
            .query_expand(query, cfg.memory.expansion_variants)
            .await
        {
            variants.extend(extra);
        }
    }

    // 2. Run BM25 + vector search over each variant, merge by max.
    let mut by_id: HashMap<String, Hit> = HashMap::new();
    let now = util::now_ts();
    let half_life = cfg.memory.temporal_half_life_days.max(1.0);

    for v in &variants {
        // BM25 — always runs; cheap, sub-millisecond.
        if let Ok(rows) = store.bm25_search(v, fetch).await {
            let (min, max) = bm25_range(&rows);
            for (mem, raw) in rows {
                let norm = normalise_bm25(raw, min, max);
                merge(&mut by_id, mem, None, Some(norm), cfg, now, half_life);
            }
        }
        // Vector — gated by [memory].vector_search so users on slow Ollama
        // hardware can opt out and get pure-BM25 recall.
        if cfg.memory.vector_search {
            if let Ok(emb) = ollama.embed(v).await {
                if let Ok(rows) = store.vector_search(&emb, fetch).await {
                    for (mem, dist) in rows {
                        // sqlite-vec returns L2 distance; convert to a similarity in [0,1].
                        let sim = 1.0 / (1.0 + dist.max(0.0));
                        merge(&mut by_id, mem, Some(sim), None, cfg, now, half_life);
                    }
                }
            }
        }
    }

    let mut hits: Vec<Hit> = by_id.into_values().collect();
    hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    hits.truncate(k);
    Ok(hits)
}

fn merge(
    map: &mut HashMap<String, Hit>,
    mem: StoredMemory,
    vector: Option<f32>,
    bm25: Option<f32>,
    cfg: &Config,
    now: i64,
    half_life: f32,
) {
    let age_days = ((now - mem.updated_at).max(0) as f32) / 86400.0;
    let decay = util::temporal_decay(age_days, half_life);
    let imp = mem.importance.clamp(0.0, 1.0);
    let base = bm25.unwrap_or(0.0) * cfg.memory.bm25_weight
        + vector.unwrap_or(0.0) * cfg.memory.vector_weight;
    // Importance gives a small lift; decay fades older memories.
    let score = base * (0.7 + 0.3 * imp) * decay;

    let id = mem.id.clone();
    match map.get_mut(&id) {
        Some(existing) => {
            existing.score = existing.score.max(score);
            if let Some(b) = bm25 {
                existing.bm25 = Some(existing.bm25.unwrap_or(0.0).max(b));
            }
            if let Some(v) = vector {
                existing.vector = Some(existing.vector.unwrap_or(0.0).max(v));
            }
        }
        None => {
            map.insert(
                id,
                Hit {
                    memory: mem,
                    score,
                    bm25,
                    vector,
                },
            );
        }
    }
}

fn bm25_range(rows: &[(StoredMemory, f32)]) -> (f32, f32) {
    if rows.is_empty() {
        return (0.0, 1.0);
    }
    let min = rows.iter().map(|(_, s)| *s).fold(f32::INFINITY, f32::min);
    let max = rows
        .iter()
        .map(|(_, s)| *s)
        .fold(f32::NEG_INFINITY, f32::max);
    (min, max)
}

/// SQLite's bm25() returns lower = better (and usually negative). Turn that
/// into a normalised 0..1 where 1 is best.
fn normalise_bm25(raw: f32, min: f32, max: f32) -> f32 {
    if (max - min).abs() < 1e-6 {
        return 1.0;
    }
    let inverted = (max - raw) / (max - min); // min -> 1, max -> 0
    inverted.clamp(0.0, 1.0)
}
