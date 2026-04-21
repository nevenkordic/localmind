//! Hybrid search — BM25 + vector fused via Reciprocal Rank Fusion (RRF),
//! with temporal decay, optional query expansion, an optional graph
//! retriever over the knowledge graph, and an optional LLM-as-judge
//! reranker pass.
//!
//! RRF notes. Prior iterations of this file did score-based CombMAX
//! fusion weighted by `bm25_weight` / `vector_weight`. The problem: BM25
//! scores and cosine similarities live in unrelated units; any weight
//! is a guess that drifts as the embedding model changes. RRF sidesteps
//! it entirely — each retriever contributes `1 / (k + rank_in_that_list)`,
//! so only the ordering matters. k = 60 is the canonical IR-literature
//! constant. The old weight keys are still accepted in config for
//! backwards compat but are ignored here.

use crate::config::Config;
use crate::llm::ollama::OllamaClient as Ollama;
use crate::llm::types::ChatMessage;
use crate::memory::kg::KgSnapshot;
use crate::memory::store::{Store, StoredMemory};
use crate::util;
use anyhow::Result;
use std::collections::HashMap;

/// Reciprocal Rank Fusion constant. 60 is the canonical value used by
/// every major IR library. Lower k gives more weight to top ranks;
/// higher k flattens. 60 is the robust default — no reason to deviate
/// without benchmarking.
const RRF_K: f32 = 60.0;

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

    // When a reranker is configured we pull a larger candidate pool so the
    // reranker has material to reorder. Otherwise stick with the 3×
    // overfetch that leaves RRF room to work but keeps latency tight.
    let fetch = if cfg.memory.rerank_model.is_empty() {
        k * 3
    } else {
        cfg.memory.rerank_fetch_k.max(k)
    };

    // 0. Graph retrieval pre-pass. Runs concurrently with the first
    //    variant's retrievers below; the result joins the RRF fusion as
    //    a third retriever. Cheap when disabled (returns empty).
    let graph_hits: Vec<(String, f32)> = if cfg.memory.graph_search {
        match graph_search(store, query, fetch).await {
            Ok(v) => v,
            Err(_) => Vec::new(),
        }
    } else {
        Vec::new()
    };

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

    // 2. Per variant: run BM25 and vector concurrently, record rank
    //    positions, then fuse via RRF. Raw normalised scores are kept on
    //    the Hit for diagnostics only.
    let mut by_id: HashMap<String, Accum> = HashMap::new();

    for v in &variants {
        let bm25_fut = async {
            let rows = store.bm25_search(v, fetch).await.unwrap_or_default();
            let (min, max) = bm25_range(&rows);
            rows.into_iter()
                .map(|(mem, raw)| (mem, normalise_bm25(raw, min, max)))
                .collect::<Vec<_>>()
        };
        let vec_fut = async {
            if !cfg.memory.vector_search {
                return Vec::new();
            }
            let Ok(emb) = ollama.embed(v).await else {
                return Vec::new();
            };
            store
                .vector_search(&emb, fetch)
                .await
                .unwrap_or_default()
                .into_iter()
                .map(|(mem, dist)| (mem, 1.0 / (1.0 + dist.max(0.0))))
                .collect::<Vec<_>>()
        };
        let (bm25_hits, vec_hits) = tokio::join!(bm25_fut, vec_fut);
        for (rank, (mem, norm)) in bm25_hits.into_iter().enumerate() {
            accumulate(&mut by_id, mem, rank, RetrieverKind::Bm25, Some(norm));
        }
        for (rank, (mem, sim)) in vec_hits.into_iter().enumerate() {
            accumulate(&mut by_id, mem, rank, RetrieverKind::Vector, Some(sim));
        }
    }

    // 2b. Fold graph hits into the same RRF accumulator. These are
    //     already ranked by PPR-aggregated memory score; rank is all
    //     fusion needs. `raw_score` stored on the Accum is just for
    //     display — the PPR number itself isn't comparable to bm25/cos.
    if !graph_hits.is_empty() {
        let ids: Vec<String> = graph_hits.iter().map(|(id, _)| id.clone()).collect();
        if let Ok(memories) = store.get_memories_by_ids(&ids).await {
            let mem_by_id: HashMap<String, StoredMemory> =
                memories.into_iter().map(|m| (m.id.clone(), m)).collect();
            for (rank, (id, _score)) in graph_hits.iter().enumerate() {
                if let Some(mem) = mem_by_id.get(id).cloned() {
                    accumulate(&mut by_id, mem, rank, RetrieverKind::Graph, None);
                }
            }
        }
    }

    // 3. Convert accumulators to final Hits. Apply temporal decay and
    //    importance multipliers on top of the RRF score.
    let now = util::now_ts();
    let half_life = cfg.memory.temporal_half_life_days.max(1.0);
    let mut hits: Vec<Hit> = by_id
        .into_values()
        .map(|a| {
            let age_days = ((now - a.memory.updated_at).max(0) as f32) / 86400.0;
            let decay = util::temporal_decay(age_days, half_life);
            let imp = a.memory.importance.clamp(0.0, 1.0);
            let score = a.rrf * (0.7 + 0.3 * imp) * decay;
            Hit {
                memory: a.memory,
                score,
                bm25: a.bm25,
                vector: a.vector,
            }
        })
        .collect();

    hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    hits.truncate(fetch.min(hits.len()));

    // 4. Optional reranker pass. When `rerank_model` is set, the top
    //    `fetch` candidates get rescored by an LLM-as-judge and the final
    //    top_k comes from that reordering.
    if !cfg.memory.rerank_model.is_empty() && hits.len() > 1 {
        // Reranker works on indices + scores — no clone of the potentially
        // large StoredMemory bodies. On failure (timeout, malformed JSON)
        // we fall through to the RRF ordering.
        if let Ok(scores) = llm_rerank_scores(ollama, &cfg.memory.rerank_model, query, &hits).await
        {
            let mut indexed: Vec<(usize, f32)> =
                scores.into_iter().enumerate().collect();
            indexed.sort_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut reordered: Vec<Hit> = Vec::with_capacity(hits.len());
            let mut taken = hits.into_iter().map(Some).collect::<Vec<_>>();
            for (idx, s) in indexed {
                if let Some(mut h) = taken.get_mut(idx).and_then(|o| o.take()) {
                    h.score = s;
                    reordered.push(h);
                }
            }
            // Any indices the judge missed — append in their original order.
            for slot in taken.into_iter().flatten() {
                reordered.push(slot);
            }
            hits = reordered;
        }
    }

    hits.truncate(k);
    Ok(hits)
}

struct Accum {
    memory: StoredMemory,
    rrf: f32,
    bm25: Option<f32>,
    vector: Option<f32>,
}

#[derive(Copy, Clone)]
enum RetrieverKind {
    Bm25,
    Vector,
    Graph,
}

fn accumulate(
    map: &mut HashMap<String, Accum>,
    mem: StoredMemory,
    rank: usize,
    kind: RetrieverKind,
    raw_score: Option<f32>,
) {
    let contribution = 1.0 / (RRF_K + rank as f32);
    let id = mem.id.clone();
    match map.get_mut(&id) {
        Some(a) => {
            a.rrf += contribution;
            match kind {
                RetrieverKind::Bm25 => {
                    a.bm25 = Some(a.bm25.unwrap_or(0.0).max(raw_score.unwrap_or(0.0)))
                }
                RetrieverKind::Vector => {
                    a.vector = Some(a.vector.unwrap_or(0.0).max(raw_score.unwrap_or(0.0)))
                }
                // Graph retrieval contributes only to the RRF sum — no
                // raw score is recorded on the accumulator because PPR
                // values are not on the same scale as bm25 / cosine and
                // would mislead the diagnostic display.
                RetrieverKind::Graph => {}
            }
        }
        None => {
            let (bm25, vector) = match kind {
                RetrieverKind::Bm25 => (raw_score, None),
                RetrieverKind::Vector => (None, raw_score),
                RetrieverKind::Graph => (None, None),
            };
            map.insert(
                id,
                Accum {
                    memory: mem,
                    rrf: contribution,
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
/// into a normalised 0..1 where 1 is best. Used only for diagnostic
/// display on the Hit — fusion itself runs off ranks, not scores.
fn normalise_bm25(raw: f32, min: f32, max: f32) -> f32 {
    if (max - min).abs() < 1e-6 {
        return 1.0;
    }
    let inverted = (max - raw) / (max - min);
    inverted.clamp(0.0, 1.0)
}

/// LLM-as-judge reranker. Sends the full candidate pool to the configured
/// small model in a single prompt and asks for integer relevance scores
/// keyed by candidate index. One round-trip keeps latency bounded even
/// for 50 candidates — per-item calls would be ~50× slower.
///
/// Returns a vector of scores aligned to the input `hits` slice. Missing
/// indices in the judge's response get a 0. Errors bubble up and the
/// caller falls back to the RRF ordering.
async fn llm_rerank_scores(
    ollama: &Ollama,
    model: &str,
    query: &str,
    hits: &[Hit],
) -> Result<Vec<f32>> {
    if hits.is_empty() {
        return Ok(Vec::new());
    }
    // Keep per-candidate content short — the judge only needs enough to
    // assess relevance, not the full document.
    const CONTENT_BUDGET: usize = 400;
    let mut block = String::new();
    for (i, h) in hits.iter().enumerate() {
        let excerpt: String = h.memory.content.chars().take(CONTENT_BUDGET).collect();
        block.push_str(&format!(
            "[{i}] {title}\n{excerpt}\n\n",
            i = i,
            title = h.memory.title,
            excerpt = excerpt
        ));
    }

    let sys = "You are a relevance judge for a search system. Read the query \
               and the candidate documents, then emit a single JSON object \
               mapping each candidate index to an integer 0-10 relevance \
               score (10 = perfect match, 0 = unrelated). Output ONLY the \
               JSON object, no prose, no code fences.";
    let user = format!(
        "Query: {query}\n\nCandidates:\n{block}\nReturn JSON like {{\"0\": 8, \"1\": 3, ...}}."
    );
    let msgs = vec![ChatMessage::system(sys), ChatMessage::user(user)];
    let reply = ollama.chat_on(&msgs, None, false, Some(model)).await?;
    let raw = reply
        .content
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();
    let scores: HashMap<String, i32> = serde_json::from_str(raw)?;

    Ok((0..hits.len())
        .map(|i| scores.get(&i.to_string()).copied().unwrap_or(0) as f32)
        .collect())
}

// ---------------------------------------------------------------------------
// Graph retrieval — Personalized PageRank over the entity graph
// ---------------------------------------------------------------------------

/// PageRank parameters. 20 iterations is plenty for graphs at this scale
/// to converge within 1e-3; damping 0.85 is the canonical value
/// (probability of continuing the random walk vs. teleporting back to the
/// seeds).
const PPR_ITERS: usize = 20;
const PPR_DAMPING: f32 = 0.85;

/// Graph retrieval. Matches query tokens against existing graph
/// entities, seeds Personalized PageRank on the entity graph from the
/// matches, then aggregates entity scores into per-memory scores via the
/// `kg_entity_memories` provenance table.
///
/// Returns at most `top_n` (memory_id, score) pairs in descending order.
/// Empty result when the graph is empty, no seeds match, or the store
/// read fails — caller always falls back to BM25 + vector.
pub(crate) async fn graph_search(
    store: &Store,
    query: &str,
    top_n: usize,
) -> Result<Vec<(String, f32)>> {
    let snap = store.load_kg_snapshot().await?;
    if snap.entities.is_empty() {
        return Ok(Vec::new());
    }
    let seeds = match_query_to_entities(query, &snap);
    if seeds.is_empty() {
        return Ok(Vec::new());
    }
    let entity_scores = personalized_pagerank(&snap, &seeds, PPR_ITERS, PPR_DAMPING);
    let memory_scores = aggregate_to_memories(&snap, &entity_scores);
    let mut ranked: Vec<(String, f32)> = memory_scores.into_iter().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    ranked.truncate(top_n);
    Ok(ranked)
}

/// Cheap keyword-style seed selection: lowercase the query, then find
/// every entity whose canonical name appears as a substring. Names
/// shorter than 3 characters are skipped (too noisy — a single-letter
/// entity would match almost any query). Works surprisingly well for
/// domain vocab and keeps us off the LLM's critical path.
fn match_query_to_entities(query: &str, snap: &KgSnapshot) -> Vec<String> {
    let q = query.to_lowercase();
    snap.entities
        .iter()
        .filter(|(_, (name_lower, _))| name_lower.len() >= 3 && q.contains(name_lower.as_str()))
        .map(|(id, _)| id.clone())
        .collect()
}

/// Power-iteration Personalized PageRank. Edges are treated as
/// bidirectional because the extracted graph is usually sparse and the
/// direction of `authored` / `contains` is a crude signal — forward /
/// backward walks both carry relatedness. Weights divide the outgoing
/// mass proportionally; isolated nodes just hold their teleport mass.
fn personalized_pagerank(
    snap: &KgSnapshot,
    seeds: &[String],
    iters: usize,
    damping: f32,
) -> HashMap<String, f32> {
    if seeds.is_empty() || snap.entities.is_empty() {
        return HashMap::new();
    }
    let n_seeds = seeds.len() as f32;
    let teleport: HashMap<String, f32> = seeds
        .iter()
        .filter(|s| snap.entities.contains_key(s.as_str()))
        .map(|s| (s.clone(), 1.0 / n_seeds))
        .collect();
    if teleport.is_empty() {
        return HashMap::new();
    }
    let mut scores: HashMap<String, f32> = teleport.clone();
    for _ in 0..iters {
        let mut next: HashMap<String, f32> = HashMap::with_capacity(scores.len());
        // Teleport component — always pulls walkers back to the seeds.
        for (k, v) in &teleport {
            *next.entry(k.clone()).or_insert(0.0) += (1.0 - damping) * v;
        }
        // Walk component — distribute each node's score among its
        // neighbours (outgoing + incoming), proportional to edge weight.
        for (node, score) in &scores {
            if *score <= 0.0 {
                continue;
            }
            let out_w: f32 = snap
                .outgoing
                .get(node)
                .map(|v| v.iter().map(|(_, w)| *w).sum())
                .unwrap_or(0.0);
            let in_w: f32 = snap
                .incoming
                .get(node)
                .map(|v| v.iter().map(|(_, w)| *w).sum())
                .unwrap_or(0.0);
            let total = out_w + in_w;
            if total <= 0.0 {
                // Dangling node (no edges). Its mass evaporates — the
                // teleport term reseeds the walk next iteration so this
                // doesn't leak score.
                continue;
            }
            if let Some(edges) = snap.outgoing.get(node) {
                for (dst, w) in edges {
                    *next.entry(dst.clone()).or_insert(0.0) +=
                        damping * score * w / total;
                }
            }
            if let Some(edges) = snap.incoming.get(node) {
                for (src, w) in edges {
                    *next.entry(src.clone()).or_insert(0.0) +=
                        damping * score * w / total;
                }
            }
        }
        scores = next;
    }
    scores
}

/// Collapse entity-level PPR scores onto the memories they came from.
/// An entity that appears in multiple memories contributes its score to
/// each of them — the "graph cue → documents" fan-out step.
fn aggregate_to_memories(
    snap: &KgSnapshot,
    entity_scores: &HashMap<String, f32>,
) -> HashMap<String, f32> {
    let mut memory_scores: HashMap<String, f32> = HashMap::new();
    for (eid, score) in entity_scores {
        if *score <= 0.0 {
            continue;
        }
        if let Some(memory_ids) = snap.entity_to_memories.get(eid) {
            for mid in memory_ids {
                *memory_scores.entry(mid.clone()).or_insert(0.0) += score;
            }
        }
    }
    memory_scores
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn mem(id: &str, importance: f32, age_secs: i64) -> StoredMemory {
        StoredMemory {
            id: id.into(),
            kind: "note".into(),
            title: id.into(),
            content: id.into(),
            source: "test".into(),
            tags: vec![],
            importance,
            created_at: 0,
            updated_at: util::now_ts() - age_secs,
            accessed_at: 0,
            access_count: 0,
        }
    }

    #[test]
    fn rrf_single_retriever_ranks_by_position() {
        let mut map = HashMap::new();
        for (rank, id) in ["a", "b", "c"].iter().enumerate() {
            accumulate(
                &mut map,
                mem(id, 1.0, 0),
                rank,
                RetrieverKind::Bm25,
                Some(1.0),
            );
        }
        let a = map.get("a").unwrap().rrf;
        let b = map.get("b").unwrap().rrf;
        let c = map.get("c").unwrap().rrf;
        assert!(a > b && b > c, "rank 0 should outrank rank 1 should outrank rank 2");
        assert!((a - 1.0 / (RRF_K + 0.0)).abs() < 1e-6);
    }

    #[test]
    fn rrf_rewards_cross_retriever_overlap() {
        let mut map = HashMap::new();
        // `a` appears at rank 0 in both retrievers.
        accumulate(&mut map, mem("a", 1.0, 0), 0, RetrieverKind::Bm25, Some(1.0));
        accumulate(&mut map, mem("a", 1.0, 0), 0, RetrieverKind::Vector, Some(1.0));
        // `b` appears only in bm25 at rank 0.
        accumulate(&mut map, mem("b", 1.0, 0), 0, RetrieverKind::Bm25, Some(1.0));
        // `c` appears only in vector at rank 0.
        accumulate(&mut map, mem("c", 1.0, 0), 0, RetrieverKind::Vector, Some(1.0));

        let a = map.get("a").unwrap().rrf;
        let b = map.get("b").unwrap().rrf;
        let c = map.get("c").unwrap().rrf;
        assert!(a > b && a > c, "cross-retriever hit should rank above single-retriever hits");
        assert!((b - c).abs() < 1e-6, "single-retriever hits at same rank should tie");
    }

    #[test]
    fn rrf_later_rank_contributes_less() {
        let mut map = HashMap::new();
        accumulate(&mut map, mem("early", 1.0, 0), 0, RetrieverKind::Bm25, Some(1.0));
        accumulate(&mut map, mem("late", 1.0, 0), 49, RetrieverKind::Bm25, Some(1.0));
        let early = map.get("early").unwrap().rrf;
        let late = map.get("late").unwrap().rrf;
        assert!(early > late);
        // Sanity-check the RRF formula: 1/(60+0) vs 1/(60+49).
        let expected_early = 1.0 / (RRF_K + 0.0);
        let expected_late = 1.0 / (RRF_K + 49.0);
        assert!((early - expected_early).abs() < 1e-6);
        assert!((late - expected_late).abs() < 1e-6);
    }

    // ----- graph retrieval / PPR unit tests ---------------------------

    fn small_graph() -> KgSnapshot {
        // 3-node triangle: a-b-c-a, each edge weight 1.
        // Plus one dangling node `d` with no edges.
        let mut snap = KgSnapshot::default();
        for (id, name) in [("a", "alpha"), ("b", "beta"), ("c", "gamma"), ("d", "delta")] {
            snap.entities
                .insert(id.into(), (name.to_string(), "tech".into()));
        }
        let mut add = |s: &str, d: &str| {
            snap.outgoing.entry(s.into()).or_default().push((d.into(), 1.0));
            snap.incoming.entry(d.into()).or_default().push((s.into(), 1.0));
        };
        add("a", "b");
        add("b", "c");
        add("c", "a");
        // Memory provenance: a and b both came from memory m1; c from m2.
        snap.entity_to_memories
            .insert("a".into(), vec!["m1".into()]);
        snap.entity_to_memories
            .insert("b".into(), vec!["m1".into()]);
        snap.entity_to_memories
            .insert("c".into(), vec!["m2".into()]);
        snap
    }

    #[test]
    fn ppr_empty_seeds_returns_empty() {
        let snap = small_graph();
        let scores = personalized_pagerank(&snap, &[], 20, 0.85);
        assert!(scores.is_empty());
    }

    #[test]
    fn ppr_seed_holds_highest_score() {
        let snap = small_graph();
        let scores = personalized_pagerank(&snap, &["a".into()], 30, 0.85);
        let a = scores.get("a").copied().unwrap_or(0.0);
        let b = scores.get("b").copied().unwrap_or(0.0);
        let c = scores.get("c").copied().unwrap_or(0.0);
        assert!(a > b, "seed should score highest: a={a} b={b}");
        assert!(a > c);
        assert!(b > 0.0 && c > 0.0, "neighbours get non-zero mass");
    }

    #[test]
    fn ppr_scores_sum_approximately_to_one() {
        // Classic PageRank sanity: scores converge toward a probability
        // distribution. Iterated enough, the sum stays close to 1.
        let snap = small_graph();
        let scores = personalized_pagerank(&snap, &["a".into()], 30, 0.85);
        let total: f32 = scores.values().sum();
        assert!((total - 1.0).abs() < 0.01, "total = {total}");
    }

    #[test]
    fn aggregate_folds_entities_into_memories() {
        let snap = small_graph();
        let mut es = HashMap::new();
        es.insert("a".to_string(), 0.5);
        es.insert("b".to_string(), 0.3);
        es.insert("c".to_string(), 0.2);
        let ms = aggregate_to_memories(&snap, &es);
        // m1 owns a + b → 0.5 + 0.3 = 0.8
        // m2 owns c → 0.2
        assert!((ms.get("m1").copied().unwrap_or(0.0) - 0.8).abs() < 1e-6);
        assert!((ms.get("m2").copied().unwrap_or(0.0) - 0.2).abs() < 1e-6);
    }

    #[test]
    fn match_query_requires_min_length() {
        let snap = small_graph();
        // 3-char+ names ("alpha", "beta", "gamma") — query contains "alpha".
        let hits = match_query_to_entities("what about alpha again", &snap);
        assert_eq!(hits, vec!["a"]);
    }

    #[test]
    fn match_query_returns_empty_when_nothing_matches() {
        let snap = small_graph();
        let hits = match_query_to_entities("pure unrelated text", &snap);
        assert!(hits.is_empty());
    }

    #[test]
    fn bm25_range_handles_empty_and_degenerate() {
        let empty: Vec<(StoredMemory, f32)> = vec![];
        let (min, max) = bm25_range(&empty);
        assert_eq!(min, 0.0);
        assert_eq!(max, 1.0);

        let single = vec![(mem("a", 1.0, 0), -2.5)];
        let (min, max) = bm25_range(&single);
        assert_eq!(min, -2.5);
        assert_eq!(max, -2.5);
        // Degenerate range → normalise returns 1.0 for everything.
        assert_eq!(normalise_bm25(-2.5, min, max), 1.0);
    }
}
