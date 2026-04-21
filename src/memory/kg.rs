//! Knowledge-graph helpers — kg_entities / kg_edges tables with a
//! provenance link back to the memory rows entities were extracted from.
//! Intentionally small for v1; the agent can insert entities via tool
//! calls and we surface them in search fallbacks.

use crate::memory::Store;
use crate::util;
use anyhow::Result;
use rusqlite::params;
use std::collections::HashMap;

/// In-memory snapshot of the knowledge graph used by graph-based
/// retrieval. Loaded per query — at current scales (≤10k entities) the
/// load cost is sub-millisecond and keeping it immutable avoids any
/// staleness when the embedding worker extracts entities concurrently.
#[derive(Debug, Default)]
pub struct KgSnapshot {
    /// entity_id -> (lowercased name, type)
    pub entities: HashMap<String, (String, String)>,
    /// entity_id -> outgoing neighbours and edge weights
    pub outgoing: HashMap<String, Vec<(String, f32)>>,
    /// entity_id -> incoming neighbours (edges are treated as
    /// bidirectional during PPR so we walk in both directions)
    pub incoming: HashMap<String, Vec<(String, f32)>>,
    /// entity_id -> memory_ids it was extracted from
    pub entity_to_memories: HashMap<String, Vec<String>>,
}

impl Store {
    pub async fn upsert_entity(&self, name: &str, etype: &str, summary: &str) -> Result<String> {
        let inner = self.inner.clone();
        let id = util::new_uuid();
        let now = util::now_ts();
        let name = name.to_string();
        let etype = etype.to_string();
        let summary = summary.to_string();
        let id_out = id.clone();
        tokio::task::spawn_blocking(move || -> Result<String> {
            let conn = inner.lock().unwrap();
            let mut stmt =
                conn.prepare("SELECT id FROM kg_entities WHERE name = ?1 AND type = ?2")?;
            let existing: Option<String> = stmt.query_row(params![name, etype], |r| r.get(0)).ok();
            if let Some(e) = existing {
                conn.execute(
                    "UPDATE kg_entities SET summary = ?1, updated_at = ?2 WHERE id = ?3",
                    params![summary, now, e],
                )?;
                return Ok(e);
            }
            conn.execute(
                "INSERT INTO kg_entities(id, name, type, summary, weight, created_at, updated_at)
                 VALUES (?1, ?2, ?3, ?4, 1.0, ?5, ?5)",
                params![id_out, name, etype, summary, now],
            )?;
            Ok(id_out)
        })
        .await?
    }

    /// Record that `entity_id` was extracted from `memory_id`. This is
    /// the provenance link graph-seeded recall uses to walk from
    /// query-matched entities back to their source memories.
    /// Idempotent — primary key is (entity_id, memory_id).
    pub async fn link_entity_memory(&self, entity_id: &str, memory_id: &str) -> Result<()> {
        let inner = self.inner.clone();
        let entity_id = entity_id.to_string();
        let memory_id = memory_id.to_string();
        tokio::task::spawn_blocking(move || -> Result<()> {
            let conn = inner.lock().unwrap();
            conn.execute(
                "INSERT OR IGNORE INTO kg_entity_memories(entity_id, memory_id) VALUES (?1, ?2)",
                params![entity_id, memory_id],
            )?;
            Ok(())
        })
        .await??;
        Ok(())
    }

    /// Load the entire entity + edge + provenance graph into memory.
    /// Cheap at this scale (three small queries). Used by graph-based
    /// retrieval each time a search runs — fresh loads mean we never
    /// read stale data while the embedding worker is writing.
    pub async fn load_kg_snapshot(&self) -> Result<KgSnapshot> {
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || -> Result<KgSnapshot> {
            let conn = inner.lock().unwrap();
            let mut snap = KgSnapshot::default();

            // Entities.
            {
                let mut stmt = conn.prepare("SELECT id, name, type FROM kg_entities")?;
                let rows = stmt.query_map([], |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, String>(2)?,
                    ))
                })?;
                for row in rows {
                    let (id, name, etype) = row?;
                    snap.entities.insert(id, (name.to_lowercase(), etype));
                }
            }

            // Edges — store both directions since PPR treats the graph as
            // bidirectional. Weight defaults to 1.0 when the row's weight
            // is null/zero (a guard against bad upserts).
            {
                let mut stmt =
                    conn.prepare("SELECT src_id, dst_id, weight FROM kg_edges")?;
                let rows = stmt.query_map([], |r| {
                    let w: f64 = r.get(2).unwrap_or(1.0);
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, String>(1)?,
                        if w <= 0.0 { 1.0 } else { w as f32 },
                    ))
                })?;
                for row in rows {
                    let (src, dst, w) = row?;
                    snap.outgoing.entry(src.clone()).or_default().push((dst.clone(), w));
                    snap.incoming.entry(dst).or_default().push((src, w));
                }
            }

            // Entity → memory provenance. An entity may span multiple
            // memories (same name + type was upserted across notes).
            {
                let mut stmt = conn.prepare(
                    "SELECT entity_id, memory_id FROM kg_entity_memories",
                )?;
                let rows = stmt.query_map([], |r| {
                    Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
                })?;
                for row in rows {
                    let (eid, mid) = row?;
                    snap.entity_to_memories.entry(eid).or_default().push(mid);
                }
            }

            Ok(snap)
        })
        .await?
    }

    pub async fn upsert_edge(
        &self,
        src_name: &str,
        src_type: &str,
        dst_name: &str,
        dst_type: &str,
        relation: &str,
    ) -> Result<()> {
        let src = self.upsert_entity(src_name, src_type, "").await?;
        let dst = self.upsert_entity(dst_name, dst_type, "").await?;
        let inner = self.inner.clone();
        let relation = relation.to_string();
        let now = util::now_ts();
        tokio::task::spawn_blocking(move || -> Result<()> {
            let conn = inner.lock().unwrap();
            conn.execute(
                "INSERT OR IGNORE INTO kg_edges(src_id, dst_id, relation, weight, created_at, updated_at)
                 VALUES (?1, ?2, ?3, 1.0, ?4, ?4)",
                params![src, dst, relation, now],
            )?;
            Ok(())
        }).await??;
        Ok(())
    }
}
