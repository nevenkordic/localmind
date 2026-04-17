//! The Store: owns the SQLite connection pool and exposes high-level
//! operations for inserting, retrieving, and maintaining memories.
//!
//! SQLite is used in WAL mode so the background embedding worker can write
//! vectors concurrently with the agent reading the FTS index.

use crate::config::Config;
use crate::util;
use anyhow::{Context, Result};
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// Input to `insert_memory`.
pub struct NewMemory {
    pub kind: String,
    pub title: String,
    pub content: String,
    pub source: String,
    pub tags: Vec<String>,
    pub importance: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredMemory {
    pub id: String,
    pub kind: String,
    pub title: String,
    pub content: String,
    pub source: String,
    pub tags: Vec<String>,
    pub importance: f32,
    pub created_at: i64,
    pub updated_at: i64,
    pub accessed_at: i64,
    pub access_count: i64,
}

#[derive(Debug, Default)]
pub struct Stats {
    pub memory_count: i64,
    pub vector_count: i64,
    pub entity_count: i64,
    pub edge_count: i64,
    pub outbox_pending: i64,
    pub db_size: i64,
}

#[derive(Debug, Default)]
pub struct OutboxHealth {
    pub pending: i64,
    pub running: i64,
    pub done: i64,
    pub failed: i64,
    /// Most recent `last_error` from a failed embedding attempt, if any.
    pub last_error: Option<String>,
}

/// Thread-safe, clonable handle to the on-disk database.
/// Writes are serialised through a Mutex<Connection>; reads are too for
/// simplicity. A single local user hits nowhere near the throughput where
/// this matters.
#[derive(Clone)]
pub struct Store {
    // Visible to sibling modules (kg.rs) but not outside the crate.
    pub(crate) inner: Arc<Mutex<Connection>>,
    pub db_path: PathBuf,
}

impl Store {
    pub async fn open(cfg: &Config) -> Result<Self> {
        let path = cfg.memory.db_path_resolved();
        let path_clone = path.clone();
        let conn = tokio::task::spawn_blocking(move || Connection::open(&path_clone)).await??;
        Ok(Self {
            inner: Arc::new(Mutex::new(conn)),
            db_path: path,
        })
    }

    /// Run embedded migrations. Idempotent — safe to call on every startup.
    pub async fn migrate(&self) -> Result<()> {
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || -> Result<()> {
            let conn = inner.lock().unwrap();
            conn.execute_batch(include_str!("../../migrations/001_init.sql"))
                .context("applying 001_init.sql")?;
            conn.execute_batch(include_str!("../../migrations/002_vector_index.sql"))
                .context("applying 002_vector_index.sql")?;
            Ok(())
        })
        .await??;
        Ok(())
    }

    pub async fn insert_memory(&self, new: &NewMemory) -> Result<String> {
        let id = util::new_uuid();
        let now = util::now_ts();
        let hash = util::sha256_hex(&new.content);
        let tags_json = serde_json::to_string(&new.tags).unwrap_or_else(|_| "[]".into());
        let inner = self.inner.clone();
        let id_out = id.clone();
        let kind = new.kind.clone();
        let title = new.title.clone();
        let content = new.content.clone();
        let source = new.source.clone();
        let importance = new.importance.clamp(0.0, 1.0);

        tokio::task::spawn_blocking(move || -> Result<()> {
            let conn = inner.lock().unwrap();
            // INSERT OR IGNORE on content_hash dedup — if the exact same
            // content was stored before, we just update accessed_at instead.
            let changed = conn.execute(
                r#"INSERT OR IGNORE INTO memories
                   (id, kind, title, content, source, tags_json, importance,
                    created_at, updated_at, accessed_at, access_count, content_hash)
                   VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?8, ?8, 0, ?9)"#,
                params![id_out, kind, title, content, source, tags_json, importance, now, hash],
            )?;
            if changed == 0 {
                // Duplicate content — update access stats on the existing row.
                conn.execute(
                    r#"UPDATE memories
                       SET accessed_at = ?1, access_count = access_count + 1
                       WHERE content_hash = ?2"#,
                    params![now, hash],
                )?;
            } else {
                // Enqueue for embedding worker.
                conn.execute(
                    r#"INSERT INTO embedding_outbox (memory_id, status, enqueued_at, updated_at)
                       VALUES (?1, 'pending', ?2, ?2)"#,
                    params![id_out, now],
                )?;
            }
            Ok(())
        })
        .await??;
        Ok(id)
    }

    #[allow(dead_code)]
    pub async fn get_memory(&self, id: &str) -> Result<Option<StoredMemory>> {
        let inner = self.inner.clone();
        let id = id.to_string();
        tokio::task::spawn_blocking(move || -> Result<Option<StoredMemory>> {
            let conn = inner.lock().unwrap();
            let mut stmt = conn.prepare(
                "SELECT id, kind, title, content, source, tags_json, importance,
                        created_at, updated_at, accessed_at, access_count
                 FROM memories WHERE id = ?1",
            )?;
            let mut rows = stmt.query(params![id])?;
            if let Some(row) = rows.next()? {
                let tags_json: String = row.get(5)?;
                Ok(Some(StoredMemory {
                    id: row.get(0)?,
                    kind: row.get(1)?,
                    title: row.get(2)?,
                    content: row.get(3)?,
                    source: row.get(4)?,
                    tags: serde_json::from_str(&tags_json).unwrap_or_default(),
                    importance: row.get(6)?,
                    created_at: row.get(7)?,
                    updated_at: row.get(8)?,
                    accessed_at: row.get(9)?,
                    access_count: row.get(10)?,
                }))
            } else {
                Ok(None)
            }
        })
        .await?
    }

    pub async fn delete_memory(&self, id: &str) -> Result<()> {
        let inner = self.inner.clone();
        let id = id.to_string();
        tokio::task::spawn_blocking(move || -> Result<()> {
            let conn = inner.lock().unwrap();
            // Cascade deletes handle memory_vectors, outbox, kg links.
            conn.execute("DELETE FROM memories WHERE id = ?1", params![id])?;
            // vec0 virtual table needs explicit delete (no FK).
            conn.execute("DELETE FROM memory_vec WHERE memory_id = ?1", params![id])?;
            Ok(())
        })
        .await??;
        Ok(())
    }

    /// List all memories of a given kind, ordered by importance then recency.
    /// Used by the `/skills` REPL command and similar audit views.
    pub async fn list_by_kind(&self, kind: &str, limit: usize) -> Result<Vec<StoredMemory>> {
        let inner = self.inner.clone();
        let kind = kind.to_string();
        tokio::task::spawn_blocking(move || -> Result<Vec<StoredMemory>> {
            let conn = inner.lock().unwrap();
            let mut stmt = conn.prepare(
                r#"SELECT id, kind, title, content, source, tags_json, importance,
                          created_at, updated_at, accessed_at, access_count
                   FROM memories WHERE kind = ?1
                   ORDER BY importance DESC, updated_at DESC
                   LIMIT ?2"#,
            )?;
            let rows = stmt.query_map(params![kind, limit as i64], |row| {
                let tags_json: String = row.get(5)?;
                Ok(StoredMemory {
                    id: row.get(0)?,
                    kind: row.get(1)?,
                    title: row.get(2)?,
                    content: row.get(3)?,
                    source: row.get(4)?,
                    tags: serde_json::from_str(&tags_json).unwrap_or_default(),
                    importance: row.get(6)?,
                    created_at: row.get(7)?,
                    updated_at: row.get(8)?,
                    accessed_at: row.get(9)?,
                    access_count: row.get(10)?,
                })
            })?;
            let mut out = Vec::new();
            for r in rows {
                out.push(r?);
            }
            Ok(out)
        })
        .await?
    }

    /// Find memories whose id starts with the given prefix. Returns up to
    /// `limit` matches so the caller can detect ambiguity.
    pub async fn find_by_id_prefix(&self, prefix: &str, limit: usize) -> Result<Vec<StoredMemory>> {
        let inner = self.inner.clone();
        let pattern = format!("{prefix}%");
        tokio::task::spawn_blocking(move || -> Result<Vec<StoredMemory>> {
            let conn = inner.lock().unwrap();
            let mut stmt = conn.prepare(
                r#"SELECT id, kind, title, content, source, tags_json, importance,
                          created_at, updated_at, accessed_at, access_count
                   FROM memories WHERE id LIKE ?1
                   ORDER BY updated_at DESC
                   LIMIT ?2"#,
            )?;
            let rows = stmt.query_map(params![pattern, limit as i64], |row| {
                let tags_json: String = row.get(5)?;
                Ok(StoredMemory {
                    id: row.get(0)?,
                    kind: row.get(1)?,
                    title: row.get(2)?,
                    content: row.get(3)?,
                    source: row.get(4)?,
                    tags: serde_json::from_str(&tags_json).unwrap_or_default(),
                    importance: row.get(6)?,
                    created_at: row.get(7)?,
                    updated_at: row.get(8)?,
                    accessed_at: row.get(9)?,
                    access_count: row.get(10)?,
                })
            })?;
            let mut out = Vec::new();
            for r in rows {
                out.push(r?);
            }
            Ok(out)
        })
        .await?
    }

    pub async fn bm25_search(&self, query: &str, limit: usize) -> Result<Vec<(StoredMemory, f32)>> {
        let inner = self.inner.clone();
        let q = sanitize_fts_query(query);
        tokio::task::spawn_blocking(move || -> Result<Vec<(StoredMemory, f32)>> {
            let conn = inner.lock().unwrap();
            let mut stmt = conn.prepare(
                r#"SELECT m.id, m.kind, m.title, m.content, m.source, m.tags_json,
                          m.importance, m.created_at, m.updated_at, m.accessed_at,
                          m.access_count, bm25(memories_fts) AS rank
                   FROM memories_fts JOIN memories m ON m.rowid = memories_fts.rowid
                   WHERE memories_fts MATCH ?1
                   ORDER BY rank
                   LIMIT ?2"#,
            )?;
            let rows = stmt.query_map(params![q, limit as i64], |row| {
                let tags_json: String = row.get(5)?;
                let rank: f64 = row.get(11)?;
                // bm25() returns a negative-ish score; smaller = better. Invert.
                Ok((
                    StoredMemory {
                        id: row.get(0)?,
                        kind: row.get(1)?,
                        title: row.get(2)?,
                        content: row.get(3)?,
                        source: row.get(4)?,
                        tags: serde_json::from_str(&tags_json).unwrap_or_default(),
                        importance: row.get(6)?,
                        created_at: row.get(7)?,
                        updated_at: row.get(8)?,
                        accessed_at: row.get(9)?,
                        access_count: row.get(10)?,
                    },
                    rank as f32,
                ))
            })?;
            let mut out = Vec::new();
            for r in rows {
                out.push(r?);
            }
            Ok(out)
        })
        .await?
    }

    /// Nearest-neighbour search via sqlite-vec's vec0 virtual table.
    pub async fn vector_search(
        &self,
        embedding: &[f32],
        limit: usize,
    ) -> Result<Vec<(StoredMemory, f32)>> {
        let inner = self.inner.clone();
        let blob = util::f32_to_blob(embedding);
        tokio::task::spawn_blocking(move || -> Result<Vec<(StoredMemory, f32)>> {
            let conn = inner.lock().unwrap();
            let mut stmt = conn.prepare(
                r#"SELECT m.id, m.kind, m.title, m.content, m.source, m.tags_json,
                          m.importance, m.created_at, m.updated_at, m.accessed_at,
                          m.access_count, v.distance
                   FROM memory_vec v JOIN memories m ON m.id = v.memory_id
                   WHERE v.embedding MATCH ?1 AND k = ?2
                   ORDER BY v.distance"#,
            )?;
            let rows = stmt.query_map(params![blob, limit as i64], |row| {
                let tags_json: String = row.get(5)?;
                let dist: f64 = row.get(11)?;
                Ok((
                    StoredMemory {
                        id: row.get(0)?,
                        kind: row.get(1)?,
                        title: row.get(2)?,
                        content: row.get(3)?,
                        source: row.get(4)?,
                        tags: serde_json::from_str(&tags_json).unwrap_or_default(),
                        importance: row.get(6)?,
                        created_at: row.get(7)?,
                        updated_at: row.get(8)?,
                        accessed_at: row.get(9)?,
                        access_count: row.get(10)?,
                    },
                    dist as f32,
                ))
            })?;
            let mut out = Vec::new();
            for r in rows {
                out.push(r?);
            }
            Ok(out)
        })
        .await?
    }

    pub async fn stats(&self) -> Result<Stats> {
        let inner = self.inner.clone();
        let path = self.db_path.clone();
        tokio::task::spawn_blocking(move || -> Result<Stats> {
            let conn = inner.lock().unwrap();
            let mut s = Stats::default();
            s.memory_count = conn.query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))?;
            s.vector_count =
                conn.query_row("SELECT COUNT(*) FROM memory_vectors", [], |r| r.get(0))?;
            s.entity_count =
                conn.query_row("SELECT COUNT(*) FROM kg_entities", [], |r| r.get(0))?;
            s.edge_count = conn.query_row("SELECT COUNT(*) FROM kg_edges", [], |r| r.get(0))?;
            s.outbox_pending = conn.query_row(
                "SELECT COUNT(*) FROM embedding_outbox WHERE status = 'pending'",
                [],
                |r| r.get(0),
            )?;
            s.db_size = std::fs::metadata(&path)
                .map(|m| m.len() as i64)
                .unwrap_or(0);
            Ok(s)
        })
        .await?
    }

    /// Per-status outbox counts plus the most recent embedding error. Used
    /// by `llm health` and `/health` to distinguish "embedder is busy" from
    /// "embedder is broken".
    pub async fn outbox_health(&self) -> Result<OutboxHealth> {
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || -> Result<OutboxHealth> {
            let conn = inner.lock().unwrap();
            let count = |status: &str| -> rusqlite::Result<i64> {
                conn.query_row(
                    "SELECT COUNT(*) FROM embedding_outbox WHERE status = ?1",
                    params![status],
                    |r| r.get(0),
                )
            };
            let last_error: Option<String> = conn
                .query_row(
                    r#"SELECT last_error FROM embedding_outbox
                       WHERE status = 'failed' AND last_error IS NOT NULL
                       ORDER BY updated_at DESC LIMIT 1"#,
                    [],
                    |r| r.get(0),
                )
                .ok();
            Ok(OutboxHealth {
                pending: count("pending")?,
                running: count("running")?,
                done: count("done")?,
                failed: count("failed")?,
                last_error,
            })
        })
        .await?
    }

    /// Rebuild the vec0 ANN index from the portable memory_vectors blob table.
    /// Useful after a backup/restore or a dimension change.
    pub async fn reindex_vectors(&self) -> Result<usize> {
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || -> Result<usize> {
            let conn = inner.lock().unwrap();
            conn.execute("DELETE FROM memory_vec", [])?;
            let mut stmt = conn.prepare("SELECT memory_id, embedding FROM memory_vectors")?;
            let mut rows = stmt.query([])?;
            let mut n = 0usize;
            while let Some(r) = rows.next()? {
                let id: String = r.get(0)?;
                let blob: Vec<u8> = r.get(1)?;
                conn.execute(
                    "INSERT INTO memory_vec(memory_id, embedding) VALUES (?1, ?2)",
                    params![id, blob],
                )?;
                n += 1;
            }
            Ok(n)
        })
        .await?
    }

    /// Used by the embedding worker to persist a computed embedding.
    pub async fn upsert_embedding(
        &self,
        memory_id: &str,
        embedding: &[f32],
        model: &str,
    ) -> Result<()> {
        let inner = self.inner.clone();
        let id = memory_id.to_string();
        let model = model.to_string();
        let blob = util::f32_to_blob(embedding);
        let dim = embedding.len() as i64;
        let norm = util::l2_norm(embedding) as f64;
        let now = util::now_ts();
        tokio::task::spawn_blocking(move || -> Result<()> {
            let conn = inner.lock().unwrap();
            conn.execute(
                r#"INSERT INTO memory_vectors(memory_id, dim, model, embedding, norm, created_at)
                   VALUES (?1, ?2, ?3, ?4, ?5, ?6)
                   ON CONFLICT(memory_id) DO UPDATE SET
                     dim=excluded.dim, model=excluded.model,
                     embedding=excluded.embedding, norm=excluded.norm,
                     created_at=excluded.created_at"#,
                params![id, dim, model, blob, norm, now],
            )?;
            conn.execute(
                "INSERT OR REPLACE INTO memory_vec(memory_id, embedding) VALUES (?1, ?2)",
                params![id, blob],
            )?;
            Ok(())
        })
        .await??;
        Ok(())
    }

    /// Claim the next pending outbox row for embedding. Returns (outbox_id, memory_id, content).
    pub async fn claim_next_embedding_job(&self) -> Result<Option<(i64, String, String)>> {
        let inner = self.inner.clone();
        let now = util::now_ts();
        tokio::task::spawn_blocking(move || -> Result<Option<(i64, String, String)>> {
            let conn = inner.lock().unwrap();
            let tx = conn.unchecked_transaction()?;
            let row = {
                let mut stmt = tx.prepare(
                    r#"SELECT o.id, o.memory_id, m.title || '\n' || m.content
                       FROM embedding_outbox o JOIN memories m ON m.id = o.memory_id
                       WHERE o.status = 'pending'
                       ORDER BY o.enqueued_at
                       LIMIT 1"#,
                )?;
                let mut rows = stmt.query([])?;
                if let Some(r) = rows.next()? {
                    let id: i64 = r.get(0)?;
                    let mid: String = r.get(1)?;
                    let text: String = r.get(2)?;
                    Some((id, mid, text))
                } else {
                    None
                }
            };
            if let Some((oid, _, _)) = &row {
                tx.execute(
                    "UPDATE embedding_outbox SET status='running', updated_at=?1, attempts = attempts + 1 WHERE id = ?2",
                    params![now, oid],
                )?;
                tx.commit()?;
            }
            Ok(row)
        }).await?
    }

    pub async fn mark_embedding_done(&self, outbox_id: i64) -> Result<()> {
        let inner = self.inner.clone();
        let now = util::now_ts();
        tokio::task::spawn_blocking(move || -> Result<()> {
            let conn = inner.lock().unwrap();
            conn.execute(
                "UPDATE embedding_outbox SET status='done', updated_at=?1 WHERE id=?2",
                params![now, outbox_id],
            )?;
            Ok(())
        })
        .await??;
        Ok(())
    }

    pub async fn mark_embedding_failed(&self, outbox_id: i64, err: &str) -> Result<()> {
        let inner = self.inner.clone();
        let now = util::now_ts();
        let err = err.to_string();
        tokio::task::spawn_blocking(move || -> Result<()> {
            let conn = inner.lock().unwrap();
            conn.execute(
                "UPDATE embedding_outbox SET status='failed', last_error=?1, updated_at=?2 WHERE id=?3",
                params![err, now, outbox_id],
            )?;
            Ok(())
        }).await??;
        Ok(())
    }
}

/// FTS5 tokenizer accepts a narrower syntax than full English. Strip anything
/// risky and wrap quoted-phrase fallback for robustness.
fn sanitize_fts_query(q: &str) -> String {
    let cleaned: String = q
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace() || matches!(*c, '_' | '-' | '.'))
        .collect();
    let trimmed = cleaned.trim();
    if trimmed.is_empty() {
        return "\"\"".into();
    }
    // Use OR between tokens to keep recall high; dedupe via set-like approach.
    let tokens: Vec<&str> = trimmed.split_whitespace().collect();
    tokens.join(" OR ")
}
