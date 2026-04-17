//! Knowledge-graph helpers — broodlink's kg_entities / kg_edges pattern.
//! Intentionally small for v1; the agent can insert entities via tool calls
//! and we surface them in search fallbacks.

use crate::memory::Store;
use crate::util;
use anyhow::Result;
use rusqlite::params;

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
