-- Vector ANN index via sqlite-vec extension.
-- The `memory_vectors` table (see 001_init.sql) remains the source of truth
-- for portability and migration. `memory_vec` is a derived ANN index that can
-- be rebuilt at any time from `memory_vectors`.

CREATE VIRTUAL TABLE IF NOT EXISTS memory_vec USING vec0(
    memory_id    TEXT PRIMARY KEY,
    embedding    float[768]           -- nomic-embed-text dimension
);

-- Entity vectors — broodlink's `broodlink_kg_entities` Qdrant collection.
CREATE VIRTUAL TABLE IF NOT EXISTS kg_entity_vec USING vec0(
    entity_id    TEXT PRIMARY KEY,
    embedding    float[768]
);

INSERT OR IGNORE INTO schema_version(version, applied_at) VALUES (2, strftime('%s','now'));
