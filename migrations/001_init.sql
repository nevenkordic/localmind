-- localmind memory schema
-- Borrowed from nevenkordic/broodlink: multi-layer memory (versioned store +
-- FTS5 full-text + vector embeddings + knowledge graph + outbox). Collapsed
-- from Dolt/Postgres/Qdrant to a single SQLite file for a single-user tool.

PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;
PRAGMA temp_store = MEMORY;

-- ---------------------------------------------------------------------------
-- Sessions & conversation turns
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sessions (
    id            TEXT PRIMARY KEY,           -- uuid
    title         TEXT NOT NULL DEFAULT '',
    started_at    INTEGER NOT NULL,           -- unix epoch seconds
    last_active   INTEGER NOT NULL,
    cwd           TEXT NOT NULL DEFAULT '',
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS messages (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role          TEXT NOT NULL,              -- system | user | assistant | tool
    content       TEXT NOT NULL,
    tool_name     TEXT,
    tool_args_json TEXT,
    created_at    INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, created_at);

-- ---------------------------------------------------------------------------
-- Core memory — equivalent to broodlink's `agent_memory` table in Dolt.
-- Each row is a durable fact, decision, or extracted summary the agent keeps.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS memories (
    id            TEXT PRIMARY KEY,           -- uuid
    kind          TEXT NOT NULL,              -- fact | decision | project | preference | note
    title         TEXT NOT NULL,
    content       TEXT NOT NULL,
    source        TEXT NOT NULL DEFAULT '',   -- where it came from (session id, url, filepath)
    tags_json     TEXT NOT NULL DEFAULT '[]',
    importance    REAL NOT NULL DEFAULT 0.5,  -- 0.0 - 1.0
    created_at    INTEGER NOT NULL,
    updated_at    INTEGER NOT NULL,
    accessed_at   INTEGER NOT NULL,
    access_count  INTEGER NOT NULL DEFAULT 0,
    content_hash  TEXT NOT NULL,              -- dedup guard (sha256)
    UNIQUE(content_hash)
);
CREATE INDEX IF NOT EXISTS idx_memories_kind ON memories(kind);
CREATE INDEX IF NOT EXISTS idx_memories_updated ON memories(updated_at);
CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance);

-- ---------------------------------------------------------------------------
-- Full-text search (BM25) — replaces broodlink's Postgres tsvector index.
-- ---------------------------------------------------------------------------
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    title,
    content,
    tags,
    content='memories',
    content_rowid='rowid',
    tokenize='porter unicode61 remove_diacritics 2'
);

-- Keep FTS in sync with memories table.
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, title, content, tags)
    VALUES (new.rowid, new.title, new.content, new.tags_json);
END;
CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, title, content, tags)
    VALUES ('delete', old.rowid, old.title, old.content, old.tags_json);
END;
CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, title, content, tags)
    VALUES ('delete', old.rowid, old.title, old.content, old.tags_json);
    INSERT INTO memories_fts(rowid, title, content, tags)
    VALUES (new.rowid, new.title, new.content, new.tags_json);
END;

-- ---------------------------------------------------------------------------
-- Vector store — replaces broodlink's Qdrant `broodlink_memory` collection.
-- Embeddings are stored as little-endian f32 blobs and scored in Rust.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS memory_vectors (
    memory_id     TEXT PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
    dim           INTEGER NOT NULL,
    model         TEXT NOT NULL,              -- embed model name
    embedding     BLOB NOT NULL,              -- f32[dim]
    norm          REAL NOT NULL,              -- l2 norm for fast cosine
    created_at    INTEGER NOT NULL
);

-- ---------------------------------------------------------------------------
-- Knowledge graph — mirrors broodlink's kg_entities / kg_edges in Postgres.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS kg_entities (
    id            TEXT PRIMARY KEY,           -- uuid
    name          TEXT NOT NULL,              -- canonical name
    type          TEXT NOT NULL,              -- person | project | system | tech | file | concept
    summary       TEXT NOT NULL DEFAULT '',
    weight        REAL NOT NULL DEFAULT 1.0,  -- decayed over time by heartbeat
    created_at    INTEGER NOT NULL,
    updated_at    INTEGER NOT NULL,
    UNIQUE(name, type)
);
CREATE INDEX IF NOT EXISTS idx_kg_entities_type ON kg_entities(type);

CREATE TABLE IF NOT EXISTS kg_edges (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    src_id        TEXT NOT NULL REFERENCES kg_entities(id) ON DELETE CASCADE,
    dst_id        TEXT NOT NULL REFERENCES kg_entities(id) ON DELETE CASCADE,
    relation      TEXT NOT NULL,              -- uses | owns | depends_on | related_to | authored
    weight        REAL NOT NULL DEFAULT 1.0,
    created_at    INTEGER NOT NULL,
    updated_at    INTEGER NOT NULL,
    UNIQUE(src_id, dst_id, relation)
);
CREATE INDEX IF NOT EXISTS idx_kg_edges_src ON kg_edges(src_id);
CREATE INDEX IF NOT EXISTS idx_kg_edges_dst ON kg_edges(dst_id);

-- Link table: which memories reference which entities.
CREATE TABLE IF NOT EXISTS kg_entity_memories (
    entity_id     TEXT NOT NULL REFERENCES kg_entities(id) ON DELETE CASCADE,
    memory_id     TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    PRIMARY KEY (entity_id, memory_id)
);

-- ---------------------------------------------------------------------------
-- Outbox — broodlink's async embedding pattern. The agent inserts a row
-- after writing a memory; the embedding worker polls, generates the vector,
-- upserts to memory_vectors, and marks the row done.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS embedding_outbox (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id     TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    status        TEXT NOT NULL DEFAULT 'pending', -- pending | running | done | failed
    attempts      INTEGER NOT NULL DEFAULT 0,
    last_error    TEXT,
    enqueued_at   INTEGER NOT NULL,
    updated_at    INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_outbox_status ON embedding_outbox(status, enqueued_at);

-- ---------------------------------------------------------------------------
-- Schema metadata
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at INTEGER NOT NULL
);
INSERT OR IGNORE INTO schema_version(version, applied_at) VALUES (1, strftime('%s','now'));
