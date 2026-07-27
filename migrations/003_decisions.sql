-- Append-only decisions ledger. Structured "what we decided and why" —
-- separate from free-form memories so the verify harness and future
-- turns can audit choices without fishing through notes.

CREATE TABLE IF NOT EXISTS decisions (
    id            TEXT PRIMARY KEY,
    decision      TEXT NOT NULL,
    reasoning     TEXT NOT NULL DEFAULT '',
    alternatives  TEXT NOT NULL DEFAULT '',
    outcome       TEXT NOT NULL DEFAULT '',
    source        TEXT NOT NULL DEFAULT '',
    created_at    INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_decisions_created ON decisions(created_at);

INSERT OR IGNORE INTO schema_version(version, applied_at) VALUES (3, strftime('%s','now'));
