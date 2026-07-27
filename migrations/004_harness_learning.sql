-- Harness run history and per-model verdicts.

CREATE TABLE IF NOT EXISTS harness_runs (
    id              TEXT PRIMARY KEY,
    formula_name    TEXT NOT NULL,
    task            TEXT NOT NULL,
    plan            TEXT NOT NULL DEFAULT '',
    result          TEXT NOT NULL DEFAULT '',
    passed          INTEGER NOT NULL DEFAULT 0,
    attempts        INTEGER NOT NULL DEFAULT 0,
    checks_passed   INTEGER NOT NULL DEFAULT 0,
    checks_json     TEXT NOT NULL DEFAULT '[]',
    skills_stored   INTEGER NOT NULL DEFAULT 0,
    decision_id     TEXT,
    created_at      INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_harness_runs_created ON harness_runs(created_at);

CREATE TABLE IF NOT EXISTS harness_verdicts (
    id          TEXT PRIMARY KEY,
    run_id      TEXT NOT NULL REFERENCES harness_runs(id) ON DELETE CASCADE,
    stage       TEXT NOT NULL,
    model       TEXT NOT NULL,
    passed      INTEGER NOT NULL,
    feedback    TEXT NOT NULL DEFAULT '',
    created_at  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_harness_verdicts_run ON harness_verdicts(run_id);

INSERT OR IGNORE INTO schema_version(version, applied_at) VALUES (4, strftime('%s','now'));
