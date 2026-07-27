-- Per-run skill attribution: which skills were primed/credited/distilled.

CREATE TABLE IF NOT EXISTS harness_skill_links (
    id          TEXT PRIMARY KEY,
    run_id      TEXT NOT NULL REFERENCES harness_runs(id) ON DELETE CASCADE,
    skill_id    TEXT NOT NULL,
    role        TEXT NOT NULL,
    passed      INTEGER NOT NULL DEFAULT 0,
    created_at  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_harness_skill_links_run ON harness_skill_links(run_id);
CREATE INDEX IF NOT EXISTS idx_harness_skill_links_skill ON harness_skill_links(skill_id);

INSERT OR IGNORE INTO schema_version(version, applied_at) VALUES (6, strftime('%s','now'));
