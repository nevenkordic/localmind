//! Smoke tests — do the basic CLI commands still work end-to-end?
//! These never reach a real Ollama (host points at 127.0.0.1:1) so they're
//! safe to run on any machine, online or offline.

mod common;

use std::process::Command;

fn run(cfg: &std::path::Path, args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_llm"))
        .arg("--config")
        .arg(cfg)
        .args(args)
        .output()
        .expect("run llm")
}

#[test]
fn version_flag_exits_clean() {
    let out = Command::new(env!("CARGO_BIN_EXE_llm"))
        .arg("--version")
        .output()
        .expect("run llm --version");
    assert!(out.status.success(), "non-zero exit");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains(env!("CARGO_PKG_VERSION")),
        "missing version: {stdout:?}"
    );
}

#[test]
fn config_show_prints_toml() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = common::testenv::write_config(dir.path(), "http://127.0.0.1:1");
    let out = run(&cfg, &["config-show"]);
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("[ollama]"));
    assert!(stdout.contains("[memory]"));
    assert!(stdout.contains("test-chat"));
}

#[test]
fn memory_add_then_search_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = common::testenv::write_config(dir.path(), "http://127.0.0.1:1");

    // Add a memory.
    let out = run(
        &cfg,
        &[
            "memory",
            "add",
            "-t",
            "smoke-test-title",
            "--kind",
            "note",
            "the quick brown fox jumps over the lazy dog",
        ],
    );
    assert!(
        out.status.success(),
        "add failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    // Search for a token from the body — BM25 should find it even with no
    // embeddings (the embedder will fail against the unreachable host).
    let out = run(&cfg, &["memory", "search", "fox"]);
    assert!(
        out.status.success(),
        "search failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("smoke-test-title") || stdout.contains("quick brown fox"),
        "search returned no matching hit: {stdout}"
    );
}

#[test]
fn memory_stats_on_fresh_db() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = common::testenv::write_config(dir.path(), "http://127.0.0.1:1");
    let out = run(&cfg, &["memory", "stats"]);
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("memories:"));
    assert!(stdout.contains("kg entities:"));
}

#[test]
fn health_command_prints_full_report() {
    // Regression: the `llm health` subcommand must report DB stats, embedder
    // outbox status, ollama reachability, and recall config in one shot.
    let dir = tempfile::tempdir().unwrap();
    let cfg = common::testenv::write_config(dir.path(), "http://127.0.0.1:1");
    let out = run(&cfg, &["health"]);
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    for needle in [
        "localmind health",
        "memory:",
        "embedder outbox:",
        "ollama:",
        "recall config:",
        "expansion_variants",
        "vector_search",
        "reachable:         NO",
    ] {
        assert!(
            stdout.contains(needle),
            "missing '{needle}' in health output:\n{stdout}"
        );
    }
}

#[test]
fn models_set_via_flags_writes_config_and_preserves_other_keys() {
    // Regression: `llm models --chat NAME` must rewrite ONLY the chat_model
    // line and leave the rest of the file (including comments) intact.
    let dir = tempfile::tempdir().unwrap();
    let cfg = common::testenv::write_config(dir.path(), "http://127.0.0.1:1");

    // Add a comment to the file so we can verify it survives the rewrite.
    let original = std::fs::read_to_string(&cfg).unwrap();
    let with_comment = original.replace(
        "[ollama]",
        "# user note: do not delete this comment\n[ollama]",
    );
    std::fs::write(&cfg, &with_comment).unwrap();

    let out = run(
        &cfg,
        &["models", "--chat", "fake-chat:42b", "--embed", "fake-embed"],
    );
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let after = std::fs::read_to_string(&cfg).unwrap();
    assert!(
        after.contains("chat_model = \"fake-chat:42b\""),
        "chat_model not updated:\n{after}"
    );
    assert!(
        after.contains("embed_model = \"fake-embed\""),
        "embed_model not updated:\n{after}"
    );
    // vision_model untouched (we didn't pass --vision).
    assert!(
        after.contains("test-vision"),
        "vision_model should be unchanged:\n{after}"
    );
    // User comment preserved (toml_edit, not toml::Value round-trip).
    assert!(
        after.contains("user note: do not delete this comment"),
        "comment was lost during rewrite:\n{after}"
    );
}

#[test]
fn memory_search_bm25_flag_is_fast_and_works_offline() {
    // Regression: `--bm25` must produce results without ever calling Ollama.
    // The unreachable host would block for `timeout_secs` if hybrid_search
    // ran, so this test passing quickly is itself the assertion.
    let dir = tempfile::tempdir().unwrap();
    let cfg = common::testenv::write_config(dir.path(), "http://127.0.0.1:1");
    let _ = run(
        &cfg,
        &[
            "memory",
            "add",
            "-t",
            "bm25-flag-test",
            "--kind",
            "note",
            "platypus walks into a bar",
        ],
    );
    let start = std::time::Instant::now();
    let out = run(&cfg, &["memory", "search", "platypus", "--bm25"]);
    let elapsed = start.elapsed();
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("bm25-flag-test") || stdout.contains("platypus"),
        "search hit missing: {stdout}"
    );
    assert!(
        elapsed < std::time::Duration::from_secs(3),
        "--bm25 took too long ({:?}), suggests it hit Ollama",
        elapsed
    );
}

#[test]
fn memory_dedup_via_content_hash() {
    // Regression: inserting the same content twice must NOT create a duplicate
    // row. The store uses sha256(content) as the dedup key.
    let dir = tempfile::tempdir().unwrap();
    let cfg = common::testenv::write_config(dir.path(), "http://127.0.0.1:1");

    for i in 0..3 {
        let out = run(
            &cfg,
            &[
                "memory",
                "add",
                "-t",
                &format!("dup-test-{i}"),
                "--kind",
                "note",
                "exact same body for dedup test",
            ],
        );
        assert!(out.status.success(), "add #{i} failed");
    }

    let out = run(&cfg, &["memory", "stats"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    // Find `memories: <N>` and assert N == 1.
    let count: i64 = stdout
        .lines()
        .find_map(|l| {
            l.strip_prefix("memories:")
                .map(|s| s.trim().parse().unwrap_or(-1))
        })
        .unwrap_or(-1);
    assert_eq!(
        count, 1,
        "expected dedup, got {count} memories\nfull output:\n{stdout}"
    );
}
