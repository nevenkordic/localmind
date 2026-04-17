//! End-to-end tests against a scripted mock Ollama. Verifies the full agent
//! pipeline: user prompt → /api/chat → optional tool dispatch → /api/chat with
//! tool result → final reply.

mod common;

use common::mock_ollama::{MockOllama, MockReply};
use tokio::process::Command;

async fn run_ask(cfg: &std::path::Path, prompt: &str) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_llm"))
        .arg("--config")
        .arg(cfg)
        .arg("ask")
        .arg(prompt)
        .arg("--mode")
        .arg("workspace-write")
        .output()
        .await
        .expect("run llm ask")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ask_returns_plain_assistant_reply() {
    let mock = MockOllama::start(vec![MockReply::chat_text("hi from the mock model")]).await;

    let dir = tempfile::tempdir().unwrap();
    let cfg = common::testenv::write_config(dir.path(), &mock.url);
    let out = run_ask(&cfg, "say hi").await;

    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("hi from the mock model"),
        "stdout: {stdout}"
    );
    assert_eq!(
        mock.remaining_chat_replies(),
        0,
        "all replies should have been consumed"
    );

    // The agent should have hit /api/chat exactly once.
    let chat_calls = mock
        .requests()
        .into_iter()
        .filter(|r| r.path.contains("/api/chat"))
        .count();
    assert_eq!(chat_calls, 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structured_tool_call_is_dispatched_and_loop_terminates() {
    // Turn 1: model asks for `search_memory`.
    // Turn 2: model returns plain text using the tool result.
    let mock = MockOllama::start(vec![
        MockReply::chat_tool_call("search_memory", serde_json::json!({"query": "anything"})),
        MockReply::chat_text("ok, nothing in memory yet"),
    ])
    .await;

    let dir = tempfile::tempdir().unwrap();
    let cfg = common::testenv::write_config(dir.path(), &mock.url);
    let out = run_ask(&cfg, "what do you remember?").await;

    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("nothing in memory yet"), "stdout: {stdout}");
    assert_eq!(
        mock.remaining_chat_replies(),
        0,
        "agent didn't make the second turn"
    );

    // Two /api/chat calls = first turn + post-tool-result turn.
    let chat_calls = mock
        .requests()
        .into_iter()
        .filter(|r| r.path.contains("/api/chat"))
        .count();
    assert_eq!(chat_calls, 2, "expected 2 chat calls, got {chat_calls}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn text_form_tool_call_is_recovered_end_to_end() {
    // Regression: qwen often emits a tool call as `name key="value"` text in
    // the message content instead of structured `tool_calls`. The agent must
    // recover, dispatch, and continue the loop.
    let mock = MockOllama::start(vec![
        MockReply::chat_text_form_tool_call(r#"search_memory query="test query" top_k=3"#),
        MockReply::chat_text("recovered and answered"),
    ])
    .await;

    let dir = tempfile::tempdir().unwrap();
    let cfg = common::testenv::write_config(dir.path(), &mock.url);
    let out = run_ask(&cfg, "look it up").await;

    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("recovered and answered"),
        "stdout: {stdout}"
    );
    assert_eq!(mock.remaining_chat_replies(), 0, "second turn never fired");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn memory_primer_injects_relevant_prior_memory() {
    // Seed a memory matching the upcoming user input via `llm memory add`,
    // then `llm ask` against the mock and verify the primer placed the
    // seeded title/body into the /api/chat request body.
    let dir = tempfile::tempdir().unwrap();

    // Step 1: seed a memory (use unreachable host so we don't need the mock yet).
    let cfg_for_seed = common::testenv::write_config(dir.path(), "http://127.0.0.1:1");
    let seed = tokio::process::Command::new(env!("CARGO_BIN_EXE_llm"))
        .arg("--config")
        .arg(&cfg_for_seed)
        .args([
            "memory",
            "add",
            "-t",
            "primer-seed-marker",
            "--kind",
            "note",
            "the secret pangolin codeword is azure-buffalo",
        ])
        .output()
        .await
        .expect("seed memory");
    assert!(
        seed.status.success(),
        "seed failed: {}",
        String::from_utf8_lossy(&seed.stderr)
    );

    // Step 2: ask against the mock — same temp dir → same memory.db.
    let mock = MockOllama::start(vec![MockReply::chat_text("ack")]).await;
    let cfg = common::testenv::write_config(dir.path(), &mock.url);
    let out = run_ask(&cfg, "what is the pangolin codeword").await;
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let chats: Vec<_> = mock
        .requests()
        .into_iter()
        .filter(|r| r.path.contains("/api/chat"))
        .collect();
    assert_eq!(chats.len(), 1);
    assert!(
        chats[0].body.contains("primer-seed-marker") || chats[0].body.contains("azure-buffalo"),
        "memory primer didn't inject the seeded memory.\nbody: {}",
        chats[0].body
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn vector_search_disabled_skips_embed_calls() {
    // Regression: when [memory].vector_search = false (test config default),
    // the agent must never POST to /api/embed during a normal ask. Skill and
    // memory primers are pure BM25; embedder outbox is empty on a fresh DB.
    let mock = MockOllama::start(vec![MockReply::chat_text("ok")]).await;
    let dir = tempfile::tempdir().unwrap();
    let cfg = common::testenv::write_config(dir.path(), &mock.url);

    let out = run_ask(&cfg, "hi").await;
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let embed_calls = mock
        .requests()
        .into_iter()
        .filter(|r| r.path.contains("/api/embed"))
        .count();
    assert_eq!(embed_calls, 0, "expected no embed calls, got {embed_calls}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn remember_directive_persists_across_sessions() {
    // Regression: "remember X" must store regardless of which model is
    // configured. Then the next session's primer must surface it.
    let dir = tempfile::tempdir().unwrap();

    let mock1 = MockOllama::start(vec![MockReply::chat_text("noted")]).await;
    let cfg1 = common::testenv::write_config(dir.path(), &mock1.url);
    let out = run_ask(
        &cfg1,
        "remember the staging deploy command is `kubectl rollout restart deploy/api`",
    )
    .await;
    assert!(
        out.status.success(),
        "session 1: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    drop(mock1);

    let mock2 = MockOllama::start(vec![MockReply::chat_text("ack")]).await;
    let cfg2 = common::testenv::write_config(dir.path(), &mock2.url);
    let out = run_ask(&cfg2, "what's the deploy command for staging").await;
    assert!(
        out.status.success(),
        "session 2: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let chats: Vec<_> = mock2
        .requests()
        .into_iter()
        .filter(|r| r.path.contains("/api/chat"))
        .collect();
    assert_eq!(chats.len(), 1);
    assert!(
        chats[0].body.contains("kubectl rollout"),
        "primer didn't surface the remembered command:\n{}",
        chats[0].body
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn name_persists_across_sessions_via_auto_extract() {
    // Regression for the user's report: telling the agent "my name is X" in
    // one session and asking "what is my name?" in the next must surface
    // the name. The agent's auto-extract stores the name unconditionally;
    // the memory primer pulls it back into the next turn's context.
    let dir = tempfile::tempdir().unwrap();

    // Session 1: user introduces themselves. Mock returns a no-op reply.
    let mock1 = MockOllama::start(vec![MockReply::chat_text("Nice to meet you!")]).await;
    let cfg1 = common::testenv::write_config(dir.path(), &mock1.url);
    let out = run_ask(&cfg1, "my name is TestUserAlpha").await;
    assert!(
        out.status.success(),
        "session 1 stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    drop(mock1);

    // Session 2: ask "what is my name?". The mock just acks. We assert the
    // name landed in the chat request body via the memory primer.
    let mock2 = MockOllama::start(vec![MockReply::chat_text("ack")]).await;
    let cfg2 = common::testenv::write_config(dir.path(), &mock2.url);
    let out = run_ask(&cfg2, "what is my name").await;
    assert!(
        out.status.success(),
        "session 2 stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let chats: Vec<_> = mock2
        .requests()
        .into_iter()
        .filter(|r| r.path.contains("/api/chat"))
        .collect();
    assert_eq!(chats.len(), 1);
    assert!(
        chats[0].body.contains("TestUserAlpha"),
        "memory primer didn't surface the auto-extracted name.\nbody: {}",
        chats[0].body
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn loose_call_recovery_fires_for_thinking_out_loud() {
    // Regression: qwen "thinks out loud" — emits `<read_only_tool> <free text>`
    // with no `=`, no JSON. We recover by mapping the remainder to the tool's
    // primary string arg.
    let mock = MockOllama::start(vec![
        MockReply::chat_text_form_tool_call("search_memory what do you know about the user"),
        MockReply::chat_text("done thinking out loud"),
    ])
    .await;

    let dir = tempfile::tempdir().unwrap();
    let cfg = common::testenv::write_config(dir.path(), &mock.url);
    let out = run_ask(&cfg, "who am I?").await;

    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("done thinking out loud"),
        "stdout: {stdout}"
    );

    // Inspect the second /api/chat call — its message log should contain a
    // tool-result for `search_memory`, proving the loose call dispatched.
    let chats: Vec<_> = mock
        .requests()
        .into_iter()
        .filter(|r| r.path.contains("/api/chat"))
        .collect();
    assert_eq!(chats.len(), 2);
    assert!(
        chats[1].body.contains("\"name\":\"search_memory\""),
        "second chat call should reference the search_memory tool result\nbody: {}",
        chats[1].body
    );
}
