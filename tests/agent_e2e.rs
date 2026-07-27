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
    // Regression: some models emit a tool call as `name key="value"` text
    // in the message content instead of structured `tool_calls`. The agent
    // must recover, dispatch, and continue the loop.
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
    // Regression: models that "think out loud" emit `<read_only_tool>
    // <free text>` with no `=`, no JSON. We recover by mapping the
    // remainder to the tool's primary string arg.
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn taught_skill_persists_and_primes_next_session() {
    // "when X, do Y" must land as kind=skill and surface on a matching turn.
    let dir = tempfile::tempdir().unwrap();

    let mock1 = MockOllama::start(vec![MockReply::chat_text("got it")]).await;
    let cfg1 = common::testenv::write_config(dir.path(), &mock1.url);
    let out = run_ask(&cfg1, "when running tests, always use cargo test --locked").await;
    assert!(
        out.status.success(),
        "teach: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    drop(mock1);

    let mock2 = MockOllama::start(vec![MockReply::chat_text("ack")]).await;
    let cfg2 = common::testenv::write_config(dir.path(), &mock2.url);
    let out = run_ask(&cfg2, "how should I go about running tests").await;
    assert!(
        out.status.success(),
        "recall: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let chats: Vec<_> = mock2
        .requests()
        .into_iter()
        .filter(|r| r.path.contains("/api/chat"))
        .collect();
    assert_eq!(chats.len(), 1);
    assert!(
        chats[0].body.contains("cargo test --locked") || chats[0].body.contains("Relevant skills"),
        "skill primer missing:\n{}",
        chats[0].body
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn harness_plan_act_verify_records_skill() {
    // plan → act → verify → distill_skills. Scripted replies:
    // 1) plan_task
    // 2) act turn (plain text)
    // 3) verify_stage JSON pass
    // 4) distill_skills JSON array
    let mock = MockOllama::start(vec![
        MockReply::chat_text("1. Write the greeting file\n2. Confirm contents"),
        MockReply::chat_text("Created hello.txt with Hello"),
        MockReply::chat_text(r#"{"pass": true, "feedback": "ok"}"#),
        MockReply::chat_text(
            r#"[{"title":"when writing a greeting file","content":"1. write hello.txt\n2. confirm contents"}]"#,
        ),
    ])
    .await;

    let dir = tempfile::tempdir().unwrap();
    let cfg = common::testenv::write_config(dir.path(), &mock.url);
    let out = Command::new(env!("CARGO_BIN_EXE_llm"))
        .arg("--config")
        .arg(&cfg)
        .arg("run")
        .arg("write a greeting file")
        .arg("--no-plan-review")
        .arg("--mode")
        .arg("workspace-write")
        .output()
        .await
        .expect("run llm run");

    assert!(
        out.status.success(),
        "stderr: {}\nstdout: {}",
        String::from_utf8_lossy(&out.stderr),
        String::from_utf8_lossy(&out.stdout)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("Created hello.txt"), "stdout: {stdout}");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("skill recorded") || stderr.contains("skill(s) recorded"),
        "stderr: {stderr}"
    );

    // Confirm the skill is searchable offline.
    let search = Command::new(env!("CARGO_BIN_EXE_llm"))
        .arg("--config")
        .arg(&cfg)
        .args(["memory", "search", "greeting", "--bm25"])
        .output()
        .await
        .expect("search");
    assert!(search.status.success());
    let search_out = String::from_utf8_lossy(&search.stdout);
    assert!(
        search_out.contains("greeting") || search_out.contains("hello.txt"),
        "skill not in memory search:\n{search_out}"
    );

    // Harness outcomes must also land as searchable LTM decisions so later
    // sessions can recall what/when/how — not only distilled skills.
    let harness_search = Command::new(env!("CARGO_BIN_EXE_llm"))
        .arg("--config")
        .arg(&cfg)
        .args(["memory", "search", "harness", "--bm25"])
        .output()
        .await
        .expect("harness search");
    assert!(harness_search.status.success());
    let harness_out = String::from_utf8_lossy(&harness_search.stdout);
    assert!(
        harness_out.contains("harness") || harness_out.contains("greeting"),
        "harness run not mirrored to LTM:\n{harness_out}"
    );

    let stats = Command::new(env!("CARGO_BIN_EXE_llm"))
        .arg("--config")
        .arg(&cfg)
        .args(["harness", "stats"])
        .output()
        .await
        .expect("harness stats");
    assert!(stats.status.success());
    let stats_out = String::from_utf8_lossy(&stats.stdout);
    assert!(
        stats_out.contains("runs:") && stats_out.contains("passed:"),
        "unexpected stats:\n{stats_out}"
    );

    let hist = Command::new(env!("CARGO_BIN_EXE_llm"))
        .arg("--config")
        .arg(&cfg)
        .args(["harness", "history", "-n", "5"])
        .output()
        .await
        .expect("harness history");
    assert!(hist.status.success());
    let hist_out = String::from_utf8_lossy(&hist.stdout);
    assert!(
        hist_out.contains("PASS") && hist_out.contains("greeting"),
        "history missing pass:\n{hist_out}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn harness_failure_stores_avoidance_note() {
    // Failed verify must store a harness-fail note for future plans.
    // max_retries defaults to 2 → plan + 3 act/verify cycles; keep replies
    // enough for one fail then exhaust? With max_retries=2 from config and
    // verify failing once we need: plan, act, verify(fail), act, verify(fail),
    // act, verify(fail) = 1 + 3*2 = 7 chat replies if each act is one turn.
    // Simpler: set test_command to a failing shell so we fail at ground
    // checks without verify LLM calls — but testenv has test_command="".
    // Use verify JSON fail with quorum_min=1 and enough scripted replies.
    let mock = MockOllama::start(vec![
        MockReply::chat_text("1. Do the wrong thing"),
        MockReply::chat_text("I did the wrong thing"),
        MockReply::chat_text(r#"{"pass": false, "feedback": "missing required output"}"#),
        MockReply::chat_text("I tried again wrongly"),
        MockReply::chat_text(r#"{"pass": false, "feedback": "still missing required output"}"#),
        MockReply::chat_text("third attempt still wrong"),
        MockReply::chat_text(r#"{"pass": false, "feedback": "never fixed"}"#),
    ])
    .await;

    let dir = tempfile::tempdir().unwrap();
    let cfg = common::testenv::write_config(dir.path(), &mock.url);
    let out = Command::new(env!("CARGO_BIN_EXE_llm"))
        .arg("--config")
        .arg(&cfg)
        .arg("run")
        .arg("produce the required output file")
        .arg("--no-plan-review")
        .arg("--mode")
        .arg("workspace-write")
        .output()
        .await
        .expect("run llm run");

    assert!(
        !out.status.success(),
        "expected harness failure; stdout={}",
        String::from_utf8_lossy(&out.stdout)
    );

    let search = Command::new(env!("CARGO_BIN_EXE_llm"))
        .arg("--config")
        .arg(&cfg)
        .args(["memory", "search", "harness fail", "--bm25"])
        .output()
        .await
        .expect("search fail note");
    assert!(search.status.success());
    let search_out = String::from_utf8_lossy(&search.stdout);
    assert!(
        search_out.contains("fail")
            || search_out.contains("Avoid")
            || search_out.contains("never fixed")
            || search_out.contains("required output"),
        "harness-fail note missing:\n{search_out}"
    );

    let hist = Command::new(env!("CARGO_BIN_EXE_llm"))
        .arg("--config")
        .arg(&cfg)
        .args(["harness", "history", "--failed"])
        .output()
        .await
        .expect("failed history");
    let hist_out = String::from_utf8_lossy(&hist.stdout);
    assert!(
        hist_out.contains("FAIL"),
        "failed history empty:\n{hist_out}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn auto_persist_records_turn_actions_for_next_session() {
    // With auto_persist=true, a tool-using turn must write an LTM note of
    // what/when/how so the next session's recent-context primer surfaces it.
    let dir = tempfile::tempdir().unwrap();

    let mock1 = MockOllama::start(vec![
        MockReply::chat_tool_call(
            "search_memory",
            serde_json::json!({"query": "deploy staging"}),
        ),
        MockReply::chat_text(
            "I checked memory for the staging deploy procedure and found nothing yet.",
        ),
    ])
    .await;
    let cfg1 = common::testenv::write_config_opts(dir.path(), &mock1.url, true);
    let out = run_ask(&cfg1, "look up how we deploy to staging").await;
    assert!(
        out.status.success(),
        "session 1: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    drop(mock1);

    let mock2 = MockOllama::start(vec![MockReply::chat_text("ack")]).await;
    let cfg2 = common::testenv::write_config_opts(dir.path(), &mock2.url, true);
    let out = run_ask(&cfg2, "what did you do about staging deploy").await;
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
        chats[0].body.contains("search_memory")
            || chats[0].body.contains("auto-persist")
            || chats[0].body.contains("How (actions)")
            || chats[0].body.contains("Recent context"),
        "recent-context primer missing auto-persisted actions:\n{}",
        chats[0].body
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn recent_context_primes_preferences_without_query_match() {
    // Preferences must appear in the always-on recent-context primer even
    // when the user's question wouldn't BM25-match the preference title.
    let dir = tempfile::tempdir().unwrap();

    let mock1 = MockOllama::start(vec![MockReply::chat_text("Nice to meet you!")]).await;
    let cfg1 = common::testenv::write_config(dir.path(), &mock1.url);
    let out = run_ask(&cfg1, "my name is ContextPrimeUser").await;
    assert!(
        out.status.success(),
        "session 1: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    drop(mock1);

    // "hi" is trivial — still injects recent preferences so the model
    // already knows the user without a matching recall query.
    let mock2 = MockOllama::start(vec![MockReply::chat_text("hello back")]).await;
    let cfg2 = common::testenv::write_config(dir.path(), &mock2.url);
    let out = run_ask(&cfg2, "hi").await;
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
        chats[0].body.contains("ContextPrimeUser"),
        "recent-context primer didn't surface preference on trivial turn:\n{}",
        chats[0].body
    );
}
