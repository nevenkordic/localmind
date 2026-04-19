//! The agent loop — think/act cycle over Ollama chat + tools.

pub mod system_prompt;

use crate::config::Config;
use crate::llm::ollama::OllamaClient as Ollama;
use crate::llm::{ChatMessage, ToolCall};
use crate::memory::{NewMemory, Store};
use crate::tools::audit::AuditLog;
use crate::tools::permissions::{PermissionManager, PermissionMode};
use crate::tools::{Registry, ToolContext};
use anyhow::Result;
use once_cell::sync::Lazy;
use regex::Regex;
use std::sync::Arc;

/// Callback the agent invokes for each streamed token from the final
/// assistant reply. The REPL writes these straight to stdout so the user
/// sees the answer materialise instead of a long spinner-then-dump wait.
pub type TokenSink = Arc<dyn Fn(&str) + Send + Sync>;

pub struct AgentRun {
    pub ctx: Arc<ToolContext>,
    pub client: Ollama,
    pub messages: Vec<ChatMessage>,
    pub max_tool_iterations: usize,
    /// Last skill primer text we injected — used to skip re-injection when the
    /// same skills would be fired on consecutive turns.
    last_primer: Option<String>,
    /// Last memory primer text we injected — same dedup as `last_primer` but
    /// for general (non-skill) memories.
    last_memory_primer: Option<String>,
    /// When set, chat calls stream tokens through this callback instead of
    /// blocking until the whole reply lands. Callers who set a sink should
    /// NOT re-print the return value of `turn()` — the content has already
    /// been rendered progressively.
    token_sink: Option<TokenSink>,
    /// When set, this turn is forced to use the named model regardless of
    /// what the router would have picked. Cleared after the turn. Set via
    /// REPL slash commands like `/retry-big` and `/fast <msg>`.
    force_model: Option<String>,
}

impl AgentRun {
    #[allow(dead_code)]
    pub fn new(
        cfg: Arc<Config>,
        store: Store,
        non_interactive: bool,
        web: bool,
        shell: bool,
    ) -> Result<Self> {
        Self::new_with_mode(cfg, store, None, non_interactive, web, shell)
    }

    pub fn new_with_mode(
        cfg: Arc<Config>,
        store: Store,
        mode_override: Option<PermissionMode>,
        non_interactive: bool,
        web: bool,
        shell: bool,
    ) -> Result<Self> {
        let audit = Arc::new(AuditLog::open()?);
        let mode = mode_override
            .unwrap_or_else(|| PermissionMode::parse(&cfg.tools.mode).unwrap_or_default());
        let perms = Arc::new(PermissionManager::new(
            mode,
            &cfg.tools.allow_rules,
            &cfg.tools.deny_rules,
            &cfg.tools.ask_rules,
            non_interactive,
        ));
        let ctx = Arc::new(ToolContext {
            cfg: cfg.clone(),
            store,
            audit,
            permissions: perms,
            web_enabled: web,
            shell_enabled: shell,
            confirm_writes: cfg.repl.confirm_writes,
        });
        let client = Ollama::new(&cfg.ollama);
        Ok(Self {
            ctx,
            client,
            messages: vec![ChatMessage::system(system_prompt::render())],
            max_tool_iterations: 12,
            last_primer: None,
            last_memory_primer: None,
            token_sink: None,
            force_model: None,
        })
    }

    /// Force the next turn to use a specific model, overriding the cascade
    /// router. Cleared automatically after the turn runs. Used by the REPL's
    /// `/retry-big` / `/fast` / `/big` slash commands.
    pub fn force_next_model(&mut self, model: impl Into<String>) {
        self.force_model = Some(model.into());
    }

    /// Enable progressive token streaming. Each content chunk from the
    /// final assistant reply is passed to `sink` as it arrives from
    /// Ollama. Intermediate tool-call iterations still use non-streaming
    /// chat — interleaved tool-narration wouldn't play nicely with the
    /// REPL's output formatting.
    pub fn set_token_sink(&mut self, sink: TokenSink) {
        self.token_sink = Some(sink);
    }

    pub fn is_streaming(&self) -> bool {
        self.token_sink.is_some()
    }

    /// Decide which model this turn should use. Precedence:
    ///   1. An explicit `force_model` set by the REPL (slash commands).
    ///   2. No routing if `fast_model` is empty / equals `chat_model` — the
    ///      user hasn't opted into cascading.
    ///   3. Heuristic: trivial / short / conversational → fast_model;
    ///      code / long / tool-implicit → chat_model.
    fn route_turn(&self, input: &str) -> String {
        if let Some(m) = &self.force_model {
            return m.clone();
        }
        let fast = self.ctx.cfg.ollama.fast_model.trim();
        let main = &self.ctx.cfg.ollama.chat_model;
        if fast.is_empty() || fast == main {
            return main.clone();
        }
        if should_use_fast(input) {
            fast.to_string()
        } else {
            main.clone()
        }
    }

    /// Send a user turn and iterate through any tool calls the model requests.
    /// Returns the final assistant text.
    pub async fn turn(&mut self, user_input: &str) -> Result<String> {
        // Auto-extract personal facts ("my name is X", "call me X",
        // "remember X") and persist BEFORE the model sees the turn. The
        // system prompt also tells qwen to do this; the regex guarantees
        // the obvious cases never depend on model cooperation.
        for (title, content, kind) in extract_facts(user_input) {
            let _ = self
                .ctx
                .store
                .insert_memory(&NewMemory {
                    kind,
                    title,
                    content: content.clone(),
                    source: "auto-extract".into(),
                    tags: vec!["user".into()],
                    importance: 0.95,
                })
                .await;
            eprintln!("  · auto-remembered: {}", content);
        }

        // Single recall pass — hybrid (BM25 + vector when enabled) so
        // semantic matches surface even when the user phrases things
        // differently from how they were stored. Then split the hits into
        // the skill primer (authoritative procedures) and the memory primer
        // (general context).
        let (skill, memory) = self.build_primers(user_input).await;
        if let Some(primer) = skill {
            if self.last_primer.as_deref() != Some(primer.as_str()) {
                self.messages.push(ChatMessage::system(primer.clone()));
                self.last_primer = Some(primer);
            }
        }
        if let Some(primer) = memory {
            if self.last_memory_primer.as_deref() != Some(primer.as_str()) {
                self.messages.push(ChatMessage::system(primer.clone()));
                self.last_memory_primer = Some(primer);
            }
        }

        self.messages
            .push(ChatMessage::user(user_input.to_string()));
        let model = self.route_turn(user_input);
        // Consume any force_model so the next turn re-routes normally.
        self.force_model = None;
        self.loop_tools(&model).await
    }

    /// Build skill + memory primers from a single recall query. Honours
    /// `[memory].vector_search`: when enabled, uses hybrid_search for
    /// semantic match; when disabled, falls back to BM25 (BM25 also fires
    /// when the query is too short for hybrid search to be meaningful).
    async fn build_primers(&self, user_input: &str) -> (Option<String>, Option<String>) {
        const MAX_SKILLS: usize = 2;
        const MAX_MEMORIES: usize = 3;
        const CONTENT_CAP: usize = 500;
        if is_trivial_turn(user_input) {
            // No BM25 or vector round-trip — the user said "hi" or similar
            // and the result of a recall against that is always useless
            // noise. Big savings on short acknowledgement turns.
            return (None, None);
        }

        // Overfetch so split-by-kind still leaves enough in each bucket.
        let fetch = (MAX_SKILLS + MAX_MEMORIES) * 3;
        let hits: Vec<(crate::memory::StoredMemory, f32)> = if self.ctx.cfg.memory.vector_search {
            // Full hybrid (BM25 + vector ANN). Embed cost is amortised by
            // the OllamaClient embedding cache.
            crate::memory::search::hybrid_search(
                &self.ctx.store,
                &self.client,
                &self.ctx.cfg,
                user_input,
                fetch,
            )
            .await
            .ok()
            .map(|hits| hits.into_iter().map(|h| (h.memory, h.score)).collect())
            .unwrap_or_default()
        } else {
            // Pure BM25 — no Ollama round-trip.
            self.ctx
                .store
                .bm25_search(user_input, fetch)
                .await
                .unwrap_or_default()
        };
        if hits.is_empty() {
            return (None, None);
        }

        let skills: Vec<_> = hits
            .iter()
            .filter(|(m, _)| m.kind == "skill")
            .take(MAX_SKILLS)
            .collect();
        let memories: Vec<_> = hits
            .iter()
            .filter(|(m, _)| m.kind != "skill")
            .take(MAX_MEMORIES)
            .collect();

        let skill_primer = if skills.is_empty() {
            None
        } else {
            let mut out = String::from(
                "Relevant skills you have been taught — follow them where applicable:\n\n",
            );
            for (m, _) in &skills {
                out.push_str(&format!("• {}\n{}\n\n", m.title, m.content.trim()));
            }
            Some(out)
        };

        let memory_primer = if memories.is_empty() {
            None
        } else {
            let mut out = String::from(
                "Memory recall for this turn (treat as authoritative for user-stated facts):\n\n",
            );
            for (m, _) in &memories {
                let body: String = m.content.chars().take(CONTENT_CAP).collect();
                out.push_str(&format!("• [{}] {}\n{}\n\n", m.kind, m.title, body.trim()));
            }
            Some(out)
        };

        (skill_primer, memory_primer)
    }

    async fn loop_tools(&mut self, model: &str) -> Result<String> {
        let specs = Registry::specs(&self.ctx);
        let color = crate::ui::color_enabled(self.ctx.cfg.repl.color);
        for _ in 0..self.max_tool_iterations {
            let spinner = crate::ui::Spinner::start("thinking", color);

            // Streaming path: if the caller installed a token sink, route
            // the reply through chat_stream so tokens land progressively.
            // The spinner stops on the first content byte; subsequent
            // tokens flow straight to the sink.
            //
            // We only stream when the iteration ends up being the final
            // assistant reply (no tool_calls). There's no reliable way to
            // know that ahead of time, but in practice:
            //   - If the model emits tool calls, it usually narrates them
            //     first ("Let me check...") — those tokens are informative
            //     output the user actually wants to see stream.
            //   - If the model emits the final reply, 100% of tokens are
            //     wanted in stream order.
            // So we stream unconditionally on every iteration when the
            // sink is set.
            let reply_res = if let Some(sink) = self.token_sink.clone() {
                // Hand the spinner to the callback via a Mutex<Option<>> so
                // the first token tick can consume and drop it. Wrap the
                // sink with a tool-call-JSON filter so accidental textual
                // tool-call emissions (some models emit BOTH a structured
                // tool_call AND a serialized JSON copy in content) don't
                // leak to the user.
                let spinner_slot = std::sync::Arc::new(std::sync::Mutex::new(Some(spinner)));
                let slot_clone = spinner_slot.clone();
                let filter = std::sync::Arc::new(std::sync::Mutex::new(
                    StreamingToolCallFilter::new(),
                ));
                let filter_clone = filter.clone();
                let sink_clone = sink.clone();
                let cb = move |chunk: &str| {
                    if let Some(sp) = slot_clone.lock().unwrap().take() {
                        sp.stop_sync();
                    }
                    let mut f = filter_clone.lock().unwrap();
                    if let Some(emit) = f.feed(chunk) {
                        sink_clone(&emit);
                    }
                };
                let r = self
                    .client
                    .chat_stream_on(&self.messages, Some(&specs), false, Some(model), cb)
                    .await;
                // Flush any remainder of the filter's buffer if the stream
                // ended while we were still deciding whether the content
                // was JSON or prose.
                if let Some(tail) = filter.lock().unwrap().flush() {
                    sink(&tail);
                }
                // If the stream ended without ever calling the callback
                // (error before the first token) the spinner is still
                // ticking — drop it now.
                if let Some(sp) = spinner_slot.lock().unwrap().take() {
                    sp.stop_sync();
                }
                r
            } else {
                let r = self
                    .client
                    .chat_on(&self.messages, Some(&specs), false, Some(model))
                    .await;
                spinner.stop().await;
                r
            };

            let mut reply = reply_res?;
            // Small models (qwen2.5-coder:3b in particular) sometimes write
            // what a tool response would LOOK like — e.g. a `<tool_response>`
            // block with "[Content of the file]" — instead of actually
            // calling the tool. Detect and either strip (if the model also
            // produced a real reply around the fabrication) or replace with
            // a clear "I didn't actually run that — escalate to chat_model"
            // error so the user sees the failure instead of fake content.
            if reply.tool_calls.as_ref().map(|v| v.is_empty()).unwrap_or(true) {
                if let Some(scrubbed) = scrub_hallucinated_tool_output(&reply.content) {
                    reply.content = scrubbed;
                }
            }
            self.messages.push(reply.clone());
            match reply.tool_calls.as_ref() {
                Some(calls) if !calls.is_empty() => {
                    for call in calls {
                        let result = dispatch(&self.ctx, call).await;
                        let content = match result {
                            Ok(s) => s,
                            Err(e) => format!("error: {e}"),
                        };
                        self.messages
                            .push(ChatMessage::tool_result(&call.function.name, content));
                    }
                }
                _ => {
                    return Ok(reply.content);
                }
            }
        }
        Ok("(max tool iterations reached)".into())
    }
}

async fn dispatch(ctx: &ToolContext, call: &ToolCall) -> Result<String> {
    let name = &call.function.name;
    let args = &call.function.arguments;
    if ctx.cfg.repl.show_tool_calls {
        crate::ui::print_tool_step(crate::ui::color_enabled(ctx.cfg.repl.color), name);
    }
    Registry::dispatch(ctx, name, args).await
}

/// Pull facts out of the user's turn. Returns `(title, content, kind)` for
/// each match — multiple matches per input are possible (e.g. "my name is X.
/// remember Y."). Strictly model-agnostic: runs regardless of which Ollama
/// model is configured, so storage never depends on the model's willingness
/// to call store_memory.
pub(crate) fn extract_facts(input: &str) -> Vec<(String, String, String)> {
    static NAME_RE: Lazy<Regex> = Lazy::new(|| {
        // "my name is X", "call me X", "you can call me X".
        Regex::new(
            r"(?i)\b(?:my name is|you can call me|call me)\s+([A-Za-z][A-Za-z'\-\s]{0,40}?)(?:[.!?,;]|$)",
        )
        .unwrap()
    });
    static REMEMBER_RE: Lazy<Regex> = Lazy::new(|| {
        // "remember X" / "remember that X" / "remember: X" — explicit user
        // signal. Anchored to start-of-input so prose like "I'll always
        // remember the day..." doesn't fire.
        Regex::new(r"(?im)^\s*remember(?:\s+that)?[:\s]+(.{3,400}?)\s*$").unwrap()
    });

    let trimmed = input.trim();
    let mut out = Vec::new();

    if let Some(c) = NAME_RE.captures(trimmed) {
        if let Some(m) = c.get(1) {
            let name = m
                .as_str()
                .trim()
                .trim_matches(|c: char| !c.is_alphanumeric());
            if !name.is_empty() {
                out.push((
                    format!("user name: {name}"),
                    format!("The user's name is {name}."),
                    "preference".into(),
                ));
            }
        }
    }

    if let Some(c) = REMEMBER_RE.captures(trimmed) {
        if let Some(m) = c.get(1) {
            let body = m.as_str().trim();
            if !body.is_empty() {
                let title: String = body.chars().take(80).collect();
                out.push((title, body.to_string(), "note".into()));
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::{
        extract_facts, is_trivial_turn, scrub_hallucinated_tool_output, should_use_fast,
        StreamingToolCallFilter,
    };

    fn first(input: &str) -> Option<(String, String, String)> {
        extract_facts(input).into_iter().next()
    }

    #[test]
    fn extracts_my_name_is() {
        let (title, content, kind) = first("my name is Neven").unwrap();
        assert!(title.contains("Neven"));
        assert!(content.contains("Neven"));
        assert_eq!(kind, "preference");
    }

    #[test]
    fn extracts_with_punctuation() {
        let (_, content, _) = first("Hi! My name is Alice Smith.").unwrap();
        assert!(content.contains("Alice Smith"));
    }

    #[test]
    fn extracts_call_me() {
        let (_, content, _) = first("you can call me Bob").unwrap();
        assert!(content.contains("Bob"));
    }

    #[test]
    fn ignores_unrelated_prose() {
        assert!(extract_facts("I am tired").is_empty());
        assert!(extract_facts("my dog ate the keyboard").is_empty());
        assert!(extract_facts("what is my name?").is_empty());
    }

    #[test]
    fn extracts_remember_explicit() {
        let (_, content, kind) = first("remember the deploy command is `kubectl rollout`").unwrap();
        assert!(content.contains("deploy command"));
        assert_eq!(kind, "note");
    }

    #[test]
    fn extracts_remember_that() {
        let (_, content, _) = first("remember that I work in PST timezone").unwrap();
        assert!(content.contains("PST"));
    }

    #[test]
    fn extracts_remember_colon() {
        let (_, content, _) = first("Remember: vector_search is off in this config").unwrap();
        assert!(content.contains("vector_search"));
    }

    #[test]
    fn remember_skipped_for_inline_use() {
        // "I'll always remember the day..." is prose, not a directive.
        // Must NOT fire — the regex anchors `remember` to start of line.
        assert!(extract_facts("I'll always remember the day we shipped").is_empty());
    }

    #[test]
    fn trivial_turns_skip_recall() {
        for w in &[
            "hi", "Hi", "Hi.", "HELLO", "yes", "ok!", "thanks", "cya", "y", "okay",
        ] {
            assert!(is_trivial_turn(w), "expected '{w}' to be trivial");
        }
    }

    #[test]
    fn substantive_turns_do_recall() {
        for w in &[
            "hi what was my deploy command",
            "remember the port number",
            "fix the failing test",
            "why",   // 3 chars but not a known interjection — let recall decide
            "what",
        ] {
            assert!(!is_trivial_turn(w), "expected '{w}' to run recall");
        }
    }

    #[test]
    fn router_fast_for_short_chatty_turns() {
        for w in &[
            "hi",
            "what time is it?",
            "who am I",
            "tell me a joke",
            "is it raining",
            "hello there friend",
        ] {
            assert!(should_use_fast(w), "expected fast path for: {w}");
        }
    }

    #[test]
    fn router_big_for_code_and_tool_work() {
        for w in &[
            "fix the failing test in src/agent/mod.rs",
            "```rust\nfn main() { }\n```",
            "refactor this function to use iterators",
            "debug why the embedder is failing",
            "please implement a retry loop",
            "run the integration test suite",
            "fetch https://example.com/data.json",
            "look at ./config/local.toml and tell me what's wrong",
        ] {
            assert!(!should_use_fast(w), "expected big model for: {w}");
        }
    }

    #[test]
    fn router_big_for_long_prompts() {
        let long = "a ".repeat(200); // 400 chars
        assert!(!should_use_fast(&long));
    }

    #[test]
    fn scrubs_fabricated_tool_response_block() {
        let fake = "<tool_response>\nPDF Content:\n[Content of the PDF file]\n</tool_response>";
        let out = scrub_hallucinated_tool_output(fake).expect("should scrub");
        assert!(out.contains("fake"));
        assert!(out.contains("/retry-big"));
        assert!(out.contains("qwen2.5-coder:7b"));
    }

    #[test]
    fn scrubs_fabrication_across_every_tool_category() {
        // One sample per tool family — if a 3B model would have fabricated
        // it, scrub should catch it.
        let samples = [
            "[Content of the file]",                   // read_file
            "[Content of the PDF]",                    // read_pdf
            "[Document content here]",                 // read_docx
            "[Spreadsheet content: rows follow]",      // read_xlsx
            "[Image description here]",                // read_image
            "[Command output: ls -la]",                // shell
            "[Shell output here]",                     // shell (variant)
            "[Webpage content from https://x.com]",    // web_fetch
            "[HTML content]",                          // web_fetch (variant)
            "[Response body: {...}]",                  // http_fetch
            "[Search results here: ...]",              // web_search
            "[Query results]",                         // search_memory
            "[DNS result: A record]",                  // dns_lookup
            "[Whois result: ...]",                     // whois
            "[Port status here]",                      // port_check
            "[Output here]",                           // generic
            "[Tool result]",                           // generic
        ];
        for s in samples {
            assert!(
                scrub_hallucinated_tool_output(s).is_some(),
                "expected scrub for: {s}"
            );
        }
    }

    #[test]
    fn leaves_legitimate_prose_alone() {
        // A reply that genuinely just answers the user without fabricating
        // tool output must not be clobbered.
        let real = "Your name is Neven Kordic based on the memory recall.";
        assert!(scrub_hallucinated_tool_output(real).is_none());
    }

    // ----- StreamingToolCallFilter tests --------------------------------

    fn run_filter(chunks: &[&str]) -> String {
        let mut f = StreamingToolCallFilter::new();
        let mut out = String::new();
        for c in chunks {
            if let Some(emit) = f.feed(c) {
                out.push_str(&emit);
            }
        }
        if let Some(tail) = f.flush() {
            out.push_str(&tail);
        }
        out
    }

    #[test]
    fn filter_drops_tool_call_json_at_start() {
        // The exact failure the user hit: qwen emits `{"name":"read_pdf",...}`
        // as text before the structured tool_call runs.
        let out = run_filter(&[
            r#"{"name":"read_pdf","arguments":{"path":"/tmp/x.pdf"}}"#,
        ]);
        assert_eq!(out, "", "tool-call JSON must not leak: got {out:?}");
    }

    #[test]
    fn filter_drops_tool_call_and_keeps_trailing_prose() {
        // Rare but possible: JSON then continuation prose on the same stream.
        let out = run_filter(&[
            r#"{"name":"read_pdf","arguments":{"path":"/tmp/x.pdf"}}"#,
            " This is the first page of the document.",
        ]);
        assert_eq!(out, " This is the first page of the document.");
    }

    #[test]
    fn filter_passes_through_plain_prose() {
        let out = run_filter(&[
            "The PDF says Knosys Limited is based in Melbourne, ",
            "with the CTO being Nicolas Passmore.",
        ]);
        assert_eq!(
            out,
            "The PDF says Knosys Limited is based in Melbourne, with the CTO being Nicolas Passmore."
        );
    }

    #[test]
    fn filter_handles_chunked_json() {
        // Real streams arrive byte-by-byte.
        let out = run_filter(&[
            r#"{"n"#,
            r#"ame":"read_pdf","arg"#,
            r#"uments":{"path":"/tmp/x.pdf"}}"#,
        ]);
        assert_eq!(out, "", "chunked JSON must still be filtered, got {out:?}");
    }

    #[test]
    fn filter_handles_json_strings_with_embedded_braces() {
        // JSON value containing braces shouldn't confuse the depth counter.
        let out = run_filter(&[r#"{"name":"x","arguments":{"raw":"}oops{"}}"#]);
        assert_eq!(out, "");
    }

    #[test]
    fn filter_emits_prose_that_starts_with_a_literal_brace() {
        // Edge case: prose genuinely starting with `{`. Commit to dropping
        // only when we see a `"name"` key within the first 256 chars.
        let long_brace_prose: String =
            "{".to_owned() + &"this is just prose with a leading brace and no quoted keys so the filter must eventually give up and emit. ".repeat(4) + "}";
        let out = run_filter(&[&long_brace_prose]);
        assert!(
            out.starts_with("{this is just prose"),
            "prose with leading brace must not be dropped, got {out:?}"
        );
    }

    #[test]
    fn filter_drops_multiple_sequential_tool_calls() {
        // The exact failure mode from the user's screenshot: three
        // write_file calls back-to-back, no prose in between.
        let out = run_filter(&[
            r#"{"name":"write_file","arguments":{"path":"/a","content":"x"}}"#,
            r#"{"name":"write_file","arguments":{"path":"/b","content":"y"}}"#,
            r#"{"name":"write_file","arguments":{"path":"/c","content":"z"}}"#,
        ]);
        assert_eq!(out, "", "consecutive tool-call JSON must all be dropped, got {out:?}");
    }

    #[test]
    fn filter_drops_interleaved_template_tokens() {
        let out = run_filter(&[
            r#"{"name":"write_file","arguments":{"path":"/a","content":"x"}}"#,
            "<|im_start|>",
            r#"{"name":"write_file","arguments":{"path":"/b","content":"y"}}"#,
            "<|im_start|>",
            r#"{"name":"write_file","arguments":{"path":"/c","content":"z"}}"#,
        ]);
        assert_eq!(out, "", "template tokens + JSON must all be dropped, got {out:?}");
    }

    #[test]
    fn filter_keeps_prose_between_tool_calls() {
        let out = run_filter(&[
            "Now let's save these files.\n",
            r#"{"name":"write_file","arguments":{"path":"/a","content":"x"}}"#,
            "\nFirst file done.\n",
            r#"{"name":"write_file","arguments":{"path":"/b","content":"y"}}"#,
            "\nAll done.",
        ]);
        // Must keep the prose and only drop the JSON blocks.
        assert!(out.contains("Now let's save these files."));
        assert!(out.contains("First file done."));
        assert!(out.contains("All done."));
        assert!(!out.contains("write_file"), "got leaked JSON: {out:?}");
    }

    #[test]
    fn filter_strips_template_tokens_in_prose() {
        let out = run_filter(&[
            "Hello there.<|im_end|>\n",
            "<|im_start|>",
            "World.",
        ]);
        assert!(out.contains("Hello there."));
        assert!(out.contains("World."));
        assert!(!out.contains("<|im_"), "template tokens must not leak: {out:?}");
    }

    #[test]
    fn filter_handles_template_token_split_across_chunks() {
        // Real stream could split the token mid-way; filter must stitch.
        let out = run_filter(&["<|im_st", "art|>hello"]);
        assert_eq!(out, "hello");
    }

    #[test]
    fn leaves_literal_bracket_mentions_alone() {
        // Messages that HAPPEN to contain square brackets but aren't fake-
        // tool-response placeholders must not trigger.
        let benign = [
            "I got a [warning] flag from the linter in [src/foo.rs].",
            "timestamp [2024-01-15T10:00:00Z]",
            "see section [1.2.3] of the spec",
            "[INFO] starting worker",
            "the array was [1, 2, 3] before the fix",
        ];
        for s in benign {
            assert!(
                scrub_hallucinated_tool_output(s).is_none(),
                "false positive on: {s}"
            );
        }
    }

    #[test]
    fn extracts_name_and_remember_together() {
        // Single input may carry multiple facts (one per line).
        let facts = extract_facts("My name is Cory.\nremember I prefer 4-space indent");
        assert!(facts
            .iter()
            .any(|(_, c, k)| k == "preference" && c.contains("Cory")));
        assert!(facts
            .iter()
            .any(|(_, c, k)| k == "note" && c.contains("4-space")));
    }
}

/// Streaming filter that drops model-emitted tool-call JSON AND chat-template
/// tokens from the output before they reach the user. Some models (notably
/// qwen2.5-coder) emit a tool call BOTH as a structured `tool_calls` field
/// AND as a serialised `{"name":"...","arguments":{...}}` block inside
/// `content`, often several back-to-back. Others leak internal chat-template
/// markers like `<|im_start|>` / `<|im_end|>` when Ollama's template handling
/// hiccups. Without this filter, streaming dumps all of it straight to stdout:
///
///     Now let's save these files.
///     {"name":"write_file","arguments":{"content":"<!DOCTYPE…","path":"…"}}
///     <|im_start|>
///     {"name":"write_file","arguments":{"content":"…","path":"…"}}
///     <|im_start|>
///     …
///
/// The filter maintains a small state machine and reruns the decision loop
/// whenever it finishes suppressing one block — so consecutive JSON objects
/// and consecutive template tokens are ALL stripped, not just the first.
/// Prose in between (when present) passes through unchanged.
pub(crate) struct StreamingToolCallFilter {
    buf: String,
    state: FilterState,
    brace_depth: i32,
    in_string: bool,
    escape: bool,
}

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
enum FilterState {
    /// Still accumulating; haven't decided what the next bytes are.
    Pending,
    /// Committed to prose; emit as bytes arrive, but scan for the start of
    /// a fresh JSON/template block so we can flip back to Pending.
    Emitting,
    /// Inside a JSON object we're dropping. Counts braces to find the close.
    DroppingJson,
    /// Inside a `<|...|>` template token. Drops until `|>`.
    DroppingTemplate,
}

const TEMPLATE_PREFIXES: &[&str] = &["<|im_start|>", "<|im_end|>", "<|endoftext|>"];

impl StreamingToolCallFilter {
    pub fn new() -> Self {
        Self {
            buf: String::new(),
            state: FilterState::Pending,
            brace_depth: 0,
            in_string: false,
            escape: false,
        }
    }

    /// Feed a streamed chunk. Returns the portion (if any) that should
    /// reach the user's sink right now.
    pub fn feed(&mut self, chunk: &str) -> Option<String> {
        self.buf.push_str(chunk);
        let mut out = String::new();
        // Process the buffer through as many state transitions as possible.
        // Each iteration either consumes bytes (produces output or drops
        // them) or breaks out when more input is needed.
        loop {
            match self.state {
                FilterState::Pending => {
                    if !self.try_decide() {
                        break; // need more bytes
                    }
                }
                FilterState::DroppingJson => {
                    if !self.consume_json() {
                        break;
                    }
                    // Closed — next bytes might be another JSON or template
                    // or prose. Go back to Pending and re-decide.
                    self.state = FilterState::Pending;
                }
                FilterState::DroppingTemplate => {
                    if !self.consume_template() {
                        break;
                    }
                    self.state = FilterState::Pending;
                }
                FilterState::Emitting => {
                    // Emit up to the start of the next JSON/template block,
                    // if any. Otherwise drain everything.
                    let restart = self.find_restart_point();
                    match restart {
                        Some(idx) if idx > 0 => {
                            out.push_str(&self.buf[..idx]);
                            self.buf.drain(..idx);
                            self.state = FilterState::Pending;
                        }
                        Some(_) => {
                            // Restart is at 0 — go straight to decision.
                            self.state = FilterState::Pending;
                        }
                        None => {
                            out.push_str(&self.buf);
                            self.buf.clear();
                            break;
                        }
                    }
                }
            }
        }
        if out.is_empty() {
            None
        } else {
            Some(out)
        }
    }

    pub fn flush(&mut self) -> Option<String> {
        // Stream ended. If we were emitting or pending prose, flush the
        // remainder. If we were mid-drop, drop it — an unterminated JSON
        // or template token is still noise.
        let out = match self.state {
            FilterState::Emitting => std::mem::take(&mut self.buf),
            FilterState::Pending => {
                let trimmed = self.buf.trim_start();
                if trimmed.is_empty()
                    || trimmed.starts_with('{')
                    || trimmed.starts_with('[')
                    || TEMPLATE_PREFIXES.iter().any(|p| trimmed.starts_with(p))
                {
                    // Looks like unfinished JSON / template / empty — drop.
                    self.buf.clear();
                    String::new()
                } else {
                    std::mem::take(&mut self.buf)
                }
            }
            _ => {
                self.buf.clear();
                String::new()
            }
        };
        if out.is_empty() {
            None
        } else {
            Some(out)
        }
    }

    /// Look at self.buf and decide whether it starts a JSON tool-call, a
    /// template token, or prose. Returns true if a decision was made (state
    /// changed away from Pending), false if we need more bytes.
    fn try_decide(&mut self) -> bool {
        // Skip leading whitespace we can safely emit later.
        let trimmed = self.buf.trim_start();
        if trimmed.is_empty() {
            return false;
        }
        // Template token at current position?
        for prefix in TEMPLATE_PREFIXES {
            if trimmed.starts_with(prefix) {
                self.state = FilterState::DroppingTemplate;
                return true;
            }
            // Partial prefix — need more bytes before we know.
            if prefix.starts_with(trimmed) && trimmed.len() < prefix.len() {
                return false;
            }
        }
        // JSON tool-call shape?
        if trimmed.starts_with('{') || trimmed.starts_with('[') {
            let has_name_key = trimmed.contains("\"name\"");
            if has_name_key {
                self.state = FilterState::DroppingJson;
                self.reset_brace_tracker();
                return true;
            }
            if trimmed.len() > 256 {
                // Long JSON-looking prose without a "name" key — emit.
                self.state = FilterState::Emitting;
                return true;
            }
            return false; // wait for more
        }
        // Everything else is prose.
        self.state = FilterState::Emitting;
        true
    }

    fn consume_json(&mut self) -> bool {
        // Walk buf tracking brace depth. Drop up through the closing brace.
        let bytes = self.buf.as_bytes();
        let mut close_at: Option<usize> = None;
        for (i, &b) in bytes.iter().enumerate() {
            if self.in_string {
                if self.escape {
                    self.escape = false;
                } else if b == b'\\' {
                    self.escape = true;
                } else if b == b'"' {
                    self.in_string = false;
                }
                continue;
            }
            match b {
                b'"' => self.in_string = true,
                b'{' | b'[' => self.brace_depth += 1,
                b'}' | b']' => {
                    self.brace_depth -= 1;
                    if self.brace_depth <= 0 {
                        close_at = Some(i + 1);
                        break;
                    }
                }
                _ => {}
            }
        }
        match close_at {
            Some(n) => {
                self.buf.drain(..n);
                true
            }
            None => {
                // Whole buffer consumed inside the JSON. Clear and wait for
                // more input.
                self.buf.clear();
                false
            }
        }
    }

    fn consume_template(&mut self) -> bool {
        // Templates end with `|>`. Find it and drop through.
        if let Some(idx) = self.buf.find("|>") {
            self.buf.drain(..idx + 2);
            true
        } else {
            // Haven't seen the close yet. Keep the last byte in case `|` is
            // split across chunks; drop everything before.
            let keep_from = self.buf.len().saturating_sub(1);
            self.buf.drain(..keep_from);
            false
        }
    }

    /// In Emitting mode, scan buf for the start of the next suppressible
    /// block so we know where to stop emitting. Returns Some(index) if
    /// found, None if the buffer is all prose.
    fn find_restart_point(&self) -> Option<usize> {
        let mut earliest: Option<usize> = None;
        // JSON tool-call boundary — `{"name"` is our signature.
        if let Some(idx) = self.buf.find("{\"name\"") {
            earliest = Some(idx);
        }
        // Template tokens.
        for prefix in TEMPLATE_PREFIXES {
            if let Some(idx) = self.buf.find(prefix) {
                earliest = Some(match earliest {
                    Some(e) => e.min(idx),
                    None => idx,
                });
            }
        }
        earliest
    }

    fn reset_brace_tracker(&mut self) {
        self.brace_depth = 0;
        self.in_string = false;
        self.escape = false;
    }
}

/// Detect when the model fabricated a tool response in its own text (a
/// failure mode on smaller models — they emit `<tool_response>...`
/// blocks or placeholder brackets instead of actually calling the tool).
///
/// Returns `Some(replacement)` when the content contains a red-flag
/// pattern. The replacement is an honest error the user can act on:
/// try `/retry-big` or configure a larger chat_model.
///
/// Returns `None` when the content looks legitimate (most turns).
///
/// Covers every tool localmind offers — not just PDF:
///   files: read_file, read_pdf, read_docx, read_xlsx, read_image
///   memory: search_memory (fake-hit fabrication)
///   web:   web_search, web_fetch, http_fetch
///   shell: shell (fake stdout)
///   net:   port_check, dns_lookup, whois (fake results)
/// New placeholders can be added here; false positives are cheap to
/// diagnose (the replacement tells the user what was flagged) whereas
/// silent hallucinations corrupt memory and mislead the user.
pub(crate) fn scrub_hallucinated_tool_output(content: &str) -> Option<String> {
    let lower = content.to_ascii_lowercase();
    // Protocol tag — NEVER legitimately present in assistant text; only the
    // tool-result message role is allowed to carry tool output.
    if lower.contains("<tool_response>") || lower.contains("</tool_response>") {
        return Some(hallucination_error("<tool_response>"));
    }
    // Placeholder brackets. Tuned against real hallucinations seen in the
    // wild; patterns are distinctive enough (no legitimate prose talks about
    // "[Content of the ___]") that false positives are rare.
    const RED_FLAGS: &[&str] = &[
        // file contents
        "[content of",        // covers "[Content of the file]", "[Content of the PDF]", etc.
        "[contents of",       // plural variant
        "[file contents]",
        "[file content]",
        "[pdf content",
        "[document content",
        "[spreadsheet content",
        "[image content",
        "[image description",
        "[extracted text]",
        // shell / command output
        "[command output",
        "[shell output",
        "[stdout here]",
        "[stderr here]",
        // web
        "[webpage content",
        "[webpage text",
        "[page content",
        "[html content",
        "[response body",
        // network / memory
        "[query results]",
        "[search results here",
        "[dns result",
        "[whois result",
        "[port status here]",
        // generic
        "[output here]",
        "[results here]",
        "[result here]",
        "[response here]",
        "[tool output]",
        "[tool result]",
    ];
    let hit = RED_FLAGS.iter().find(|flag| lower.contains(*flag))?;
    Some(hallucination_error(hit))
}

fn hallucination_error(flag: &str) -> String {
    format!(
        "(localmind: the model emitted a fake `{flag}` block instead of actually \
         calling a tool — no real result was produced. This usually means the current \
         chat_model is too small for reliable tool use. Try `/retry-big` to re-run on \
         [ollama].chat_model, or switch to a larger model with \
         `llm models --chat qwen2.5-coder:7b`.)"
    )
}

/// Cascade router heuristic: true if this turn should run on `fast_model`.
/// Intentionally loose — the goal is to route the easy 80% to the small
/// model. Wrong calls are recoverable via `/retry-big`.
///
/// Fast path when ALL of:
///   - Trimmed input is ≤ 300 chars (longer prompts almost always want
///     the capable model).
///   - No code-ish tokens (triple backticks, file extensions, path
///     separators, the `::` / `=>` / `->` / `()` sigils).
///   - No tool-implicit verbs ("fix", "refactor", "debug", "fetch",
///     "run", "implement", "build", "search web", etc.).
/// Trivial turns ("hi", "ok", "thanks") trivially qualify.
pub(crate) fn should_use_fast(input: &str) -> bool {
    let t = input.trim();
    if is_trivial_turn(t) {
        return true;
    }
    if t.chars().count() > 300 {
        return false;
    }
    const CODE_MARKERS: &[&str] = &[
        "```", ".rs", ".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".java",
        ".rb", ".cpp", ".c ", "::", "=>", "->", "()", "{}", "[]", "&&", "||",
    ];
    if CODE_MARKERS.iter().any(|m| t.contains(m)) {
        return false;
    }
    // Slash / path-like — likely file reference.
    if t.contains('/') && t.split_whitespace().any(|w| w.contains('/') && w.len() > 3) {
        return false;
    }
    const TOOL_VERBS: &[&str] = &[
        "fix ", "refactor", "debug", "implement", "build",
        "read file", "open file", "run ", "execute", "shell ",
        "fetch ", "download", "scrape", "curl ", "http",
        "search web", "look up online", "google", "wget",
        "deploy", "commit", "git ",
    ];
    let lower = t.to_lowercase();
    if TOOL_VERBS.iter().any(|w| lower.contains(w)) {
        return false;
    }
    true
}

/// Skip memory recall for interjection-only turns. Recall against "hi" or
/// "ok" almost always surfaces irrelevant past mentions of the same tokens
/// and costs an Ollama embed round-trip. For substantive messages we always
/// recall.
pub(crate) fn is_trivial_turn(input: &str) -> bool {
    let t = input.trim().trim_end_matches(|c: char| matches!(c, '.' | '!' | '?' | ',' | ';'));
    if t.len() < 3 {
        return true;
    }
    // Only bare interjections — if the user wrote anything more than the
    // greeting word, run recall (e.g. "hi, what was my deploy command" is
    // NOT trivial).
    if t.contains(char::is_whitespace) {
        return false;
    }
    const TRIVIAL: &[&str] = &[
        "hi", "hey", "hello", "yo", "sup",
        "ok", "okay", "kk", "k",
        "yes", "yep", "yeah", "yup", "sure",
        "no", "nope", "nah",
        "thanks", "thx", "ty",
        "bye", "goodbye", "cya",
        "cool", "nice", "neat",
    ];
    let lower = t.to_lowercase();
    TRIVIAL.iter().any(|w| *w == lower)
}

/// Convenience for `localmind ask "<prompt>"` — single turn, non-interactive.
pub async fn one_shot(
    cfg: Config,
    store: Store,
    prompt: &str,
    mode: Option<PermissionMode>,
) -> Result<()> {
    let mut run = AgentRun::new_with_mode(Arc::new(cfg), store, mode, true, true, true)?;
    let out = run.turn(prompt).await?;
    println!("{out}");
    Ok(())
}
