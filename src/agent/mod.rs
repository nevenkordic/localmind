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
    /// When set, messages are persisted under this session id so the next
    /// REPL launch in the same cwd can resume. `None` = in-memory only.
    session_id: Option<String>,
    /// How many of `messages` have been flushed to the session store.
    /// Only the delta past this index is written on each flush.
    persisted_up_to: usize,
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
            session_id: None,
            persisted_up_to: 1, // system prompt never needs persisting
        })
    }

    /// Bind this agent to a persistent session id. After this, every new
    /// message pushed onto `self.messages` will be flushed to the store
    /// by `persist_new_messages`.
    pub fn set_session(&mut self, id: String) {
        self.session_id = Some(id);
    }

    /// Wipe the persisted session for this cwd AND the in-memory history,
    /// leaving only the system prompt at index 0. The session id stays
    /// bound so future messages are saved into the (now-empty) session
    /// record rather than creating a new row. Primer dedup trackers are
    /// cleared so recall runs fresh on the next turn.
    pub fn reset_to_fresh_session(&mut self) {
        if let Some(id) = &self.session_id {
            let id = id.clone();
            let store = self.ctx.store.clone();
            tokio::spawn(async move {
                let _ = store.clear_session_messages(&id).await;
            });
        }
        self.messages.truncate(1);
        self.persisted_up_to = 1;
        self.last_primer = None;
        self.last_memory_primer = None;
    }

    /// Replace the in-memory message list with `loaded` (appended after
    /// the system prompt at index 0) and mark everything as already
    /// persisted so we don't double-write on the first turn.
    pub fn restore_messages(&mut self, loaded: Vec<ChatMessage>) {
        // Keep self.messages[0] (fresh system prompt from this boot) and
        // append the historical tail. The system prompt itself is not
        // persisted — it rebuilds on every launch.
        self.messages.truncate(1);
        self.messages.extend(loaded);
        self.persisted_up_to = self.messages.len();
    }

    /// Flush any newly-pushed messages to the session store. Safe to call
    /// repeatedly; each call writes only the delta since the last flush.
    /// No-op when no session is bound.
    pub async fn persist_new_messages(&mut self) {
        let Some(session_id) = self.session_id.clone() else {
            return;
        };
        let start = self.persisted_up_to;
        let end = self.messages.len();
        if start >= end {
            return;
        }
        for i in start..end {
            let m = &self.messages[i];
            let extras = Self::message_extras_json(m);
            if let Err(e) = self
                .ctx
                .store
                .append_message(
                    &session_id,
                    &m.role,
                    &m.content,
                    m.name.as_deref(),
                    extras.as_deref(),
                )
                .await
            {
                // Log but don't fail the turn — persistence is a
                // convenience, not correctness-critical.
                tracing::warn!("persist_new_messages: {e}");
                break;
            }
        }
        self.persisted_up_to = end;
    }

    /// Serialize the fields of a ChatMessage that don't fit the flat
    /// (role, content, tool_name) schema into a JSON blob — tool_calls
    /// on assistant messages, tool_call_id on tool-role responses. Used
    /// in both persistence and reconstruction paths.
    fn message_extras_json(m: &ChatMessage) -> Option<String> {
        if m.tool_calls.is_none() && m.tool_call_id.is_none() {
            return None;
        }
        let mut obj = serde_json::Map::new();
        if let Some(calls) = &m.tool_calls {
            obj.insert("tool_calls".into(), serde_json::to_value(calls).ok()?);
        }
        if let Some(id) = &m.tool_call_id {
            obj.insert("tool_call_id".into(), serde_json::Value::String(id.clone()));
        }
        serde_json::to_string(&serde_json::Value::Object(obj)).ok()
    }

    /// Inverse of message_extras_json — rebuild a ChatMessage from the
    /// persisted row tuple (role, content, tool_name, extras_json).
    pub fn message_from_persisted(
        role: String,
        content: String,
        tool_name: Option<String>,
        extras_json: Option<String>,
    ) -> ChatMessage {
        let (tool_calls, tool_call_id) = extras_json
            .as_deref()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(s).ok())
            .map(|v| {
                let calls = v
                    .get("tool_calls")
                    .and_then(|c| serde_json::from_value(c.clone()).ok());
                let id = v
                    .get("tool_call_id")
                    .and_then(|i| i.as_str())
                    .map(|s| s.to_string());
                (calls, id)
            })
            .unwrap_or((None, None));
        ChatMessage {
            role,
            content,
            images: None,
            tool_calls,
            tool_call_id,
            name: tool_name,
        }
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

    /// Rough token estimate — char count / 4 plus a flat per-message
    /// overhead for role / formatting. Matches the industry rule-of-thumb
    /// close enough to drive compaction decisions without pulling in a
    /// tokenizer crate.
    fn estimate_tokens(&self) -> usize {
        const PER_MESSAGE_OVERHEAD: usize = 10;
        self.messages
            .iter()
            .map(|m| {
                let body = m.content.len()
                    + m.tool_calls
                        .as_ref()
                        .map(|v| v.len() * 80)
                        .unwrap_or(0);
                PER_MESSAGE_OVERHEAD + body / 4
            })
            .sum()
    }

    /// Check whether `auto_compact_at` has been crossed; if so, run
    /// compaction. Silent no-op when disabled (threshold = 0) or when
    /// history is still short enough to fit comfortably.
    async fn maybe_compact(&mut self) {
        let threshold = self.ctx.cfg.ollama.auto_compact_at;
        if threshold <= 0.0 {
            return;
        }
        let budget = self.ctx.cfg.ollama.num_ctx as f32;
        let used = self.estimate_tokens() as f32;
        if used < threshold * budget {
            return;
        }
        if let Err(e) = self.compact_now(/*forced=*/ false).await {
            eprintln!("  \x1b[2m(auto-compact failed: {e}; continuing without)\x1b[0m");
        }
    }

    /// Replace the middle of `self.messages` with a summary system
    /// message. Keeps the main system prompt (index 0) and the last
    /// `compact_keep_tail` messages verbatim. Tail boundary is snapped
    /// forward to the next `user` message so a tool_call / tool_response
    /// pair never gets split.
    ///
    /// `forced = true` bypasses the "is there anything worth compacting?"
    /// guard — used by the `/compact` slash command.
    pub async fn compact_now(&mut self, forced: bool) -> Result<()> {
        let keep_tail = self.ctx.cfg.ollama.compact_keep_tail.max(2);
        // Need at least: system prompt + some middle + tail. If the
        // history is already short, there's nothing to do.
        if self.messages.len() < keep_tail + 3 {
            if forced {
                eprintln!("  \x1b[2m· already compact ({} messages)\x1b[0m", self.messages.len());
            }
            return Ok(());
        }

        // Candidate split: everything in messages[1..split] becomes the
        // summary; messages[split..] is kept verbatim.
        let candidate = self.messages.len().saturating_sub(keep_tail);
        let split = self.next_user_boundary(candidate).unwrap_or(candidate);
        if split <= 1 {
            return Ok(());
        }

        // Render the middle as plain text for the summariser. Tool-call
        // JSON is stringified inline so the summariser sees what actually
        // happened, not opaque structs.
        let mut transcript = String::new();
        for m in &self.messages[1..split] {
            transcript.push_str(&format!("[{}] {}\n", m.role, m.content.trim()));
            if let Some(calls) = &m.tool_calls {
                for c in calls {
                    transcript.push_str(&format!(
                        "  <tool_call name={} args={}>\n",
                        c.function.name,
                        serde_json::to_string(&c.function.arguments).unwrap_or_default()
                    ));
                }
            }
        }

        let before_count = self.messages.len();
        let before_tokens = self.estimate_tokens();

        let summary = self
            .client
            .summarize_history(&transcript)
            .await
            .map_err(|e| anyhow::anyhow!("summarize failed: {e}"))?;

        let placeholder = ChatMessage::system(format!(
            "[Earlier conversation summary — replaces {} messages]\n{}",
            split - 1,
            summary.trim()
        ));

        // Splice: messages[0] + placeholder + messages[split..].
        let tail: Vec<ChatMessage> = self.messages.split_off(split);
        self.messages.truncate(1); // keep only the main system prompt
        self.messages.push(placeholder);
        self.messages.extend(tail);

        // Primers are inside the summary now — clear the dedup trackers
        // so the next turn's recall can reinject them if still relevant.
        self.last_primer = None;
        self.last_memory_primer = None;

        let after_tokens = self.estimate_tokens();
        eprintln!(
            "  \x1b[2m· auto-compacted {} messages (~{} → {} tokens)\x1b[0m",
            before_count - self.messages.len(),
            before_tokens,
            after_tokens,
        );
        Ok(())
    }

    /// Walk forward from `from` until the next `user`-role message; that
    /// sits on a clean turn boundary (never between an assistant
    /// tool_call and its tool response). `None` if there's no user
    /// message at/after `from`.
    fn next_user_boundary(&self, from: usize) -> Option<usize> {
        (from..self.messages.len()).find(|&i| self.messages[i].role == "user")
    }

    /// Send a user turn and iterate through any tool calls the model requests.
    /// Returns the final assistant text.
    pub async fn turn(&mut self, user_input: &str) -> Result<String> {
        // Auto-compact history if it's pushing num_ctx. Runs only at turn
        // boundaries — compacting mid-tool-loop would break the
        // assistant-tool_call → tool-result pairing and confuse the model.
        self.maybe_compact().await;

        // Auto-extract personal facts ("my name is X", "call me X",
        // "remember X") and persist BEFORE the model sees the turn. The
        // system prompt also tells the model to do this; the regex guarantees
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
        // One-shot guard — we nudge at most once per turn on detected
        // narrated-write hallucinations, so a repeatedly-misbehaving
        // model can't pin the loop against max_tool_iterations.
        let mut already_nudged_fake_write = false;
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
                let filter =
                    std::sync::Arc::new(std::sync::Mutex::new(StreamingToolCallFilter::new()));
                // Pipeline: Ollama → StreamingToolCallFilter (strips JSON /
                // template tokens) → MarkdownHighlighter (colours fenced
                // code blocks) → user's sink (stdout).
                let filter_clone = filter.clone();
                let md = std::sync::Arc::new(std::sync::Mutex::new(MarkdownHighlighter::new()));
                let md_clone = md.clone();
                let sink_clone = sink.clone();
                let sink_for_md = sink.clone();
                let cb = move |chunk: &str| {
                    if let Some(sp) = slot_clone.lock().unwrap().take() {
                        sp.stop_sync();
                    }
                    let mut f = filter_clone.lock().unwrap();
                    if let Some(emit) = f.feed(chunk) {
                        let mut h = md_clone.lock().unwrap();
                        let sink_inner = sink_clone.clone();
                        h.feed(&emit, move |s| sink_inner(s));
                    }
                };
                let r = self
                    .client
                    .chat_stream_on(&self.messages, Some(&specs), false, Some(model), cb)
                    .await;
                // Flush any remainder of the filter's buffer if the stream
                // ended while we were still deciding whether the content
                // was JSON or prose, then flush the markdown highlighter.
                if let Some(tail) = filter.lock().unwrap().flush() {
                    let mut h = md.lock().unwrap();
                    let sink_inner = sink_for_md.clone();
                    h.feed(&tail, move |s| sink_inner(s));
                }
                {
                    let mut h = md.lock().unwrap();
                    let sink_inner = sink.clone();
                    h.flush(move |s| sink_inner(s));
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
            // Small models sometimes write
            // what a tool response would LOOK like — e.g. a `<tool_response>`
            // block with "[Content of the file]" — instead of actually
            // calling the tool. Detect and either strip (if the model also
            // produced a real reply around the fabrication) or replace with
            // a clear "I didn't actually run that — escalate to chat_model"
            // error so the user sees the failure instead of fake content.
            if reply
                .tool_calls
                .as_ref()
                .map(|v| v.is_empty())
                .unwrap_or(true)
            {
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
                    // Hallucination check: some models narrate file
                    // writes as rendered diff output in prose instead of
                    // calling write_file. If we spot 3+ diff-style rows
                    // and made no tool calls, nudge the model to actually
                    // call the tool. Fires at most once per turn.
                    if !already_nudged_fake_write {
                        if let Some(count) = detect_narrated_write(&reply.content) {
                            already_nudged_fake_write = true;
                            eprintln!(
                                "  \x1b[2m· detected narrated write ({count} fenced-code lines, no write_file call) — nudging model\x1b[0m"
                            );
                            self.messages.push(ChatMessage::user(
                                "STOP describing file contents in code fences. You MUST call the \
                                 write_file tool directly with a structured tool_call to actually \
                                 create the file. Do not paste file bodies as assistant prose. If \
                                 you genuinely cannot call the tool, say so in one short sentence. \
                                 Call the tool now.".to_string()
                            ));
                            continue;
                        }
                    }
                    return Ok(reply.content);
                }
            }
        }
        Ok("(max tool iterations reached)".into())
    }
}

/// Detect the "narrated write" hallucination: the model describes
/// creating a file by pasting its content into a fenced code block
/// instead of calling `write_file`. Fires when either:
///
///   (a) any fenced code block carries a fake structured tool call
///       (shape `{"name": "write_file" | "create_dir" | ...}`), or
///   (b) the reply has ≥ FENCE_LINE_THRESHOLD lines of fenced code with
///       no accompanying tool call.
///
/// Returns the offending line count when suspicious, `None` otherwise.
/// Threshold keeps short explanatory snippets (`one-liner in ```bash```)
/// from false-firing. The caller has already confirmed no structured
/// tool_calls were made.
pub(crate) fn detect_narrated_write(reply: &str) -> Option<usize> {
    const FENCE_LINE_THRESHOLD: usize = 12;
    static FENCE_RE: Lazy<Regex> = Lazy::new(|| {
        // Matches ```<optional-lang>\n<body>\n``` as a balanced block.
        // (?s) so `.` crosses newlines; non-greedy body captures up to
        // the nearest closing fence.
        Regex::new(r"(?s)```[^\n`]{0,30}\n(.*?)\n```").unwrap()
    });
    static FAKE_TOOL_CALL_RE: Lazy<Regex> =
        // `"name"` key paired with a known side-effecting tool name in
        // the same block is a strong signal the model is describing a
        // call, not making one.
        Lazy::new(|| {
            Regex::new(
                r#""name"\s*:\s*"(write_file|create_dir|shell|web_fetch|http_fetch|zip_create|zip_extract)""#,
            )
            .unwrap()
        });
    let mut fence_line_count = 0usize;
    for caps in FENCE_RE.captures_iter(reply) {
        let body = caps.get(1).map(|m| m.as_str()).unwrap_or("");
        if FAKE_TOOL_CALL_RE.is_match(body) {
            return Some(body.lines().count().max(1));
        }
        fence_line_count += body.lines().count();
    }
    if fence_line_count >= FENCE_LINE_THRESHOLD {
        Some(fence_line_count)
    } else {
        None
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
        detect_narrated_write, extract_facts, is_trivial_turn, scrub_hallucinated_tool_output,
        should_use_fast, MarkdownHighlighter, StreamingToolCallFilter,
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
            "why", // 3 chars but not a known interjection — let recall decide
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
            "[Content of the file]",                // read_file
            "[Content of the PDF]",                 // read_pdf
            "[Document content here]",              // read_docx
            "[Spreadsheet content: rows follow]",   // read_xlsx
            "[Image description here]",             // read_image
            "[Command output: ls -la]",             // shell
            "[Shell output here]",                  // shell (variant)
            "[Webpage content from https://x.com]", // web_fetch
            "[HTML content]",                       // web_fetch (variant)
            "[Response body: {...}]",               // http_fetch
            "[Search results here: ...]",           // web_search
            "[Query results]",                      // search_memory
            "[DNS result: A record]",               // dns_lookup
            "[Whois result: ...]",                  // whois
            "[Port status here]",                   // port_check
            "[Output here]",                        // generic
            "[Tool result]",                        // generic
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

    // ----- detect_narrated_write tests ---------------------------------

    #[test]
    fn narrated_write_fires_on_fake_tool_call_json() {
        // The exact pathology from the wild: model pastes a JSON block
        // describing a tool call instead of emitting a structured call.
        let reply = "### Step 1\n```json\n{\n  \"name\": \"write_file\",\n  \"arguments\": {\n    \"path\": \"/tmp/x.html\"\n  }\n}\n```\n";
        assert!(detect_narrated_write(reply).is_some());
    }

    #[test]
    fn narrated_write_fires_on_large_code_fence() {
        // A substantial fenced code block with no accompanying tool
        // call, on a turn where one was expected, is our softer signal.
        let mut reply = String::from("Here's the HTML:\n```html\n");
        for i in 0..15 {
            reply.push_str(&format!("<div class=\"row-{i}\">content</div>\n"));
        }
        reply.push_str("```\n");
        assert!(detect_narrated_write(&reply).is_some());
    }

    #[test]
    fn narrated_write_ignores_short_snippet() {
        // A one-liner in a fence (e.g. `ls -la`) shouldn't fire.
        let reply = "Run this to check:\n```bash\nls -la ~/Desktop\n```";
        assert_eq!(detect_narrated_write(reply), None);
    }

    #[test]
    fn narrated_write_ignores_plain_prose() {
        let reply = "I created the file for you. Open it in your editor.";
        assert_eq!(detect_narrated_write(reply), None);
    }

    #[test]
    fn narrated_write_ignores_numbered_list() {
        let reply = "1. First do this\n2. Then do that\n3. Finally run tests";
        assert_eq!(detect_narrated_write(reply), None);
    }

    #[test]
    fn narrated_write_detects_unrelated_json_with_tool_name() {
        // A config snippet mentioning `write_file` in a `"name"` key
        // would fire — that's acceptable since the caller has already
        // gated on tool_calls being empty. False positives on genuine
        // user-facing JSON examples are rare and benign (the nudge
        // just asks the model to call the tool for real).
        let reply = "```json\n{\"name\": \"write_file\"}\n```";
        assert!(detect_narrated_write(reply).is_some());
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
        // The exact failure the user hit: the model emits `{"name":"read_pdf",...}`
        // as text before the structured tool_call runs.
        let out = run_filter(&[r#"{"name":"read_pdf","arguments":{"path":"/tmp/x.pdf"}}"#]);
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
        assert_eq!(
            out, "",
            "consecutive tool-call JSON must all be dropped, got {out:?}"
        );
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
        assert_eq!(
            out, "",
            "template tokens + JSON must all be dropped, got {out:?}"
        );
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
        let out = run_filter(&["Hello there.<|im_end|>\n", "<|im_start|>", "World."]);
        assert!(out.contains("Hello there."));
        assert!(out.contains("World."));
        assert!(
            !out.contains("<|im_"),
            "template tokens must not leak: {out:?}"
        );
    }

    #[test]
    fn filter_drops_tool_response_block_inline() {
        let out = run_filter(&[
            "Sure, reading the file now.\n",
            "<tool_response>\nPDF Content:\n[Content of the PDF file]\n</tool_response>",
            "\nDone.",
        ]);
        assert!(out.contains("Sure, reading the file now."));
        assert!(out.contains("Done."));
        assert!(
            !out.contains("<tool_response>") && !out.contains("</tool_response>"),
            "tags must be fully stripped, got: {out:?}"
        );
        assert!(
            !out.contains("[Content of the PDF file]"),
            "body of the fake tool response must be dropped, got: {out:?}"
        );
    }

    #[test]
    fn filter_drops_tool_response_split_across_chunks() {
        let out = run_filter(&["<tool_", "response>\nfake body\n</tool_", "response>"]);
        assert_eq!(
            out, "",
            "split tool_response must still be dropped: {out:?}"
        );
    }

    #[test]
    fn filter_drops_orphan_closing_tool_response_tag() {
        // Some models emit only the closing tag. Don't leak it.
        let out = run_filter(&["hello</tool_response> world"]);
        assert!(out.contains("hello"));
        assert!(out.contains(" world"));
        assert!(!out.contains("</tool_response>"));
    }

    #[test]
    fn filter_handles_template_token_split_across_chunks() {
        // Real stream could split the token mid-way; filter must stitch.
        let out = run_filter(&["<|im_st", "art|>hello"]);
        assert_eq!(out, "hello");
    }

    // ----- MarkdownHighlighter ---------------------------------------

    fn run_md(chunks: &[&str]) -> String {
        let mut h = MarkdownHighlighter::new();
        let mut out = String::new();
        for c in chunks {
            let collected = std::cell::RefCell::new(String::new());
            h.feed(c, |s| collected.borrow_mut().push_str(s));
            out.push_str(&collected.borrow());
        }
        let collected = std::cell::RefCell::new(String::new());
        h.flush(|s| collected.borrow_mut().push_str(s));
        out.push_str(&collected.borrow());
        out
    }

    #[test]
    fn md_prose_passes_through_unchanged() {
        let out = run_md(&["Hello world.\nHow are you?\n"]);
        assert_eq!(out, "Hello world.\nHow are you?\n");
    }

    #[test]
    fn md_fenced_code_gets_highlighted_with_gutter() {
        let out = run_md(&["Here is Rust:\n", "```rust\n", "fn main() {}\n", "```\n"]);
        // Fence markers are swallowed — user asked for them hidden.
        assert!(!out.contains("```"));
        assert!(
            out.contains("   1"),
            "expected line-number gutter, got: {out:?}"
        );
        // Non-diff fences render as context rows: gutter + space sign,
        // no + marker and no row background. This keeps casual code
        // examples in chat from looking like proposed file writes.
        assert!(
            !out.contains("\x1b[48;2;12;36;12m"),
            "non-diff fence must NOT use added-row background, got: {out:?}"
        );
    }

    #[test]
    fn md_fenced_block_without_lang_still_gutters() {
        let out = run_md(&["```\n", "some plaintext\n", "```\n"]);
        assert!(out.contains("   1"));
        assert!(!out.contains("```"));
    }

    #[test]
    fn md_handles_chunked_fence_open() {
        // Split the opening fence across chunk boundaries — the "could be
        // fence prefix" guard should hold bytes until we see the newline.
        let out = run_md(&["``", "`rust\n", "fn main() {}\n", "```\n"]);
        // Now that fences are swallowed, the output is just the highlighted
        // code line (no fence markers).
        assert!(
            !out.contains("```"),
            "fence markers must be hidden: {out:?}"
        );
        assert!(out.contains("   1"));
    }

    #[test]
    fn md_diff_fence_renders_plus_minus_colors() {
        // ```diff fences get per-line semantics: + green, - red, space
        // context. The user asked for visible red removals in replies.
        let out = run_md(&[
            "```diff\n",
            "+ added line\n",
            "- removed line\n",
            "  context line\n",
            "```\n",
        ]);
        // Fence markers stripped.
        assert!(!out.contains("```"));
        // Add row: green bg (12/36/12).
        assert!(
            out.contains("\x1b[48;2;12;36;12m"),
            "expected green bg on insert row: {out:?}"
        );
        // Remove row: red bg (48/12/12).
        assert!(
            out.contains("\x1b[48;2;48;12;12m"),
            "expected red bg on delete row: {out:?}"
        );
        // Context row: no bg colour code but a space sign.
        assert!(out.contains("context line"));
        // The +/- prefix chars are stripped from the body — check the
        // body doesn't contain a literal `+ added` pair.
        assert!(!out.contains("+ added"));
    }

    #[test]
    fn md_multiple_fences_in_one_stream() {
        let out = run_md(&[
            "First block:\n",
            "```js\n",
            "console.log(1);\n",
            "```\n",
            "Then more:\n",
            "```py\n",
            "print(2)\n",
            "```\n",
            "Done.\n",
        ]);
        // Each code block's line counter resets to 1 — so the first-line
        // gutter `   1` appears exactly twice across the rendered output.
        assert_eq!(
            out.matches("   1").count(),
            2,
            "expected 2 first-line gutters, got: {out:?}"
        );
        assert!(out.contains("Done.\n"));
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
/// tokens from the output before they reach the user. Some models emit a
/// tool call BOTH as a structured `tool_calls` field
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

#[derive(PartialEq, Eq, Clone, Debug)]
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
    /// Inside a block-level tag (`<tool_response>...</tool_response>`).
    /// Drops bytes until the closing tag appears. String is the close
    /// tag literal (e.g. `</tool_response>`).
    DroppingBlock(String),
}

const TEMPLATE_PREFIXES: &[&str] = &["<|im_start|>", "<|im_end|>", "<|endoftext|>"];

/// Block-level tags the model sometimes emits as text even though they're
/// meant to come from the protocol layer. Everything between the open
/// and close tag is discarded. Close tags appearing orphan (without a
/// matching open) are also dropped. Add new pairs as new failure modes
/// surface.
const BLOCK_TAG_PAIRS: &[(&str, &str)] = &[
    ("<tool_response>", "</tool_response>"),
    // Reasoning-model chain-of-thought. Strip the think block so the
    // transcript shows the final reply only.
    ("<think>", "</think>"),
];
const ORPHAN_CLOSE_TAGS: &[&str] = &["</tool_response>", "</think>"];

// ANSI colour constants shared by the markdown-code renderer and the
// write_file diff preamble so the two look-and-feels stay in sync. The
// greens and reds are deliberately dark (RGB ~12/36/12, 48/12/12) so they
// read as a gentle row-fill rather than a glaring block.
pub(crate) const GUTTER_FG: &str = "\x1b[2;37m"; // dim grey — context / no-change
                                                 // Gutter + marker share the same shade per row-type so the line-number
                                                 // and the +/- marker look uniform. Previously the gutter was dimmed and
                                                 // looked a half-step lighter than the marker; matched now.
pub(crate) const GUTTER_FG_ADD: &str = "\x1b[32m"; // green for added-line gutter
pub(crate) const GUTTER_FG_DEL: &str = "\x1b[31m"; // red for removed-line gutter
pub(crate) const ADDED_BG: &str = "\x1b[48;2;12;36;12m"; // muted dark green
pub(crate) const REMOVED_BG: &str = "\x1b[48;2;48;12;12m"; // muted dark red
pub(crate) const MARKER_FG_ADD: &str = "\x1b[32m"; // regular green (not bright)
pub(crate) const MARKER_FG_DEL: &str = "\x1b[31m"; // regular red
pub(crate) const FG_RESET: &str = "\x1b[39m"; // reset foreground only — keeps bg
pub(crate) const CLEAR_EOL: &str = "\x1b[K"; // fills remainder of line with current bg
pub(crate) const RESET: &str = "\x1b[0m"; // full reset

/// Markdown-aware syntax highlighter for streamed assistant replies. Wraps
/// a downstream sink and intercepts ```lang code fences, highlighting
/// their contents via syntect before forwarding. Prose outside fences
/// passes through char-by-char so the streaming UX stays snappy; code
/// inside fences renders line-by-line (we need a full line for syntect
/// to tokenise).
///
/// Detection is deliberately lenient:
///   * Fence is a line whose first non-whitespace run is 3+ backticks
///   * Language tag is optional (whitespace-terminated after the ticks)
///   * Closing fence is any line whose first non-whitespace run is ≥ the
///     opening count of backticks
///
/// When the language is absent or unknown, the code still gets a dim
/// gutter and plain-text styling — never worse than the previous raw
/// output.
pub(crate) struct MarkdownHighlighter {
    state: MdState,
    /// Buffer of bytes from the current line that haven't been flushed yet.
    line_buf: String,
    /// Language tag from the opening fence, used for syntect lookup.
    lang: String,
    /// One-off initialised syntect data; shared across all fences in a
    /// session so we don't reload the syntax set per block.
    ps: syntect::parsing::SyntaxSet,
    ts: syntect::highlighting::ThemeSet,
    /// 1-based line counter inside the current code fence. Reset on fence
    /// open so each block numbers from 1.
    code_lineno: usize,
}

#[derive(PartialEq, Eq)]
enum MdState {
    Prose,
    InFence,
}

impl MarkdownHighlighter {
    pub fn new() -> Self {
        Self {
            state: MdState::Prose,
            line_buf: String::new(),
            lang: String::new(),
            ps: syntect::parsing::SyntaxSet::load_defaults_newlines(),
            ts: syntect::highlighting::ThemeSet::load_defaults(),
            code_lineno: 0,
        }
    }

    /// Feed a chunk of post-filter prose and flush everything we can to
    /// `sink`. Stateful across calls.
    ///
    /// Strategy: in Prose mode, characters are forwarded as soon as we
    /// know the current line cannot be a fence opener (as soon as we see
    /// any non-whitespace, non-backtick char). Until then, bytes sit in
    /// `line_buf` so we can recognise a `\`\`\`` fence even when it's
    /// split across chunks. In fence mode everything is line-buffered —
    /// syntect needs whole lines.
    pub fn feed<F: FnMut(&str)>(&mut self, chunk: &str, mut sink: F) {
        for ch in chunk.chars() {
            match self.state {
                MdState::Prose => self.feed_prose_char(ch, &mut sink),
                MdState::InFence => self.feed_fence_char(ch, &mut sink),
            }
        }
    }

    pub fn flush<F: FnMut(&str)>(&mut self, mut sink: F) {
        if self.line_buf.is_empty() {
            return;
        }
        let tail = std::mem::take(&mut self.line_buf);
        match self.state {
            MdState::InFence => {
                // Unterminated code block — highlight what we have.
                self.code_lineno += 1;
                let rendered = self.highlight_line(&tail);
                sink(&rendered);
            }
            MdState::Prose => {
                // Held fence prefix that never resolved; emit literal.
                sink(&tail);
            }
        }
    }

    fn feed_prose_char<F: FnMut(&str)>(&mut self, ch: char, sink: &mut F) {
        if ch == '\n' {
            // End of line. If line_buf holds anything, it's either a
            // complete fence line or an unflushed all-whitespace/backtick
            // prefix.
            if !self.line_buf.is_empty() {
                let held = std::mem::take(&mut self.line_buf);
                if let Some(lang) = parse_fence(held.trim_start()) {
                    // Open a fence. SWALLOW the ```lang line + its trailing
                    // newline — the user asked for the markers hidden; the
                    // highlighted body alone speaks for itself.
                    self.state = MdState::InFence;
                    self.lang = lang;
                    self.code_lineno = 0;
                    return;
                }
                // Not a fence — flush the held bytes as normal prose.
                sink(&held);
            }
            sink("\n");
            return;
        }

        if self.line_buf.is_empty() {
            // Beginning of a line. Only whitespace or backtick could
            // possibly lead to a fence; everything else is definitely
            // prose and can be forwarded immediately.
            if ch.is_whitespace() || ch == '`' {
                self.line_buf.push(ch);
            } else {
                sink(&ch.to_string());
            }
            return;
        }

        // Mid-line, still holding a potential fence prefix. Append and
        // check whether it could STILL be a fence.
        self.line_buf.push(ch);
        if !could_be_fence_prefix(&self.line_buf) {
            // Committed to prose for this line — flush everything held.
            let flush = std::mem::take(&mut self.line_buf);
            sink(&flush);
        }
    }

    fn feed_fence_char<F: FnMut(&str)>(&mut self, ch: char, sink: &mut F) {
        self.line_buf.push(ch);
        if ch != '\n' {
            return;
        }
        let line = std::mem::take(&mut self.line_buf);
        if parse_fence(line.trim()).is_some() {
            // Closing fence — swallow it for the same reason the opener
            // is swallowed; the green rows themselves mark the block.
            self.state = MdState::Prose;
            self.lang.clear();
            return;
        }
        // A code line. Highlight it with a gutter.
        self.code_lineno += 1;
        let rendered = self.highlight_line(&line);
        sink(&rendered);
    }

    fn highlight_line(&self, line: &str) -> String {
        use syntect::easy::HighlightLines;
        use syntect::util::as_24_bit_terminal_escaped;

        let body_line = line.strip_suffix('\n').unwrap_or(line);
        let trailing_nl = if line.ends_with('\n') { "\n" } else { "" };

        // `diff` fences get per-line +/- semantics: the first char decides
        // insert / delete / context, the rest is highlighted as-is. Any
        // other language renders as context (line number gutter only, no
        // + marker) — treating every code fence as "added diff rows"
        // made casual code examples in chat look like proposed writes,
        // confusing both readers and downstream hallucination detection.
        let (tag, body_content) = if self.lang == "diff" || self.lang == "patch" {
            classify_diff_line(body_line)
        } else {
            (DiffTag::Context, body_line)
        };

        // Pick a syntax highlighter: for diff content, attempt the
        // sub-language if we can infer one; else plain text. For normal
        // fences, use the declared language.
        let syntax = if self.lang == "diff" || self.lang == "patch" {
            self.ps.find_syntax_plain_text()
        } else if self.lang.is_empty() {
            self.ps.find_syntax_plain_text()
        } else {
            self.ps
                .find_syntax_by_token(&self.lang)
                .unwrap_or_else(|| self.ps.find_syntax_plain_text())
        };
        let theme = &self.ts.themes["base16-ocean.dark"];
        let mut h = HighlightLines::new(syntax, theme);
        let raw_body = match h.highlight_line(body_content, &self.ps) {
            Ok(ranges) => as_24_bit_terminal_escaped(&ranges[..], false),
            Err(_) => body_content.to_string(),
        };
        // Syntect emits `\x1b[0m` between ranges; swap for fg-only reset
        // so our row background stays active.
        let body = raw_body.replace("\x1b[0m", "\x1b[39m");

        let (bg, gutter_fg, marker_fg, sign) = match tag {
            DiffTag::Insert => (ADDED_BG, GUTTER_FG_ADD, MARKER_FG_ADD, "+"),
            DiffTag::Delete => (REMOVED_BG, GUTTER_FG_DEL, MARKER_FG_DEL, "-"),
            DiffTag::Context => ("", GUTTER_FG, "\x1b[2;37m", " "),
        };

        if bg.is_empty() {
            // Context row — no bg fill.
            format!(
                "{gutter_fg}{lineno:>4}{RESET} {marker_fg}{sign}{RESET} {body}\x1b[0m{trailing_nl}",
                lineno = self.code_lineno,
            )
        } else {
            format!(
                "{bg}{gutter_fg}{lineno:>4}{FG_RESET} {marker_fg}{sign}{FG_RESET} {body}{CLEAR_EOL}{RESET}{trailing_nl}",
                lineno = self.code_lineno,
            )
        }
    }
}

/// Per-line classification inside a ```diff fence. The first printable
/// char of the line decides insertion / deletion / context; that char is
/// stripped before highlighting so the body itself isn't polluted with
/// the marker.
#[derive(Clone, Copy, Debug)]
enum DiffTag {
    Insert,
    Delete,
    Context,
}

fn classify_diff_line(line: &str) -> (DiffTag, &str) {
    // Standard unified-diff prefixes. We treat unrecognised starts as
    // context so rogue lines don't blow up the render.
    if let Some(rest) = line.strip_prefix('+') {
        (DiffTag::Insert, rest)
    } else if let Some(rest) = line.strip_prefix('-') {
        (DiffTag::Delete, rest)
    } else if let Some(rest) = line.strip_prefix(' ') {
        (DiffTag::Context, rest)
    } else {
        (DiffTag::Context, line)
    }
}

/// Is `line_buf` still possibly the start of a fence line? We hold off on
/// forwarding to the sink while this is true so that `\`\`\`rust\n` at the
/// start of a new line doesn't leak the first ticks before we recognise
/// the fence.
fn could_be_fence_prefix(line_buf: &str) -> bool {
    // Only applies at the start of a line (nothing before the last \n in
    // the cumulative stream). We approximate by checking the current line
    // buffer: if it's whitespace + up to 3 backticks + optional tag chars,
    // it COULD be a fence.
    let s = line_buf.trim_start();
    if s.is_empty() {
        return true;
    }
    // Fence requires at least one backtick at the start.
    let ticks = s.chars().take_while(|c| *c == '`').count();
    if ticks == 0 {
        return false;
    }
    // If we've seen < 3 backticks and nothing else, keep holding.
    if ticks < 3 && ticks == s.chars().count() {
        return true;
    }
    // 3+ backticks followed by a language tag — still developing.
    if ticks >= 3 {
        // Everything after the ticks must be identifier-ish (or whitespace
        // leading to a newline). We don't have the newline yet, so assume
        // it's still a fence candidate.
        let rest = &s[ticks..];
        if rest.is_empty()
            || rest
                .chars()
                .all(|c| c.is_alphanumeric() || c == '_' || c == '-' || c == '+')
        {
            return true;
        }
    }
    false
}

/// Parse a fence line. Returns `Some(lang)` if the trimmed line opens /
/// closes a fence (`lang` empty for an untagged fence), None otherwise.
fn parse_fence(trimmed: &str) -> Option<String> {
    let ticks = trimmed.chars().take_while(|c| *c == '`').count();
    if ticks < 3 {
        return None;
    }
    let rest: String = trimmed
        .chars()
        .skip(ticks)
        .take_while(|c| !c.is_whitespace())
        .collect();
    // Anything AFTER the tag must be whitespace only — fences don't carry
    // arbitrary prose.
    let after_tag: String = trimmed.chars().skip(ticks + rest.chars().count()).collect();
    if !after_tag.chars().all(|c| c.is_whitespace()) {
        return None;
    }
    Some(rest)
}

/// Walk backwards from `at` until we find a UTF-8 char boundary. Used by
/// the streaming filter to slice `buf` at safe byte positions — without
/// this, a multi-byte char straddling the split point would cause a panic
/// on the slice. `str::is_char_boundary` is fast; the loop terminates
/// within 4 iterations since UTF-8 code points are at most 4 bytes.
fn prev_char_boundary(s: &str, at: usize) -> usize {
    let mut cut = at.min(s.len());
    while cut > 0 && !s.is_char_boundary(cut) {
        cut -= 1;
    }
    cut
}

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
            match &self.state {
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
                FilterState::DroppingBlock(close_tag) => {
                    let close_tag = close_tag.clone();
                    if !self.consume_block(&close_tag) {
                        break;
                    }
                    self.state = FilterState::Pending;
                }
                FilterState::Emitting => {
                    // Emit up to the start of the next JSON/template block,
                    // if any. Otherwise drain everything except a small
                    // holdback tail — a chunk boundary might land mid-way
                    // through a restart pattern (buf ends with `{` while
                    // the next chunk starts with `"name":...`); holding the
                    // last bytes back until we see the next chunk keeps
                    // those patterns from leaking.
                    let restart = self.find_restart_point();
                    match restart {
                        Some(idx) if idx > 0 => {
                            let cut = prev_char_boundary(&self.buf, idx);
                            out.push_str(&self.buf[..cut]);
                            self.buf.drain(..cut);
                            self.state = FilterState::Pending;
                        }
                        Some(_) => {
                            // Restart is at 0 — go straight to decision.
                            self.state = FilterState::Pending;
                        }
                        None => {
                            // Holdback = longest possible partial prefix of
                            // any restart pattern. Templates are ≤13 chars;
                            // `{"name"` is 7. 16 covers both with headroom.
                            const HOLDBACK: usize = 16;
                            if self.buf.len() > HOLDBACK {
                                let cut = prev_char_boundary(&self.buf, self.buf.len() - HOLDBACK);
                                out.push_str(&self.buf[..cut]);
                                self.buf.drain(..cut);
                            }
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
        // remainder INCLUDING the holdback tail — no more chunks will
        // arrive, so nothing to recognise. If we were mid-drop, drop it:
        // an unterminated JSON or template token is still noise.
        let out = match self.state {
            FilterState::Emitting => std::mem::take(&mut self.buf),
            FilterState::Pending => {
                let trimmed = self.buf.trim_start();
                let looks_unfinished_unsafe = trimmed.is_empty()
                    || trimmed.starts_with('{')
                    || trimmed.starts_with('[')
                    || TEMPLATE_PREFIXES.iter().any(|p| trimmed.starts_with(p))
                    || BLOCK_TAG_PAIRS
                        .iter()
                        .any(|(open, _)| trimmed.starts_with(open))
                    || ORPHAN_CLOSE_TAGS.iter().any(|t| trimmed.starts_with(t));
                if looks_unfinished_unsafe {
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
        let trimmed = self.buf.trim_start();
        if trimmed.is_empty() {
            return false;
        }
        // Block-level tags (open → drop through matching close).
        for (open, close) in BLOCK_TAG_PAIRS {
            if trimmed.starts_with(open) {
                self.state = FilterState::DroppingBlock((*close).to_string());
                return true;
            }
            if open.starts_with(trimmed) && trimmed.len() < open.len() {
                return false;
            }
        }
        // Orphan close tags — a stray `</tool_response>` with no matching
        // open. Drop it as a unit to avoid leaking the literal text.
        for tag in ORPHAN_CLOSE_TAGS {
            if trimmed.starts_with(tag) {
                // Drain the leading whitespace + tag, back to Pending.
                let skip = self.buf.len() - trimmed.len() + tag.len();
                self.buf.drain(..skip);
                return true;
            }
            if tag.starts_with(trimmed) && trimmed.len() < tag.len() {
                return false;
            }
        }
        // Inline template tokens like <|im_start|>.
        for prefix in TEMPLATE_PREFIXES {
            if trimmed.starts_with(prefix) {
                self.state = FilterState::DroppingTemplate;
                return true;
            }
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
                self.state = FilterState::Emitting;
                return true;
            }
            return false;
        }
        self.state = FilterState::Emitting;
        true
    }

    fn consume_block(&mut self, close_tag: &str) -> bool {
        if let Some(idx) = self.buf.find(close_tag) {
            self.buf.drain(..idx + close_tag.len());
            true
        } else {
            // Hold back the last (close_tag.len() - 1) chars in case the
            // close tag is split across chunks; drop everything before.
            let keep = close_tag.len().saturating_sub(1);
            let drain_to = self.buf.len().saturating_sub(keep);
            if drain_to > 0 {
                let cut = prev_char_boundary(&self.buf, drain_to);
                self.buf.drain(..cut);
            }
            false
        }
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
        let mut record = |idx: usize| {
            earliest = Some(match earliest {
                Some(e) => e.min(idx),
                None => idx,
            });
        };
        if let Some(idx) = self.buf.find("{\"name\"") {
            record(idx);
        }
        for prefix in TEMPLATE_PREFIXES {
            if let Some(idx) = self.buf.find(prefix) {
                record(idx);
            }
        }
        for (open, _) in BLOCK_TAG_PAIRS {
            if let Some(idx) = self.buf.find(open) {
                record(idx);
            }
        }
        for tag in ORPHAN_CLOSE_TAGS {
            if let Some(idx) = self.buf.find(tag) {
                record(idx);
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
        "[content of",  // covers "[Content of the file]", "[Content of the PDF]", etc.
        "[contents of", // plural variant
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
/// Cascade-router classifier. Default is `false` — route to `chat_model`.
///
/// Returns `true` (→ `fast_model`) only on POSITIVE evidence of
/// triviality: a bare interjection, or a short lookup / question with
/// no side-effect verbs. This "safe default" inverts the previous
/// "fast unless we spotted a tool verb" logic, which leaked any
/// unclassified substantive request to a small model that can't
/// reliably tool-call.
pub(crate) fn should_use_fast(input: &str) -> bool {
    let t = input.trim();

    // Signal 1: bare interjection ("ok", "thanks", "hi") — always fast.
    if is_trivial_turn(t) {
        return true;
    }

    // From here on, only route to fast on short positive lookups /
    // questions. Everything else stays on chat_model.

    if t.chars().count() > 120 {
        return false;
    }

    let lower = t.to_lowercase();

    // Any whiff of a side-effect or file-ish token kills the fast path,
    // regardless of length. This is the safety net — if we ever add
    // more routing heuristics above, a stray action verb still forces
    // chat_model.
    const SIDE_EFFECT_MARKERS: &[&str] = &[
        // file / io
        "create ", "write ", "save ", "edit ", "modify ", "delete ",
        "remove ", "rename ", "append ", "mkdir", "touch ", "chmod",
        "chown", "upload", "download", "zip", "extract", "archive",
        // code
        "fix ", "refactor", "debug", "implement", "build", "compile",
        "install", "uninstall", "upgrade", "update ", "lint", "format ",
        "test ", "tests", "benchmark", "profile", "patch ", "diff ",
        "generate", "render ", "convert", "translate ", "parse ",
        "review",
        // shell / web / network
        "run ", "execute", "shell ", "fetch ", "scrape", "curl ",
        "wget", "http", "search web", "look up", "google", "deploy",
        "commit", " push ", " pull ", "merge ", "rebase", "branch",
        "checkout", "clone ", "fork ",
        // content creation
        "website", "webpage", "web page", "html", "css ", "javascript",
        "typescript", "json ", "yaml", "markdown", "script ",
        "readme", "documentation", "pdf", "docx", "xlsx",
        // code markers
        "```", ".rs", ".py", ".ts", ".tsx", ".js", ".jsx", ".go",
        ".java", ".rb", ".cpp", "::", "=>", "->", "()", "{}", "[]",
        "&&", "||",
    ];
    if SIDE_EFFECT_MARKERS.iter().any(|m| lower.contains(m)) {
        return false;
    }
    // Path-like token (`/Users/...`, `src/agent/mod.rs`) → chat_model.
    if t.contains('/') && t.split_whitespace().any(|w| w.contains('/') && w.len() > 3) {
        return false;
    }

    // Positive triviality signals — short, pure lookup / question.
    const TRIVIAL_LEADERS: &[&str] = &[
        "what is ", "what's ", "whats ", "what time", "what day",
        "what year", "what month", "who is ", "who's ", "whos ",
        "who am ", "when is ", "when's ", "where is ", "where's ",
        "why is ", "why's ", "how does ", "how do you ", "how do i say ",
        "how many ", "how much ", "is it ", "are you ", "can you ",
        "do you ", "does ", "did ", "define ", "explain ", "tell me ",
        "remind me ",
    ];
    if TRIVIAL_LEADERS.iter().any(|p| lower.starts_with(p)) {
        return true;
    }

    // Conversational filler / acknowledgement phrases.
    const FILLER: &[&str] = &[
        "how are you", "how's it going", "good morning", "good afternoon",
        "good evening", "good night", "thank you", "thanks for",
        "you there", "got it", "nice one", "hello ", "hey ", "hi there",
    ];
    if FILLER.iter().any(|p| lower.contains(p)) {
        return true;
    }

    // Default: chat_model. Better to over-invest than to break the turn.
    false
}

/// Skip memory recall for interjection-only turns. Recall against "hi" or
/// "ok" almost always surfaces irrelevant past mentions of the same tokens
/// and costs an Ollama embed round-trip. For substantive messages we always
/// recall.
pub(crate) fn is_trivial_turn(input: &str) -> bool {
    let t = input
        .trim()
        .trim_end_matches(|c: char| matches!(c, '.' | '!' | '?' | ',' | ';'));
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
        "hi", "hey", "hello", "yo", "sup", "ok", "okay", "kk", "k", "yes", "yep", "yeah", "yup",
        "sure", "no", "nope", "nah", "thanks", "thx", "ty", "bye", "goodbye", "cya", "cool",
        "nice", "neat",
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
