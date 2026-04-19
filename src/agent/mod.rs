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
        })
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
        self.loop_tools().await
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

    async fn loop_tools(&mut self) -> Result<String> {
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
                // the first token tick can consume and drop it.
                let spinner_slot = std::sync::Arc::new(std::sync::Mutex::new(Some(spinner)));
                let slot_clone = spinner_slot.clone();
                let cb = move |chunk: &str| {
                    if let Some(sp) = slot_clone.lock().unwrap().take() {
                        sp.stop_sync();
                    }
                    sink(chunk);
                };
                let r = self
                    .client
                    .chat_stream(&self.messages, Some(&specs), false, cb)
                    .await;
                // If the stream ended without ever calling the callback
                // (error before the first token) the spinner is still
                // ticking — drop it now.
                if let Some(sp) = spinner_slot.lock().unwrap().take() {
                    sp.stop_sync();
                }
                r
            } else {
                let r = self.client.chat(&self.messages, Some(&specs), false).await;
                spinner.stop().await;
                r
            };

            let reply = reply_res?;
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
    use super::{extract_facts, is_trivial_turn};

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
