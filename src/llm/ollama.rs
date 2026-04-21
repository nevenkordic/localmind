//! Minimal Ollama HTTP client — chat (with tool calling), embeddings, health.
//!
//! We speak to a local Ollama server (default http://127.0.0.1:11434) using
//! the native /api/chat and /api/embed endpoints. No telemetry, no analytics,
//! no cloud fallback.

use crate::config::OllamaConfig;
use crate::llm::types::*;
use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Output schema for `extract_entities`. Kept intentionally shallow —
/// callers build the graph from these by calling
/// `Store::upsert_entity` / `upsert_edge`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ExtractedEntity {
    pub name: String,
    #[serde(rename = "type")]
    pub etype: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ExtractedEdge {
    pub src: String,
    pub src_type: String,
    pub dst: String,
    pub dst_type: String,
    pub relation: String,
}

/// First-response watchdog. Bounds how long a `/api/chat` call can sit with
/// no reply before we treat it as a hung / corrupt model and surface an
/// actionable error. Kept as a constant rather than a config knob because
/// the failure it catches (0 bytes after 120s on a local loopback call) is
/// never what anyone wants — users who need longer total generation time
/// should raise `[ollama] timeout_secs` instead, which bounds the reqwest
/// client itself.
const CHAT_WATCHDOG: Duration = Duration::from_secs(120);

#[derive(Clone)]
pub struct OllamaClient {
    http: reqwest::Client,
    host: String,
    chat_model: String,
    fast_model: String,
    embed_model: String,
    vision_model: String,
    num_ctx: u32,
    keep_alive: String,
    temperature: f32,
    top_p: f32,
    /// Embedding cache. Sized for ~768KB at the default 256-entry cap with
    /// 768-dim float vectors. Shared across clones of the client so the
    /// agent loop and CLI commands hit the same warm cache when run in the
    /// same process.
    embed_cache: Arc<Mutex<EmbedCache>>,
}

struct EmbedCache {
    map: HashMap<String, Vec<f32>>,
    order: VecDeque<String>,
    cap: usize,
}

impl EmbedCache {
    fn new(cap: usize) -> Self {
        Self {
            map: HashMap::with_capacity(cap),
            order: VecDeque::with_capacity(cap),
            cap,
        }
    }
    fn get(&self, key: &str) -> Option<Vec<f32>> {
        self.map.get(key).cloned()
    }
    fn put(&mut self, key: String, value: Vec<f32>) {
        if self.map.contains_key(&key) {
            return;
        }
        if self.map.len() >= self.cap {
            if let Some(oldest) = self.order.pop_front() {
                self.map.remove(&oldest);
            }
        }
        self.order.push_back(key.clone());
        self.map.insert(key, value);
    }
    fn len(&self) -> usize {
        self.map.len()
    }
}

impl OllamaClient {
    pub fn new(cfg: &OllamaConfig) -> Self {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(cfg.timeout_secs))
            // Identify ourselves clearly — avoids "mysterious unknown agent" in logs.
            .user_agent(format!(
                "localmind/{} (work-tool)",
                env!("CARGO_PKG_VERSION")
            ))
            .build()
            .expect("reqwest client");
        Self {
            http,
            host: cfg.host.trim_end_matches('/').to_string(),
            chat_model: cfg.chat_model.clone(),
            fast_model: cfg.fast_model.clone(),
            embed_model: cfg.embed_model.clone(),
            vision_model: cfg.vision_model.clone(),
            num_ctx: cfg.num_ctx,
            keep_alive: cfg.keep_alive.clone(),
            temperature: cfg.temperature,
            top_p: cfg.top_p,
            embed_cache: Arc::new(Mutex::new(EmbedCache::new(256))),
        }
    }

    /// Check that Ollama is reachable and list available models.
    pub async fn health(&self) -> Result<Vec<String>> {
        #[derive(Deserialize)]
        struct Tags {
            models: Vec<Model>,
        }
        #[derive(Deserialize)]
        struct Model {
            name: String,
        }
        let url = format!("{}/api/tags", self.host);
        let resp = self.http.get(&url).send().await.context("GET /api/tags")?;
        if !resp.status().is_success() {
            return Err(anyhow!("ollama /api/tags returned {}", resp.status()));
        }
        let tags: Tags = resp.json().await?;
        Ok(tags.models.into_iter().map(|m| m.name).collect())
    }

    /// Non-streaming chat call. Returns the assistant message. Uses the
    /// configured chat / vision model; callers that need per-turn model
    /// routing (cascade fast→capable) should use `chat_on` instead.
    pub async fn chat(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[ToolSpec]>,
        use_vision: bool,
    ) -> Result<ChatMessage> {
        self.chat_on(messages, tools, use_vision, None).await
    }

    /// Non-streaming chat call with an explicit model override. `None` falls
    /// back to `use_vision ? vision_model : chat_model` — same behaviour
    /// as `chat`. Pass `Some(name)` to route this single call to a
    /// different model (e.g. a 3B fast path).
    pub async fn chat_on(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[ToolSpec]>,
        use_vision: bool,
        model_override: Option<&str>,
    ) -> Result<ChatMessage> {
        #[derive(Serialize)]
        struct Req<'a> {
            model: &'a str,
            messages: &'a [ChatMessage],
            stream: bool,
            #[serde(skip_serializing_if = "Option::is_none")]
            tools: Option<&'a [ToolSpec]>,
            options: Options,
            /// Overrides Ollama's default 5-minute model unload window —
            /// see OllamaConfig::keep_alive.
            keep_alive: &'a str,
        }
        #[derive(Serialize)]
        struct Options {
            num_ctx: u32,
            temperature: f32,
            top_p: f32,
        }
        #[derive(Deserialize)]
        struct Resp {
            message: ChatMessage,
            #[serde(default)]
            #[allow(dead_code)]
            done: bool,
        }
        let model: &str = match model_override {
            Some(m) if !m.is_empty() => m,
            _ if use_vision => &self.vision_model,
            _ => &self.chat_model,
        };
        let url = format!("{}/api/chat", self.host);
        let body = Req {
            model,
            messages,
            stream: false,
            tools,
            options: Options {
                num_ctx: self.num_ctx,
                temperature: self.temperature,
                top_p: self.top_p,
            },
            keep_alive: &self.keep_alive,
        };
        // Watchdog. If Ollama goes unresponsive (corrupt model, hung worker),
        // the non-streaming request would otherwise sit on the reqwest
        // timeout (600s by default) — long enough to look like a lockup in
        // the REPL. A 120s bound catches real hangs while still leaving
        // plenty of room for legitimate cold-load + first response on 7B
        // and even 30B-class models on modern hardware.
        let call = async {
            let resp = self
                .http
                .post(&url)
                .json(&body)
                .send()
                .await
                .context("POST /api/chat")?;
            if !resp.status().is_success() {
                let status = resp.status();
                let text = resp.text().await.unwrap_or_default();
                return Err(anyhow!("ollama /api/chat {}: {}", status, text));
            }
            let parsed: Resp = resp.json().await.context("parsing chat response")?;
            Ok::<Resp, anyhow::Error>(parsed)
        };
        let parsed = match tokio::time::timeout(CHAT_WATCHDOG, call).await {
            Ok(r) => r?,
            Err(_) => {
                return Err(anyhow!(
                    "no response from Ollama after {}s — model '{model}' may be hung or corrupt.\n  \
                     Try:\n    ollama ps                       # is a worker stuck?\n    \
                     ollama rm {model} && ollama pull {model}\n  \
                     (Or raise [ollama] timeout_secs if you expect legitimately slow generations.)",
                    CHAT_WATCHDOG.as_secs()
                ));
            }
        };
        Ok(recover_text_tool_calls(parsed.message))
    }

    /// Streaming chat call. Calls `on_token` for each content chunk as it
    /// arrives, then returns the fully-assembled message once Ollama reports
    /// `done: true`. Same request body as `chat` with `stream: true` and
    /// newline-delimited JSON response framing.
    ///
    /// `on_token` runs on the request's tokio task — it must not block.
    /// Expect a few-to-many calls per response; Ollama emits one chunk per
    /// generated token in practice. Tool calls, if any, arrive in the final
    /// chunk and are returned via the `ChatMessage.tool_calls` field — the
    /// callback sees only user-visible content.
    #[allow(dead_code)]
    pub async fn chat_stream<F>(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[ToolSpec]>,
        use_vision: bool,
        on_token: F,
    ) -> Result<ChatMessage>
    where
        F: FnMut(&str),
    {
        self.chat_stream_on(messages, tools, use_vision, None, on_token)
            .await
    }

    /// Streaming chat with an explicit model override. Same semantics as
    /// `chat_stream`, but the caller picks the model (e.g. the cascade
    /// router routing a short turn to `fast_model`).
    pub async fn chat_stream_on<F>(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[ToolSpec]>,
        use_vision: bool,
        model_override: Option<&str>,
        mut on_token: F,
    ) -> Result<ChatMessage>
    where
        F: FnMut(&str),
    {
        use futures::StreamExt;
        #[derive(Serialize)]
        struct Req<'a> {
            model: &'a str,
            messages: &'a [ChatMessage],
            stream: bool,
            #[serde(skip_serializing_if = "Option::is_none")]
            tools: Option<&'a [ToolSpec]>,
            options: Options,
            keep_alive: &'a str,
        }
        #[derive(Serialize)]
        struct Options {
            num_ctx: u32,
            temperature: f32,
            top_p: f32,
        }
        #[derive(Deserialize)]
        struct Chunk {
            #[serde(default)]
            message: Option<ChatMessage>,
            #[serde(default)]
            done: bool,
        }
        let model: &str = match model_override {
            Some(m) if !m.is_empty() => m,
            _ if use_vision => &self.vision_model,
            _ => &self.chat_model,
        };
        let url = format!("{}/api/chat", self.host);
        let body = Req {
            model,
            messages,
            stream: true,
            tools,
            options: Options {
                num_ctx: self.num_ctx,
                temperature: self.temperature,
                top_p: self.top_p,
            },
            keep_alive: &self.keep_alive,
        };

        // Same watchdog as the non-streaming path, but it bounds the TTFB
        // (time to first byte) — once streaming starts, the reqwest-level
        // timeout keeps guard for the rest.
        let resp = match tokio::time::timeout(
            CHAT_WATCHDOG,
            self.http.post(&url).json(&body).send(),
        )
        .await
        {
            Ok(r) => r.context("POST /api/chat")?,
            Err(_) => {
                return Err(anyhow!(
                    "no response from Ollama after {}s — model '{model}' may be hung or corrupt.\n  \
                     Try: ollama ps ; ollama rm {model} && ollama pull {model}",
                    CHAT_WATCHDOG.as_secs()
                ));
            }
        };
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(anyhow!("ollama /api/chat {}: {}", status, text));
        }

        // Ollama emits newline-delimited JSON: one `{"message": {...}, "done": bool}`
        // object per chunk. Buffer bytes and split on '\n' so a chunk split
        // mid-object doesn't get parsed in two halves.
        let mut stream = resp.bytes_stream();
        let mut buf: Vec<u8> = Vec::new();
        let mut aggregated = ChatMessage::assistant(String::new());
        while let Some(chunk) = stream.next().await {
            let bytes = chunk.context("reading chat stream")?;
            buf.extend_from_slice(&bytes);
            // Drain complete lines.
            while let Some(pos) = buf.iter().position(|&b| b == b'\n') {
                let line: Vec<u8> = buf.drain(..=pos).collect();
                let trimmed = &line[..line.len().saturating_sub(1)]; // drop '\n'
                if trimmed.is_empty() {
                    continue;
                }
                let parsed: Chunk = match serde_json::from_slice(trimmed) {
                    Ok(c) => c,
                    // Ollama shouldn't emit partial lines, but tolerate one
                    // so we don't abort the whole stream on a transient glitch.
                    Err(e) => {
                        tracing::debug!("chat stream: unparseable chunk: {e}");
                        continue;
                    }
                };
                if let Some(m) = parsed.message {
                    // Every chunk carries only the newly-generated fragment
                    // of content; concatenate into the running total.
                    if !m.content.is_empty() {
                        on_token(&m.content);
                        aggregated.content.push_str(&m.content);
                    }
                    // Tool calls arrive in the `done: true` chunk; take the
                    // latest non-empty set we see.
                    if let Some(tc) = m.tool_calls {
                        if !tc.is_empty() {
                            aggregated.tool_calls = Some(tc);
                        }
                    }
                }
                if parsed.done {
                    return Ok(recover_text_tool_calls(aggregated));
                }
            }
        }
        // Stream ended without a `done: true` — treat any parsed content as
        // the final reply rather than erroring.
        Ok(recover_text_tool_calls(aggregated))
    }

    /// Generate an embedding for a single text. Cached on (model, text) so
    /// repeated lookups (e.g. user re-runs the same `memory search`) skip
    /// the Ollama round-trip entirely.
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let key = cache_key(&self.embed_model, text);
        if let Some(hit) = self.embed_cache.lock().unwrap().get(&key) {
            return Ok(hit);
        }
        #[derive(Serialize)]
        struct Req<'a> {
            model: &'a str,
            input: &'a str,
            keep_alive: &'a str,
        }
        #[derive(Deserialize)]
        struct Resp {
            embeddings: Vec<Vec<f32>>,
        }
        let url = format!("{}/api/embed", self.host);
        let body = Req {
            model: &self.embed_model,
            input: text,
            keep_alive: &self.keep_alive,
        };
        let resp = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .context("POST /api/embed")?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(anyhow!("ollama /api/embed {}: {}", status, text));
        }
        let parsed: Resp = resp.json().await.context("parsing embed response")?;
        let vec = parsed
            .embeddings
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("ollama returned no embeddings"))?;
        self.embed_cache.lock().unwrap().put(key, vec.clone());
        Ok(vec)
    }

    /// Diagnostic — surfaces cache size for `llm health` if we want to add it.
    #[allow(dead_code)]
    pub fn embed_cache_size(&self) -> usize {
        self.embed_cache.lock().unwrap().len()
    }

    /// Ask the chat model to produce N paraphrasings of a query — used by
    /// hybrid search for recall via query expansion.
    pub async fn query_expand(&self, query: &str, n: usize) -> Result<Vec<String>> {
        if n == 0 {
            return Ok(vec![]);
        }
        let prompt = format!(
            "Rewrite the following search query into {} diverse paraphrasings. \
             Return ONLY a JSON array of strings, no prose, no code fences. \
             Query: {}",
            n, query,
        );
        let msgs = vec![
            ChatMessage::system("You rewrite search queries. Output JSON only."),
            ChatMessage::user(prompt),
        ];
        // Route expansion to fast_model when configured — paraphrase is a
        // tiny task and doesn't need chat_model's depth. Keeps `memory
        // search` off the critical path of a slow reasoning model.
        let model_override = if self.fast_model.is_empty() {
            None
        } else {
            Some(self.fast_model.as_str())
        };
        let reply = self.chat_on(&msgs, None, false, model_override).await?;
        let content = reply
            .content
            .trim()
            .trim_start_matches("```json")
            .trim_end_matches("```")
            .trim();
        match serde_json::from_str::<Vec<String>>(content) {
            Ok(v) => Ok(v.into_iter().take(n).collect()),
            Err(_) => Ok(vec![]),
        }
    }

    /// Summarise a run of past conversation messages into one compact
    /// narrative. Used by the agent's auto-compaction path when the live
    /// message history approaches `num_ctx`. Prompt is tuned to preserve
    /// decisions, tool outcomes, and file paths — the things future turns
    /// usually care about — while dropping redundant narration.
    pub async fn summarize_history(&self, transcript: &str) -> Result<String> {
        let sys = "You compress conversation histories for a long-running coding \
                   assistant. Preserve: user goals, decisions made, file paths \
                   touched, commands run, errors hit, and anything the assistant \
                   promised to do. Drop: chit-chat, partial drafts superseded by \
                   later versions, repeated tool output. Emit a bulleted \
                   summary, no preamble, no code fences. Max 400 words.";
        let user = format!("Conversation history to compress:\n\n{transcript}");
        let msgs = vec![ChatMessage::system(sys), ChatMessage::user(user)];
        let reply = self.chat(&msgs, None, false).await?;
        Ok(reply.content.trim().to_string())
    }

    /// Extract entities + relationships from a stored memory. Used by
    /// the embedding worker to populate `kg_entities` / `kg_edges` so
    /// graph-based retrieval can seed queries by entity. The model is
    /// asked for strict JSON; malformed output returns an empty pair so
    /// extraction is never fatal to the embedding pipeline.
    pub async fn extract_entities(
        &self,
        title: &str,
        content: &str,
    ) -> Result<(Vec<ExtractedEntity>, Vec<ExtractedEdge>)> {
        const INPUT_BUDGET: usize = 2000;
        let excerpt: String = content.chars().take(INPUT_BUDGET).collect();
        let sys = "You extract a compact knowledge graph from a note. Output \
                   ONLY a JSON object with two arrays:\n\
                     \"entities\": [{\"name\": str, \"type\": str}, ...]\n\
                     \"edges\": [{\"src\": str, \"src_type\": str, \"dst\": \
                   str, \"dst_type\": str, \"relation\": str}, ...]\n\
                   Entity types: person | project | file | system | tech | \
                   concept | place | org.\n\
                   Edge relations: uses | owns | depends_on | related_to | \
                   authored | located_in | contains | creates | mentions.\n\
                   Names MUST be canonical (no articles, singular form, \
                   human-readable). At most 8 entities and 12 edges. If no \
                   meaningful entities exist, return empty arrays.\n\
                   No prose, no code fences.";
        let user = format!("Title: {title}\n\nNote:\n{excerpt}");
        let msgs = vec![ChatMessage::system(sys), ChatMessage::user(user)];
        let reply = self.chat(&msgs, None, false).await?;
        let raw = reply
            .content
            .trim()
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim();
        #[derive(Deserialize)]
        struct Parsed {
            #[serde(default)]
            entities: Vec<ExtractedEntity>,
            #[serde(default)]
            edges: Vec<ExtractedEdge>,
        }
        match serde_json::from_str::<Parsed>(raw) {
            Ok(p) => Ok((p.entities, p.edges)),
            Err(_) => Ok((Vec::new(), Vec::new())),
        }
    }

    /// Contextual enrichment before embedding. Ask the chat model for a
    /// one-sentence context line that situates this memory inside the
    /// user's broader history — topic, date, what kind of thing it is.
    /// The line is prepended to the content before embedding so short or
    /// ambiguous notes become findable by semantically adjacent queries.
    ///
    /// Returns an empty string on any failure; callers should fall back
    /// to the raw content in that case.
    pub async fn contextualize(&self, title: &str, content: &str) -> Result<String> {
        // Keep the prompt cheap — small model, terse instruction, small
        // input budget. Aim for ~50-100 tokens of context.
        const INPUT_BUDGET: usize = 1500;
        let excerpt: String = content.chars().take(INPUT_BUDGET).collect();
        let sys = "You produce ONE short sentence (<= 25 words) that situates \
                   the given note: its topic area, what kind of thing it is, \
                   and any entities named. No preamble, no quotes, just the \
                   sentence.";
        let user = format!("Title: {title}\n\nNote:\n{excerpt}");
        let msgs = vec![ChatMessage::system(sys), ChatMessage::user(user)];
        let reply = self.chat(&msgs, None, false).await?;
        Ok(reply.content.trim().trim_matches('"').to_string())
    }
}

// ---------------------------------------------------------------------------
// Text-form tool-call recovery
//
// Some Ollama models (certain coder templates in particular) emit
// tool calls as plain JSON in the message `content` instead of the structured
// `tool_calls` field. This function:
//
//   1. leaves well-formed structured calls untouched;
//   2. strips ```json fences and a leading <tool_call> wrapper;
//   3. scans the content for one or more `{"name": ..., "arguments": {...}}`
//      objects (either concatenated or as a JSON array) and promotes them to
//      real `tool_calls` entries;
//   4. on a successful promotion, clears the content so the model's duplicate
//      text isn't echoed to the user.
//
// The recovery is intentionally conservative — if *any* part of the content
// outside the recognised tool-call region is non-whitespace prose, we leave
// the original message alone and let the agent loop treat it as a final
// assistant answer.
// ---------------------------------------------------------------------------

fn recover_text_tool_calls(mut msg: ChatMessage) -> ChatMessage {
    if msg
        .tool_calls
        .as_ref()
        .map(|v| !v.is_empty())
        .unwrap_or(false)
    {
        return msg;
    }
    let raw = msg.content.trim();
    if raw.is_empty() {
        return msg;
    }
    // Strip fences and common wrapper tags.
    let cleaned = strip_wrappers(raw);
    let cleaned = cleaned.trim();
    if cleaned.is_empty() {
        return msg;
    }

    let calls = extract_tool_calls(cleaned);
    if !calls.is_empty() {
        msg.tool_calls = Some(calls);
        msg.content.clear();
    }
    msg
}

fn strip_wrappers(s: &str) -> String {
    let mut s = s.trim().to_string();
    // ```json ... ``` or ``` ... ```
    if let Some(rest) = s.strip_prefix("```json") {
        s = rest.trim_start().to_string();
    } else if let Some(rest) = s.strip_prefix("```") {
        s = rest.trim_start().to_string();
    }
    if let Some(idx) = s.rfind("```") {
        s.truncate(idx);
    }
    // <tool_call> ... </tool_call>
    if let Some(rest) = s.trim_start().strip_prefix("<tool_call>") {
        s = rest.to_string();
    }
    if let Some(idx) = s.rfind("</tool_call>") {
        s.truncate(idx);
    }
    s
}

fn extract_tool_calls(s: &str) -> Vec<ToolCall> {
    let trimmed = s.trim();
    // Try 1: the whole body is a JSON array of calls.
    if let Ok(v) = serde_json::from_str::<Vec<ToolFunctionCall>>(trimmed) {
        return v.into_iter().map(wrap_tool_call).collect();
    }
    // Try 2: the whole body is a single call object.
    if let Some(tc) = try_parse_tool_call(trimmed) {
        return vec![tc];
    }
    // Try 3: scan the full string for embedded tool calls. Walks every `{`,
    // finds its matching `}`, and checks whether the slice parses as a tool
    // call (must have a non-empty `name` and an `arguments` field). This
    // handles models that prefix / interleave tool calls with chat prose, or
    // wrap them with tags we didn't strip. Non-matching braces are skipped.
    let mut out = Vec::new();
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'{' {
            i += 1;
            continue;
        }
        let end = match find_matching_brace(bytes, i) {
            Some(e) => e,
            None => break,
        };
        let slice = &s[i..=end];
        if let Some(tc) = try_parse_tool_call(slice) {
            out.push(tc);
            i = end + 1;
        } else {
            i += 1;
        }
    }

    // Try 4: function-call syntax — `name key="value" key=value ...`.
    // Some Ollama chat templates emit calls in this shape instead of JSON.
    // Only fires if no JSON form was recovered above. The line must be ENTIRELY
    // a call (identifier + at least one `key=value` pair, nothing else) so
    // ordinary prose doesn't get mistaken for a tool invocation.
    if out.is_empty() {
        for line in s.lines() {
            if let Some(tc) = try_parse_call_syntax(line) {
                out.push(tc);
            }
        }
    }

    // Try 5: loose CLI syntax — `<read_only_tool> <free text>`. Catches
    // "thinking out loud" like `search_memory what do you know about the user`.
    // Restricted to read-only tools with an unambiguous primary string arg so
    // the recovery can't trigger destructive actions (shell, write_file, etc.).
    if out.is_empty() {
        for line in s.lines() {
            if let Some(tc) = try_parse_loose_call(line) {
                out.push(tc);
            }
        }
    }
    out
}

/// Parse a single line shaped like `tool_name key="value" key=value ...`.
/// Returns None unless the whole line consists of an identifier followed by
/// one or more `key=value` tokens — bare identifiers, trailing prose, or any
/// token without `=` cause a refusal so we don't fabricate calls from prose.
fn try_parse_call_syntax(line: &str) -> Option<ToolCall> {
    let line = line.trim();
    if line.is_empty() {
        return None;
    }
    // Identifier: ASCII letters / digits / underscore, must start with a letter
    // or underscore (digits-first would be a number, not a tool name).
    let bytes = line.as_bytes();
    let mut i = 0;
    let first = *bytes.first()?;
    if !(first.is_ascii_alphabetic() || first == b'_') {
        return None;
    }
    while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
        i += 1;
    }
    let name = &line[..i];
    if name.len() < 2 {
        return None;
    }
    // Must be followed by at least one whitespace separator.
    if i >= bytes.len() || !bytes[i].is_ascii_whitespace() {
        return None;
    }
    let rest = line[i..].trim_start();
    if rest.is_empty() {
        return None;
    }
    let args = parse_call_args(rest)?;
    if args.is_empty() {
        return None;
    }
    Some(ToolCall {
        id: None,
        r#type: "function".into(),
        function: ToolFunctionCall {
            name: name.to_string(),
            arguments: serde_json::Value::Object(args),
        },
    })
}

/// Parse `key=value key="value with spaces" key=true ...` into a JSON object.
/// Returns None if any token is malformed (so the caller can fall back to
/// treating the line as ordinary prose).
fn parse_call_args(s: &str) -> Option<serde_json::Map<String, serde_json::Value>> {
    let mut args = serde_json::Map::new();
    let bytes = s.as_bytes();
    let n = bytes.len();
    let mut i = 0;
    loop {
        while i < n && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= n {
            break;
        }
        // Key: identifier.
        let key_start = i;
        if !(bytes[i].is_ascii_alphabetic() || bytes[i] == b'_') {
            return None;
        }
        while i < n && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
            i += 1;
        }
        let key = &s[key_start..i];
        if i >= n || bytes[i] != b'=' {
            return None;
        }
        i += 1;
        if i >= n {
            return None;
        }
        // Value: quoted string, JSON array/object, or bareword.
        let value = if bytes[i] == b'"' || bytes[i] == b'\'' {
            let quote = bytes[i];
            i += 1;
            let mut out = String::new();
            let mut chunk_start = i;
            while i < n && bytes[i] != quote {
                if bytes[i] == b'\\' && i + 1 < n {
                    if i > chunk_start {
                        out.push_str(&s[chunk_start..i]);
                    }
                    let real = match bytes[i + 1] {
                        b'"' => '"',
                        b'\'' => '\'',
                        b'\\' => '\\',
                        b'n' => '\n',
                        b't' => '\t',
                        b'r' => '\r',
                        c => c as char,
                    };
                    out.push(real);
                    i += 2;
                    chunk_start = i;
                } else {
                    i += 1;
                }
            }
            if i >= n {
                return None; // unterminated string
            }
            if i > chunk_start {
                out.push_str(&s[chunk_start..i]);
            }
            i += 1; // closing quote
            serde_json::Value::String(out)
        } else if bytes[i] == b'[' || bytes[i] == b'{' {
            // JSON array or object value, e.g. tags=["a","b"] or meta={"k":1}.
            let close = if bytes[i] == b'[' { b']' } else { b'}' };
            let end = find_matching_bracket(bytes, i, bytes[i], close)?;
            let slice = &s[i..=end];
            let v: serde_json::Value = serde_json::from_str(slice).ok()?;
            i = end + 1;
            v
        } else {
            let v_start = i;
            while i < n && !bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            parse_bare_value(&s[v_start..i])
        };
        args.insert(key.to_string(), value);
    }
    Some(args)
}

/// Read-only tools whose primary parameter is a single string. When a
/// model emits `<tool> <free text>` (no `=`, no JSON braces) we can
/// safely promote the remainder into this parameter without risking a
/// destructive action.
const LOOSE_CALL_TOOLS: &[(&str, &str)] = &[
    ("search_memory", "query"),
    ("read_file", "path"),
    ("read_pdf", "path"),
    ("read_docx", "path"),
    ("read_xlsx", "path"),
    ("read_image", "path"),
    ("list_dir", "path"),
    ("dns_lookup", "name"),
    ("web_search", "query"),
    ("web_fetch", "url"),
    ("whois", "query"),
];

/// Loose recovery: `<tool> <rest of line>` where `<tool>` is in the whitelist
/// above. The whole remainder becomes the value of the tool's primary string
/// arg. Refuses lines containing `=`, `{`, or `[` so the structured / JSON
/// parsers above always win.
fn try_parse_loose_call(line: &str) -> Option<ToolCall> {
    let line = line.trim();
    if line.is_empty() {
        return None;
    }
    let mut parts = line.splitn(2, char::is_whitespace);
    let name = parts.next()?;
    let rest = parts.next()?.trim();
    if rest.is_empty() {
        return None;
    }
    if rest.contains('=') || rest.starts_with('{') || rest.starts_with('[') {
        return None;
    }
    let arg_name = LOOSE_CALL_TOOLS
        .iter()
        .find_map(|(t, a)| if *t == name { Some(*a) } else { None })?;
    let mut args = serde_json::Map::new();
    args.insert(
        arg_name.to_string(),
        serde_json::Value::String(rest.to_string()),
    );
    Some(ToolCall {
        id: None,
        r#type: "function".into(),
        function: ToolFunctionCall {
            name: name.to_string(),
            arguments: serde_json::Value::Object(args),
        },
    })
}

fn parse_bare_value(s: &str) -> serde_json::Value {
    match s {
        "true" => serde_json::Value::Bool(true),
        "false" => serde_json::Value::Bool(false),
        "null" => serde_json::Value::Null,
        _ => {
            if let Ok(n) = s.parse::<i64>() {
                return serde_json::Value::from(n);
            }
            if let Ok(f) = s.parse::<f64>() {
                return serde_json::Value::from(f);
            }
            serde_json::Value::String(s.to_string())
        }
    }
}

/// Parse a single JSON object as a tool call. Requires a non-empty `name`
/// and an `arguments` field (object, array, or null). Returns None for
/// anything else so random `{...}` fragments in prose don't become spurious
/// tool calls.
fn try_parse_tool_call(slice: &str) -> Option<ToolCall> {
    let v: serde_json::Value = serde_json::from_str(slice).ok()?;
    let obj = v.as_object()?;
    let name = obj.get("name")?.as_str()?;
    if name.is_empty() {
        return None;
    }
    let args = obj.get("arguments")?.clone();
    if !(args.is_object() || args.is_array() || args.is_null()) {
        return None;
    }
    Some(ToolCall {
        id: None,
        r#type: "function".into(),
        function: ToolFunctionCall {
            name: name.to_string(),
            arguments: args,
        },
    })
}

fn cache_key(model: &str, text: &str) -> String {
    // `\u{1F}` (unit separator) — can't appear in a model name or normal text,
    // so we don't need to worry about colliding keys.
    format!("{model}\u{1F}{text}")
}

fn wrap_tool_call(f: ToolFunctionCall) -> ToolCall {
    ToolCall {
        id: None,
        r#type: "function".into(),
        function: f,
    }
}

/// Given `s` and the index of an opening `{`, return the index of the
/// matching closing `}`. Respects string literals and escape sequences so
/// braces inside JSON string values don't throw off the counter.
fn find_matching_brace(s: &[u8], start: usize) -> Option<usize> {
    find_matching_bracket(s, start, b'{', b'}')
}

/// Generalised bracket matcher — used for both `{...}` and `[...]` so the
/// call-syntax parser can pull JSON arrays out of values.
fn find_matching_bracket(s: &[u8], start: usize, open: u8, close: u8) -> Option<usize> {
    if s.get(start) != Some(&open) {
        return None;
    }
    let mut depth = 0i32;
    let mut in_str = false;
    let mut escape = false;
    for (i, &c) in s.iter().enumerate().skip(start) {
        if in_str {
            if escape {
                escape = false;
            } else if c == b'\\' {
                escape = true;
            } else if c == b'"' {
                in_str = false;
            }
            continue;
        }
        if c == b'"' {
            in_str = true;
        } else if c == open {
            depth += 1;
        } else if c == close {
            depth -= 1;
            if depth == 0 {
                return Some(i);
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(content: &str) -> ChatMessage {
        ChatMessage {
            role: "assistant".into(),
            content: content.into(),
            images: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    #[test]
    fn recovers_single_object() {
        let m = recover_text_tool_calls(msg(
            r#"{"name": "dns_lookup", "arguments": {"name": "github.com"}}"#,
        ));
        let calls = m.tool_calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "dns_lookup");
        assert!(m.content.is_empty());
    }

    #[test]
    fn recovers_two_concat_objects() {
        let m = recover_text_tool_calls(msg(
            r#"{"name": "dns_lookup", "arguments": {"name": "github.com"}}
{"name": "listening_ports", "arguments": {"protocol": "tcp"}}"#,
        ));
        let calls = m.tool_calls.unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "dns_lookup");
        assert_eq!(calls[1].function.name, "listening_ports");
    }

    #[test]
    fn recovers_json_array() {
        let m = recover_text_tool_calls(msg(
            r#"[{"name":"a","arguments":{}},{"name":"b","arguments":{}}]"#,
        ));
        assert_eq!(m.tool_calls.unwrap().len(), 2);
    }

    #[test]
    fn strips_code_fences() {
        let m = recover_text_tool_calls(msg(
            "```json\n{\"name\":\"dns_lookup\",\"arguments\":{\"name\":\"x\"}}\n```",
        ));
        assert_eq!(m.tool_calls.unwrap()[0].function.name, "dns_lookup");
    }

    #[test]
    fn leaves_prose_alone() {
        let m = recover_text_tool_calls(msg("Hello, I can help with that. No tool needed."));
        assert!(m.tool_calls.is_none());
        assert!(!m.content.is_empty());
    }

    #[test]
    fn leaves_structured_calls_alone() {
        let mut m = msg("");
        m.tool_calls = Some(vec![wrap_tool_call(ToolFunctionCall {
            name: "dns_lookup".into(),
            arguments: serde_json::json!({"name": "x"}),
        })]);
        let out = recover_text_tool_calls(m);
        assert_eq!(out.tool_calls.unwrap().len(), 1);
    }

    #[test]
    fn handles_nested_braces_in_args() {
        let m = recover_text_tool_calls(msg(
            r#"{"name":"x","arguments":{"nested":{"a":1,"b":"}"}}}"#,
        ));
        assert_eq!(m.tool_calls.unwrap()[0].function.name, "x");
    }

    #[test]
    fn recovers_call_after_prose() {
        // Mirrors the real failure: the model says "Sure, I'll create..."
        // then emits the tool call.
        let m = recover_text_tool_calls(msg(
            r#"Sure, I'll create a zip archive of the specified directory.
{"name": "zip_create", "arguments": {"archive_path":"/tmp/src.zip","inputs":["/Users/neven/workbuddy/localmind/src"]}}"#,
        ));
        let calls = m.tool_calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "zip_create");
        assert!(m.content.is_empty());
    }

    #[test]
    fn ignores_random_braces_in_prose() {
        // A model reply that mentions `{foo}` or `{}` without a real call
        // should NOT spawn a spurious tool call.
        let m =
            recover_text_tool_calls(msg("I think the answer is {42}. Nothing more to do here."));
        assert!(m.tool_calls.is_none());
        assert!(!m.content.is_empty());
    }

    #[test]
    fn rejects_object_without_arguments() {
        // Just {"name": "x"} without `arguments` should not be treated as a
        // tool call.
        let m = recover_text_tool_calls(msg(r#"My name is {"name": "bob"}. Done."#));
        assert!(m.tool_calls.is_none());
    }

    // -----------------------------------------------------------------------
    // Function-call syntax fallback — `name key="value" key=value`. Some
    // Ollama chat templates emit tool calls in this shape rather than JSON.
    // -----------------------------------------------------------------------

    #[test]
    fn recovers_call_syntax_quoted_and_int() {
        // The exact failure from the user's session.
        let m = recover_text_tool_calls(msg(r#"search_memory query="project details" top_k=5"#));
        let calls = m.tool_calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "search_memory");
        let args = calls[0].function.arguments.as_object().unwrap();
        assert_eq!(args.get("query").unwrap(), "project details");
        assert_eq!(args.get("top_k").unwrap(), 5);
        assert!(m.content.is_empty());
    }

    #[test]
    fn recovers_call_syntax_bool_and_float() {
        let m = recover_text_tool_calls(msg(
            r#"port_check host=example.com port=443 verbose=true ratio=0.5"#,
        ));
        let args = m.tool_calls.unwrap()[0].function.arguments.clone();
        let obj = args.as_object().unwrap();
        assert_eq!(obj.get("host").unwrap(), "example.com");
        assert_eq!(obj.get("port").unwrap(), 443);
        assert_eq!(obj.get("verbose").unwrap(), true);
        assert_eq!(obj.get("ratio").unwrap().as_f64().unwrap(), 0.5);
    }

    #[test]
    fn call_syntax_handles_escaped_quote() {
        let m = recover_text_tool_calls(msg(r#"web_search query="say \"hi\" to me""#));
        let args = m.tool_calls.unwrap()[0].function.arguments.clone();
        assert_eq!(args.get("query").unwrap(), "say \"hi\" to me");
    }

    #[test]
    fn call_syntax_skipped_for_prose() {
        // Bare prose with no `=` tokens — must not become a tool call.
        let m = recover_text_tool_calls(msg("I will use search_memory next to look that up."));
        assert!(m.tool_calls.is_none());
        assert!(!m.content.is_empty());
    }

    #[test]
    fn call_syntax_skipped_for_trailing_prose() {
        // Trailing prose after the args breaks the strict parse — refused so
        // we don't half-recover a malformed line.
        let m = recover_text_tool_calls(msg(r#"search_memory query="x" then summarise"#));
        assert!(m.tool_calls.is_none());
    }

    #[test]
    fn call_syntax_skipped_for_bare_identifier() {
        // `search_memory` alone (no args) should NOT match — too easy to
        // confuse with prose like a sentence-starting word.
        let m = recover_text_tool_calls(msg("search_memory"));
        assert!(m.tool_calls.is_none());
    }

    #[test]
    fn call_syntax_handles_json_array_value() {
        // The exact shape that was leaking — the model emits `tags=["a","b"]`.
        let m = recover_text_tool_calls(msg(
            r#"store_memory title="X" content="Y" kind=preference importance=0.9 tags=["creator", "user-identity"]"#,
        ));
        let calls = m.tool_calls.unwrap();
        assert_eq!(calls.len(), 1);
        let args = calls[0].function.arguments.as_object().unwrap();
        assert_eq!(args.get("kind").unwrap(), "preference");
        let tags = args.get("tags").unwrap().as_array().unwrap();
        assert_eq!(tags.len(), 2);
        assert_eq!(tags[0], "creator");
        assert_eq!(tags[1], "user-identity");
    }

    #[test]
    fn call_syntax_handles_json_object_value() {
        let m = recover_text_tool_calls(msg(
            r#"kg_link src_name="A" src_type=person dst_name="B" dst_type=project relation=owns meta={"weight": 0.8, "note": "hi"}"#,
        ));
        let args = m.tool_calls.unwrap()[0].function.arguments.clone();
        let meta = args.get("meta").unwrap().as_object().unwrap();
        assert_eq!(meta.get("weight").unwrap().as_f64().unwrap(), 0.8);
        assert_eq!(meta.get("note").unwrap(), "hi");
    }

    // -----------------------------------------------------------------------
    // Loose CLI syntax — `<read_only_tool> <free text>`. Catches the
    // "thinking out loud" failure mode where the model names a tool and
    // then a natural-language argument with no `=`.
    // -----------------------------------------------------------------------

    #[test]
    fn loose_call_recovers_search_memory() {
        let m = recover_text_tool_calls(msg("search_memory what do you know about the user"));
        let calls = m.tool_calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "search_memory");
        assert_eq!(
            calls[0].function.arguments.get("query").unwrap(),
            "what do you know about the user"
        );
        assert!(m.content.is_empty());
    }

    #[test]
    fn loose_call_recovers_read_file() {
        let m = recover_text_tool_calls(msg("read_file /etc/hosts"));
        let calls = m.tool_calls.unwrap();
        assert_eq!(calls[0].function.name, "read_file");
        assert_eq!(
            calls[0].function.arguments.get("path").unwrap(),
            "/etc/hosts"
        );
    }

    #[test]
    fn loose_call_skipped_for_destructive_tool() {
        // `shell` is not in the whitelist — must not become a loose call,
        // even though `rm -rf /tmp/x` parses cleanly.
        let m = recover_text_tool_calls(msg("shell rm -rf /tmp/x"));
        assert!(m.tool_calls.is_none());
        assert!(!m.content.is_empty());
    }

    #[test]
    fn loose_call_skipped_when_structured_form_matches() {
        // Quoted form should win over loose recovery. `search_memory query="x"`
        // is the canonical key=value form; loose path must not also fire.
        let m = recover_text_tool_calls(msg(r#"search_memory query="hi""#));
        let calls = m.tool_calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.arguments.get("query").unwrap(), "hi");
    }

    #[test]
    fn loose_call_skipped_for_unknown_tool() {
        let m = recover_text_tool_calls(msg("nonexistent_tool blah blah blah"));
        assert!(m.tool_calls.is_none());
    }

    #[test]
    fn call_syntax_recovered_after_prose_line() {
        // Mirrors `recovers_call_after_prose` but for the non-JSON shape.
        let m = recover_text_tool_calls(msg(
            "I'll search memory for relevant context.\nsearch_memory query=\"context\" top_k=3",
        ));
        let calls = m.tool_calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "search_memory");
    }
}
