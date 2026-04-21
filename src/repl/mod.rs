//! Interactive REPL.
//!
//! Multiline input, command history persisted per-user (mode 0600 on Unix),
//! slash commands for common operations, graceful Ctrl-C.

use crate::agent::AgentRun;
use crate::config::Config;
use crate::memory::Store;
use crate::tools::permissions::PermissionMode;
use anyhow::Result;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use std::sync::Arc;

pub async fn run(
    cfg: Config,
    store: Store,
    _resume: Option<String>,
    mode_override: Option<PermissionMode>,
) -> Result<()> {
    banner(&cfg, mode_override);
    let mut agent = AgentRun::new_with_mode(
        Arc::new(cfg.clone()),
        store.clone(),
        mode_override,
        false,
        !cfg.web.brave_api_key.is_empty(),
        true,
    )?;

    // Session persistence: keep one session per cwd. If a recent session
    // exists (within the last week), resume its last N messages for
    // short-term continuity. N is kept small by default because large
    // tails of prior-session assistant style can anchor the model on
    // patterns unrelated to the user's next turn.
    const SESSION_MAX_AGE_SECS: i64 = 7 * 86400;
    let resume_limit = cfg.ollama.resume_messages;
    let cwd = std::env::current_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_default();
    if let Ok((session_id, created)) = store
        .session_get_or_create(&cwd, SESSION_MAX_AGE_SECS)
        .await
    {
        agent.set_session(session_id.clone());
        if !created && resume_limit > 0 {
            if let Ok(rows) = store.load_recent_messages(&session_id, resume_limit).await {
                if !rows.is_empty() {
                    let n = rows.len();
                    let restored: Vec<_> = rows
                        .into_iter()
                        .map(|(role, content, tool_name, extras_json)| {
                            AgentRun::message_from_persisted(role, content, tool_name, extras_json)
                        })
                        .collect();
                    agent.restore_messages(restored);
                    eprintln!(
                        "  \x1b[2m· resumed session ({n} messages from last run — type /new to start fresh)\x1b[0m"
                    );
                }
            }
        }
    }

    // Stream tokens to stdout as Ollama generates them so the reply
    // materialises in real time instead of dropping at the end in a wall.
    // Stderr keeps the spinner + tool-step indicators so piping the REPL
    // (`llm | tee`) still captures just the model output.
    agent.set_token_sink(std::sync::Arc::new(|chunk: &str| {
        use std::io::Write;
        let mut out = std::io::stdout().lock();
        let _ = out.write_all(chunk.as_bytes());
        let _ = out.flush();
    }));

    let hist_path = history_path();
    let mut rl = DefaultEditor::new()?;
    if let Some(p) = &hist_path {
        let _ = rl.load_history(p);
    }

    // Remember the last user turn so `/retry-big` can re-run it on the
    // capable model after the router routed it to the fast one.
    let mut last_user_turn: Option<String> = None;

    loop {
        let line = rl.readline("localmind> ");
        match line {
            Ok(l) => {
                let l = l.trim();
                if l.is_empty() {
                    continue;
                }
                let _ = rl.add_history_entry(l);

                // Slash-command dispatch, with a guard against file-path
                // inputs like `/Users/neven/Desktop/...`. If the first
                // token contains another `/` after the leading one, it's
                // a path — send it through as normal user input instead
                // of printing a confusing "unknown command" error.
                if l.starts_with('/') && !looks_like_path(l) {
                    match handle_slash(l, &cfg, &mut agent, last_user_turn.as_deref()).await {
                        SlashAction::Continue => {}
                        SlashAction::Quit => break,
                        SlashAction::RunTurn(input) => {
                            last_user_turn = Some(input.clone());
                            run_turn(&mut agent, &input).await;
                        }
                    }
                    continue;
                }

                last_user_turn = Some(l.to_string());
                run_turn(&mut agent, l).await;
            }
            Err(ReadlineError::Interrupted) => {
                eprintln!("(^C — type /quit to exit)");
            }
            Err(ReadlineError::Eof) => break,
            Err(e) => {
                eprintln!("readline: {e}");
                break;
            }
        }
    }
    if let Some(p) = &hist_path {
        let _ = rl.save_history(p);
        // The history file may capture sensitive REPL input (`/remember <secret>`,
        // pasted credentials, etc.). Lock it to user-only on Unix; Windows uses
        // ACL inheritance from the user's profile dir which already restricts.
        chmod_user_only(p);
    }
    Ok(())
}

#[cfg(unix)]
fn chmod_user_only(p: &std::path::Path) {
    use std::os::unix::fs::PermissionsExt;
    let _ = std::fs::set_permissions(p, std::fs::Permissions::from_mode(0o600));
}

#[cfg(not(unix))]
fn chmod_user_only(_p: &std::path::Path) {}

/// Heuristic: is `line` a filesystem path starting with `/` rather than
/// a slash command? True when the first whitespace-delimited token
/// contains a second `/` — real slash commands are alphanumeric-only
/// after the leading slash (`/help`, `/retry-big`, etc.), so `/Users/…`
/// is unambiguously a path.
fn looks_like_path(line: &str) -> bool {
    let first = line.split_whitespace().next().unwrap_or("");
    // `first` is at least `/` given the caller guarded on starts_with('/').
    first[1..].contains('/')
}

enum SlashAction {
    /// Slash command handled; keep the REPL running.
    Continue,
    /// User asked to quit.
    Quit,
    /// Slash command resolved into a full turn the REPL should run (e.g.
    /// `/retry-big` replays the last input on the capable model).
    RunTurn(String),
}

async fn run_turn(agent: &mut AgentRun, input: &str) {
    println!();
    // Snapshot is_streaming BEFORE the turn borrow — we can't read it while
    // turn_fut holds `&mut agent` across the await.
    let streaming = agent.is_streaming();
    // Race the turn against Ctrl-C. Dropping the turn future cancels the
    // in-flight reqwest call to Ollama — the server sees the dropped
    // connection and stops generating. Partial messages already pushed to
    // `agent.messages` stay in history so context isn't lost; the user
    // just didn't get a final reply this round.
    // Run the turn in an inner scope so its mutable borrow on `agent` is
    // released before we try to flush the message delta to the store.
    let (outcome, interrupted) = {
        let turn_fut = agent.turn(input);
        tokio::pin!(turn_fut);
        let ctrl_c = tokio::signal::ctrl_c();
        tokio::pin!(ctrl_c);
        tokio::select! {
            r = &mut turn_fut => (Some(r), false),
            _ = &mut ctrl_c => (None, true),
        }
    };

    match outcome {
        Some(Ok(reply)) => {
            if !streaming {
                println!("{reply}");
            }
            println!();
            println!();
        }
        Some(Err(e)) => eprintln!("! {e}"),
        None => {
            eprintln!();
            eprintln!("\x1b[1;33m(interrupted — press enter for a new prompt)\x1b[0m");
        }
    }

    // Flush the delta regardless of outcome — a half-finished turn still
    // has partial messages on agent.messages that should survive a
    // relaunch. No-op when no session is bound.
    let _ = interrupted;
    agent.persist_new_messages().await;
}

async fn handle_slash(
    line: &str,
    cfg: &Config,
    agent: &mut AgentRun,
    last_user_turn: Option<&str>,
) -> SlashAction {
    let mut parts = line[1..].split_whitespace();
    let cmd = parts.next().unwrap_or("");
    match cmd {
        "quit" | "exit" | "q" => SlashAction::Quit,
        "retry-big" | "big" | "escalate" => {
            let Some(last) = last_user_turn else {
                eprintln!("(no previous turn to retry)");
                return SlashAction::Continue;
            };
            let chat = cfg.ollama.chat_model.clone();
            eprintln!("\x1b[2m(retrying on {chat})\x1b[0m");
            agent.force_next_model(chat);
            return SlashAction::RunTurn(last.to_string());
        }
        "fast" => {
            // `/fast <message>` forces the fast_model for this one turn.
            let fast = cfg.ollama.fast_model.trim();
            if fast.is_empty() {
                eprintln!("(no [ollama].fast_model configured — edit config/local.toml)");
                return SlashAction::Continue;
            }
            let rest = parts.collect::<Vec<_>>().join(" ");
            if rest.is_empty() {
                eprintln!("usage: /fast <message>");
                return SlashAction::Continue;
            }
            agent.force_next_model(fast.to_string());
            return SlashAction::RunTurn(rest);
        }
        "help" | "?" => {
            println!("Slash commands:");
            println!("  /help                - this message");
            println!("  /quit                - exit the REPL");
            println!("  /init                - scan the current directory and store a project profile in memory");
            println!("  /stats               - memory database stats");
            println!("  /health              - DB + embedder outbox + Ollama reachability");
            println!(
                "  /model               - show current chat/vision/embed models (use `llm models` to change)"
            );
            println!(
                "  /skills              - list everything you've taught the agent (kind=skill)"
            );
            println!(
                "  /forget <id-prefix>  - delete a memory by id prefix (8 chars usually enough)"
            );
            println!("  /remember <fact>     - explicitly store a fact (when the model didn't)");
            println!(
                "  /recall <query>      - preview what the memory primer would surface (debug)"
            );
            println!(
                "  /context             - show the message log being sent to the model (debug)"
            );
            println!("  /audit               - print path to audit log");
            println!("  /config              - show effective configuration");
            println!("  /tools               - list advertised tools");
            println!("  /mode                - show the current permission mode");
            println!("  /mode <ro|ww|full>   - change the permission mode");
            println!("  /fast <message>      - force this turn on [ollama].fast_model");
            println!("  /retry-big           - re-run the last turn on [ollama].chat_model");
            SlashAction::Continue
        }
        "init" => {
            if let Err(e) = run_init(agent).await {
                eprintln!("! /init failed: {e}");
            }
            SlashAction::Continue
        }
        "compact" => {
            if let Err(e) = agent.compact_now(true).await {
                eprintln!("! /compact failed: {e}");
            }
            SlashAction::Continue
        }
        "new" | "clear" | "reset" => {
            // Wipe the persisted session for this cwd AND the in-memory
            // history, leaving only a fresh system prompt. Next turn
            // starts clean — useful when the model has drifted or the
            // user is switching topics.
            agent.reset_to_fresh_session();
            println!("· new session — history cleared");
            SlashAction::Continue
        }
        "mode" => {
            match parts.next() {
                None => {
                    println!("mode: {}", agent.ctx.permissions.mode().as_str());
                }
                Some(v) => match PermissionMode::parse(v) {
                    Some(m) => {
                        agent.ctx.permissions.set_mode(m);
                        println!("mode: {}", m.as_str());
                    }
                    None => eprintln!(
                        "unknown mode '{v}'. Use: read-only | workspace-write | unrestricted"
                    ),
                },
            }
            SlashAction::Continue
        }
        "stats" => {
            match agent.ctx.store.stats().await {
                Ok(s) => println!(
                    "memories={} vectors={} entities={} edges={} pending={} db_size={}",
                    s.memory_count,
                    s.vector_count,
                    s.entity_count,
                    s.edge_count,
                    s.outbox_pending,
                    s.db_size
                ),
                Err(e) => eprintln!("{e}"),
            }
            SlashAction::Continue
        }
        "health" => {
            print!("{}", crate::health::report(cfg, &agent.ctx.store).await);
            SlashAction::Continue
        }
        "model" | "models" => {
            println!("current models:");
            println!("  chat    {}", cfg.ollama.chat_model);
            println!("  vision  {}", cfg.ollama.vision_model);
            println!("  embed   {}", cfg.ollama.embed_model);
            println!();
            println!("To change: exit, then run `llm models` (interactive picker)");
            println!("           or `llm models --chat <name>` to set non-interactively.");
            SlashAction::Continue
        }
        "skills" => {
            match agent.ctx.store.list_by_kind("skill", 100).await {
                Ok(skills) if skills.is_empty() => {
                    println!(
                        "(no skills yet — teach me something with \"from now on when X, do Y\")"
                    );
                }
                Ok(skills) => {
                    println!("{} skill(s):", skills.len());
                    for s in &skills {
                        let id_short: String = s.id.chars().take(8).collect();
                        println!("  {id_short}  [{:.2}]  {}", s.importance, s.title);
                    }
                }
                Err(e) => eprintln!("{e}"),
            }
            SlashAction::Continue
        }
        "recall" => {
            let query: String = parts.collect::<Vec<_>>().join(" ");
            if query.trim().is_empty() {
                eprintln!("usage: /recall <query>");
            } else {
                let client = crate::llm::ollama::OllamaClient::new(&cfg.ollama);
                match crate::memory::search::hybrid_search(
                    &agent.ctx.store,
                    &client,
                    cfg,
                    &query,
                    8,
                )
                .await
                {
                    Ok(hits) if hits.is_empty() => {
                        println!(
                            "(no hits — try simpler keywords or /skills to see what's stored)"
                        );
                    }
                    Ok(hits) => {
                        println!("{} hit(s) for '{query}':", hits.len());
                        for h in &hits {
                            let id_short: String = h.memory.id.chars().take(8).collect();
                            println!(
                                "  {id_short}  [{:.3}  bm25={:.3} vec={:.3}]  [{}] {}",
                                h.score,
                                h.bm25.unwrap_or(0.0),
                                h.vector.unwrap_or(0.0),
                                h.memory.kind,
                                h.memory.title,
                            );
                        }
                    }
                    Err(e) => eprintln!("{e}"),
                }
            }
            SlashAction::Continue
        }
        "context" => {
            let total: usize = agent.messages.iter().map(|m| m.content.len()).sum();
            println!(
                "active context: {} messages, ~{} chars (~{} tokens estimated)",
                agent.messages.len(),
                total,
                total / 4,
            );
            // Show roles + previews; long messages truncated for readability.
            let start = agent.messages.len().saturating_sub(8);
            for (i, m) in agent.messages.iter().enumerate().skip(start) {
                let preview: String = m.content.chars().take(180).collect();
                let suffix = if m.content.chars().count() > 180 {
                    "…"
                } else {
                    ""
                };
                println!("  [{i}] {}: {preview}{suffix}", m.role);
            }
            if start > 0 {
                println!(
                    "  (omitted {} older messages — use /stats for db counts)",
                    start
                );
            }
            SlashAction::Continue
        }
        "remember" => {
            let text: String = parts.collect::<Vec<_>>().join(" ");
            if text.trim().is_empty() {
                eprintln!("usage: /remember <fact to persist as memory>");
            } else {
                let title: String = text.chars().take(80).collect();
                let new = crate::memory::NewMemory {
                    kind: "preference".into(),
                    title,
                    content: text.clone(),
                    source: "/remember".into(),
                    tags: vec!["user".into()],
                    importance: 0.95,
                };
                match agent.ctx.store.insert_memory(&new).await {
                    Ok(id) => {
                        let id_short: String = id.chars().take(8).collect();
                        println!("remembered ({id_short}): {text}");
                    }
                    Err(e) => eprintln!("{e}"),
                }
            }
            SlashAction::Continue
        }
        "forget" => {
            match parts.next() {
                None => eprintln!("usage: /forget <id-prefix>  (run /skills to see ids)"),
                Some(prefix) if prefix.len() < 4 => {
                    eprintln!("id-prefix too short — use at least 4 characters to avoid mistakes");
                }
                Some(prefix) => match agent.ctx.store.find_by_id_prefix(prefix, 5).await {
                    Ok(matches) => match matches.len() {
                        0 => eprintln!("no memory matches id prefix '{prefix}'"),
                        1 => {
                            let id = matches[0].id.clone();
                            let title = matches[0].title.clone();
                            match agent.ctx.store.delete_memory(&id).await {
                                Ok(()) => println!("forgot: {title}"),
                                Err(e) => eprintln!("delete failed: {e}"),
                            }
                        }
                        n => {
                            eprintln!("ambiguous: {n} matches for '{prefix}' — be more specific");
                            for m in matches {
                                let id_short: String = m.id.chars().take(12).collect();
                                eprintln!("  {id_short}  {}  ({})", m.title, m.kind);
                            }
                        }
                    },
                    Err(e) => eprintln!("{e}"),
                },
            }
            SlashAction::Continue
        }
        "audit" => {
            println!("{}", agent.ctx.audit.path().display());
            SlashAction::Continue
        }
        "config" => {
            if let Ok(s) = cfg.pretty() {
                println!("{s}");
            }
            SlashAction::Continue
        }
        "tools" => {
            for t in crate::tools::Registry::specs(&agent.ctx) {
                println!(
                    "  - {}  \t{}",
                    t.function.name,
                    crate::util::truncate(&t.function.description, 80)
                );
            }
            SlashAction::Continue
        }
        other => {
            eprintln!("unknown command: /{other}");
            SlashAction::Continue
        }
    }
}

fn banner(cfg: &Config, mode_override: Option<PermissionMode>) {
    let mode =
        mode_override.unwrap_or_else(|| PermissionMode::parse(&cfg.tools.mode).unwrap_or_default());
    let color_on = cfg.repl.color
        && std::env::var_os("NO_COLOR").is_none()
        && std::env::var("TERM").map(|t| t != "dumb").unwrap_or(true);

    // Gold/amber border palette. Softer than pure orange (214) so the frame
    // frames the content without screaming.
    let gold = if color_on { "\x1b[38;5;214m" } else { "" };
    let dim = if color_on { "\x1b[2m" } else { "" };
    let bold = if color_on { "\x1b[1m" } else { "" };
    let cyan = if color_on { "\x1b[38;5;81m" } else { "" };
    let peach = if color_on { "\x1b[38;5;216m" } else { "" };
    let rst = if color_on { "\x1b[0m" } else { "" };

    // Detect user for the welcome line.
    let username = std::env::var("USER")
        .or_else(|_| std::env::var("USERNAME"))
        .unwrap_or_else(|_| "friend".into());
    let cwd = std::env::current_dir()
        .ok()
        .map(|p| collapse_home(&p.to_string_lossy()))
        .unwrap_or_else(|| "~".into());

    // Layout — two panels separated by a vertical rule. Width auto-adapts
    // to the terminal with a sensible floor/ceiling so the box always
    // reads cleanly.
    let total = term_width().clamp(70, 110);
    let inner = total.saturating_sub(4); // 2 border + 2 pad (one each side)
    let left_w = (inner * 52) / 100; // ~52% for left panel
    let right_w = inner.saturating_sub(left_w + 3); // rest minus " │ "

    // Left panel content. Coloured strings are kept aligned via visible-width
    // aware padding in render_row.
    let model_line = format!(
        "{bold}{}{rst}",
        truncate_middle(&cfg.ollama.chat_model, left_w - 2)
    );
    let mode_line = mode_badge_dim(mode, color_on);
    let cwd_line = format!("{dim}{}{rst}", truncate_middle(&cwd, left_w - 2));

    let left: Vec<String> = vec![
        String::new(),
        format!(
            "{bold}Welcome back, {peach}{username}{rst}{bold}!{rst}",
            peach = peach,
        ),
        String::new(),
        // Friendly little mascot — a curious face, centred in the left
        // panel. In cyan so it pops against the gold frame.
        center(
            &format!("{cyan}   ▄▀▀▀▀▄{rst}", cyan = cyan, rst = rst),
            left_w,
        ),
        center(
            &format!("{cyan}  ( ◕ ◡ ◕ ){rst}", cyan = cyan, rst = rst),
            left_w,
        ),
        center(
            &format!("{cyan}   ▀▄▄▄▄▀{rst}", cyan = cyan, rst = rst),
            left_w,
        ),
        String::new(),
        model_line,
        mode_line,
        cwd_line,
        String::new(),
    ];

    // Right panel — tips + a memory/activity snapshot. Keeping this static
    // for now; future work can query audit.log for real recent activity.
    let tips_header = format!("{bold}Tips for getting started{rst}");
    let mem_header = format!("{bold}Memory & activity{rst}");
    let data_dir = directories::ProjectDirs::from("com", "calligoit", "localmind")
        .map(|p| collapse_home(&p.data_dir().to_string_lossy()))
        .unwrap_or_else(|| "~/.local/share/localmind".into());
    let data_line = format!("{dim}{}{rst}", truncate_middle(&data_dir, right_w));

    let right: Vec<String> = vec![
        tips_header,
        format!("{cyan}/help{rst}     slash commands"),
        format!("{cyan}/models{rst}   switch models"),
        format!("{cyan}/mode{rst}     change permission mode"),
        format!("{cyan}/quit{rst}     exit"),
        String::new(),
        mem_header,
        data_line,
        format!("{dim}audit log + memory db live here{rst}"),
        String::new(),
        String::new(),
    ];

    // Pad columns to equal height so the box bottom lines up.
    let rows = left.len().max(right.len());
    let pad_vec = |v: Vec<String>, n: usize| -> Vec<String> {
        let mut v = v;
        while v.len() < n {
            v.push(String::new());
        }
        v
    };
    let left = pad_vec(left, rows);
    let right = pad_vec(right, rows);

    // Frame characters and title.
    let title = format!(" localmind v{} ", env!("CARGO_PKG_VERSION"));
    let title_cells = visible_width(&title);
    let top_dashes = "─".repeat(total.saturating_sub(title_cells + 3));

    println!();
    println!("  {gold}╭─{bold}{title}{rst}{gold}{top_dashes}╮{rst}");
    for (l, r) in left.iter().zip(right.iter()) {
        let l_padded = pad_visible(l, left_w);
        let r_padded = pad_visible(r, right_w);
        println!("  {gold}│{rst} {l_padded} {gold}│{rst} {r_padded} {gold}│{rst}");
    }
    let bottom = "─".repeat(total.saturating_sub(2));
    println!("  {gold}╰{bottom}╯{rst}");
    println!();
}

/// A dim "mode: <mode>" line that sits under the model in the left panel.
/// The mode name gets the same coloured badge style the old banner used,
/// keeping the semantic (green = read-only, amber = workspace-write, red =
/// unrestricted) visible at a glance.
fn mode_badge_dim(mode: PermissionMode, color: bool) -> String {
    if !color {
        return format!("mode: {}", mode.as_str());
    }
    let label = format!(" {} ", mode.as_str());
    let painted = match mode {
        PermissionMode::ReadOnly => format!("\x1b[48;5;29;38;5;231;1m{label}\x1b[0m"),
        PermissionMode::WorkspaceWrite => format!("\x1b[48;5;178;38;5;232;1m{label}\x1b[0m"),
        PermissionMode::Unrestricted => format!("\x1b[48;5;160;38;5;231;1m{label}\x1b[0m"),
    };
    painted
}

/// Count visible characters in a string, skipping ANSI SGR escape sequences.
/// Used by the banner so coloured panel content aligns against the frame.
fn visible_width(s: &str) -> usize {
    let mut w = 0;
    let mut in_esc = false;
    for c in s.chars() {
        if in_esc {
            if c == 'm' {
                in_esc = false;
            }
            continue;
        }
        if c == '\x1b' {
            in_esc = true;
            continue;
        }
        w += 1;
    }
    w
}

/// Right-pad `s` with spaces so its visible width equals `w`. Truncates via
/// middle-ellipsis if already too long.
fn pad_visible(s: &str, w: usize) -> String {
    let vis = visible_width(s);
    if vis >= w {
        // Can't safely truncate colour-coded strings mid-escape; fall back
        // to the raw string (terminal will wrap).
        return s.to_string();
    }
    format!("{}{}", s, " ".repeat(w - vis))
}

/// Center `s` inside a column of width `w`, padding both sides with spaces.
fn center(s: &str, w: usize) -> String {
    let vis = visible_width(s);
    if vis >= w {
        return s.to_string();
    }
    let total_pad = w - vis;
    let left = total_pad / 2;
    let right = total_pad - left;
    format!("{}{}{}", " ".repeat(left), s, " ".repeat(right))
}

/// Same as `center`, but adds `extra_cells` to the reported visible width —
/// use this when the string contains a double-width glyph (emoji, CJK) so
/// padding still lines up against the frame.
fn center_visual(s: &str, w: usize, extra_cells: usize) -> String {
    let vis = visible_width(s) + extra_cells;
    if vis >= w {
        return s.to_string();
    }
    let total_pad = w - vis;
    let left = total_pad / 2;
    let right = total_pad - left;
    format!("{}{}{}", " ".repeat(left), s, " ".repeat(right))
}

// ---------------------------------------------------------------------------
// Banner helpers — terminal width, path tidying.
// (The old Theme struct lived here; the boxed banner uses inline colour
// constants now so the struct's wrapper methods were dead weight.)
// ---------------------------------------------------------------------------

fn term_width() -> usize {
    std::env::var("COLUMNS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&w| w >= 40)
        .unwrap_or(80)
}

fn collapse_home(path: &str) -> String {
    if let Some(home) = std::env::var_os("HOME") {
        let h = home.to_string_lossy();
        if !h.is_empty() && path.starts_with(&*h) {
            return format!("~{}", &path[h.len()..]);
        }
    }
    path.to_string()
}

/// Middle-truncate a string with a single ellipsis so the head and tail remain
/// visible. Returns the original string unchanged when it already fits.
fn truncate_middle(s: &str, max: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= max || max < 4 {
        return s.to_string();
    }
    let keep = (max - 1) / 2;
    let head: String = chars.iter().take(keep).collect();
    let tail: String = chars.iter().skip(chars.len() - (max - 1 - keep)).collect();
    format!("{head}…{tail}")
}

fn history_path() -> Option<std::path::PathBuf> {
    directories::ProjectDirs::from("com", "calligoit", "localmind").map(|p| {
        let dir = p.data_dir().to_path_buf();
        let _ = std::fs::create_dir_all(&dir);
        dir.join("history.txt")
    })
}

// ---------------------------------------------------------------------------
// /init — scan the current directory and persist a project profile to memory.
// Stores the profile via `store_memory` so it's retrievable in future
// sessions via the auto-primer or an explicit `search_memory` call.
// ---------------------------------------------------------------------------

async fn run_init(agent: &mut AgentRun) -> anyhow::Result<()> {
    use std::path::PathBuf;

    let cwd: PathBuf = std::env::current_dir()?;
    println!("/init: scanning {}", cwd.display());

    let tree = scan_tree(&cwd, 2, 120)?;
    let files = read_key_files(&cwd, 10 * 1024, 30 * 1024)?;

    let prompt = format!(
        "/init — build a persistent project profile for this working directory.\n\
         \n\
         PROJECT PATH: {cwd}\n\
         \n\
         TOP-LEVEL TREE (depth 2, {tree_lines} lines):\n\
         {tree}\n\
         \n\
         KEY FILES (truncated):\n\
         {files}\n\
         \n\
         Produce a concise project profile covering:\n\
         1. What this project is (1-2 sentences)\n\
         2. Primary language(s) and framework(s)\n\
         3. Build / test / run commands\n\
         4. Entry points and key modules\n\
         5. Conventions and patterns worth remembering\n\
         6. Anything unusual about the toolchain or layout\n\
         \n\
         Then immediately call the `store_memory` tool with:\n\
         - title:      a short project name (e.g. \"localmind\")\n\
         - kind:       \"project\"\n\
         - tags:       [\"init\", \"project-profile\", <primary-language>]\n\
         - content:    the full profile you just drafted\n\
         - importance: 0.9\n\
         \n\
         Do NOT print the profile as a chat reply — put it entirely inside the \
         `content` argument of `store_memory`. After the tool call succeeds, \
         reply with a single short sentence confirming what was stored.",
        cwd = cwd.display(),
        tree_lines = tree.lines().count(),
        tree = tree,
        files = files,
    );

    let reply = agent.turn(&prompt).await?;
    println!();
    println!("{reply}");
    println!();
    Ok(())
}

/// Walk `root` up to `max_depth` directories deep, honouring a few
/// commonly-noisy-folder skips (`.git`, `node_modules`, `target`, `dist`,
/// etc.) and capped at `max_entries` total lines.
fn scan_tree(
    root: &std::path::Path,
    max_depth: usize,
    max_entries: usize,
) -> anyhow::Result<String> {
    const SKIP_DIRS: &[&str] = &[
        ".git",
        "node_modules",
        "target",
        "dist",
        "build",
        ".venv",
        "venv",
        "__pycache__",
        ".next",
        ".cache",
        ".idea",
        ".vscode",
    ];
    let mut out = String::new();
    let mut count = 0usize;
    fn walk(
        dir: &std::path::Path,
        root: &std::path::Path,
        depth: usize,
        max_depth: usize,
        max_entries: usize,
        skip: &[&str],
        count: &mut usize,
        out: &mut String,
    ) -> std::io::Result<()> {
        if *count >= max_entries || depth > max_depth {
            return Ok(());
        }
        let mut entries: Vec<_> = std::fs::read_dir(dir)?.filter_map(Result::ok).collect();
        entries.sort_by_key(|e| e.file_name());
        for entry in entries {
            if *count >= max_entries {
                return Ok(());
            }
            let name = entry.file_name();
            let name_s = name.to_string_lossy();
            if name_s.starts_with('.') && name_s != ".gitignore" && name_s != ".env.example" {
                continue;
            }
            if skip.iter().any(|s| *s == name_s) {
                continue;
            }
            let p = entry.path();
            let rel = p.strip_prefix(root).unwrap_or(&p);
            let indent = "  ".repeat(depth);
            let kind = if p.is_dir() { "DIR " } else { "FILE" };
            out.push_str(&format!("{indent}{kind}  {}\n", rel.display()));
            *count += 1;
            if p.is_dir() {
                walk(
                    &p,
                    root,
                    depth + 1,
                    max_depth,
                    max_entries,
                    skip,
                    count,
                    out,
                )?;
            }
        }
        Ok(())
    }
    walk(
        root,
        root,
        0,
        max_depth,
        max_entries,
        SKIP_DIRS,
        &mut count,
        &mut out,
    )?;
    Ok(out)
}

/// Read a predefined set of "this is what kind of project it is" files, cap
/// each individual file at `per_file_cap` bytes and the aggregate at
/// `total_cap` bytes so the prompt stays bounded.
fn read_key_files(
    root: &std::path::Path,
    per_file_cap: usize,
    total_cap: usize,
) -> anyhow::Result<String> {
    const CANDIDATES: &[&str] = &[
        // rust
        "Cargo.toml",
        // node / ts
        "package.json",
        "tsconfig.json",
        "pnpm-workspace.yaml",
        // python
        "pyproject.toml",
        "requirements.txt",
        "setup.cfg",
        "setup.py",
        // go
        "go.mod",
        // java / kotlin
        "pom.xml",
        "build.gradle",
        "build.gradle.kts",
        "settings.gradle",
        // php / ruby
        "composer.json",
        "Gemfile",
        // c / c++
        "CMakeLists.txt",
        "Makefile",
        // infra
        "Dockerfile",
        "docker-compose.yml",
        "docker-compose.yaml",
        // docs & conventions
        "README.md",
        "README.rst",
        "README",
        ".gitignore",
        ".editorconfig",
        ".env.example",
        // lockfile hint (just the first few lines for dep count)
        "Cargo.lock",
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
    ];

    let mut out = String::new();
    let mut total = 0usize;
    for name in CANDIDATES {
        if total >= total_cap {
            break;
        }
        let p = root.join(name);
        if !p.is_file() {
            continue;
        }
        let bytes = match std::fs::read(&p) {
            Ok(b) => b,
            Err(_) => continue,
        };
        let take = bytes
            .len()
            .min(per_file_cap)
            .min(total_cap.saturating_sub(total));
        let text = String::from_utf8_lossy(&bytes[..take]).into_owned();
        let truncated = bytes.len() > take;
        out.push_str(&format!("----- {name} -----\n"));
        out.push_str(&text);
        if truncated {
            out.push_str(&format!(
                "\n... (truncated, {} of {} bytes shown)\n",
                take,
                bytes.len()
            ));
        } else {
            out.push('\n');
        }
        total += take;
    }
    if out.is_empty() {
        out.push_str("(no standard project-config files found)\n");
    }
    Ok(out)
}
