//! The tool registry — converts the model's tool_call into a real action,
//! gated on permissions and deny-globs, logged to audit.

use crate::config::Config;
use crate::llm::ToolSpec;
use crate::memory::Store;
use crate::tools::audit::AuditLog;
use crate::tools::permissions::{Decision, PermissionManager, PermissionMode, ValidatorNote};
use crate::tools::shell_validation::ValidationResult;
use anyhow::{anyhow, Result};
use serde_json::Value;
use std::path::Path;
use std::sync::Arc;

pub struct ToolContext {
    pub cfg: Arc<Config>,
    pub store: Store,
    pub audit: Arc<AuditLog>,
    pub permissions: Arc<PermissionManager>,
    pub web_enabled: bool,
    pub shell_enabled: bool,
    pub confirm_writes: bool,
}

pub struct Registry;

impl Registry {
    /// Full set of JSON-schema tool specs to advertise to the model.
    pub fn specs(ctx: &ToolContext) -> Vec<ToolSpec> {
        let mut specs = vec![
            read_file_spec(),
            write_file_spec(),
            list_dir_spec(),
            create_dir_spec(),
            read_pdf_spec(),
            read_docx_spec(),
            read_xlsx_spec(),
            read_image_spec(),
            search_memory_spec(),
            store_memory_spec(),
            kg_link_spec(),
            // Networking / admin tools — always advertised; the permission
            // layer decides whether each individual call is allowed.
            port_check_spec(),
            dns_lookup_spec(),
            listening_ports_spec(),
            whois_spec(),
            http_fetch_spec(),
            // Archive tools — cross-platform zip / unzip.
            zip_create_spec(),
            zip_extract_spec(),
        ];
        if ctx.web_enabled && !ctx.cfg.web.brave_api_key.is_empty() {
            specs.push(web_search_spec());
            specs.push(web_fetch_spec());
        }
        if ctx.shell_enabled {
            specs.push(shell_spec());
        }
        specs
    }

    pub async fn dispatch(ctx: &ToolContext, name: &str, args: &Value) -> Result<String> {
        // Deny-glob check for path-bearing tools, regardless of permission.
        if let Some(path) = path_arg(args) {
            if crate::tools::file_ops::is_denied(&ctx.cfg.tools.deny_globs, &path) {
                ctx.audit.record(name, args, "deny-glob", "path denied");
                return Err(anyhow!("path denied by configured deny-glob: {path}"));
            }
        }

        // Mode-level guard-rail BEFORE prompting. ReadOnly refuses mutating
        // tools outright; WorkspaceWrite refuses writes that clearly escape
        // the workspace.
        let mode = ctx.permissions.mode();
        if let Some(block) = mode_guard(mode, name, args, &ctx.cfg.tools.workspace_root) {
            ctx.audit.record(name, args, "mode-block", &block);
            return Err(anyhow!(
                "refused by permission mode ({}): {}",
                mode.as_str(),
                block
            ));
        }

        // Shell validator → may block or warn. The warn is rendered inside
        // the permission prompt so the user knows why it's risky.
        let mut validator_note: Option<ValidatorNote> = None;
        if name == "shell" {
            if let Some(cmd) = args.get("command").and_then(|v| v.as_str()) {
                match crate::tools::shell::precheck(ctx, cmd) {
                    ValidationResult::Block { reason } => {
                        ctx.audit.record(name, args, "validator-block", &reason);
                        return Err(anyhow!("shell refused by validator: {reason}"));
                    }
                    ValidationResult::Warn { message } => {
                        validator_note = Some(ValidatorNote::Warn(message));
                    }
                    ValidationResult::Allow => {}
                }
            }
        }

        let scope = describe_scope(name, args);
        // Tools that always prompt (side effects or outbound network):
        //   shell, web_search, web_fetch, http_fetch, port_check, whois
        // write_file prompts only when confirm_writes is on.
        // Local read-only inventory (listening_ports, dns_lookup) doesn't prompt.
        let needs_permission = matches!(
            name,
            "shell" | "web_search" | "web_fetch" | "http_fetch" | "port_check" | "whois"
        ) || ((matches!(
            name,
            "write_file" | "zip_create" | "zip_extract" | "create_dir"
        )) && ctx.confirm_writes);

        if needs_permission {
            // For destructive tools, print a rich preamble (full-width
            // diff, file header, +N/-M summary) BEFORE the permission
            // prompt. Keeps the prompt box short and lets the diff
            // breathe across the terminal width.
            print_preamble(name, args);
            match ctx.permissions.ask(name, &scope, validator_note) {
                Decision::Allow | Decision::AllowSession => {}
                Decision::AllowPersistent(rule) => {
                    // Save to the user's config so future sessions also skip
                    // this prompt, then activate in-process so the rest of
                    // this session benefits too.
                    if let Err(e) = persist_allow_rule(&rule) {
                        eprintln!(
                            "  (could not save permanent grant to config: {e:#} — granting for this session only)"
                        );
                    } else {
                        eprintln!("  \x1b[1;32m✓\x1b[0m saved: allow_rules += {rule}");
                    }
                    ctx.permissions.add_allow_rule(&rule);
                }
                Decision::Deny => {
                    ctx.audit.record(name, args, "deny", "user denied");
                    return Err(anyhow!("user denied tool call: {name}"));
                }
                Decision::Edit(reason) => {
                    ctx.audit.record(name, args, "edit", &reason);
                    return Err(anyhow!("user redirected: {reason}"));
                }
            }
        }

        let result = match name {
            "read_file" => crate::tools::file_ops::read_file(ctx, args).await,
            "write_file" => crate::tools::file_ops::write_file(ctx, args).await,
            "list_dir" => crate::tools::file_ops::list_dir(ctx, args).await,
            "create_dir" => crate::tools::file_ops::create_dir(ctx, args).await,
            "read_pdf" => crate::tools::pdf::read_pdf(ctx, args).await,
            "read_docx" => crate::tools::docs::read_docx(ctx, args).await,
            "read_xlsx" => crate::tools::docs::read_xlsx(ctx, args).await,
            "read_image" => crate::tools::image::read_image(ctx, args).await,
            "search_memory" => crate::tools::memory_tools::search_memory(ctx, args).await,
            "store_memory" => crate::tools::memory_tools::store_memory(ctx, args).await,
            "kg_link" => crate::tools::memory_tools::kg_link(ctx, args).await,
            "web_search" => crate::tools::web::web_search(ctx, args).await,
            "web_fetch" => crate::tools::web::web_fetch(ctx, args).await,
            "shell" => crate::tools::shell::run(ctx, args).await,
            "port_check" => crate::tools::net::port_check(ctx, args).await,
            "dns_lookup" => crate::tools::net::dns_lookup(ctx, args).await,
            "listening_ports" => crate::tools::net::listening_ports(ctx, args).await,
            "whois" => crate::tools::net::whois(ctx, args).await,
            "http_fetch" => crate::tools::net::http_fetch(ctx, args).await,
            "zip_create" => crate::tools::archive::zip_create(ctx, args).await,
            "zip_extract" => crate::tools::archive::zip_extract(ctx, args).await,
            other => Err(anyhow!("unknown tool: {other}")),
        };

        let (decision, summary) = match &result {
            Ok(r) => ("allow", crate::util::truncate(r, 200)),
            Err(e) => ("error", crate::util::truncate(&e.to_string(), 200)),
        };
        ctx.audit.record(name, args, decision, &summary);
        result
    }
}

fn path_arg(args: &Value) -> Option<String> {
    args.get("path")
        .and_then(|v| v.as_str())
        .map(str::to_string)
}

/// PermissionMode gate before any prompting.
/// Returns `Some(reason)` to refuse outright.
fn mode_guard(
    mode: PermissionMode,
    name: &str,
    args: &Value,
    workspace_root: &str,
) -> Option<String> {
    match mode {
        PermissionMode::ReadOnly => match name {
            "write_file" => Some("write_file disabled in read-only mode".into()),
            "shell" => None, // handled by shell_validation (richer reasons)
            "web_search" | "web_fetch" => Some(format!("{name} disabled in read-only mode")),
            // http_fetch can mutate remote state (POST/PUT/DELETE); block it.
            // port_check / whois initiate outbound connections — also blocked
            // under read-only to keep the session fully passive.
            "http_fetch" | "port_check" | "whois" => {
                Some(format!("{name} disabled in read-only mode"))
            }
            "zip_create" | "zip_extract" | "create_dir" => {
                Some(format!("{name} disabled in read-only mode"))
            }
            "store_memory" | "kg_link" => Some(format!("{name} disabled in read-only mode")),
            _ => None,
        },
        PermissionMode::WorkspaceWrite => {
            if workspace_root.is_empty() {
                return None;
            }
            let outside = |key: &str| -> Option<String> {
                let path = args.get(key).and_then(|v| v.as_str()).unwrap_or("");
                let p = Path::new(path);
                if p.is_absolute() && !p.starts_with(workspace_root) {
                    Some(format!(
                        "{name}.{key} outside workspace_root ({workspace_root}) — switch to unrestricted mode to allow"
                    ))
                } else {
                    None
                }
            };
            match name {
                "write_file" => outside("path"),
                "create_dir" => outside("path"),
                "zip_create" => outside("archive_path"),
                "zip_extract" => outside("dest_dir"),
                _ => None,
            }
        }
        PermissionMode::Unrestricted => None,
    }
}

fn describe_scope(name: &str, args: &Value) -> String {
    match name {
        // Intentionally short now — the full diff is rendered by
        // print_preamble before this fires, so the prompt box stays tight.
        "write_file" => format!(
            "write {}",
            args.get("path").and_then(|v| v.as_str()).unwrap_or("?")
        ),
        "create_dir" => format!(
            "create directory: {}",
            args.get("path").and_then(|v| v.as_str()).unwrap_or("?")
        ),
        "shell" => format!(
            "command: {}",
            crate::util::truncate(
                args.get("command").and_then(|v| v.as_str()).unwrap_or(""),
                400
            )
        ),
        "web_search" => format!(
            "search: {}",
            args.get("query").and_then(|v| v.as_str()).unwrap_or("")
        ),
        "web_fetch" => format!(
            "fetch: {}",
            args.get("url").and_then(|v| v.as_str()).unwrap_or("")
        ),
        "http_fetch" => format!(
            "{} {}",
            args.get("method")
                .and_then(|v| v.as_str())
                .unwrap_or("GET")
                .to_uppercase(),
            args.get("url").and_then(|v| v.as_str()).unwrap_or("")
        ),
        "port_check" => format!(
            "connect: {}:{}",
            args.get("host").and_then(|v| v.as_str()).unwrap_or("?"),
            args.get("port").and_then(|v| v.as_u64()).unwrap_or(0)
        ),
        "whois" => format!(
            "whois: {}",
            args.get("query").and_then(|v| v.as_str()).unwrap_or("")
        ),
        "zip_create" => format!(
            "create archive: {}\ninputs: {}",
            args.get("archive_path")
                .and_then(|v| v.as_str())
                .unwrap_or("?"),
            args.get("inputs")
                .and_then(|v| v.as_array())
                .map(|a| a
                    .iter()
                    .filter_map(|x| x.as_str())
                    .collect::<Vec<_>>()
                    .join(", "))
                .unwrap_or_default()
        ),
        "zip_extract" => format!(
            "extract: {}\n    into: {}",
            args.get("archive_path")
                .and_then(|v| v.as_str())
                .unwrap_or("?"),
            args.get("dest_dir").and_then(|v| v.as_str()).unwrap_or("?")
        ),
        _ => serde_json::to_string(args).unwrap_or_default(),
    }
}

/// Print a rich-format context block to stderr BEFORE the permission
/// prompt fires. Currently only `write_file` and `create_dir` have one;
/// other tools (shell, web, etc.) don't need a preamble — the short scope
/// inside the prompt box is enough.
///
/// For write_file, the preamble renders:
///   ● Update(path)  |  ● Create(path)
///      └─ Added N lines, removed M lines
///
///   123      unchanged context line
///   124 -    removed line            (red on dark-red background)
///   124 +    added line              (green on dark-green background)
///
/// Using a full-width render outside the prompt box so long lines don't
/// wrap awkwardly against the `┃` gutter.
pub(crate) fn print_preamble(name: &str, args: &Value) {
    match name {
        "write_file" => {
            let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("?");
            let new_content = args
                .get("content")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let resolved = Path::new(path);
            let exists = resolved.exists();
            let old_content = std::fs::read_to_string(resolved).unwrap_or_default();

            let (verb, summary) = if exists {
                let (plus, minus) = count_changes(&old_content, new_content);
                (
                    "Update",
                    format!("Added {plus} lines, removed {minus} lines"),
                )
            } else {
                ("Create", format!("New file, {} lines", new_content.lines().count()))
            };

            eprintln!();
            eprintln!("\x1b[1;36m●\x1b[0m \x1b[1m{verb}(\x1b[36m{path}\x1b[0m\x1b[1m)\x1b[0m");
            eprintln!("  \x1b[2m└─ {summary}\x1b[0m");
            eprintln!();
            let diff = render_diff(&old_content, new_content, path, 60);
            for line in diff.lines() {
                eprintln!("  {line}");
            }
            eprintln!();
        }
        "create_dir" => {
            let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("?");
            eprintln!();
            eprintln!("\x1b[1;36m●\x1b[0m \x1b[1mCreate directory(\x1b[36m{path}\x1b[0m\x1b[1m)\x1b[0m");
            eprintln!();
        }
        _ => {}
    }
}

/// Syntax-highlighted, background-coloured unified diff of `old` vs `new`.
/// Matches the visual style of modern diff UIs: a dim line-number gutter,
/// a red-tinted row background for deletions, green-tinted for insertions,
/// and syntect-powered foreground colouring based on the language
/// detected from `path` (file extension). Context lines get no background
/// and a dim foreground.
///
///   199      "http_fetch" | "port_check" ...        ← dim context
///   202 -    "zip_create" | "zip_extract" => ...    ← red bg, red sign
///   202 +    "zip_create" | "zip_extract" | ...     ← green bg, green sign
///
/// Truncated to `max_lines`. `path` is used only for syntax detection —
/// pass an empty string to disable highlighting and get plain-colored text.
pub(crate) fn render_diff(old: &str, new: &str, path: &str, max_lines: usize) -> String {
    use similar::{ChangeTag, TextDiff};
    let diff = TextDiff::from_lines(old, new);

    let max_line_no = old.lines().count().max(new.lines().count()).max(1);
    let gutter_width = max_line_no.to_string().len();

    // Syntax highlighter setup. We build these per-call because the diff
    // render isn't on a hot path and the one-time cost of the syntax set
    // (loaded from embedded binary data) is <50ms on first call, then
    // cached in syntect's static tables for subsequent calls.
    let ps = syntect::parsing::SyntaxSet::load_defaults_newlines();
    let ts = syntect::highlighting::ThemeSet::load_defaults();
    // base16-ocean.dark has sensible defaults and reads well on both dark
    // and light terminals; users who want true theme integration can
    // customise later.
    let theme = &ts.themes["base16-ocean.dark"];
    let syntax = detect_syntax_for_path(&ps, path);

    let mut out = String::new();
    let mut rendered = 0usize;
    let mut skipped = 0usize;
    for change in diff.iter_all_changes() {
        if rendered >= max_lines {
            skipped += 1;
            continue;
        }
        let lineno = match change.tag() {
            ChangeTag::Delete => change.old_index(),
            ChangeTag::Insert => change.new_index(),
            ChangeTag::Equal => change.new_index(),
        };
        let lineno = lineno.map(|n| n + 1).unwrap_or(0);
        let lineno_str = format!("{:>width$}", lineno, width = gutter_width);

        // Background + sign colouring per tag. Background uses 256-colour
        // codes (52 = dark red, 22 = dark green) so the row stays visible
        // without overpowering the foreground.
        let (sign, bg_ansi, sign_fg) = match change.tag() {
            ChangeTag::Delete => ("-", "\x1b[48;5;52m", "\x1b[91m"),
            ChangeTag::Insert => ("+", "\x1b[48;5;22m", "\x1b[92m"),
            ChangeTag::Equal => (" ", "", "\x1b[2;37m"),
        };

        let raw = change.to_string();
        let body = raw.strip_suffix('\n').unwrap_or(&raw);
        // Highlight the body if we have a syntax; else fall back to plain.
        let highlighted_body = highlight_line(syntax, theme, &ps, body);

        // Gutter (line number + separator) is always dim, no background.
        // Then row: background + sign + highlighted body + reset.
        out.push_str(&format!(
            "\x1b[2;37m{lineno_str} \u{2502}\x1b[0m {bg_ansi}{sign_fg}{sign}\x1b[39m {highlighted_body}\x1b[0m\n"
        ));
        rendered += 1;
    }
    if skipped > 0 {
        out.push_str(&format!(
            "\x1b[2m… ({} more lines not shown — full diff in audit log)\x1b[0m\n",
            skipped
        ));
    }
    if out.ends_with('\n') {
        out.pop();
    }
    out
}

/// Pick a syntect syntax definition from a file path. Falls back to plain
/// text when the extension is unknown or the path is empty.
fn detect_syntax_for_path<'a>(
    ps: &'a syntect::parsing::SyntaxSet,
    path: &str,
) -> &'a syntect::parsing::SyntaxReference {
    use std::path::Path;
    if path.is_empty() {
        return ps.find_syntax_plain_text();
    }
    if let Some(ext) = Path::new(path).extension().and_then(|e| e.to_str()) {
        if let Some(s) = ps.find_syntax_by_extension(ext) {
            return s;
        }
    }
    // Try first line detection (shebangs etc.) if we had content here;
    // we don't, so just plain text.
    ps.find_syntax_plain_text()
}

/// Highlight a single line of source. Returns an ANSI-coloured string.
/// Syntax-highlighting failures (corrupt theme, etc.) downgrade to
/// returning the unstyled line rather than failing the whole diff render.
fn highlight_line(
    syntax: &syntect::parsing::SyntaxReference,
    theme: &syntect::highlighting::Theme,
    ps: &syntect::parsing::SyntaxSet,
    line: &str,
) -> String {
    use syntect::easy::HighlightLines;
    use syntect::util::as_24_bit_terminal_escaped;
    let mut h = HighlightLines::new(syntax, theme);
    match h.highlight_line(line, ps) {
        Ok(ranges) => as_24_bit_terminal_escaped(&ranges[..], false),
        Err(_) => line.to_string(),
    }
}

/// Count added / removed lines between `old` and `new`. Cheap — the full
/// diff iterator already computes this; we just tally counts for the
/// header line in the scope string.
fn count_changes(old: &str, new: &str) -> (usize, usize) {
    use similar::{ChangeTag, TextDiff};
    let diff = TextDiff::from_lines(old, new);
    let mut plus = 0;
    let mut minus = 0;
    for change in diff.iter_all_changes() {
        match change.tag() {
            ChangeTag::Insert => plus += 1,
            ChangeTag::Delete => minus += 1,
            ChangeTag::Equal => {}
        }
    }
    (plus, minus)
}

/// Append a new entry to `[tools].allow_rules` in the user's config file.
/// Uses `Config::ensure_writable_path` + toml_edit so comments and other
/// unrelated settings are preserved. If the config has no [tools] section
/// or no allow_rules array yet, both are created with sensible defaults.
fn persist_allow_rule(rule: &str) -> Result<()> {
    let path = Config::ensure_writable_path(None)?;
    let raw = std::fs::read_to_string(&path)
        .map_err(|e| anyhow!("reading {}: {e}", path.display()))?;
    let mut doc: toml_edit::DocumentMut = raw
        .parse()
        .map_err(|e| anyhow!("parsing TOML in {}: {e}", path.display()))?;

    // Ensure [tools] exists.
    if !doc.contains_key("tools") {
        doc["tools"] = toml_edit::Item::Table(toml_edit::Table::new());
    }
    let tools = doc["tools"]
        .as_table_mut()
        .ok_or_else(|| anyhow!("[tools] is not a table"))?;

    // Ensure allow_rules exists as an array.
    if !tools.contains_key("allow_rules") {
        tools["allow_rules"] = toml_edit::value(toml_edit::Array::new());
    }
    let arr = tools["allow_rules"]
        .as_array_mut()
        .ok_or_else(|| anyhow!("[tools].allow_rules is not an array"))?;

    // Idempotent: don't duplicate an existing entry.
    let already_present = arr
        .iter()
        .any(|v| v.as_str().map(|s| s == rule).unwrap_or(false));
    if !already_present {
        arr.push(rule);
    }

    std::fs::write(&path, doc.to_string())
        .map_err(|e| anyhow!("writing {}: {e}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod persist_tests {
    use super::*;

    // The real persist_allow_rule uses Config::ensure_writable_path, which
    // would clobber the user's actual config dir during a test. We inline a
    // copy that takes an explicit path so the behaviour (toml_edit preserves
    // comments + appends to allow_rules idempotently) is still covered.
    fn persist_to(path: &std::path::Path, rule: &str) -> Result<()> {
        let raw = std::fs::read_to_string(path)?;
        let mut doc: toml_edit::DocumentMut = raw.parse()?;
        if !doc.contains_key("tools") {
            doc["tools"] = toml_edit::Item::Table(toml_edit::Table::new());
        }
        let tools = doc["tools"].as_table_mut().unwrap();
        if !tools.contains_key("allow_rules") {
            tools["allow_rules"] = toml_edit::value(toml_edit::Array::new());
        }
        let arr = tools["allow_rules"].as_array_mut().unwrap();
        let dup = arr.iter().any(|v| v.as_str() == Some(rule));
        if !dup {
            arr.push(rule);
        }
        std::fs::write(path, doc.to_string())?;
        Ok(())
    }

    #[test]
    fn appends_to_existing_allow_rules_and_preserves_comments() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("local.toml");
        std::fs::write(
            &path,
            "# top-of-file comment\n[tools]\n# explaining comment\nallow_rules = [\"shell(git status)\"]\n",
        )
        .unwrap();
        persist_to(&path, "write_file(write to: /tmp/x*)").unwrap();
        let raw = std::fs::read_to_string(&path).unwrap();
        assert!(raw.contains("# top-of-file comment"));
        assert!(raw.contains("# explaining comment"));
        assert!(raw.contains("shell(git status)"));
        assert!(raw.contains("write_file(write to: /tmp/x*)"));
    }

    #[test]
    fn creates_missing_tools_section_and_array() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("local.toml");
        std::fs::write(&path, "[ollama]\nhost = \"http://127.0.0.1:11434\"\n").unwrap();
        persist_to(&path, "shell(ls*)").unwrap();
        let raw = std::fs::read_to_string(&path).unwrap();
        assert!(raw.contains("[tools]"));
        assert!(raw.contains("shell(ls*)"));
    }

    #[test]
    fn deduplicates_repeated_grants() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("local.toml");
        std::fs::write(&path, "[tools]\nallow_rules = []\n").unwrap();
        persist_to(&path, "shell(ls*)").unwrap();
        persist_to(&path, "shell(ls*)").unwrap();
        let raw = std::fs::read_to_string(&path).unwrap();
        // Count occurrences; should be exactly one.
        assert_eq!(raw.matches("shell(ls*)").count(), 1);
    }
}

// ---------------------------------------------------------------------------
// Tool spec helpers — JSON schema for each advertised tool.
// ---------------------------------------------------------------------------
fn spec(name: &str, desc: &str, params: Value) -> ToolSpec {
    ToolSpec {
        r#type: "function".into(),
        function: crate::llm::types::ToolFunctionSpec {
            name: name.into(),
            description: desc.into(),
            parameters: params,
        },
    }
}

fn read_file_spec() -> ToolSpec {
    spec("read_file", "Read a UTF-8 text file from the user's filesystem and return its contents. Reject if the path matches any deny-glob.", serde_json::json!({
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute or workspace-relative path"},
            "max_bytes": {"type": "integer", "description": "Max bytes to read, default 200000"}
        },
        "required": ["path"]
    }))
}
fn write_file_spec() -> ToolSpec {
    spec(
        "write_file",
        "Create or overwrite a text file. Requires user approval.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
                "create_parents": {"type": "boolean", "description": "Create missing parent dirs"}
            },
            "required": ["path", "content"]
        }),
    )
}
fn list_dir_spec() -> ToolSpec {
    spec(
        "list_dir",
        "List files and subdirectories of a directory (non-recursive by default).",
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "recursive": {"type": "boolean"},
                "max_entries": {"type": "integer"}
            },
            "required": ["path"]
        }),
    )
}
fn create_dir_spec() -> ToolSpec {
    spec(
        "create_dir",
        "Create a directory (and any missing parents, like `mkdir -p`). \
         Idempotent — succeeds if the directory already exists. Subject to \
         workspace_root and the same permission prompt as write_file.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or workspace-relative directory path"}
            },
            "required": ["path"]
        }),
    )
}
fn read_pdf_spec() -> ToolSpec {
    spec(
        "read_pdf",
        "Extract text from a PDF file.",
        serde_json::json!({
            "type": "object",
            "properties": {"path": {"type": "string"}}, "required": ["path"]
        }),
    )
}
fn read_docx_spec() -> ToolSpec {
    spec(
        "read_docx",
        "Extract text from a Microsoft Word .docx file.",
        serde_json::json!({
            "type": "object",
            "properties": {"path": {"type": "string"}}, "required": ["path"]
        }),
    )
}
fn read_xlsx_spec() -> ToolSpec {
    spec(
        "read_xlsx",
        "Extract CSV-like text from an Excel .xlsx file. Returns sheets separated by headers.",
        serde_json::json!({
            "type": "object",
            "properties": {"path": {"type": "string"}, "sheet": {"type": "string"}}, "required": ["path"]
        }),
    )
}
fn read_image_spec() -> ToolSpec {
    spec("read_image", "Load an image file, send it to the vision model, and return the model's description of what the image contains.", serde_json::json!({
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "prompt": {"type": "string", "description": "Optional question or task about the image"}
        },
        "required": ["path"]
    }))
}
fn search_memory_spec() -> ToolSpec {
    spec(
        "search_memory",
        "Hybrid (BM25 + vector) search over the agent's persistent memory.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "default": 8}
            },
            "required": ["query"]
        }),
    )
}
fn store_memory_spec() -> ToolSpec {
    spec("store_memory", "Persist a fact, decision, note, or skill to long-term memory. Use kind=\"skill\" for procedures the user TEACHES you (\"from now on when X, do Y\") — title it after when the skill should fire, put the steps in content, set importance ~0.85. Skills are auto-retrieved at the start of each turn.", serde_json::json!({
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "content": {"type": "string"},
            "kind": {"type": "string", "enum": ["fact","decision","project","preference","note","skill"]},
            "tags": {"type": "array", "items": {"type": "string"}},
            "importance": {"type": "number", "minimum": 0, "maximum": 1}
        },
        "required": ["title", "content"]
    }))
}
fn kg_link_spec() -> ToolSpec {
    spec(
        "kg_link",
        "Add an edge to the knowledge graph: (src, relation, dst).",
        serde_json::json!({
            "type": "object",
            "properties": {
                "src_name": {"type": "string"},
                "src_type": {"type": "string"},
                "dst_name": {"type": "string"},
                "dst_type": {"type": "string"},
                "relation": {"type": "string"}
            },
            "required": ["src_name", "src_type", "dst_name", "dst_type", "relation"]
        }),
    )
}
fn web_search_spec() -> ToolSpec {
    spec(
        "web_search",
        "Search the web via the Brave Search API. Requires user approval each call.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "count": {"type": "integer"}
            },
            "required": ["query"]
        }),
    )
}
fn web_fetch_spec() -> ToolSpec {
    spec(
        "web_fetch",
        "Fetch the text of a specific URL. Requires user approval each call.",
        serde_json::json!({
            "type": "object",
            "properties": {"url": {"type": "string"}}, "required": ["url"]
        }),
    )
}
fn shell_spec() -> ToolSpec {
    spec(
        "shell",
        "Run a shell command (bash on Unix, cmd on Windows). Requires user approval each call.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "cwd": {"type": "string"},
                "timeout_secs": {"type": "integer"}
            },
            "required": ["command"]
        }),
    )
}

// ---------------------------------------------------------------------------
// Networking tools — cross-platform native-Rust replacements for nc / ss /
// whois / curl / host. See src/tools/net.rs for the implementations.
// ---------------------------------------------------------------------------

fn port_check_spec() -> ToolSpec {
    spec(
        "port_check",
        "TCP reachability test — the cross-platform equivalent of `nc -zv host port`. \
         Resolves the host via the OS resolver and opens a TCP connection with a timeout. \
         Does not send any payload. Requires user approval each call.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "host": {"type": "string", "description": "Hostname or IP address"},
                "port": {"type": "integer", "minimum": 1, "maximum": 65535},
                "timeout_ms": {"type": "integer", "description": "Connect timeout in ms (default 3000)"}
            },
            "required": ["host", "port"]
        }),
    )
}

fn dns_lookup_spec() -> ToolSpec {
    spec(
        "dns_lookup",
        "Resolve a hostname via the OS resolver. Returns separate A (IPv4) and AAAA (IPv6) lists. \
         Local read-only — no permission prompt.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
            "required": ["name"]
        }),
    )
}

fn listening_ports_spec() -> ToolSpec {
    spec(
        "listening_ports",
        "Inventory local TCP listeners and/or UDP sockets — the cross-platform equivalent of \
         `ss -tulpn` on Linux or `netstat -ano` on Windows. Uses OS socket-table APIs directly \
         (iphlpapi on Windows, libproc on macOS, /proc/net on Linux). Local read-only — no prompt.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "protocol": {
                    "type": "string",
                    "enum": ["tcp", "udp", "both"],
                    "description": "Which protocol to list; default tcp"
                }
            }
        }),
    )
}

fn whois_spec() -> ToolSpec {
    spec(
        "whois",
        "Query WHOIS for a domain, IP, or ASN. Connects to whois.iana.org (or the given server) \
         on TCP/43 and follows `refer:` / `whois:` / `ReferralServer:` hints to reach the \
         authoritative registry. Returns the referral chain plus the final response. \
         Requires user approval each call.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "server": {"type": "string", "description": "Override the starting WHOIS server"},
                "max_hops": {"type": "integer", "description": "Max referrals to follow (default 3)"},
                "timeout_secs": {"type": "integer", "description": "Per-hop timeout (default 10)"}
            },
            "required": ["query"]
        }),
    )
}

fn http_fetch_spec() -> ToolSpec {
    spec(
        "http_fetch",
        "Make an arbitrary HTTP request — the cross-platform equivalent of `curl` / `wget`. \
         Supports GET, HEAD, POST, PUT, DELETE, custom headers, a request body, and a response \
         size cap. Binary bodies are reported as a size, not decoded. Requires user approval \
         each call and is disabled in read-only mode.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "method": {
                    "type": "string",
                    "enum": ["GET", "HEAD", "POST", "PUT", "DELETE"],
                    "description": "HTTP method; default GET"
                },
                "headers": {
                    "type": "object",
                    "description": "Header name → value map",
                    "additionalProperties": {"type": "string"}
                },
                "body": {"type": "string"},
                "timeout_secs": {"type": "integer", "description": "Request timeout (default 30)"},
                "max_bytes": {"type": "integer", "description": "Cap on body size returned (default 1000000)"}
            },
            "required": ["url"]
        }),
    )
}

// ---------------------------------------------------------------------------
// Archive tools — cross-platform zip / unzip.
// ---------------------------------------------------------------------------

fn zip_create_spec() -> ToolSpec {
    spec(
        "zip_create",
        "Pack files and/or directories into a .zip archive. Directories are \
         recursed. Use `base_dir` to control the entry names inside the \
         archive (paths are stored relative to this directory). Honours \
         deny-glob and workspace boundaries. Requires user approval when \
         confirm_writes is on.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "archive_path": {"type": "string", "description": "Path to the .zip to create"},
                "inputs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Files or directories to include"
                },
                "base_dir": {"type": "string", "description": "Strip this prefix from entry names"},
                "compression": {
                    "type": "string",
                    "enum": ["stored", "deflated"],
                    "description": "Compression method; default deflated"
                }
            },
            "required": ["archive_path", "inputs"]
        }),
    )
}

fn zip_extract_spec() -> ToolSpec {
    spec(
        "zip_extract",
        "Extract a .zip archive into a destination directory with a zip-slip \
         guard — entries that would escape `dest_dir` are refused. Existing \
         files are skipped unless `overwrite` is true. Optional \
         `strip_components` drops leading path segments like `tar --strip-components`.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "archive_path": {"type": "string"},
                "dest_dir": {"type": "string"},
                "overwrite": {"type": "boolean", "description": "Overwrite existing files (default false)"},
                "strip_components": {"type": "integer", "description": "Leading path parts to drop"}
            },
            "required": ["archive_path", "dest_dir"]
        }),
    )
}

/// Helper used from Path-bearing tools.
pub fn resolve_path(ctx: &ToolContext, p: &str) -> std::path::PathBuf {
    let path = Path::new(p);
    if path.is_absolute() {
        path.to_path_buf()
    } else if ctx.cfg.tools.workspace_root.is_empty() {
        std::env::current_dir()
            .unwrap_or_else(|_| ".".into())
            .join(path)
    } else {
        Path::new(&ctx.cfg.tools.workspace_root).join(path)
    }
}
