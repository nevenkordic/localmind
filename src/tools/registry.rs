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
        ) || ((matches!(name, "write_file" | "zip_create" | "zip_extract"))
            && ctx.confirm_writes);

        if needs_permission {
            match ctx.permissions.ask(name, &scope, validator_note) {
                Decision::Allow | Decision::AllowSession => {}
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
            "zip_create" | "zip_extract" => Some(format!("{name} disabled in read-only mode")),
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
        "write_file" => format!(
            "write to: {}\nsize: {} chars",
            args.get("path").and_then(|v| v.as_str()).unwrap_or("?"),
            args.get("content")
                .and_then(|v| v.as_str())
                .map(|s| s.len())
                .unwrap_or(0)
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
