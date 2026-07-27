//! Ground-truth checks for the verify harness — objective evidence beyond
//! model self-report.

use anyhow::{Context, Result};
use serde::Serialize;
use std::path::Path;
use std::process::Stdio;
use tokio::process::Command;

#[derive(Debug, Clone, Serialize)]
pub struct CheckResult {
    pub name: String,
    pub passed: bool,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct CheckReport {
    pub results: Vec<CheckResult>,
}

impl CheckReport {
    pub fn all_passed(&self) -> bool {
        self.results.iter().all(|r| r.passed)
    }

    pub fn summary(&self) -> String {
        let mut out = String::new();
        for r in &self.results {
            let mark = if r.passed { "PASS" } else { "FAIL" };
            out.push_str(&format!("[{mark}] {}: {}\n", r.name, r.detail));
        }
        out
    }
}

/// Extra act-retry instructions when ground checks fail. Keeps the model
/// from spawning more http.servers when the port is already fine, and
/// forces full HTML/CSS rewrites when substance floors fail.
pub fn retry_guidance(report: &CheckReport, opts: &CheckOptions) -> String {
    let mut out = report.summary();
    let server_ok = report
        .results
        .iter()
        .any(|r| r.name == "detached_server" && r.passed);
    let substance_fail = report.results.iter().any(|r| {
        (r.name == "html_substance" || r.name == "css_substance") && !r.passed
    });

    if server_ok {
        out.push_str(
            "\nSERVER ALREADY UP — do NOT call shell/http.server again. \
             Reuse the existing URL. ONLY rewrite files.\n",
        );
    }
    if substance_fail {
        out.push_str(&format!(
            "\nREWRITE ALL of index.html, about.html (or second page), AND styles.css \
             via write_file in this attempt — partial fixes fail. Hard floors: each \
             HTML ≥ {} bytes, CSS ≥ {} bytes. Expand with: distinctive brand name, \
             real multi-paragraph copy (not lorem), hero + ≥2 sections + footer, \
             CSS :root variables, Google Font @import, non-flat background, hover \
             states, and a @media block. Aim ~20-40% over the floor so truncation \
             still passes.\n",
            opts.min_html_bytes, opts.min_css_bytes
        ));
    }
    out
}

/// Options for post-act ground-truth checks.
#[derive(Debug, Clone, Default)]
pub struct CheckOptions {
    pub test_command: String,
    /// Fail when the audit tail has no tool calls (act claimed work but
    /// touched nothing).
    pub require_tool_use: bool,
    /// Paths that must exist after act (relative or absolute).
    pub check_paths: Vec<String>,
    /// Require that a detached http.server (or similar) from this act is
    /// still listening — parses `likely_port=` markers from the audit log
    /// and TCP-connects to each.
    pub require_detached_server: bool,
    /// Each `.html` path written in the audit must be at least this many bytes.
    pub min_html_bytes: u64,
    /// Each `.css` path written in the audit must be at least this many bytes.
    pub min_css_bytes: u64,
}

/// Run configured ground-truth checks after the act stage.
///
/// `act_audit` — tools from this attempt only (for `require_tool_use`).
/// `evidence_audit` — broader window (typically since harness start) used
/// for detached_server + file substance so a retry that only fixes one
/// file still sees the server/CSS from earlier attempts. Pass the same
/// string twice when you don't need the split.
pub async fn run_checks(opts: &CheckOptions, act_audit: &str) -> CheckReport {
    run_checks_with_evidence(opts, act_audit, act_audit).await
}

pub async fn run_checks_with_evidence(
    opts: &CheckOptions,
    act_audit: &str,
    evidence_audit: &str,
) -> CheckReport {
    let mut results = Vec::new();

    if !opts.test_command.trim().is_empty() {
        results.push(run_shell_check("test_command", &opts.test_command).await);
    }

    if opts.require_tool_use {
        let tool_calls = count_audit_tool_calls(act_audit);
        results.push(CheckResult {
            name: "require_tool_use".into(),
            passed: tool_calls > 0,
            detail: if tool_calls > 0 {
                format!("{tool_calls} tool call(s) in audit log")
            } else {
                "no tool calls in audit log — act did not use tools".into()
            },
        });
    }

    for path in &opts.check_paths {
        let p = path.trim();
        if p.is_empty() {
            continue;
        }
        let exists = Path::new(p).exists();
        results.push(CheckResult {
            name: format!("path:{p}"),
            passed: exists,
            detail: if exists {
                "exists".into()
            } else {
                "missing after act".into()
            },
        });
    }

    if opts.require_detached_server {
        results.push(check_detached_server(evidence_audit).await);
    }

    if opts.min_html_bytes > 0 || opts.min_css_bytes > 0 {
        results.extend(check_written_file_substance(
            evidence_audit,
            opts.min_html_bytes,
            opts.min_css_bytes,
        ));
    }

    // Informational audit summary — only counts as a hard check when it's
    // the sole configured signal and require_tool_use already covered it.
    // Otherwise keep it as a soft always-pass breadcrumb for verify prompts.
    if results.is_empty() {
        results.push(CheckResult {
            name: "audit_log".into(),
            passed: true,
            detail: if act_audit.trim().is_empty() {
                "no tool calls in audit log (informational; set require_tool_use or test_command for hard checks)".into()
            } else {
                format!("{} bytes of recent audit activity (informational)", act_audit.len())
            },
        });
    }

    CheckReport { results }
}

/// Paths from write_file / create_dir in the audit must meet size floors
/// so a "server is up" stub site cannot pass.
///
/// Also expands to sibling `.html` / `.css` in the same directories — so a
/// retry that only rewrites one short page still gets judged against the
/// full site on disk (prior-attempt CSS/HTML still counts).
fn check_written_file_substance(
    audit_tail: &str,
    min_html: u64,
    min_css: u64,
) -> Vec<CheckResult> {
    let paths = expand_site_artifact_paths(&paths_written_in_audit(audit_tail));
    let html: Vec<_> = paths
        .iter()
        .filter(|p| p.ends_with(".html") || p.ends_with(".htm"))
        .cloned()
        .collect();
    let css: Vec<_> = paths
        .iter()
        .filter(|p| p.ends_with(".css"))
        .cloned()
        .collect();

    let mut out = Vec::new();
    if min_html > 0 {
        if html.is_empty() {
            out.push(CheckResult {
                name: "html_substance".into(),
                passed: false,
                detail: format!(
                    "no .html files found for this run — need real pages (≥ {min_html} bytes each)"
                ),
            });
        } else {
            for p in &html {
                out.push(file_size_check("html_substance", p, min_html));
            }
        }
    }
    if min_css > 0 {
        if css.is_empty() {
            out.push(CheckResult {
                name: "css_substance".into(),
                passed: false,
                detail: format!(
                    "no .css files found for this run — need a real stylesheet (≥ {min_css} bytes)"
                ),
            });
        } else {
            for p in &css {
                out.push(file_size_check("css_substance", p, min_css));
            }
        }
    }
    out
}

/// From write/create paths in the audit, include sibling site artifacts on
/// disk so retries don't "forget" CSS/HTML written earlier in the run.
fn expand_site_artifact_paths(written: &[String]) -> Vec<String> {
    use std::collections::BTreeSet;
    let mut out: BTreeSet<String> = BTreeSet::new();
    let mut dirs: BTreeSet<std::path::PathBuf> = BTreeSet::new();

    for p in written {
        let path = Path::new(p);
        let lower = p.to_ascii_lowercase();
        if lower.ends_with(".html") || lower.ends_with(".htm") || lower.ends_with(".css") {
            out.insert(p.clone());
        }
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                dirs.insert(parent.to_path_buf());
            }
        }
        // create_dir targets (no extension) — scan that folder for artifacts.
        if path.extension().is_none() {
            dirs.insert(path.to_path_buf());
        }
    }

    for dir in dirs {
        let entries = match std::fs::read_dir(&dir) {
            Ok(e) => e,
            Err(_) => continue,
        };
        for ent in entries.flatten() {
            let p = ent.path();
            let Some(name) = p.file_name().and_then(|s| s.to_str()) else {
                continue;
            };
            let lower = name.to_ascii_lowercase();
            if lower.ends_with(".html") || lower.ends_with(".htm") || lower.ends_with(".css") {
                out.insert(p.to_string_lossy().to_string());
            }
        }
    }
    out.into_iter().collect()
}

fn file_size_check(name: &str, path: &str, min_bytes: u64) -> CheckResult {
    match std::fs::metadata(path) {
        Ok(meta) => {
            let n = meta.len();
            if n >= min_bytes {
                CheckResult {
                    name: name.into(),
                    passed: true,
                    detail: format!("{path} is {n} bytes (≥ {min_bytes})"),
                }
            } else {
                let short = min_bytes.saturating_sub(n);
                CheckResult {
                    name: name.into(),
                    passed: false,
                    detail: format!(
                        "{path} is only {n} bytes — need ≥ {min_bytes} \
                         (short by {short}). Rewrite via write_file with more \
                         real sections/copy/CSS rules — do not restart the server."
                    ),
                }
            }
        }
        Err(_) => CheckResult {
            name: name.into(),
            passed: false,
            detail: format!("{path} missing after write_file"),
        },
    }
}

/// Extract file/dir paths from write_file and create_dir tool calls in audit JSONL.
pub fn paths_written_in_audit(audit_tail: &str) -> Vec<String> {
    let mut paths = Vec::new();
    for line in audit_tail.lines() {
        let is_write = line.contains("\"write_file\"") || line.contains("write_file");
        let is_mkdir = line.contains("\"create_dir\"") || line.contains("create_dir");
        if !is_write && !is_mkdir {
            continue;
        }
        if let Some(path) = extract_json_string_field(line, "path") {
            if !path.is_empty() && !paths.contains(&path) {
                paths.push(path);
            }
        }
    }
    paths
}

fn extract_json_string_field(line: &str, field: &str) -> Option<String> {
    let patterns = [
        format!("\"{field}\":\""),
        format!("\"{field}\": \""),
    ];
    for pat in &patterns {
        if let Some(idx) = line.find(pat.as_str()) {
            let rest = &line[idx + pat.len()..];
            let mut out = String::new();
            let mut chars = rest.chars().peekable();
            while let Some(c) = chars.next() {
                if c == '\\' {
                    if let Some(n) = chars.next() {
                        out.push(n);
                    }
                    continue;
                }
                if c == '"' {
                    break;
                }
                out.push(c);
            }
            if !out.is_empty() {
                return Some(out);
            }
        }
    }
    None
}

async fn check_detached_server(audit_tail: &str) -> CheckResult {
    let ports = crate::tools::shell::ports_claimed_in_audit(audit_tail);
    if ports.is_empty() {
        return CheckResult {
            name: "detached_server".into(),
            passed: false,
            detail: "no detached server with likely_port= in audit — call shell with \
                     `python3 -m http.server <port>` (auto-detaches) and confirm it binds"
                .into(),
        };
    }
    let mut up = Vec::new();
    let mut down = Vec::new();
    for p in &ports {
        if crate::tools::shell::port_is_open(*p).await {
            up.push(*p);
        } else {
            down.push(*p);
        }
    }
    if !up.is_empty() && down.is_empty() {
        CheckResult {
            name: "detached_server".into(),
            passed: true,
            detail: format!(
                "listening on {} — http://127.0.0.1:{}/",
                up.iter()
                    .map(|p| p.to_string())
                    .collect::<Vec<_>>()
                    .join(", "),
                up[0]
            ),
        }
    } else if !up.is_empty() {
        CheckResult {
            name: "detached_server".into(),
            passed: true,
            detail: format!("listening on {up:?}; not yet up: {down:?}"),
        }
    } else {
        CheckResult {
            name: "detached_server".into(),
            passed: false,
            detail: format!(
                "claimed ports {ports:?} but nothing accepts TCP — restart \
                 `python3 -m http.server` on a free port (try listening_ports first)"
            ),
        }
    }
}

fn count_audit_tool_calls(audit_tail: &str) -> usize {
    audit_tail
        .lines()
        .filter(|l| {
            let t = l.trim();
            !t.is_empty() && (t.contains("\"tool\"") || t.contains("\"event\":\"tool_call\""))
        })
        .count()
}

async fn run_shell_check(name: &str, command: &str) -> CheckResult {
    match Command::new("sh")
        .arg("-lc")
        .arg(command)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
    {
        Ok(out) => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            let stderr = String::from_utf8_lossy(&out.stderr);
            let detail = format!(
                "exit={} stdout={} stderr={}",
                out.status.code().unwrap_or(-1),
                crate::util::truncate(stdout.trim(), 400),
                crate::util::truncate(stderr.trim(), 200)
            );
            CheckResult {
                name: name.into(),
                passed: out.status.success(),
                detail,
            }
        }
        Err(e) => CheckResult {
            name: name.into(),
            passed: false,
            detail: format!("failed to run check: {e}"),
        },
    }
}

/// Read the last `max_lines` from the audit log for verifier context.
pub fn read_audit_tail(path: &Path, max_lines: usize) -> Result<String> {
    read_audit_tail_since(path, 0, max_lines)
}

/// Like [`read_audit_tail`], but only keeps events with `ts >= since_ts`.
/// Used by the harness so plan-scout / prior-run noise doesn't satisfy
/// `require_tool_use` or leak stale `likely_port=` markers.
pub fn read_audit_tail_since(path: &Path, since_ts: i64, max_lines: usize) -> Result<String> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("reading audit log {}", path.display()))?;
    let mut kept: Vec<&str> = Vec::new();
    for line in raw.lines() {
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        if since_ts > 0 {
            let ts = line_ts(t).unwrap_or(0);
            if ts < since_ts {
                continue;
            }
        }
        kept.push(t);
    }
    if kept.is_empty() {
        return Ok(String::new());
    }
    let start = kept.len().saturating_sub(max_lines);
    Ok(kept[start..].join("\n"))
}

fn line_ts(line: &str) -> Option<i64> {
    // Fast path: {"ts":1234567890,...
    let idx = line.find("\"ts\"")?;
    let rest = &line[idx + 4..];
    let rest = rest.trim_start_matches(|c: char| c == ':' || c.is_whitespace());
    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    digits.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn require_tool_use_fails_on_empty_audit() {
        let opts = CheckOptions {
            require_tool_use: true,
            ..Default::default()
        };
        let report = run_checks(&opts, "").await;
        assert!(!report.all_passed());
        assert!(report.results.iter().any(|r| r.name == "require_tool_use"));
    }

    #[tokio::test]
    async fn require_tool_use_passes_with_audit_line() {
        let opts = CheckOptions {
            require_tool_use: true,
            ..Default::default()
        };
        let line = r#"{"ts":1,"event":"tool_call","tool":"read_file","args":{},"decision":"allow","result_summary":"ok"}"#;
        let report = run_checks(&opts, line).await;
        assert!(report.all_passed());
    }

    #[tokio::test]
    async fn check_paths_detects_missing() {
        let opts = CheckOptions {
            check_paths: vec!["/no/such/path/localmind-missing-xyz".into()],
            ..Default::default()
        };
        let report = run_checks(&opts, "").await;
        assert!(!report.all_passed());
    }

    #[tokio::test]
    async fn detached_server_fails_without_port_marker() {
        let opts = CheckOptions {
            require_detached_server: true,
            ..Default::default()
        };
        let report = run_checks(&opts, r#"{"tool":"write_file"}"#).await;
        assert!(!report.all_passed());
        assert!(report
            .results
            .iter()
            .any(|r| r.name == "detached_server" && !r.passed));
    }

    #[tokio::test]
    async fn html_substance_rejects_stub() {
        let dir = tempfile::tempdir().unwrap();
        let stub = dir.path().join("index.html");
        std::fs::write(&stub, "<h1>hi</h1>").unwrap();
        let audit = format!(
            r#"{{"ts":1,"tool":"write_file","args":{{"path":"{}"}}}}"#,
            stub.display()
        );
        let opts = CheckOptions {
            min_html_bytes: 500,
            ..Default::default()
        };
        let report = run_checks(&opts, &audit).await;
        assert!(!report.all_passed());
        assert!(report
            .results
            .iter()
            .any(|r| r.name == "html_substance" && !r.passed));
    }

    #[tokio::test]
    async fn substance_finds_sibling_css_on_disk() {
        let dir = tempfile::tempdir().unwrap();
        let html = dir.path().join("about.html");
        let css = dir.path().join("styles.css");
        std::fs::write(&html, "x".repeat(4000)).unwrap();
        std::fs::write(&css, "y".repeat(3000)).unwrap();
        // Act only rewrote about.html — CSS should still count via sibling scan.
        let audit = format!(
            r#"{{"ts":1,"tool":"write_file","args":{{"path":"{}"}}}}"#,
            html.display()
        );
        let opts = CheckOptions {
            min_html_bytes: 2000,
            min_css_bytes: 1800,
            ..Default::default()
        };
        let report = run_checks(&opts, &audit).await;
        assert!(
            report.all_passed(),
            "expected sibling CSS to count: {}",
            report.summary()
        );
    }

    #[test]
    fn retry_guidance_blocks_extra_servers_when_up() {
        let report = CheckReport {
            results: vec![
                CheckResult {
                    name: "detached_server".into(),
                    passed: true,
                    detail: "listening on 8084".into(),
                },
                CheckResult {
                    name: "html_substance".into(),
                    passed: false,
                    detail: "index.html is only 1479 bytes".into(),
                },
            ],
        };
        let opts = CheckOptions {
            min_html_bytes: 2000,
            min_css_bytes: 1800,
            ..Default::default()
        };
        let g = retry_guidance(&report, &opts);
        assert!(g.contains("SERVER ALREADY UP"), "{g}");
        assert!(g.contains("REWRITE ALL"), "{g}");
        assert!(g.contains("2000"), "{g}");
    }
}
