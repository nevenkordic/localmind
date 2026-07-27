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

/// Options for post-act ground-truth checks.
#[derive(Debug, Clone, Default)]
pub struct CheckOptions {
    pub test_command: String,
    /// Fail when the audit tail has no tool calls (act claimed work but
    /// touched nothing).
    pub require_tool_use: bool,
    /// Paths that must exist after act (relative or absolute).
    pub check_paths: Vec<String>,
}

/// Run configured ground-truth checks after the act stage.
pub async fn run_checks(opts: &CheckOptions, audit_tail: &str) -> CheckReport {
    let mut results = Vec::new();

    if !opts.test_command.trim().is_empty() {
        results.push(run_shell_check("test_command", &opts.test_command).await);
    }

    if opts.require_tool_use {
        let tool_calls = count_audit_tool_calls(audit_tail);
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

    // Informational audit summary — only counts as a hard check when it's
    // the sole configured signal and require_tool_use already covered it.
    // Otherwise keep it as a soft always-pass breadcrumb for verify prompts.
    if results.is_empty() {
        results.push(CheckResult {
            name: "audit_log".into(),
            passed: true,
            detail: if audit_tail.trim().is_empty() {
                "no tool calls in audit log (informational; set require_tool_use or test_command for hard checks)".into()
            } else {
                format!("{} bytes of recent audit activity (informational)", audit_tail.len())
            },
        });
    }

    CheckReport { results }
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
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("reading audit log {}", path.display()))?;
    let lines: Vec<&str> = raw.lines().collect();
    if lines.is_empty() {
        return Ok(String::new());
    }
    let start = lines.len().saturating_sub(max_lines);
    Ok(lines[start..].join("\n"))
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
}
