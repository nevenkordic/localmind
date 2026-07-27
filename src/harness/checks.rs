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

/// Run configured ground-truth checks after the act stage.
pub async fn run_checks(test_command: &str, audit_tail: &str) -> CheckReport {
    let mut results = Vec::new();

    if !test_command.trim().is_empty() {
        results.push(run_shell_check("test_command", test_command).await);
    }

    results.push(CheckResult {
        name: "audit_log".into(),
        passed: true,
        detail: if audit_tail.trim().is_empty() {
            "no tool calls in audit log (informational)".into()
        } else {
            format!("{} bytes of recent audit activity", audit_tail.len())
        },
    });

    CheckReport { results }
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
