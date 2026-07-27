//! Append-only JSONL audit log of every tool call.
//! Intentionally plain-text so your IT team can `tail -f` it.

use anyhow::{Context, Result};
use serde::Serialize;
use serde_json::Value;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;

pub struct AuditLog {
    path: PathBuf,
    inner: Mutex<()>,
}

#[derive(Serialize)]
struct Entry<'a> {
    ts: i64,
    event: &'a str,
    tool: &'a str,
    args: Value,
    decision: &'a str,
    result_summary: &'a str,
}

impl AuditLog {
    pub fn open() -> Result<Self> {
        let base = directories::ProjectDirs::from("com", "calligoit", "localmind")
            .map(|p| p.data_dir().to_path_buf())
            .unwrap_or_else(|| PathBuf::from("./data"));
        std::fs::create_dir_all(&base).ok();
        Ok(Self {
            path: base.join("audit.log"),
            inner: Mutex::new(()),
        })
    }

    pub fn record(
        &self,
        tool: &str,
        args: &serde_json::Value,
        decision: &str,
        result_summary: &str,
    ) {
        let entry = Entry {
            ts: crate::util::now_ts(),
            event: "tool_call",
            tool,
            args: redact_args(args),
            decision,
            result_summary,
        };
        let line = serde_json::to_string(&entry).unwrap_or_default();
        let _guard = self.inner.lock();
        if let Ok(mut f) = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
        {
            let _ = writeln!(f, "{line}");
        }
    }

    pub fn path(&self) -> &std::path::Path {
        &self.path
    }

    /// Last `max_lines` of the JSONL audit log (newest at the end).
    pub fn read_tail(&self, max_lines: usize) -> Result<String> {
        let raw = std::fs::read_to_string(&self.path)
            .with_context(|| format!("reading audit log {}", self.path.display()))?;
        let lines: Vec<&str> = raw.lines().collect();
        if lines.is_empty() {
            return Ok(String::new());
        }
        let start = lines.len().saturating_sub(max_lines.max(1));
        Ok(lines[start..].join("\n"))
    }
}

/// Walk the args JSON and truncate any large string field — the agent could
/// be writing a private key, a long doc, or a `/remember`'d secret. Without
/// this, the args land verbatim in audit.log and persist forever.
const MAX_STRING_LEN: usize = 1024;

fn redact_args(v: &Value) -> Value {
    match v {
        Value::String(s) => Value::String(truncate_string(s)),
        Value::Array(arr) => Value::Array(arr.iter().map(redact_args).collect()),
        Value::Object(obj) => {
            let mut out = serde_json::Map::with_capacity(obj.len());
            for (k, vv) in obj {
                out.insert(k.clone(), redact_args(vv));
            }
            Value::Object(out)
        }
        _ => v.clone(),
    }
}

fn truncate_string(s: &str) -> String {
    let n = s.chars().count();
    if n <= MAX_STRING_LEN {
        return s.to_string();
    }
    let head: String = s.chars().take(MAX_STRING_LEN).collect();
    format!("{head}…[truncated, original was {n} chars]")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redacts_long_string_field() {
        let big = "x".repeat(5000);
        let v = serde_json::json!({ "content": big, "path": "/tmp/x" });
        let r = redact_args(&v);
        let content = r.get("content").unwrap().as_str().unwrap();
        assert!(content.chars().count() < 5000);
        assert!(content.contains("truncated"));
        assert_eq!(r.get("path").unwrap().as_str().unwrap(), "/tmp/x");
    }

    #[test]
    fn redacts_recursively() {
        let big = "y".repeat(2000);
        let v = serde_json::json!({ "outer": { "inner": [big, "short"] } });
        let r = redact_args(&v);
        let arr = r["outer"]["inner"].as_array().unwrap();
        assert!(arr[0].as_str().unwrap().contains("truncated"));
        assert_eq!(arr[1].as_str().unwrap(), "short");
    }

    #[test]
    fn leaves_small_payload_alone() {
        let v = serde_json::json!({ "command": "ls -la", "cwd": "/tmp" });
        assert_eq!(redact_args(&v), v);
    }

    #[test]
    fn read_tail_returns_last_lines() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("audit.log");
        std::fs::write(&path, "line1\nline2\nline3\n").unwrap();
        let log = AuditLog {
            path,
            inner: Mutex::new(()),
        };
        let tail = log.read_tail(2).unwrap();
        assert_eq!(tail, "line2\nline3");
    }
}
