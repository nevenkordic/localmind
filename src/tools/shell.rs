//! Shell execution — deliberately boring. No hidden windows, no encoded
//! commands. The user approves each call via the permission prompt, which
//! surfaces any validator warning inline (e.g. `rm -rf`, fork bomb).
//!
//! On Windows: `cmd /c <command>`. On Unix: `sh -c <command>`.
//! stdout, stderr, exit code are captured as plain text.

use crate::tools::registry::ToolContext;
use crate::tools::shell_validation::{self, ValidationResult};
use anyhow::{anyhow, Result};
use serde_json::Value;
use std::time::Duration;
use tokio::process::Command;

/// Inspect a command against the current PermissionMode and return either a
/// hard block, a warning to surface in the prompt, or nothing.
pub fn precheck(ctx: &ToolContext, command: &str) -> ValidationResult {
    shell_validation::validate_command(command, ctx.permissions.mode())
}

pub async fn run(ctx: &ToolContext, args: &Value) -> Result<String> {
    let command = args
        .get("command")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("missing command"))?;
    let cwd = args.get("cwd").and_then(|v| v.as_str()).map(str::to_string);
    // 60s default — the model can pass a larger value when it knows a
    // command is long-running. Lower default keeps a runaway shell from
    // hogging a turn for two minutes.
    let timeout_secs = args
        .get("timeout_secs")
        .and_then(|v| v.as_u64())
        .unwrap_or(60);

    // Optional allow-list: if shell_allow_regex is set, the command must match.
    if !ctx.cfg.tools.shell_allow_regex.is_empty() {
        let re = regex::Regex::new(&ctx.cfg.tools.shell_allow_regex)
            .map_err(|e| anyhow!("invalid shell_allow_regex: {e}"))?;
        if !re.is_match(command) {
            return Err(anyhow!("command not permitted by shell_allow_regex"));
        }
    }

    let mut cmd = if cfg!(windows) {
        let mut c = Command::new("cmd");
        c.arg("/c").arg(command);
        c
    } else {
        let mut c = Command::new("sh");
        c.arg("-c").arg(command);
        c
    };
    if let Some(dir) = cwd {
        cmd.current_dir(dir);
    }
    cmd.kill_on_drop(true);

    let fut = cmd.output();
    let out = tokio::time::timeout(Duration::from_secs(timeout_secs), fut)
        .await
        .map_err(|_| anyhow!("shell command timed out after {timeout_secs}s"))??;

    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    let code = out.status.code().unwrap_or(-1);
    Ok(format!(
        "exit={code}\n--- stdout ---\n{}\n--- stderr ---\n{}",
        crate::util::truncate(&stdout, 30_000),
        crate::util::truncate(&stderr, 10_000),
    ))
}
