//! Shell execution — deliberately boring. No hidden windows, no encoded
//! commands. The user approves each call via the permission prompt, which
//! surfaces any validator warning inline (e.g. `rm -rf`, fork bomb).
//!
//! On Windows: `cmd /c <command>`. On Unix: `sh -c <command>`.
//! stdout, stderr, exit code are captured as plain text.
//!
//! Long-running / background commands (trailing `&`, `nohup`,
//! `python -m http.server`, etc.) are detached so the agent turn is not
//! blocked waiting for a process that never exits. Detached PIDs are
//! tracked for optional cleanup on REPL quit. When a listen port can be
//! guessed, we poll until it accepts TCP (or fail with stderr).

use crate::tools::registry::ToolContext;
use crate::tools::shell_validation::{self, ValidationResult};
use anyhow::{anyhow, Result};
use once_cell::sync::Lazy;
use serde_json::Value;
use std::process::Stdio;
use std::sync::Mutex;
use std::time::Duration;
use tokio::net::TcpStream;
use tokio::process::{Child, Command};

/// PIDs of processes we intentionally detached this session. Cleaned up
/// by [`cleanup_detached`] on REPL quit (best-effort).
static DETACHED_PIDS: Lazy<Mutex<Vec<u32>>> = Lazy::new(|| Mutex::new(Vec::new()));

/// Inspect a command against the current PermissionMode and return either a
/// hard block, a warning to surface in the permission prompt, or nothing.
pub fn precheck(ctx: &ToolContext, command: &str) -> ValidationResult {
    shell_validation::validate_command(command, ctx.permissions.mode())
}

pub async fn run(ctx: &ToolContext, args: &Value) -> Result<String> {
    let command = args
        .get("command")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("missing command"))?;
    let cwd = args.get("cwd").and_then(|v| v.as_str()).map(str::to_string);
    // Explicit detach flag OR auto-detect background / long-running servers.
    let detach = args
        .get("detach")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
        || shell_validation::wants_detach(command);
    // 60s default — the model can pass a larger value when it knows a
    // command is long-running. Lower default keeps a runaway shell from
    // hogging a turn for two minutes. Ignored when detaching.
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

    if detach {
        return run_detached(command, cwd.as_deref()).await;
    }

    let mut cmd = spawn_shell(command);
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

/// Spawn a long-running command without waiting. When we can guess a listen
/// port, poll until TCP accepts (or the process exits) so callers get a
/// honest success/failure instead of a silent dead server.
async fn run_detached(command: &str, cwd: Option<&str>) -> Result<String> {
    let cleaned = shell_validation::strip_trailing_ampersand(command);
    let port = shell_validation::guess_listen_port(command);

    // Capture stderr so bind failures ("Address already in use") surface.
    let stderr_path = std::env::temp_dir().join(format!(
        "localmind-detach-{}.err",
        std::process::id()
    ));
    let stderr_file = std::fs::File::create(&stderr_path)
        .map_err(|e| anyhow!("cannot create detach stderr log: {e}"))?;

    let mut cmd = spawn_shell(&cleaned);
    if let Some(dir) = cwd {
        cmd.current_dir(dir);
    }
    cmd.stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::from(stderr_file))
        .kill_on_drop(false);

    // New process group so the child survives if the parent shell exits and
    // so cleanup can signal the whole tree.
    #[cfg(unix)]
    {
        cmd.process_group(0);
    }

    let mut child = cmd
        .spawn()
        .map_err(|e| anyhow!("failed to spawn detached command: {e}"))?;
    let pid = child
        .id()
        .ok_or_else(|| anyhow!("detached child has no pid"))?;

    if let Ok(mut guard) = DETACHED_PIDS.lock() {
        guard.push(pid);
    }

    let ready = wait_until_ready(&mut child, port).await?;
    let stderr_txt = std::fs::read_to_string(&stderr_path).unwrap_or_default();
    let _ = std::fs::remove_file(&stderr_path);

    if !ready.alive {
        let _ = remove_tracked_pid(pid);
        return Err(anyhow!(
            "detached process exited immediately (pid={pid}).\n--- stderr ---\n{}",
            crate::util::truncate(stderr_txt.trim(), 2000)
        ));
    }

    if let Some(p) = port {
        if !ready.port_up {
            // Kill the orphaned process — it isn't serving.
            kill_pid(pid);
            let _ = remove_tracked_pid(pid);
            return Err(anyhow!(
                "detached process running (pid={pid}) but nothing is listening on port {p}.\n\
                 --- stderr ---\n{}",
                crate::util::truncate(stderr_txt.trim(), 2000)
            ));
        }
    }

    // Keep the Child alive; kill_on_drop is false.
    std::mem::forget(child);

    let mut msg = format!(
        "detached pid={pid}\ncommand: {cleaned}\n\
         Process is running in the background — do NOT wait for it. \
         Tell the user the URL."
    );
    if let Some(p) = port {
        msg.push_str(&format!(
            "\nlikely_port={p}\nurl_hint=http://127.0.0.1:{p}/\nport_check=listening"
        ));
    }
    if !stderr_txt.trim().is_empty() {
        msg.push_str(&format!(
            "\n--- stderr (non-fatal) ---\n{}",
            crate::util::truncate(stderr_txt.trim(), 500)
        ));
    }
    Ok(msg)
}

struct ReadyState {
    alive: bool,
    port_up: bool,
}

async fn wait_until_ready(child: &mut Child, port: Option<u16>) -> Result<ReadyState> {
    // ~2.5s total — enough for python -m http.server to bind on a quiet machine.
    for _ in 0..25 {
        tokio::time::sleep(Duration::from_millis(100)).await;
        if child.try_wait()?.is_some() {
            return Ok(ReadyState {
                alive: false,
                port_up: false,
            });
        }
        if let Some(p) = port {
            if port_is_open(p).await {
                return Ok(ReadyState {
                    alive: true,
                    port_up: true,
                });
            }
        }
    }
    // Still alive after settle window.
    let alive = child.try_wait()?.is_none();
    let port_up = match port {
        Some(p) => port_is_open(p).await,
        None => true, // nothing to verify
    };
    Ok(ReadyState { alive, port_up })
}

/// True when something accepts TCP on 127.0.0.1:`port`.
pub async fn port_is_open(port: u16) -> bool {
    tokio::time::timeout(
        Duration::from_millis(200),
        TcpStream::connect(("127.0.0.1", port)),
    )
    .await
    .ok()
    .and_then(Result::ok)
    .is_some()
}

fn spawn_shell(command: &str) -> Command {
    if cfg!(windows) {
        let mut c = Command::new("cmd");
        c.arg("/c").arg(command);
        c
    } else {
        let mut c = Command::new("sh");
        c.arg("-c").arg(command);
        c
    }
}

fn remove_tracked_pid(pid: u32) -> Result<(), ()> {
    if let Ok(mut g) = DETACHED_PIDS.lock() {
        g.retain(|p| *p != pid);
    }
    Ok(())
}

fn kill_pid(pid: u32) {
    if cfg!(windows) {
        let _ = std::process::Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/T", "/F"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    } else {
        let _ = std::process::Command::new("kill")
            .args(["-TERM", &format!("-{pid}")])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        let _ = std::process::Command::new("kill")
            .args(["-TERM", &pid.to_string()])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
}

/// Best-effort kill of processes we detached this session. Called on
/// `/quit`. Safe to call multiple times; ignores already-exited PIDs.
pub fn cleanup_detached() {
    let pids: Vec<u32> = DETACHED_PIDS
        .lock()
        .map(|mut g| std::mem::take(&mut *g))
        .unwrap_or_default();
    if pids.is_empty() {
        return;
    }
    eprintln!("  · cleaning up {} detached shell process(es)", pids.len());
    for pid in pids {
        kill_pid(pid);
    }
}

/// Parse recent audit text for detached-server markers (`likely_port=N`
/// / `url_hint=…`) produced by [`run_detached`].
pub fn ports_claimed_in_audit(audit_tail: &str) -> Vec<u16> {
    let mut ports = Vec::new();
    static RE: Lazy<regex::Regex> = Lazy::new(|| {
        regex::Regex::new(r"(?:likely_port=|url_hint=http://127\.0\.0\.1:)(\d{2,5})").unwrap()
    });
    for caps in RE.captures_iter(audit_tail) {
        if let Ok(p) = caps[1].parse::<u16>() {
            if p > 0 && !ports.contains(&p) {
                ports.push(p);
            }
        }
    }
    ports
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::shell_validation;

    #[test]
    fn strip_ampersand_helpers() {
        assert_eq!(
            shell_validation::strip_trailing_ampersand("python3 -m http.server 8080 &"),
            "python3 -m http.server 8080"
        );
        assert_eq!(
            shell_validation::strip_trailing_ampersand("echo hi && true"),
            "echo hi && true"
        );
    }

    #[test]
    fn guess_port_from_http_server() {
        assert_eq!(
            shell_validation::guess_listen_port("python3 -m http.server 8082 &"),
            Some(8082)
        );
        assert_eq!(
            shell_validation::guess_listen_port("python3 -m http.server"),
            Some(8000)
        );
    }

    #[test]
    fn parse_ports_from_audit_summary() {
        let audit = r#"{"tool":"shell","result_summary":"detached pid=1\nlikely_port=8088\nurl_hint=http://127.0.0.1:8088/\nport_check=listening"}"#;
        assert_eq!(ports_claimed_in_audit(audit), vec![8088]);
    }
}
