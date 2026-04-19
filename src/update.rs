//! Non-blocking update checker.
//!
//! Queries `api.github.com/repos/.../releases/latest` at most once every
//! `check_interval_hours`, caches the result in `<data_dir>/update-check.json`,
//! and prints a one-line banner on startup when the cached tag is newer than
//! the compiled version. The HTTP call runs on a fire-and-forget tokio task
//! with a short timeout, so it never adds latency to the user's REPL or
//! `ask` calls — the *first* time a user runs after a new release lands, the
//! banner is absent; subsequent runs show it.
//!
//! Opt-out: `[updates] check = false` in `local.toml` or
//! `LOCALMIND_NO_UPDATE_CHECK=1` in the environment.

use crate::config::Config;
use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;

const REPO: &str = "nevenkordic/localmind";
const CACHE_FILE: &str = "update-check.json";
const FETCH_TIMEOUT: Duration = Duration::from_secs(3);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cached {
    pub checked_at: DateTime<Utc>,
    pub latest_tag: String,
    #[serde(default)]
    pub html_url: String,
}

fn cache_path() -> Option<PathBuf> {
    directories::ProjectDirs::from("com", "calligoit", "localmind")
        .map(|p| p.data_dir().join(CACHE_FILE))
}

pub fn current_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Compare "v0.1.0" / "0.2.3" / "v1.2.3-rc1" style tags. Strips a leading
/// `v`, takes only the numeric `a.b.c` prefix (so pre-release / build-meta
/// suffixes are compared by their core only). Returns None on anything that
/// doesn't parse — the caller treats that as "not newer" so a malformed tag
/// never produces a false upgrade prompt.
fn parse_version(s: &str) -> Option<(u64, u64, u64)> {
    let s = s.trim().trim_start_matches('v').trim_start_matches('V');
    let core: String = s
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '.')
        .collect();
    let mut parts = core.split('.');
    let a: u64 = parts.next()?.parse().ok()?;
    let b: u64 = parts.next()?.parse().ok()?;
    let c: u64 = parts.next().unwrap_or("0").parse().ok()?;
    Some((a, b, c))
}

pub fn is_newer(current: &str, latest: &str) -> bool {
    match (parse_version(current), parse_version(latest)) {
        (Some(a), Some(b)) => b > a,
        _ => false,
    }
}

fn read_cache() -> Option<Cached> {
    let p = cache_path()?;
    let raw = std::fs::read_to_string(&p).ok()?;
    serde_json::from_str(&raw).ok()
}

fn write_cache(c: &Cached) -> Result<()> {
    let p = cache_path().context("no data dir available for update cache")?;
    if let Some(parent) = p.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let raw = serde_json::to_string_pretty(c)?;
    std::fs::write(&p, raw)?;
    Ok(())
}

#[derive(Debug, Deserialize)]
struct GhRelease {
    tag_name: String,
    #[serde(default)]
    html_url: String,
}

async fn fetch_latest() -> Result<Cached> {
    let url = format!("https://api.github.com/repos/{REPO}/releases/latest");
    let client = reqwest::Client::builder()
        .timeout(FETCH_TIMEOUT)
        .user_agent(format!("localmind-updatecheck/{}", current_version()))
        .build()?;
    let resp = client
        .get(&url)
        .header("Accept", "application/vnd.github+json")
        .send()
        .await
        .context("GET releases/latest")?;
    if !resp.status().is_success() {
        anyhow::bail!("github returned {}", resp.status());
    }
    let gh: GhRelease = resp.json().await.context("parsing release json")?;
    Ok(Cached {
        checked_at: Utc::now(),
        latest_tag: gh.tag_name,
        html_url: gh.html_url,
    })
}

fn is_stale(cache: Option<&Cached>, interval_hours: u64) -> bool {
    match cache {
        None => true,
        Some(c) => {
            let age = Utc::now().signed_duration_since(c.checked_at);
            // Future timestamp → clock skew, treat as stale. Compare in
            // seconds (not hours) because chrono's num_hours truncates toward
            // zero, so a -3599s duration reports 0 and would slip past a
            // `num_hours() < 0` check.
            if age.num_seconds() < 0 {
                return true;
            }
            age.num_seconds() as u64 >= interval_hours.saturating_mul(3600)
        }
    }
}

fn banner_from(cache: &Cached) -> Option<String> {
    if is_newer(current_version(), &cache.latest_tag) {
        Some(format!(
            "\x1b[1;36m↑ {tag}\x1b[0m available (you have {cur}) — run \x1b[1mllm update\x1b[0m to upgrade",
            tag = cache.latest_tag,
            cur = current_version()
        ))
    } else {
        None
    }
}

fn spawn_refresh() {
    tokio::spawn(async move {
        match fetch_latest().await {
            Ok(c) => {
                if let Err(e) = write_cache(&c) {
                    tracing::debug!("update cache write failed: {e:#}");
                } else {
                    tracing::debug!("update check cached: latest = {}", c.latest_tag);
                }
            }
            Err(e) => tracing::debug!("update check failed: {e:#}"),
        }
    });
}

/// Called near the start of `main`. Prints the banner (if a previously-cached
/// release is newer than the running binary) and, if the cache is stale,
/// spawns an async refresh for next time. Never blocks.
pub fn maybe_check_and_print(cfg: &Config) {
    if std::env::var("LOCALMIND_NO_UPDATE_CHECK").is_ok() {
        return;
    }
    if !cfg.updates.check {
        return;
    }
    let cache = read_cache();
    if let Some(c) = &cache {
        if let Some(line) = banner_from(c) {
            // Write to stderr so command output on stdout isn't polluted
            // (important for `llm ask "..."` in scripts).
            eprintln!("{line}");
        }
    }
    if is_stale(cache.as_ref(), cfg.updates.check_interval_hours) {
        spawn_refresh();
    }
}

/// Synchronous check used by `llm update` so we can tell the user whether
/// they're already on latest before re-running the installer. Unlike
/// `maybe_check_and_print`, this *does* block on the HTTP call — but only
/// for FETCH_TIMEOUT at worst.
pub async fn fetch_and_cache() -> Result<Cached> {
    let c = fetch_latest().await?;
    let _ = write_cache(&c);
    Ok(c)
}

/// Kick off the installer one-liner for the current platform. Inherits
/// stdin/stdout/stderr so sudo prompts from the installer still work.
pub fn run_installer() -> Result<()> {
    let url = format!("https://raw.githubusercontent.com/{REPO}/main/install.sh");

    #[cfg(not(windows))]
    {
        let cmd = format!("curl -fsSL {url} | sh");
        eprintln!("\x1b[1;36m==>\x1b[0m {cmd}");
        let status = std::process::Command::new("sh")
            .arg("-c")
            .arg(&cmd)
            .status()
            .context("spawning sh for installer")?;
        if !status.success() {
            anyhow::bail!(
                "installer exited with {}",
                status
                    .code()
                    .map(|c| c.to_string())
                    .unwrap_or_else(|| "signal".into())
            );
        }
        Ok(())
    }

    #[cfg(windows)]
    {
        // Windows users: point at the PowerShell installer instead.
        let ps_url = format!("https://raw.githubusercontent.com/{REPO}/main/scripts/install.ps1");
        let cmd = format!("irm {ps_url} | iex");
        eprintln!("==> powershell -NoProfile -Command \"{cmd}\"");
        let status = std::process::Command::new("powershell")
            .args(["-NoProfile", "-Command", &cmd])
            .status()
            .context("spawning powershell for installer")?;
        if !status.success() {
            anyhow::bail!("installer exited with {:?}", status.code());
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_parse_strips_v_prefix() {
        assert_eq!(parse_version("v0.1.0"), Some((0, 1, 0)));
        assert_eq!(parse_version("V1.2.3"), Some((1, 2, 3)));
        assert_eq!(parse_version("0.0.1"), Some((0, 0, 1)));
    }

    #[test]
    fn version_parse_tolerates_prerelease_suffix() {
        // We deliberately compare the numeric core only. This means
        // v0.2.0-rc1 and v0.2.0 compare equal, which is intentional: users
        // running a release binary shouldn't get nagged to "upgrade" to a
        // pre-release.
        assert_eq!(parse_version("v0.2.0-rc1"), Some((0, 2, 0)));
        assert_eq!(parse_version("1.2.3+build.42"), Some((1, 2, 3)));
    }

    #[test]
    fn version_parse_accepts_two_segments() {
        // GitHub occasionally gets tagged `v1.2` without a patch component.
        assert_eq!(parse_version("v1.2"), Some((1, 2, 0)));
    }

    #[test]
    fn version_parse_rejects_garbage() {
        assert_eq!(parse_version(""), None);
        assert_eq!(parse_version("latest"), None);
        assert_eq!(parse_version("vX.Y"), None);
    }

    #[test]
    fn is_newer_handles_patch_bump() {
        assert!(is_newer("0.1.0", "0.1.1"));
        assert!(!is_newer("0.1.1", "0.1.0"));
        assert!(!is_newer("0.1.0", "0.1.0"));
    }

    #[test]
    fn is_newer_handles_minor_bump() {
        assert!(is_newer("v0.1.9", "v0.2.0"));
    }

    #[test]
    fn is_newer_is_false_on_parse_fail() {
        // A malformed remote tag must never trigger an "upgrade" nag —
        // otherwise a typo in a future release blasts every user with a
        // bogus banner.
        assert!(!is_newer("0.1.0", "cool-new-thing"));
        assert!(!is_newer("not-a-version", "1.0.0"));
    }

    #[test]
    fn banner_only_when_newer() {
        let c_new = Cached {
            checked_at: Utc::now(),
            latest_tag: "v999.0.0".into(),
            html_url: String::new(),
        };
        assert!(banner_from(&c_new).is_some());
        let c_same = Cached {
            checked_at: Utc::now(),
            latest_tag: format!("v{}", current_version()),
            html_url: String::new(),
        };
        assert!(banner_from(&c_same).is_none());
    }

    #[test]
    fn stale_when_no_cache() {
        assert!(is_stale(None, 24));
    }

    #[test]
    fn stale_when_older_than_interval() {
        let c = Cached {
            checked_at: Utc::now() - chrono::Duration::hours(48),
            latest_tag: "v0.0.0".into(),
            html_url: String::new(),
        };
        assert!(is_stale(Some(&c), 24));
    }

    #[test]
    fn fresh_when_within_interval() {
        let c = Cached {
            checked_at: Utc::now() - chrono::Duration::hours(2),
            latest_tag: "v0.0.0".into(),
            html_url: String::new(),
        };
        assert!(!is_stale(Some(&c), 24));
    }

    #[test]
    fn clock_skew_future_timestamp_treated_as_stale() {
        // Someone's clock jumped backwards after the last check. The cached
        // timestamp is now in the future — treat it as stale so we refresh
        // instead of trusting it forever.
        let c = Cached {
            checked_at: Utc::now() + chrono::Duration::hours(1),
            latest_tag: "v0.0.0".into(),
            html_url: String::new(),
        };
        assert!(is_stale(Some(&c), 24));
    }
}
