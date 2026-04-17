//! File read / write / list — gated by deny-globs and workspace root.

use crate::tools::registry::{resolve_path, ToolContext};
use anyhow::{anyhow, Result};
use regex::Regex;
use serde_json::Value;

pub async fn read_file(ctx: &ToolContext, args: &Value) -> Result<String> {
    let path = args
        .get("path")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("missing path"))?;
    let max_bytes = args
        .get("max_bytes")
        .and_then(|v| v.as_u64())
        .unwrap_or(200_000) as usize;
    let resolved = resolve_path(ctx, path);

    let path_str = resolved.to_string_lossy().into_owned();
    let bytes = tokio::task::spawn_blocking(move || -> Result<Vec<u8>> {
        let meta = std::fs::metadata(&path_str)?;
        let size = meta.len() as usize;
        let to_read = size.min(max_bytes);
        use std::io::Read;
        let mut f = std::fs::File::open(&path_str)?;
        let mut buf = vec![0u8; to_read];
        f.read_exact(&mut buf).or_else(|_| -> Result<()> {
            // Allow short reads.
            Ok(())
        })?;
        Ok(buf)
    })
    .await??;

    let text = String::from_utf8_lossy(&bytes).into_owned();
    Ok(text)
}

pub async fn write_file(ctx: &ToolContext, args: &Value) -> Result<String> {
    let path = args
        .get("path")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("missing path"))?;
    let content = args
        .get("content")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("missing content"))?;
    let create_parents = args
        .get("create_parents")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    let resolved = resolve_path(ctx, path);
    let path_str = resolved.clone();
    let content = content.to_string();
    tokio::task::spawn_blocking(move || -> Result<()> {
        if create_parents {
            if let Some(parent) = path_str.parent() {
                std::fs::create_dir_all(parent)?;
            }
        }
        std::fs::write(&path_str, content)?;
        Ok(())
    })
    .await??;
    Ok(format!("wrote {}", resolved.display()))
}

pub async fn list_dir(ctx: &ToolContext, args: &Value) -> Result<String> {
    let path = args
        .get("path")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("missing path"))?;
    let recursive = args
        .get("recursive")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let max = args
        .get("max_entries")
        .and_then(|v| v.as_u64())
        .unwrap_or(500) as usize;
    let resolved = resolve_path(ctx, path);
    let deny = ctx.cfg.tools.deny_globs.clone();
    let out = tokio::task::spawn_blocking(move || -> Result<String> {
        let mut out = String::new();
        let mut count = 0usize;
        let walk = |dir: &std::path::Path,
                    out: &mut String,
                    count: &mut usize,
                    deny: &[String],
                    recursive: bool|
         -> Result<()> {
            fn walk_inner(
                dir: &std::path::Path,
                out: &mut String,
                count: &mut usize,
                deny: &[String],
                recursive: bool,
                max: usize,
            ) -> Result<()> {
                for entry in std::fs::read_dir(dir)? {
                    if *count >= max {
                        return Ok(());
                    }
                    let entry = entry?;
                    let p = entry.path();
                    let s = p.to_string_lossy().into_owned();
                    if is_denied(deny, &s) {
                        continue;
                    }
                    let kind = if p.is_dir() { "DIR " } else { "FILE" };
                    out.push_str(&format!("{kind}  {}\n", s));
                    *count += 1;
                    if recursive && p.is_dir() {
                        walk_inner(&p, out, count, deny, recursive, max)?;
                    }
                }
                Ok(())
            }
            walk_inner(dir, out, count, deny, recursive, max)
        };
        walk(&resolved, &mut out, &mut count, &deny, recursive)?;
        Ok(out)
    })
    .await??;
    Ok(out)
}

/// Match a path string against the configured deny-glob list.
/// Also includes a hard-coded credential/secret deny-list that is ALWAYS on.
pub fn is_denied(globs: &[String], path: &str) -> bool {
    let norm = path.replace('\\', "/");
    // Hard-coded: never match these, no matter what config says.
    let always_deny: &[&str] = &[
        // unix user secrets
        "/.ssh/",
        "/.gnupg/",
        "/.aws/",
        "/.azure/",
        "/.config/gcloud/",
        "/.kube/config",
        "/.docker/config.json",
        // package manager / language credentials
        "/.netrc",
        "/.pypirc",
        "/.npmrc",
        "/.git-credentials",
        "/.cargo/credentials",
        // system password / privilege files (Linux + macOS + BSD)
        "/etc/shadow",
        "/etc/gshadow",
        "/etc/sudoers",
        "/etc/master.passwd",
        "/etc/spwd.db",
        "/private/etc/master.passwd",
        "/private/etc/sudoers",
        // ssh / pgp key files + common private-key extensions
        "/id_rsa",
        "/id_dsa",
        "/id_ecdsa",
        "/id_ed25519",
        "/secring.gpg",
        ".pem",
        ".key",
        ".pkcs12",
        ".p12",
        ".pfx",
        // windows secrets
        "/appdata/roaming/microsoft/credentials/",
        "/appdata/local/microsoft/credentials/",
        "/appdata/roaming/microsoft/protect/",
        "/appdata/local/microsoft/vault/",
        "/appdata/roaming/microsoft/crypto/",
        // browser profile storage
        "/cookies",
        "/login data",
        "/web data",
    ];
    let lower = norm.to_lowercase();
    for needle in always_deny {
        if lower.contains(needle) {
            return true;
        }
    }
    // Configured globs — we implement a cheap glob via regex-ified wildcards.
    for g in globs {
        if glob_match(g, &norm) {
            return true;
        }
    }
    false
}

fn glob_match(pattern: &str, path: &str) -> bool {
    // Translate a simple glob ("**", "*", "?") into a regex.
    let mut re = String::from("(?i)^");
    let bytes = pattern.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i] as char;
        match c {
            '*' if i + 1 < bytes.len() && bytes[i + 1] as char == '*' => {
                re.push_str(".*");
                i += 2;
                if i < bytes.len() && bytes[i] as char == '/' {
                    i += 1;
                }
            }
            '*' => {
                re.push_str("[^/]*");
                i += 1;
            }
            '?' => {
                re.push_str("[^/]");
                i += 1;
            }
            '.' | '+' | '(' | ')' | '|' | '^' | '$' | '{' | '}' | '[' | ']' | '\\' => {
                re.push('\\');
                re.push(c);
                i += 1;
            }
            other => {
                re.push(other);
                i += 1;
            }
        }
    }
    re.push('$');
    Regex::new(&re).map(|r| r.is_match(path)).unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::is_denied;

    #[test]
    fn deny_list_blocks_classic_secrets() {
        let none: &[String] = &[];
        assert!(is_denied(none, "/Users/x/.ssh/id_rsa"));
        assert!(is_denied(none, "/Users/x/.aws/credentials"));
        assert!(is_denied(none, "/home/x/.kube/config"));
    }

    #[test]
    fn deny_list_blocks_expanded_credentials() {
        let none: &[String] = &[];
        assert!(is_denied(none, "/Users/x/.netrc"));
        assert!(is_denied(none, "/Users/x/.git-credentials"));
        assert!(is_denied(none, "/Users/x/.npmrc"));
        assert!(is_denied(none, "/Users/x/.pypirc"));
        assert!(is_denied(none, "/Users/x/.cargo/credentials"));
    }

    #[test]
    fn deny_list_blocks_system_password_files() {
        let none: &[String] = &[];
        assert!(is_denied(none, "/etc/shadow"));
        assert!(is_denied(none, "/etc/sudoers"));
        assert!(is_denied(none, "/etc/master.passwd"));
        assert!(is_denied(none, "/private/etc/master.passwd"));
    }

    #[test]
    fn deny_list_blocks_private_key_extensions() {
        let none: &[String] = &[];
        assert!(is_denied(none, "/Users/x/Downloads/server.pem"));
        assert!(is_denied(none, "/Users/x/certs/api.key"));
        assert!(is_denied(none, "/Users/x/Downloads/identity.p12"));
    }

    #[test]
    fn deny_list_does_not_block_normal_files() {
        let none: &[String] = &[];
        assert!(!is_denied(none, "/Users/x/code/main.rs"));
        assert!(!is_denied(none, "/etc/hosts"));
        assert!(!is_denied(none, "/Users/x/notes.txt"));
    }
}
