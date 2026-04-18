//! Startup preflight — verifies Ollama is reachable and the configured chat
//! and embed models are actually pulled before the REPL starts taking input.
//!
//! When the chat model is missing and stdin is a TTY, offer to switch the
//! config to an installed model *right now* so the user doesn't hit the
//! `ollama /api/chat 404: model not found` error on their first turn. The
//! switch writes back to the same config file `llm models --chat <name>`
//! uses, so the change persists across runs.

use crate::config::{Config, EMBEDDED_DEFAULT_CONFIG};
use crate::llm::ollama::OllamaClient;
use crate::models::write_models;
use anyhow::{Context, Result};
use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};

/// Run the preflight check and (if appropriate) interactively prompt the
/// user to switch models. Mutates `cfg` in place when the user accepts the
/// switch — so the REPL banner and first turn both see the new value
/// without a second parse. Returns `true` if everything looks good after
/// the check, `false` if the user should continue at their own risk.
pub async fn check(cfg: &mut Config, explicit_config: Option<&Path>) -> bool {
    let client = OllamaClient::new(&cfg.ollama);
    let models = match client.health().await {
        Ok(m) => m,
        Err(e) => {
            eprintln!(
                "\x1b[1;33mwarning:\x1b[0m Ollama unreachable at {}: {e}",
                cfg.ollama.host
            );
            eprintln!("  Start it:");
            if cfg!(target_os = "macos") {
                eprintln!("    open -a Ollama       (or: brew services start ollama)");
            } else if cfg!(target_os = "linux") {
                eprintln!("    systemctl start ollama  (or: ollama serve &)");
            } else {
                eprintln!("    ollama serve");
            }
            eprintln!();
            return false;
        }
    };

    let chat_missing = !is_installed(&models, &cfg.ollama.chat_model);
    let embed_missing = !is_installed(&models, &cfg.ollama.embed_model);
    if !chat_missing && !embed_missing {
        return true;
    }

    eprintln!();
    if chat_missing {
        eprintln!(
            "\x1b[1;33mwarning:\x1b[0m chat model '\x1b[1m{}\x1b[0m' is not pulled.",
            cfg.ollama.chat_model
        );
    }
    if embed_missing {
        eprintln!(
            "\x1b[1;33mwarning:\x1b[0m embed model '\x1b[1m{}\x1b[0m' is not pulled.",
            cfg.ollama.embed_model
        );
    }

    if models.is_empty() {
        eprintln!("  Ollama has no models installed yet.");
    } else {
        let preview: Vec<String> = models.iter().take(8).cloned().collect();
        let more = if models.len() > 8 {
            format!(", … +{}", models.len() - 8)
        } else {
            String::new()
        };
        eprintln!("  Installed: {}{}", preview.join(", "), more);
    }

    // Offer interactive switch when a TTY is attached. In non-TTY use
    // (pipes, `llm ask` inside scripts, CI), fall back to the informational
    // remedies so the process stays deterministic.
    let interactive = std::io::stdin().is_terminal() && std::io::stderr().is_terminal();
    let fallback = if chat_missing {
        pick_chat_fallback(&models)
    } else {
        None
    };

    if interactive && chat_missing {
        if let Some(fb) = &fallback {
            match prompt_switch(&cfg.ollama.chat_model, fb) {
                Prompt::Yes => match apply_chat_model(cfg, explicit_config, fb) {
                    Ok(written_to) => {
                        eprintln!(
                            "\x1b[1;32m✓\x1b[0m chat model set to \x1b[1m{fb}\x1b[0m (saved to {})",
                            written_to.display()
                        );
                        eprintln!();
                        return true;
                    }
                    Err(e) => {
                        eprintln!("\x1b[1;31merror:\x1b[0m could not update config: {e:#}");
                        eprintln!(
                            "  Run manually: \x1b[1mllm models --chat {fb}\x1b[0m"
                        );
                    }
                },
                Prompt::No | Prompt::Unreadable => {
                    // fall through to the passive remedy list.
                }
            }
        }
    }

    // Passive remedies — shown when no interactive switch happened, or when
    // the user declined, or when no fallback is installable.
    eprintln!("  Fix one of:");
    if chat_missing {
        eprintln!("    ollama pull {}", cfg.ollama.chat_model);
    }
    if embed_missing {
        eprintln!("    ollama pull {}", cfg.ollama.embed_model);
    }
    if let Some(fb) = &fallback {
        eprintln!(
            "    llm models --chat {fb}   # use this installed model instead"
        );
    }
    eprintln!("    llm models                              # interactive picker");
    eprintln!();
    false
}

enum Prompt {
    Yes,
    No,
    Unreadable,
}

fn prompt_switch(current: &str, fallback: &str) -> Prompt {
    eprint!(
        "Switch chat model from \x1b[1m{current}\x1b[0m to \x1b[1m{fallback}\x1b[0m now? [Y/n] "
    );
    let _ = std::io::stderr().flush();
    let mut line = String::new();
    match std::io::stdin().read_line(&mut line) {
        Ok(0) => Prompt::Unreadable, // EOF — not really interactive
        Ok(_) => {
            let ans = line.trim().to_lowercase();
            if ans.is_empty() || ans == "y" || ans == "yes" {
                Prompt::Yes
            } else {
                Prompt::No
            }
        }
        Err(_) => Prompt::Unreadable,
    }
}

/// Write the selected chat model back to disk and mutate `cfg` in place.
/// Target file: whatever `Config::source_path` would read from next time.
/// If nothing exists yet (fresh curl install with no cwd config), seed a
/// user-scope config at the platform config dir from the embedded default
/// template so the choice persists across runs.
fn apply_chat_model(
    cfg: &mut Config,
    explicit_config: Option<&Path>,
    value: &str,
) -> Result<PathBuf> {
    let path = match Config::source_path(explicit_config) {
        Some(p) if p.exists() => p,
        _ => {
            let dir = directories::ProjectDirs::from("com", "calligoit", "localmind")
                .map(|p| p.config_dir().to_path_buf())
                .context("no platform config dir")?;
            std::fs::create_dir_all(&dir)
                .with_context(|| format!("creating {}", dir.display()))?;
            let p = dir.join("config.toml");
            if !p.exists() {
                std::fs::write(&p, EMBEDDED_DEFAULT_CONFIG)
                    .with_context(|| format!("seeding {}", p.display()))?;
            }
            p
        }
    };
    write_models(&path, Some(value), None, None)?;
    cfg.ollama.chat_model = value.to_string();
    Ok(path)
}

fn is_installed(models: &[String], needle: &str) -> bool {
    if needle.is_empty() {
        return true;
    }
    if models.iter().any(|m| m == needle) {
        return true;
    }
    if !needle.contains(':') {
        let with_latest = format!("{needle}:latest");
        return models.iter().any(|m| m == &with_latest);
    }
    false
}

fn pick_chat_fallback(models: &[String]) -> Option<String> {
    let preferred = [
        "qwen2.5-coder:7b",
        "qwen2.5-coder:14b",
        "qwen2.5-coder:32b",
        "qwen2.5:7b",
    ];
    for p in preferred {
        if let Some(m) = models.iter().find(|m| m.as_str() == p) {
            return Some(m.clone());
        }
    }
    if let Some(m) = models.iter().find(|m| m.starts_with("qwen") && !m.contains("embed")) {
        return Some(m.clone());
    }
    models.iter().find(|m| !m.contains("embed")).cloned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_match_is_installed() {
        let m = vec!["qwen2.5-coder:7b".into(), "nomic-embed-text".into()];
        assert!(is_installed(&m, "qwen2.5-coder:7b"));
        assert!(is_installed(&m, "nomic-embed-text"));
    }

    #[test]
    fn missing_tag_is_not_installed() {
        let m = vec!["qwen2.5-coder:7b".into()];
        assert!(!is_installed(&m, "qwen2.5-coder:32b"));
    }

    #[test]
    fn empty_needle_treated_as_installed() {
        assert!(is_installed(&[], ""));
    }

    #[test]
    fn base_name_matches_latest_tag() {
        let m = vec!["mymodel:latest".into()];
        assert!(is_installed(&m, "mymodel"));
    }

    #[test]
    fn fallback_prefers_qwen_coder_7b() {
        let m = vec![
            "nomic-embed-text".into(),
            "qwen2.5-coder:7b".into(),
            "llama3:70b".into(),
        ];
        assert_eq!(pick_chat_fallback(&m).unwrap(), "qwen2.5-coder:7b");
    }

    #[test]
    fn fallback_falls_back_to_any_non_embed_model() {
        let m = vec!["nomic-embed-text".into(), "llama3:8b".into()];
        assert_eq!(pick_chat_fallback(&m).unwrap(), "llama3:8b");
    }

    #[test]
    fn fallback_none_when_only_embed_models() {
        let m = vec!["nomic-embed-text".into(), "mxbai-embed-large".into()];
        assert!(pick_chat_fallback(&m).is_none());
    }

    // apply_chat_model is exercised via an integration-ish test that
    // writes to a temp file. It drives the same code path as the
    // interactive "Y" answer so we catch toml_edit regressions.
    #[test]
    fn apply_chat_model_writes_to_explicit_config() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("local.toml");
        std::fs::write(&path, EMBEDDED_DEFAULT_CONFIG).unwrap();
        let mut cfg: Config = toml::from_str(EMBEDDED_DEFAULT_CONFIG).unwrap();
        let written = apply_chat_model(&mut cfg, Some(&path), "llama3:8b").unwrap();
        assert_eq!(written, path);
        assert_eq!(cfg.ollama.chat_model, "llama3:8b");
        let on_disk = std::fs::read_to_string(&path).unwrap();
        assert!(on_disk.contains("chat_model = \"llama3:8b\""));
        // The [ollama] section header and at least one other comment must
        // survive the rewrite — a naive serde round-trip would blow them
        // all away, and the point of using toml_edit is exactly this.
        assert!(on_disk.contains("[ollama]"));
        assert!(on_disk.contains("# URL of your local Ollama server."));
    }
}
