//! Startup preflight — verifies Ollama is reachable and the configured chat
//! and embed models are actually pulled before the REPL starts taking input.
//!
//! A user whose config points at `qwen2.5-coder:32b` but hasn't pulled it
//! used to get a bare `ollama /api/chat 404: model not found` on their first
//! turn. Preflight catches that at startup and prints an actionable remedy
//! so they never see the 404.
//!
//! Prints warnings but never fails — the user might have pulled a model in
//! another terminal mid-session, or Ollama might come up shortly after.

use crate::config::Config;
use crate::llm::ollama::OllamaClient;

/// Run the preflight check and print any warnings to stderr. Returns `true`
/// if everything looks good, `false` if the user should see a warning
/// (caller decides whether to continue — right now everyone continues).
pub async fn check(cfg: &Config) -> bool {
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

    eprintln!("  Fix one of:");
    if chat_missing {
        eprintln!("    ollama pull {}", cfg.ollama.chat_model);
    }
    if embed_missing {
        eprintln!("    ollama pull {}", cfg.ollama.embed_model);
    }
    if let Some(fb) = chat_missing.then(|| pick_chat_fallback(&models)).flatten() {
        eprintln!(
            "    llm models --chat {}   # use this installed model instead",
            fb
        );
    }
    eprintln!("    llm models                              # interactive picker");
    eprintln!();
    false
}

fn is_installed(models: &[String], needle: &str) -> bool {
    if needle.is_empty() {
        return true; // user explicitly cleared — their problem, don't nag
    }
    // Ollama tags are case-sensitive and include the `:tag` suffix. Match
    // exactly on the full name first, then tolerate the `latest` alias
    // (ollama treats `qwen:7b` and `qwen:7b-latest` as the same, but the
    // default install never yields the suffixed form anyway).
    if models.iter().any(|m| m == needle) {
        return true;
    }
    if !needle.contains(':') {
        // User wrote `qwen2.5-coder` without a tag — ollama normalises to
        // `:latest`. Match anything with that bare base.
        let with_latest = format!("{needle}:latest");
        return models.iter().any(|m| m == &with_latest);
    }
    false
}

/// Pick a sensible chat fallback from what's actually installed so the user
/// gets a concrete `llm models --chat <name>` suggestion. Preference order:
/// exact family matches (qwen2.5-coder sizes), then any qwen-coder, then any
/// model whose name doesn't contain "embed".
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
        // Edge case — user blanked out chat_model. Don't warn.
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
}
