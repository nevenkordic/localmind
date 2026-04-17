//! `llm models` — pick which Ollama models the agent uses.
//!
//! Three modes:
//!
//!   llm models                        interactive picker (numbered prompts)
//!   llm models --list                 list installed models, mark current
//!   llm models --chat NAME ...        non-interactive set (one or more roles)
//!
//! Writes changes back to the config file via `toml_edit`, which preserves
//! comments and surrounding formatting — important since users hand-edit
//! `config/local.toml` and we don't want to nuke their inline notes.

use crate::config::Config;
use crate::llm::ollama::OllamaClient;
use anyhow::{anyhow, Context, Result};
use std::io::Write;
use std::path::Path;

#[derive(Debug, Clone, Copy)]
pub enum Role {
    Chat,
    Vision,
    Embed,
}

impl Role {
    fn as_str(self) -> &'static str {
        match self {
            Role::Chat => "chat",
            Role::Vision => "vision",
            Role::Embed => "embed",
        }
    }
    fn toml_key(self) -> &'static str {
        match self {
            Role::Chat => "chat_model",
            Role::Vision => "vision_model",
            Role::Embed => "embed_model",
        }
    }
}

pub struct CliArgs {
    pub list_only: bool,
    pub chat: Option<String>,
    pub vision: Option<String>,
    pub embed: Option<String>,
}

pub async fn run(cfg: &Config, config_path: Option<&Path>, args: CliArgs) -> Result<()> {
    let path = config_path
        .map(|p| p.to_path_buf())
        .or_else(|| Config::source_path(None))
        .ok_or_else(|| anyhow!("no config file resolved — pass --config <path>"))?;

    // Non-interactive set: any of --chat/--vision/--embed bypasses the picker.
    if args.chat.is_some() || args.vision.is_some() || args.embed.is_some() {
        write_models(
            &path,
            args.chat.as_deref(),
            args.vision.as_deref(),
            args.embed.as_deref(),
        )?;
        println!("Updated {}:", path.display());
        if let Some(c) = &args.chat {
            println!("  chat_model   = {c}");
        }
        if let Some(v) = &args.vision {
            println!("  vision_model = {v}");
        }
        if let Some(e) = &args.embed {
            println!("  embed_model  = {e}");
        }
        return Ok(());
    }

    let client = OllamaClient::new(&cfg.ollama);
    let installed = client
        .health()
        .await
        .with_context(|| format!("ollama unreachable at {}", cfg.ollama.host))?;
    if installed.is_empty() {
        println!("No models installed. Run `ollama pull <model>` first.");
        return Ok(());
    }

    print_list(&installed, &cfg.ollama);

    if args.list_only {
        return Ok(());
    }

    println!();
    println!("Pick by number (or press Enter to keep the current value).");
    let chat = pick_role(Role::Chat, &installed, &cfg.ollama.chat_model)?;
    let vision = pick_role(Role::Vision, &installed, &cfg.ollama.vision_model)?;
    let embed = pick_role(Role::Embed, &installed, &cfg.ollama.embed_model)?;

    println!();
    println!("New selection:");
    println!("  chat:   {chat}");
    println!("  vision: {vision}");
    println!("  embed:  {embed}");
    print!("Write to {}? [y/N]: ", path.display());
    let _ = std::io::stdout().flush();
    let mut ans = String::new();
    std::io::stdin().read_line(&mut ans)?;
    if !matches!(ans.trim().to_ascii_lowercase().as_str(), "y" | "yes") {
        println!("(not saved — config unchanged)");
        return Ok(());
    }
    write_models(&path, Some(&chat), Some(&vision), Some(&embed))?;
    println!("Saved.");
    Ok(())
}

fn print_list(installed: &[String], ollama: &crate::config::OllamaConfig) {
    println!("Installed Ollama models:");
    let width = installed.len().to_string().len();
    for (i, m) in installed.iter().enumerate() {
        let mut tags = Vec::new();
        if m == &ollama.chat_model {
            tags.push("chat");
        }
        if m == &ollama.vision_model {
            tags.push("vision");
        }
        if m == &ollama.embed_model {
            tags.push("embed");
        }
        let suffix = if tags.is_empty() {
            String::new()
        } else {
            format!("   ← {}", tags.join(", "))
        };
        println!("  {:>width$}. {m}{suffix}", i + 1, width = width);
    }
}

fn pick_role(role: Role, installed: &[String], current: &str) -> Result<String> {
    print!("  {} (current: {}): ", role.as_str(), current);
    let _ = std::io::stdout().flush();
    let mut line = String::new();
    std::io::stdin().read_line(&mut line)?;
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return Ok(current.to_string());
    }
    // Accept either a number (1-based) or an exact model name.
    if let Ok(n) = trimmed.parse::<usize>() {
        if n == 0 || n > installed.len() {
            anyhow::bail!("number {n} out of range (1..={})", installed.len());
        }
        return Ok(installed[n - 1].clone());
    }
    if installed.iter().any(|m| m == trimmed) {
        return Ok(trimmed.to_string());
    }
    anyhow::bail!("'{trimmed}' is not in the installed list — try a number")
}

/// Update one or more `[ollama].*_model` values in the given TOML file,
/// preserving every other line, comment, and whitespace via `toml_edit`.
pub fn write_models(
    path: &Path,
    chat: Option<&str>,
    vision: Option<&str>,
    embed: Option<&str>,
) -> Result<()> {
    let raw =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let mut doc: toml_edit::DocumentMut = raw
        .parse()
        .with_context(|| format!("parsing TOML in {}", path.display()))?;
    let ollama = doc
        .get_mut("ollama")
        .and_then(|v| v.as_table_mut())
        .ok_or_else(|| anyhow!("config has no [ollama] section"))?;
    for (role, value) in [
        (Role::Chat, chat),
        (Role::Vision, vision),
        (Role::Embed, embed),
    ] {
        if let Some(v) = value {
            ollama.insert(role.toml_key(), toml_edit::value(v));
        }
    }
    std::fs::write(path, doc.to_string()).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}
