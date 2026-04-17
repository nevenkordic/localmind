//! Interactive permission prompts + a lightweight tiered mode.
//!
//! Three pieces sit here:
//!
//!   1. [`PermissionMode`] — session-wide trust tier.
//!        ReadOnly    — file writes, shell, web are refused.
//!        WorkspaceWrite (default) — reads anywhere; writes inside workspace_root;
//!                      shell/web still prompt; destructive things warn.
//!        Unrestricted — shell/web still prompt but with no extra guard-rails.
//!
//!   2. [`Rule`] — `tool(matcher)` entries parsed from config. Three lists:
//!        allow (auto-approve), deny (hard refuse), ask (always prompt even
//!        if the user picked `[a]lways this session`).
//!
//!   3. [`PermissionManager::ask`] — the interactive prompt, with optional
//!      validator warnings surfaced inline.
//!
//! Matcher syntax:
//!   exact                  `shell(git status)`
//!   prefix wildcard        `shell(git *)`
//!   suffix wildcard        `read_file(* .key)`
//!   both / "contains"      `shell(* rm -rf *)`
//!   bare tool name         `web_search` = any invocation
//!
//! Credential deny-globs are still enforced separately in `file_ops::is_denied`.

use std::collections::HashSet;
use std::io::{self, Write};
use std::sync::Mutex;

// ---------------------------------------------------------------------------
// PermissionMode
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PermissionMode {
    ReadOnly,
    WorkspaceWrite,
    Unrestricted,
}

impl PermissionMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ReadOnly => "read-only",
            Self::WorkspaceWrite => "workspace-write",
            Self::Unrestricted => "unrestricted",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "ro" | "read-only" | "readonly" => Some(Self::ReadOnly),
            "" | "ww" | "workspace" | "workspace-write" | "default" => Some(Self::WorkspaceWrite),
            "full" | "unrestricted" | "danger" | "danger-full-access" => Some(Self::Unrestricted),
            _ => None,
        }
    }
}

impl Default for PermissionMode {
    fn default() -> Self {
        Self::WorkspaceWrite
    }
}

// ---------------------------------------------------------------------------
// Rule parsing
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Rule {
    pub raw: String,
    tool: String,
    matcher: Matcher,
}

#[derive(Debug, Clone)]
enum Matcher {
    Any,
    Exact(String),
    Prefix(String),   // "foo*"
    Suffix(String),   // "*foo"
    Contains(String), // "*foo*"
}

impl Rule {
    pub fn parse(raw: &str) -> Option<Self> {
        let raw = raw.trim();
        if raw.is_empty() {
            return None;
        }
        if let Some(lp) = raw.find('(') {
            if !raw.ends_with(')') {
                return None;
            }
            let tool = raw[..lp].trim().to_string();
            let body = raw[lp + 1..raw.len() - 1].trim();
            if tool.is_empty() {
                return None;
            }
            let matcher = Matcher::parse(body);
            return Some(Self {
                raw: raw.to_string(),
                tool,
                matcher,
            });
        }
        Some(Self {
            raw: raw.to_string(),
            tool: raw.to_string(),
            matcher: Matcher::Any,
        })
    }

    pub fn matches(&self, tool: &str, scope: &str) -> bool {
        if self.tool != tool {
            return false;
        }
        self.matcher.matches(scope)
    }
}

impl Matcher {
    fn parse(s: &str) -> Self {
        if s.is_empty() || s == "*" {
            return Self::Any;
        }
        let starts = s.starts_with('*');
        let ends = s.ends_with('*');
        match (starts, ends) {
            (true, true) => Self::Contains(s.trim_matches('*').to_string()),
            (true, false) => Self::Suffix(s.trim_start_matches('*').to_string()),
            (false, true) => Self::Prefix(s.trim_end_matches('*').to_string()),
            (false, false) => Self::Exact(s.to_string()),
        }
    }

    fn matches(&self, scope: &str) -> bool {
        match self {
            Self::Any => true,
            Self::Exact(s) => scope == s,
            Self::Prefix(p) => scope.starts_with(p),
            Self::Suffix(s) => scope.ends_with(s),
            Self::Contains(c) => scope.contains(c),
        }
    }
}

// ---------------------------------------------------------------------------
// Decisions
// ---------------------------------------------------------------------------

pub enum Decision {
    Allow,
    AllowSession,
    Deny,
    Edit(String),
}

/// Verdict from the validator (e.g. shell destructive-command detector) that
/// should be surfaced in the prompt.
///
/// `Block` variants are currently handled by the registry before this enum is
/// built, but the variant is kept so future validators can pass hard-blocks
/// through the same channel.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum ValidatorNote {
    Block(String),
    /// Warn — render inside the prompt so the user sees why this is risky.
    Warn(String),
}

// ---------------------------------------------------------------------------
// PermissionManager
// ---------------------------------------------------------------------------

pub struct PermissionManager {
    mode: Mutex<PermissionMode>,
    session_allow: Mutex<HashSet<String>>,
    allow_rules: Vec<Rule>,
    deny_rules: Vec<Rule>,
    ask_rules: Vec<Rule>,
    non_interactive: bool,
}

impl PermissionManager {
    pub fn new(
        mode: PermissionMode,
        allow: &[String],
        deny: &[String],
        ask: &[String],
        non_interactive: bool,
    ) -> Self {
        Self {
            mode: Mutex::new(mode),
            session_allow: Mutex::new(HashSet::new()),
            allow_rules: allow.iter().filter_map(|r| Rule::parse(r)).collect(),
            deny_rules: deny.iter().filter_map(|r| Rule::parse(r)).collect(),
            ask_rules: ask.iter().filter_map(|r| Rule::parse(r)).collect(),
            non_interactive,
        }
    }

    pub fn mode(&self) -> PermissionMode {
        *self.mode.lock().unwrap()
    }

    pub fn set_mode(&self, mode: PermissionMode) {
        *self.mode.lock().unwrap() = mode;
    }

    /// Ask the user whether to allow this tool invocation.
    /// `scope` is a short human-readable summary (we also match rules on it).
    /// `note` lets the caller surface a validator warning inside the prompt.
    pub fn ask(&self, tool: &str, scope: &str, note: Option<ValidatorNote>) -> Decision {
        // Validator hard-block trumps everything.
        if let Some(ValidatorNote::Block(reason)) = &note {
            eprintln!("  blocked: {reason}");
            return Decision::Deny;
        }

        // Hardcoded deny rules first.
        if let Some(rule) = self.deny_rules.iter().find(|r| r.matches(tool, scope)) {
            eprintln!("  denied by rule: {}", rule.raw);
            return Decision::Deny;
        }

        // Ask rules always prompt, even if the user picked "always this session"
        // or an allow-rule matches. Check this BEFORE the fast-paths below.
        let forced_ask = self.ask_rules.iter().any(|r| r.matches(tool, scope));

        // Allow rules auto-approve (unless ask-rule forces a prompt).
        if !forced_ask && self.allow_rules.iter().any(|r| r.matches(tool, scope)) {
            return Decision::Allow;
        }

        // Session-allow fast path (unless ask-rule forces).
        if !forced_ask {
            let key = tool.to_string();
            let guard = self.session_allow.lock().unwrap();
            if guard.contains(&key) || guard.contains(&format!("{tool}:{scope}")) {
                return Decision::Allow;
            }
        }

        if self.non_interactive {
            return Decision::Allow;
        }

        eprintln!();
        eprintln!("  ┏━ permission request ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        eprintln!("  ┃ tool:  {tool}");
        for line in scope.lines() {
            eprintln!("  ┃        {line}");
        }
        eprintln!("  ┃ mode:  {}", self.mode().as_str());
        if let Some(ValidatorNote::Warn(w)) = &note {
            eprintln!("  ┃ warn:  {w}");
        }
        if forced_ask {
            eprintln!("  ┃ note:  matched ask-rule; will always prompt");
        }
        eprintln!("  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        let hint = if forced_ask {
            "[y]es  [n]o  [e]dit(reason)"
        } else {
            "[y]es  [a]lways this session  [n]o  [e]dit(reason)"
        };
        eprint!("  allow?  {hint}: ");
        let _ = io::stderr().flush();

        let mut line = String::new();
        if io::stdin().read_line(&mut line).is_err() {
            return Decision::Deny;
        }
        let choice = line.trim().to_lowercase();
        match choice.as_str() {
            "y" | "yes" | "" => Decision::Allow,
            "a" | "always" if !forced_ask => {
                self.session_allow.lock().unwrap().insert(tool.to_string());
                Decision::AllowSession
            }
            "a" | "always" => {
                eprintln!("  (ask-rule in effect — 'always' is disabled)");
                Decision::Allow
            }
            "n" | "no" => Decision::Deny,
            s if s.starts_with('e') => {
                let reason = s.trim_start_matches('e').trim().to_string();
                let reason = if reason.is_empty() {
                    "User requested a different approach.".to_string()
                } else {
                    reason
                };
                Decision::Edit(reason)
            }
            _ => Decision::Deny,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_bare_tool_name() {
        let r = Rule::parse("web_search").unwrap();
        assert!(r.matches("web_search", "anything"));
        assert!(!r.matches("shell", "web_search"));
    }

    #[test]
    fn parses_exact_match() {
        let r = Rule::parse("shell(git status)").unwrap();
        assert!(r.matches("shell", "git status"));
        assert!(!r.matches("shell", "git log"));
    }

    #[test]
    fn parses_prefix_wildcard() {
        let r = Rule::parse("shell(git push*)").unwrap();
        assert!(r.matches("shell", "git push origin main"));
        assert!(!r.matches("shell", "git pull"));
    }

    #[test]
    fn parses_contains() {
        let r = Rule::parse("shell(*rm -rf*)").unwrap();
        assert!(r.matches("shell", "sudo rm -rf /tmp"));
    }

    #[test]
    fn mode_parse() {
        assert_eq!(PermissionMode::parse("ro"), Some(PermissionMode::ReadOnly));
        assert_eq!(
            PermissionMode::parse("workspace-write"),
            Some(PermissionMode::WorkspaceWrite)
        );
        assert_eq!(
            PermissionMode::parse("unrestricted"),
            Some(PermissionMode::Unrestricted)
        );
        assert_eq!(PermissionMode::parse("bogus"), None);
    }
}
