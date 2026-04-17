//! Shell-command validation pipeline — destructive-pattern detection, mode
//! enforcement (ReadOnly / WorkspaceWrite / Unrestricted), sed safety, path
//! heuristics, and semantic classification. Used by the `shell` tool to
//! produce Allow / Warn / Block verdicts, which the permission prompt
//! surfaces to the user.

use crate::tools::permissions::PermissionMode;

/// Result of validating a shell command before execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationResult {
    /// Command is safe to execute.
    Allow,
    /// Command is refused regardless of user approval.
    Block { reason: String },
    /// Command looks risky — surface the warning in the permission prompt.
    Warn { message: String },
}

/// Semantic classification of a shell command's intent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum CommandIntent {
    ReadOnly,
    Write,
    Destructive,
    Network,
    ProcessManagement,
    PackageManagement,
    SystemAdmin,
    Unknown,
}

// ---------------------------------------------------------------------------
// Command tables
// ---------------------------------------------------------------------------

const WRITE_COMMANDS: &[&str] = &[
    "cp", "mv", "rm", "mkdir", "rmdir", "touch", "chmod", "chown", "chgrp", "ln", "install", "tee",
    "truncate", "shred", "mkfifo", "mknod", "dd",
];

const STATE_MODIFYING_COMMANDS: &[&str] = &[
    "apt",
    "apt-get",
    "yum",
    "dnf",
    "pacman",
    "brew",
    "pip",
    "pip3",
    "npm",
    "yarn",
    "pnpm",
    "bun",
    "cargo",
    "gem",
    "go",
    "rustup",
    "docker",
    "systemctl",
    "service",
    "mount",
    "umount",
    "kill",
    "pkill",
    "killall",
    "reboot",
    "shutdown",
    "halt",
    "poweroff",
    "useradd",
    "userdel",
    "usermod",
    "groupadd",
    "groupdel",
    "crontab",
    "at",
];

const WRITE_REDIRECTIONS: &[&str] = &[">", ">>", ">&"];

const GIT_READ_ONLY_SUBCOMMANDS: &[&str] = &[
    "status",
    "log",
    "diff",
    "show",
    "branch",
    "tag",
    "stash",
    "remote",
    "fetch",
    "ls-files",
    "ls-tree",
    "cat-file",
    "rev-parse",
    "describe",
    "shortlog",
    "blame",
    "bisect",
    "reflog",
    "config",
];

/// Curated destructive patterns. First match wins; each has a human-friendly
/// reason so the permission prompt can show *why* the command is risky.
const DESTRUCTIVE_PATTERNS: &[(&str, &str)] = &[
    (
        "rm -rf /",
        "Recursive forced deletion at root — will destroy the system",
    ),
    (
        "rm -rf ~",
        "Recursive forced deletion of the home directory",
    ),
    (
        "rm -rf *",
        "Recursive forced deletion of every file in the current directory",
    ),
    (
        "rm -rf .",
        "Recursive forced deletion of the current directory",
    ),
    (
        "mkfs",
        "Filesystem creation will destroy data on the target device",
    ),
    (
        "dd if=",
        "Direct disk write — can overwrite partitions or devices",
    ),
    ("> /dev/sd", "Writing to a raw disk device"),
    (
        "chmod -R 777",
        "Recursively setting world-writable permissions",
    ),
    ("chmod -R 000", "Recursively removing all permissions"),
    (":(){ :|:& };:", "Fork bomb — will crash the system"),
];

const ALWAYS_DESTRUCTIVE_COMMANDS: &[&str] = &["shred", "wipefs"];

const SEMANTIC_READ_ONLY_COMMANDS: &[&str] = &[
    "ls",
    "cat",
    "head",
    "tail",
    "less",
    "more",
    "wc",
    "sort",
    "uniq",
    "grep",
    "egrep",
    "fgrep",
    "find",
    "which",
    "whereis",
    "whatis",
    "man",
    "info",
    "file",
    "stat",
    "du",
    "df",
    "free",
    "uptime",
    "uname",
    "hostname",
    "whoami",
    "id",
    "groups",
    "env",
    "printenv",
    "echo",
    "printf",
    "date",
    "cal",
    "bc",
    "expr",
    "test",
    "true",
    "false",
    "pwd",
    "tree",
    "diff",
    "cmp",
    "md5sum",
    "sha256sum",
    "sha1sum",
    "xxd",
    "od",
    "hexdump",
    "strings",
    "readlink",
    "realpath",
    "basename",
    "dirname",
    "seq",
    "yes",
    "tput",
    "column",
    "jq",
    "yq",
    "xargs",
    "tr",
    "cut",
    "paste",
    "awk",
    "sed",
];

const NETWORK_COMMANDS: &[&str] = &[
    "curl",
    "wget",
    "ssh",
    "scp",
    "rsync",
    "ftp",
    "sftp",
    "nc",
    "ncat",
    "telnet",
    "ping",
    "traceroute",
    "dig",
    "nslookup",
    "host",
    "whois",
    "ifconfig",
    "ip",
    "netstat",
    "ss",
    "nmap",
];

const PROCESS_COMMANDS: &[&str] = &[
    "kill", "pkill", "killall", "ps", "top", "htop", "bg", "fg", "jobs", "nohup", "disown", "wait",
    "nice", "renice",
];

const PACKAGE_COMMANDS: &[&str] = &[
    "apt", "apt-get", "yum", "dnf", "pacman", "brew", "pip", "pip3", "npm", "yarn", "pnpm", "bun",
    "cargo", "gem", "go", "rustup", "snap", "flatpak",
];

const SYSTEM_ADMIN_COMMANDS: &[&str] = &[
    "sudo",
    "su",
    "chroot",
    "mount",
    "umount",
    "fdisk",
    "parted",
    "lsblk",
    "blkid",
    "systemctl",
    "service",
    "journalctl",
    "dmesg",
    "modprobe",
    "insmod",
    "rmmod",
    "iptables",
    "ufw",
    "firewall-cmd",
    "sysctl",
    "crontab",
    "at",
    "useradd",
    "userdel",
    "usermod",
    "groupadd",
    "groupdel",
    "passwd",
    "visudo",
];

const SYSTEM_PATHS: &[&str] = &[
    "/etc/", "/usr/", "/var/", "/boot/", "/sys/", "/proc/", "/dev/", "/sbin/", "/lib/", "/opt/",
];

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

/// Run every check. Returns the first non-Allow result.
#[must_use]
pub fn validate_command(command: &str, mode: PermissionMode) -> ValidationResult {
    let v = validate_mode(command, mode);
    if v != ValidationResult::Allow {
        return v;
    }
    let v = validate_sed(command, mode);
    if v != ValidationResult::Allow {
        return v;
    }
    check_destructive(command)
}

// ---------------------------------------------------------------------------
// Read-only enforcement
// ---------------------------------------------------------------------------

fn validate_read_only(command: &str) -> ValidationResult {
    let first = extract_first_command(command);

    for &w in WRITE_COMMANDS {
        if first == w {
            return ValidationResult::Block {
                reason: format!("'{w}' modifies the filesystem; not allowed in read-only mode"),
            };
        }
    }
    for &s in STATE_MODIFYING_COMMANDS {
        if first == s {
            return ValidationResult::Block {
                reason: format!("'{s}' modifies system state; not allowed in read-only mode"),
            };
        }
    }
    if first == "sudo" {
        let inner = extract_sudo_inner(command);
        if !inner.is_empty() {
            let r = validate_read_only(inner);
            if r != ValidationResult::Allow {
                return r;
            }
        }
    }
    for &r in WRITE_REDIRECTIONS {
        if command.contains(r) {
            return ValidationResult::Block {
                reason: format!("write redirection '{r}' not allowed in read-only mode"),
            };
        }
    }
    if first == "git" {
        return validate_git_read_only(command);
    }
    ValidationResult::Allow
}

fn validate_git_read_only(command: &str) -> ValidationResult {
    let parts: Vec<&str> = command.split_whitespace().collect();
    let sub = parts.iter().skip(1).find(|p| !p.starts_with('-'));
    match sub {
        Some(&s) if GIT_READ_ONLY_SUBCOMMANDS.contains(&s) => ValidationResult::Allow,
        Some(&s) => ValidationResult::Block {
            reason: format!("git '{s}' modifies repository state; not allowed in read-only mode"),
        },
        None => ValidationResult::Allow,
    }
}

// ---------------------------------------------------------------------------
// Mode validation
// ---------------------------------------------------------------------------

fn validate_mode(command: &str, mode: PermissionMode) -> ValidationResult {
    match mode {
        PermissionMode::ReadOnly => validate_read_only(command),
        PermissionMode::WorkspaceWrite => {
            if targets_outside_workspace(command) {
                return ValidationResult::Warn {
                    message: "command targets system paths outside the workspace".into(),
                };
            }
            ValidationResult::Allow
        }
        PermissionMode::Unrestricted => ValidationResult::Allow,
    }
}

fn targets_outside_workspace(command: &str) -> bool {
    let first = extract_first_command(command);
    let is_write = WRITE_COMMANDS.contains(&first.as_str())
        || STATE_MODIFYING_COMMANDS.contains(&first.as_str());
    if !is_write {
        return false;
    }
    SYSTEM_PATHS.iter().any(|p| command.contains(p))
}

// ---------------------------------------------------------------------------
// sed validation
// ---------------------------------------------------------------------------

fn validate_sed(command: &str, mode: PermissionMode) -> ValidationResult {
    let first = extract_first_command(command);
    if first != "sed" {
        return ValidationResult::Allow;
    }
    if mode == PermissionMode::ReadOnly && command.contains(" -i") {
        return ValidationResult::Block {
            reason: "sed -i (in-place editing) not allowed in read-only mode".into(),
        };
    }
    ValidationResult::Allow
}

// ---------------------------------------------------------------------------
// Destructive patterns
// ---------------------------------------------------------------------------

fn check_destructive(command: &str) -> ValidationResult {
    for &(pat, warning) in DESTRUCTIVE_PATTERNS {
        if command.contains(pat) {
            return ValidationResult::Warn {
                message: format!("destructive: {warning}"),
            };
        }
    }
    let first = extract_first_command(command);
    for &c in ALWAYS_DESTRUCTIVE_COMMANDS {
        if first == c {
            return ValidationResult::Warn {
                message: format!("'{c}' is inherently destructive and may cause data loss"),
            };
        }
    }
    if command.contains("rm ") && command.contains("-r") && command.contains("-f") {
        return ValidationResult::Warn {
            message: "recursive forced deletion — verify the target path is correct".into(),
        };
    }
    ValidationResult::Allow
}

// ---------------------------------------------------------------------------
// Classifier (exposed for future policy hooks, currently not wired in)
// ---------------------------------------------------------------------------

#[allow(dead_code)]
#[must_use]
pub fn classify_command(command: &str) -> CommandIntent {
    let first = extract_first_command(command);
    if SEMANTIC_READ_ONLY_COMMANDS.contains(&first.as_str()) {
        if first == "sed" && command.contains(" -i") {
            return CommandIntent::Write;
        }
        return CommandIntent::ReadOnly;
    }
    if ALWAYS_DESTRUCTIVE_COMMANDS.contains(&first.as_str()) || first == "rm" {
        return CommandIntent::Destructive;
    }
    if WRITE_COMMANDS.contains(&first.as_str()) {
        return CommandIntent::Write;
    }
    if NETWORK_COMMANDS.contains(&first.as_str()) {
        return CommandIntent::Network;
    }
    if PROCESS_COMMANDS.contains(&first.as_str()) {
        return CommandIntent::ProcessManagement;
    }
    if PACKAGE_COMMANDS.contains(&first.as_str()) {
        return CommandIntent::PackageManagement;
    }
    if SYSTEM_ADMIN_COMMANDS.contains(&first.as_str()) {
        return CommandIntent::SystemAdmin;
    }
    if first == "git" {
        let parts: Vec<&str> = command.split_whitespace().collect();
        let sub = parts.iter().skip(1).find(|p| !p.starts_with('-'));
        return match sub {
            Some(&s) if GIT_READ_ONLY_SUBCOMMANDS.contains(&s) => CommandIntent::ReadOnly,
            _ => CommandIntent::Write,
        };
    }
    CommandIntent::Unknown
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract the first bare command, skipping leading `KEY=value` env vars.
fn extract_first_command(command: &str) -> String {
    let mut remaining = command.trim();
    loop {
        let next = remaining.trim_start();
        if let Some(eq) = next.find('=') {
            let before = &next[..eq];
            if !before.is_empty()
                && before
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_')
            {
                let after = &next[eq + 1..];
                if let Some(end) = find_end_of_value(after) {
                    remaining = &after[end..];
                    continue;
                }
                return String::new();
            }
        }
        break;
    }
    remaining
        .split_whitespace()
        .next()
        .unwrap_or("")
        .to_string()
}

/// Return the substring starting at the command following `sudo` (sudo flags skipped).
fn extract_sudo_inner(command: &str) -> &str {
    let parts: Vec<&str> = command.split_whitespace().collect();
    let Some(sudo_idx) = parts.iter().position(|&p| p == "sudo") else {
        return "";
    };
    for &part in &parts[sudo_idx + 1..] {
        if !part.starts_with('-') {
            let offset = command.find(part).unwrap_or(0);
            return &command[offset..];
        }
    }
    ""
}

fn find_end_of_value(s: &str) -> Option<usize> {
    let s_trim = s.trim_start();
    if s_trim.is_empty() {
        return None;
    }
    let leading = s.len() - s_trim.len();
    let bytes = s_trim.as_bytes();
    let first = bytes[0];
    if first == b'"' || first == b'\'' {
        let quote = first;
        let mut i = 1;
        while i < bytes.len() {
            if bytes[i] == quote && (i == 0 || bytes[i - 1] != b'\\') {
                i += 1;
                while i < bytes.len() && !bytes[i].is_ascii_whitespace() {
                    i += 1;
                }
                return if i < bytes.len() {
                    Some(leading + i)
                } else {
                    None
                };
            }
            i += 1;
        }
        return None;
    }
    // Unquoted: value runs until whitespace.
    let end = bytes.iter().position(|b| b.is_ascii_whitespace())?;
    Some(leading + end)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allows_plain_ls_in_read_only() {
        let r = validate_command("ls -la /tmp", PermissionMode::ReadOnly);
        assert_eq!(r, ValidationResult::Allow);
    }

    #[test]
    fn blocks_rm_in_read_only() {
        let r = validate_command("rm file.txt", PermissionMode::ReadOnly);
        assert!(matches!(r, ValidationResult::Block { .. }));
    }

    #[test]
    fn blocks_redirection_in_read_only() {
        let r = validate_command("echo hi > /tmp/x", PermissionMode::ReadOnly);
        assert!(matches!(r, ValidationResult::Block { .. }));
    }

    #[test]
    fn blocks_sudo_rm_in_read_only() {
        let r = validate_command("sudo rm -rf /tmp/foo", PermissionMode::ReadOnly);
        assert!(matches!(r, ValidationResult::Block { .. }));
    }

    #[test]
    fn warns_rm_rf_slash() {
        let r = validate_command("rm -rf /", PermissionMode::Unrestricted);
        assert!(matches!(r, ValidationResult::Warn { .. }));
    }

    #[test]
    fn warns_fork_bomb() {
        let r = validate_command(":(){ :|:& };:", PermissionMode::Unrestricted);
        assert!(matches!(r, ValidationResult::Warn { .. }));
    }

    #[test]
    fn warns_system_path_write_under_workspace_write() {
        let r = validate_command("rm /etc/hosts", PermissionMode::WorkspaceWrite);
        assert!(matches!(r, ValidationResult::Warn { .. }));
    }

    #[test]
    fn allows_git_log_in_read_only() {
        let r = validate_command("git log --oneline", PermissionMode::ReadOnly);
        assert_eq!(r, ValidationResult::Allow);
    }

    #[test]
    fn blocks_git_commit_in_read_only() {
        let r = validate_command("git commit -m hi", PermissionMode::ReadOnly);
        assert!(matches!(r, ValidationResult::Block { .. }));
    }

    #[test]
    fn blocks_sed_i_in_read_only() {
        let r = validate_command("sed -i s/a/b/ f", PermissionMode::ReadOnly);
        assert!(matches!(r, ValidationResult::Block { .. }));
    }

    #[test]
    fn extract_skips_env_prefix() {
        assert_eq!(extract_first_command("FOO=1 BAR=2 rm -rf /tmp"), "rm");
        assert_eq!(extract_first_command("FOO=\"hi there\" cat x"), "cat");
    }

    #[test]
    fn classify_basics() {
        assert_eq!(classify_command("ls -la"), CommandIntent::ReadOnly);
        assert_eq!(classify_command("rm file"), CommandIntent::Destructive);
        assert_eq!(classify_command("curl https://x"), CommandIntent::Network);
        assert_eq!(classify_command("sudo -s"), CommandIntent::SystemAdmin);
        assert_eq!(
            classify_command("npm install"),
            CommandIntent::PackageManagement
        );
        assert_eq!(classify_command("git log"), CommandIntent::ReadOnly);
        assert_eq!(classify_command("git push"), CommandIntent::Write);
        assert_eq!(classify_command("somerandom --opt"), CommandIntent::Unknown);
    }
}
