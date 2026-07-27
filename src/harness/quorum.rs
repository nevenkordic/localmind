//! Quorum helpers — model pool resolution and vote counting.

use crate::config::Config;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuorumPolicy {
    Majority,
    Unanimous,
}

impl QuorumPolicy {
    pub fn parse(s: &str) -> Self {
        match s.trim().to_lowercase().as_str() {
            "unanimous" => Self::Unanimous,
            _ => Self::Majority,
        }
    }

    pub fn passes(&self, yes: usize, total: usize) -> bool {
        if total == 0 {
            return false;
        }
        match self {
            Self::Unanimous => yes == total,
            Self::Majority => yes * 2 > total,
        }
    }
}

#[derive(Debug, Clone)]
pub struct VerdictVote {
    pub model: String,
    pub passed: bool,
    pub feedback: String,
}

/// Build the verifier model pool for quorum. Distinct models only.
pub fn verifier_models(cfg: &Config) -> Vec<String> {
    let mut out = Vec::new();
    for m in cfg.harness.verify_models.iter() {
        let m = m.trim();
        if !m.is_empty() && !out.iter().any(|x| x == m) {
            out.push(m.to_string());
        }
    }
    if out.is_empty() {
        for m in [
            cfg.harness.verify_model.as_str(),
            cfg.ollama.fast_model.as_str(),
            cfg.ollama.chat_model.as_str(),
        ] {
            let m = m.trim();
            if !m.is_empty() && !out.iter().any(|x| x == m) {
                out.push(m.to_string());
            }
        }
    }
    out
}

pub fn quorum_met(
    cfg: &Config,
    votes: &[VerdictVote],
    policy: QuorumPolicy,
    pool_size: usize,
) -> (bool, String) {
    let min = cfg.harness.quorum_min.max(1);
    let effective_min = min.min(pool_size.max(1));
    if cfg.harness.require_distinct_models && pool_size < min {
        eprintln!(
            "  ! quorum: only {pool_size} distinct verifier model(s); using effective_min={effective_min}"
        );
    }
    if votes.is_empty() {
        return (false, "quorum not met: no verifier votes".into());
    }
    let yes = votes.iter().filter(|v| v.passed).count();
    let total = votes.len();
    let passed = policy.passes(yes, total) && total >= effective_min.min(1);
    let summary: String = votes
        .iter()
        .map(|v| {
            format!(
                "{} {}: {}",
                v.model,
                if v.passed { "PASS" } else { "FAIL" },
                crate::util::truncate(&v.feedback, 120)
            )
        })
        .collect::<Vec<_>>()
        .join("; ");
    let feedback = format!("quorum {yes}/{total} ({passed}): {summary}");
    (passed, feedback)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn majority_needs_strict_majority() {
        let p = QuorumPolicy::Majority;
        assert!(p.passes(2, 3));
        assert!(!p.passes(1, 3));
        assert!(p.passes(1, 1));
    }

    #[test]
    fn unanimous_needs_all() {
        let p = QuorumPolicy::Unanimous;
        assert!(p.passes(2, 2));
        assert!(!p.passes(1, 2));
    }
}
