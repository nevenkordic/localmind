//! Multi-model verify harness — plan → act → verify with shared SQLite
//! memory. Local models consult each other; learnings are recorded as
//! skills so the next run does not rediscover the same procedure.

use crate::agent::AgentRun;
use crate::config::Config;
use crate::llm::ollama::OllamaClient;
use crate::memory::{NewMemory, Store};
use crate::tools::permissions::PermissionMode;
use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;
use std::sync::Arc;

/// Built-in default formula used by `llm run` when no --formula is given.
pub const DEFAULT_FORMULA: &str = include_str!("../../formulas/verify.toml");

#[derive(Debug, Clone, Deserialize)]
pub struct Formula {
    pub formula: FormulaMeta,
    #[serde(default)]
    pub stages: Vec<Stage>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FormulaMeta {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default = "default_retries")]
    pub max_verify_retries: usize,
}
fn default_retries() -> usize {
    2
}

#[derive(Debug, Clone, Deserialize)]
pub struct Stage {
    pub name: String,
    /// planner | worker | verifier — picks which Ollama model to use.
    #[serde(default = "default_role")]
    pub role: String,
}
fn default_role() -> String {
    "worker".into()
}

#[derive(Debug)]
pub struct HarnessResult {
    pub plan: String,
    pub result: String,
    pub passed: bool,
    pub attempts: usize,
    pub skills_stored: usize,
    pub decision_id: Option<String>,
}

impl Formula {
    pub fn parse(raw: &str) -> Result<Self> {
        let f: Formula = toml::from_str(raw).context("parsing formula TOML")?;
        if f.formula.name.trim().is_empty() {
            anyhow::bail!("formula.name is required");
        }
        Ok(f)
    }

    pub fn load_path(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("reading formula {}", path.display()))?;
        Self::parse(&raw)
    }

    pub fn default_verify() -> Result<Self> {
        Self::parse(DEFAULT_FORMULA)
    }
}

/// Resolve which model a stage role should call.
fn model_for_role(cfg: &Config, role: &str) -> Option<String> {
    let fast = cfg.ollama.fast_model.trim();
    let chat = cfg.ollama.chat_model.trim();
    let verify = cfg.harness.verify_model.trim();
    match role {
        "planner" => {
            if !fast.is_empty() {
                Some(fast.to_string())
            } else {
                None // chat_model default
            }
        }
        "verifier" => {
            if !verify.is_empty() {
                Some(verify.to_string())
            } else if !fast.is_empty() && fast != chat {
                // Prefer a distinct fast model for verify so worker and
                // verifier are not the same weights when possible.
                Some(fast.to_string())
            } else {
                None
            }
        }
        _ => None, // worker → chat_model
    }
}

/// Run plan → act → verify. On verify failure, re-run act with feedback
/// up to `max_verify_retries`. On success, log a decision and distill
/// reusable skills into SQLite.
pub async fn run(
    cfg: Config,
    store: Store,
    formula: Formula,
    task: &str,
    mode: Option<PermissionMode>,
) -> Result<HarnessResult> {
    let cfg = Arc::new(cfg);
    let client = OllamaClient::new(&cfg.ollama);
    let max_retries = formula
        .formula
        .max_verify_retries
        .max(cfg.harness.max_retries);

    eprintln!(
        "· harness `{}` — plan → act → verify (max {} retries)",
        formula.formula.name, max_retries
    );

    // Shared memory primer so every stage sees the same skills/facts.
    let memory_primer = build_memory_primer(&store, &client, &cfg, task).await;

    // ---- PLAN ----------------------------------------------------------
    let plan_model = model_for_role(&cfg, "planner");
    eprintln!(
        "  · plan ({})",
        plan_model.as_deref().unwrap_or(&cfg.ollama.chat_model)
    );
    let plan = client
        .plan_task(task, &memory_primer, plan_model.as_deref())
        .await
        .context("plan stage")?;

    // ---- ACT / VERIFY loop --------------------------------------------
    let mut feedback = String::new();
    let mut result = String::new();
    let mut passed = false;
    let mut attempts = 0usize;

    for attempt in 0..=max_retries {
        attempts = attempt + 1;
        let act_prompt = if feedback.is_empty() {
            format!(
                "You are the ACT stage of a verified local harness.\n\
                 Follow the PLAN. Use tools as needed. When done, summarise \
                 what you did.\n\nTASK:\n{task}\n\nPLAN:\n{plan}"
            )
        } else {
            format!(
                "You are the ACT stage of a verified local harness.\n\
                 Previous attempt failed verification. Fix the issues in \
                 the feedback, then summarise what you changed.\n\n\
                 TASK:\n{task}\n\nPLAN:\n{plan}\n\n\
                 VERIFIER FEEDBACK:\n{feedback}"
            )
        };

        eprintln!("  · act attempt {attempts} ({})", cfg.ollama.chat_model);
        let mut agent =
            AgentRun::new_with_mode(cfg.clone(), store.clone(), mode, true, true, true)?;
        result = agent.turn(&act_prompt).await.context("act stage")?;

        let verify_model = model_for_role(&cfg, "verifier");
        eprintln!(
            "  · verify ({})",
            verify_model.as_deref().unwrap_or(&cfg.ollama.chat_model)
        );
        let (ok, fb) = client
            .verify_stage(task, &plan, &result, verify_model.as_deref())
            .await
            .context("verify stage")?;
        if ok {
            passed = true;
            eprintln!("  · verify PASS");
            break;
        }
        feedback = fb;
        eprintln!("  · verify FAIL: {}", crate::util::truncate(&feedback, 160));
        if attempt == max_retries {
            break;
        }
    }

    // ---- DECISION + SKILLS --------------------------------------------
    let outcome = if passed { "passed" } else { "failed" };
    let decision_id = store
        .insert_decision(
            &format!("{}: {task}", formula.formula.name),
            &format!("plan:\n{plan}"),
            &feedback,
            &format!("{outcome} after {attempts} attempt(s)"),
            "harness",
        )
        .await
        .ok();

    let mut skills_stored = 0usize;
    if passed && cfg.harness.auto_skill {
        let transcript = format!(
            "TASK:\n{task}\n\nPLAN:\n{plan}\n\nACT RESULT:\n{result}\n\nOUTCOME: {outcome}"
        );
        match client.distill_skills(&transcript).await {
            Ok(skills) => {
                for (title, content) in skills {
                    match store
                        .insert_memory(&NewMemory {
                            kind: "skill".into(),
                            title: title.clone(),
                            content,
                            source: "harness-distill".into(),
                            tags: vec!["auto-skill".into(), formula.formula.name.clone()],
                            importance: 0.85,
                        })
                        .await
                    {
                        Ok(_) => {
                            skills_stored += 1;
                            eprintln!("  · skill recorded: {title}");
                        }
                        Err(e) => tracing::warn!("skill store failed: {e}"),
                    }
                }
            }
            Err(e) => tracing::warn!("skill distill failed: {e}"),
        }
    }

    Ok(HarnessResult {
        plan,
        result,
        passed,
        attempts,
        skills_stored,
        decision_id,
    })
}

async fn build_memory_primer(
    store: &Store,
    client: &OllamaClient,
    cfg: &Config,
    task: &str,
) -> String {
    match crate::memory::search::hybrid_search(store, client, cfg, task, cfg.memory.top_k).await {
        Ok(hits) if !hits.is_empty() => {
            let mut out = String::new();
            for h in hits.iter().take(8) {
                out.push_str(&format!(
                    "- [{}] {}: {}\n",
                    h.memory.kind, h.memory.title, h.memory.content
                ));
            }
            out
        }
        _ => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_default_formula() {
        let f = Formula::default_verify().expect("default formula");
        assert_eq!(f.formula.name, "verify");
        assert!(f.formula.max_verify_retries >= 1);
    }

    #[test]
    fn rejects_empty_name() {
        let raw = r#"
[formula]
name = ""
"#;
        assert!(Formula::parse(raw).is_err());
    }
}
