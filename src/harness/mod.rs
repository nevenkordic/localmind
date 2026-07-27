//! Multi-model verify harness — plan → review → act → verify with quorum,
//! ground-truth checks, and continuous skill learning.

mod checks;
mod quorum;

use crate::agent::AgentRun;
use crate::config::Config;
use crate::llm::ollama::OllamaClient;
use crate::memory::{NewMemory, Store};
use crate::tools::audit::AuditLog;
use crate::tools::permissions::PermissionMode;
use anyhow::{Context, Result};
use quorum::{quorum_met, verifier_models, QuorumPolicy, VerdictVote};
use serde::Deserialize;
use std::path::Path;
use std::sync::Arc;

/// Built-in default formula used by `llm run` when no --formula is given.
pub const DEFAULT_FORMULA: &str = include_str!("../../formulas/verify.toml");

#[derive(Debug, Clone, Default)]
pub struct RunOptions {
    pub test_command: Option<String>,
    pub plan_review: Option<bool>,
    pub quorum_min: Option<usize>,
}

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
    #[serde(default = "default_role")]
    pub role: String,
}
fn default_role() -> String {
    "worker".into()
}

#[derive(Debug)]
pub struct HarnessResult {
    pub run_id: String,
    pub plan: String,
    pub result: String,
    pub passed: bool,
    pub attempts: usize,
    pub checks_passed: bool,
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

fn model_for_role(cfg: &Config, role: &str) -> Option<String> {
    let fast = cfg.ollama.fast_model.trim();
    let chat = cfg.ollama.chat_model.trim();
    let verify = cfg.harness.verify_model.trim();
    match role {
        "planner" => {
            if !fast.is_empty() {
                Some(fast.to_string())
            } else {
                None
            }
        }
        "verifier" => {
            if !verify.is_empty() {
                Some(verify.to_string())
            } else if !fast.is_empty() && fast != chat {
                Some(fast.to_string())
            } else {
                None
            }
        }
        _ => None,
    }
}

async fn run_quorum<F, Fut>(
    cfg: &Config,
    models: &[String],
    policy: QuorumPolicy,
    stage: &str,
    mut call: F,
) -> Result<(bool, String, Vec<VerdictVote>)>
where
    F: FnMut(String) -> Fut,
    Fut: std::future::Future<Output = Result<(bool, String)>>,
{
    let mut votes = Vec::new();
    for model in models {
        let m = model.clone();
        let (ok, fb) = call(m.clone()).await?;
        votes.push(VerdictVote {
            model: m,
            passed: ok,
            feedback: fb,
        });
    }
    let (passed, summary) = quorum_met(cfg, &votes, policy, models.len());
    eprintln!("  · {stage} quorum: {summary}");
    Ok((passed, summary, votes))
}

/// Run plan → review → act → verify with quorum and ground-truth checks.
pub async fn run(
    cfg: Config,
    store: Store,
    formula: Formula,
    task: &str,
    mode: Option<PermissionMode>,
    opts: RunOptions,
) -> Result<HarnessResult> {
    let cfg = Arc::new(cfg);
    let mut cfg_mut = (*cfg).clone();
    if let Some(n) = opts.quorum_min {
        cfg_mut.harness.quorum_min = n;
    }
    let cfg = Arc::new(cfg_mut);

    let client = OllamaClient::new(&cfg.ollama);
    let policy = QuorumPolicy::parse(&cfg.harness.quorum_policy);
    let mut max_retries = formula
        .formula
        .max_verify_retries
        .max(cfg.harness.max_retries);
    let plan_review = opts.plan_review.unwrap_or(cfg.harness.plan_review);
    let test_command = opts
        .test_command
        .as_deref()
        .unwrap_or(cfg.harness.test_command.as_str())
        .to_string();
    let verifiers = verifier_models(&cfg);

    // Adaptive retries from recent harness_runs — low pass rate → one more try.
    if cfg.harness.adaptive_retries {
        if let Ok(recent) = store.list_harness_runs(10, false, None).await {
            if recent.len() >= 3 {
                let pass_n = recent.iter().filter(|r| r.passed).count();
                let rate = pass_n as f64 / recent.len() as f64;
                if rate < 0.5 {
                    let bumped = (max_retries + 1).min(5);
                    if bumped > max_retries {
                        eprintln!(
                            "  · adaptive: recent pass rate {:.0}% → max_retries {max_retries}→{bumped}",
                            rate * 100.0
                        );
                        max_retries = bumped;
                    }
                }
            }
        }
    }

    if cfg.harness.require_distinct_models && verifiers.len() < cfg.harness.quorum_min {
        anyhow::bail!(
            "quorum requires {} distinct verifier models, only {} available ({:?}). \
             Set [harness].verify_models, lower quorum_min, or set require_distinct_models=false",
            cfg.harness.quorum_min,
            verifiers.len(),
            verifiers
        );
    }

    eprintln!(
        "· harness `{}` — plan → review → act → verify (quorum {}, max {} retries)",
        formula.formula.name, cfg.harness.quorum_min, max_retries
    );

    let (memory_primer, primed_skill_ids) = build_memory_primer(&store, &client, &cfg, task).await;
    let mut plan_verdicts: Vec<VerdictVote> = Vec::new();
    let mut verify_verdicts: Vec<VerdictVote> = Vec::new();
    // Skills seen during plan; act-primed ids replace this for attribution.
    let mut outcome_skill_ids = primed_skill_ids;
    let mut distilled_this_run: Vec<String> = Vec::new();

    let check_opts = checks::CheckOptions {
        test_command: test_command.clone(),
        require_tool_use: cfg.harness.require_tool_use,
        check_paths: cfg.harness.check_paths.clone(),
    };

    // ---- PLAN SCOUT (read-only tools) ---------------------------------
    eprintln!("  · plan scout (read-only)");
    let scout_findings = {
        let mut scout = AgentRun::new_with_mode(
            cfg.clone(),
            store.clone(),
            Some(PermissionMode::ReadOnly),
            true,
            false, // no web
            false, // no shell
        )?;
        scout.max_tool_iterations = 4;
        let scout_prompt = format!(
            "You are gathering context for a planning stage. Use ONLY \
             read-only tools (search_memory, list_decisions, \
             list_recent_actions, read_file, list_dir) to learn what is \
             relevant to the TASK. Do not modify anything. When done, \
             summarise findings in under 200 words — what exists, prior \
             decisions/failures, and pitfalls to avoid.\n\nTASK:\n{task}"
        );
        match scout.turn(&scout_prompt).await {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!("plan scout failed: {e}");
                String::new()
            }
        }
    };
    let enriched_primer = if scout_findings.trim().is_empty() {
        memory_primer.clone()
    } else {
        format!(
            "{memory_primer}\n\nSCOUT FINDINGS (read-only gather):\n{}",
            crate::util::truncate(&scout_findings, 2500)
        )
    };

    // ---- PLAN ----------------------------------------------------------
    let plan_model = model_for_role(&cfg, "planner");
    eprintln!(
        "  · plan ({})",
        plan_model.as_deref().unwrap_or(&cfg.ollama.chat_model)
    );
    let plan = client
        .plan_task(task, &enriched_primer, plan_model.as_deref())
        .await
        .context("plan stage")?;

    // ---- PLAN REVIEW QUORUM --------------------------------------------
    if plan_review {
        eprintln!("  · plan review quorum ({} models)", verifiers.len());
        let primer = enriched_primer.clone();
        let task_owned = task.to_string();
        let (ok, feedback, votes) = run_quorum(&cfg, &verifiers, policy, "plan_review", |model| {
            let client = client.clone();
            let plan = plan.clone();
            let primer = primer.clone();
            let task = task_owned.clone();
            async move {
                client
                    .review_plan(&task, &plan, &primer, Some(model.as_str()))
                    .await
            }
        })
        .await?;
        plan_verdicts = votes;
        if !ok {
            anyhow::bail!("plan review quorum failed: {feedback}");
        }
    }

    // ---- ACT / VERIFY loop --------------------------------------------
    let audit = AuditLog::open()?;
    let mut feedback = String::new();
    let mut result = String::new();
    let mut passed = false;
    let mut checks_passed = false;
    let mut attempts = 0usize;
    let mut checks_json = "[]".to_string();

    // One AgentRun for all act retries, bound to the cwd session so STM
    // (and auto_persist LTM) carry prior attempt context + tool history.
    let mut act_agent =
        AgentRun::new_with_mode(cfg.clone(), store.clone(), mode, true, true, true)?;
    const SESSION_MAX_AGE_SECS: i64 = 7 * 86400;
    let resume_limit = cfg.ollama.resume_messages;
    let cwd = std::env::current_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_default();
    if let Ok((session_id, created)) = store
        .session_get_or_create(&cwd, SESSION_MAX_AGE_SECS)
        .await
    {
        act_agent.set_session(session_id.clone());
        if !created && resume_limit > 0 {
            if let Ok(rows) = store.load_recent_messages(&session_id, resume_limit).await {
                if !rows.is_empty() {
                    let restored: Vec<_> = rows
                        .into_iter()
                        .map(|(role, content, tool_name, extras_json)| {
                            AgentRun::message_from_persisted(role, content, tool_name, extras_json)
                        })
                        .collect();
                    act_agent.restore_messages(restored);
                }
            }
        }
    }

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
        result = act_agent.turn(&act_prompt).await.context("act stage")?;
        act_agent.persist_new_messages().await;
        // Prefer skills the act agent actually primed over plan-only hits.
        let act_skills = act_agent.take_primed_skill_ids();
        if !act_skills.is_empty() {
            outcome_skill_ids = act_skills;
        }

        let audit_tail = checks::read_audit_tail(audit.path(), 40).unwrap_or_default();
        let check_report = checks::run_checks(&check_opts, &audit_tail).await;
        checks_passed = check_report.all_passed();
        checks_json = serde_json::to_string(&check_report.results).unwrap_or_else(|_| "[]".into());
        eprintln!(
            "  · ground checks: {}",
            if checks_passed { "PASS" } else { "FAIL" }
        );
        if !checks_passed {
            feedback = check_report.summary();
            if attempt == max_retries {
                break;
            }
            continue;
        }

        let evidence = format!(
            "{}\n\nAUDIT TAIL:\n{}",
            check_report.summary(),
            crate::util::truncate(&audit_tail, 3000)
        );
        let task_owned = task.to_string();
        let plan_owned = plan.clone();
        let result_owned = result.clone();
        let (ok, summary, votes) = run_quorum(&cfg, &verifiers, policy, "verify", |model| {
            let client = client.clone();
            let task = task_owned.clone();
            let plan = plan_owned.clone();
            let result = result_owned.clone();
            let evidence = evidence.clone();
            async move {
                client
                    .verify_stage(&task, &plan, &result, &evidence, Some(model.as_str()))
                    .await
            }
        })
        .await?;
        verify_verdicts = votes;
        if ok {
            passed = true;
            break;
        }
        feedback = summary;
        if attempt == max_retries {
            break;
        }
    }

    let outcome = if passed { "passed" } else { "failed" };
    let decision_id = store
        .insert_decision(
            &format!("{}: {task}", formula.formula.name),
            &format!("plan:\n{plan}"),
            &feedback,
            &format!("{outcome} after {attempts} attempt(s); checks_passed={checks_passed}"),
            "harness",
        )
        .await
        .ok();

    // Mirror into searchable LTM so future sessions recall what the harness
    // did, when, and how — not only via distilled skills.
    let when = crate::util::format_ts(crate::util::now_ts());
    let harness_title = format!(
        "harness {}: {}",
        formula.formula.name,
        crate::util::truncate(task, 60)
    );
    let harness_content = format!(
        "When: {when}\nTask: {task}\nPlan:\n{}\nResult:\n{}\nOutcome: {outcome} after {attempts} attempt(s); checks_passed={checks_passed}\nFeedback: {}",
        crate::util::truncate(&plan, 1500),
        crate::util::truncate(&result, 1500),
        crate::util::truncate(&feedback, 500),
    );
    let _ = store
        .insert_memory(&NewMemory {
            kind: "decision".into(),
            title: harness_title,
            content: harness_content,
            source: "harness".into(),
            tags: vec![
                "harness".into(),
                formula.formula.name.clone(),
                outcome.into(),
            ],
            importance: if passed { 0.8 } else { 0.55 },
            trust_tier: Some("auto".into()),
            cwd: Some(Store::current_scope_key()),
        })
        .await;

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
                            trust_tier: Some("auto".into()),
                            cwd: Some(Store::current_scope_key()),
                        })
                        .await
                    {
                        Ok(id) => {
                            skills_stored += 1;
                            distilled_this_run.push(id);
                            eprintln!("  · skill recorded: {title}");
                        }
                        Err(e) => tracing::warn!("skill store failed: {e}"),
                    }
                }
            }
            Err(e) => tracing::warn!("skill distill failed: {e}"),
        }
    }

    // On failure, store a dedicated note so future plans can avoid repeating
    // the same mistake (higher importance than a generic harness decision).
    if !passed {
        let fail_title = format!("harness fail: {}", crate::util::truncate(task, 60));
        let fail_content = format!(
            "When: {when}\nFAILED TASK: {task}\nPlan:\n{}\nResult:\n{}\nAttempts: {attempts}\nChecks passed: {checks_passed}\nVerifier feedback:\n{}\n\nAvoid repeating this approach unless the feedback is addressed.",
            crate::util::truncate(&plan, 1200),
            crate::util::truncate(&result, 1200),
            crate::util::truncate(&feedback, 800),
        );
        let _ = store
            .insert_memory(&NewMemory {
                kind: "note".into(),
                title: fail_title,
                content: fail_content,
                source: "harness-fail".into(),
                tags: vec![
                    "harness".into(),
                    "fail".into(),
                    formula.formula.name.clone(),
                ],
                importance: 0.75,
                trust_tier: Some("auto".into()),
                cwd: Some(Store::current_scope_key()),
            })
            .await;
    }

    // Attribute outcomes to skills that were in act context. Skip skills
    // distilled on this same run — they need a later successful use to promote.
    let mut link_rows: Vec<(String, &str)> = Vec::new();
    for id in &outcome_skill_ids {
        link_rows.push((id.clone(), "primed"));
        if distilled_this_run.iter().any(|d| d == id) {
            continue;
        }
        link_rows.push((id.clone(), "credited"));
        let _ = store
            .record_skill_outcome(
                id,
                passed,
                cfg.harness.skill_promote_after,
                cfg.harness.skill_demote_after,
            )
            .await;
    }
    for id in &distilled_this_run {
        link_rows.push((id.clone(), "distilled"));
    }

    let run_id = store
        .insert_harness_run(
            &formula.formula.name,
            task,
            &plan,
            &result,
            passed,
            attempts,
            checks_passed,
            &checks_json,
            skills_stored,
            decision_id.as_deref(),
        )
        .await?;

    let _ = store
        .insert_harness_skill_links(&run_id, &link_rows, passed)
        .await;

    for v in plan_verdicts {
        let _ = store
            .insert_harness_verdict(&run_id, "plan_review", &v.model, v.passed, &v.feedback)
            .await;
    }
    for v in verify_verdicts {
        let _ = store
            .insert_harness_verdict(&run_id, "verify", &v.model, v.passed, &v.feedback)
            .await;
    }

    Ok(HarnessResult {
        run_id,
        plan,
        result,
        passed,
        attempts,
        checks_passed,
        skills_stored,
        decision_id,
    })
}

async fn build_memory_primer(
    store: &Store,
    client: &OllamaClient,
    cfg: &Config,
    task: &str,
) -> (String, Vec<String>) {
    match crate::memory::search::hybrid_search(store, client, cfg, task, cfg.memory.top_k).await {
        Ok(hits) if !hits.is_empty() => {
            let mut out = String::new();
            let mut skill_ids = Vec::new();
            for h in hits.iter().take(8) {
                if h.memory.trust_tier == "ignored" {
                    continue;
                }
                let tier = &h.memory.trust_tier;
                let label = match tier.as_str() {
                    "user" => "USER",
                    "verified" => "VERIFIED",
                    _ => "AUTO",
                };
                if h.memory.kind == "skill" {
                    skill_ids.push(h.memory.id.clone());
                }
                out.push_str(&format!(
                    "- [{label} {}] {}: {}\n",
                    h.memory.kind, h.memory.title, h.memory.content
                ));
            }
            (out, skill_ids)
        }
        _ => (String::new(), Vec::new()),
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
