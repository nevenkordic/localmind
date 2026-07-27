//! Multi-model verify harness — plan → review → act → verify with quorum,
//! ground-truth checks, and continuous skill learning.

pub mod checks;
mod formula;
mod quorum;
mod select;

pub use formula::{evaluate_condition, render_prompt, StageInput};
pub use select::resolve as resolve_formula;

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

/// Built-in default formula used by `llm run` when no intent-specific
/// formula matches the task (see [`resolve_formula`]).
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
    /// Stages (localmind name). Broodlink formulas use `steps` — accepted
    /// as an alias so the same TOML shape can be shared.
    #[serde(default, alias = "steps")]
    pub stages: Vec<Stage>,
    /// Optional recovery stage when the run fails (broodlink `on_failure`).
    #[serde(default)]
    pub on_failure: Option<Stage>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FormulaMeta {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default = "default_retries")]
    pub max_verify_retries: usize,
    /// Substrings — at least one must appear in the task (case-insensitive)
    /// for auto-selection from `./formulas/*.toml`.
    #[serde(default)]
    pub match_any: Vec<String>,
    /// Substrings — all must appear for auto-selection.
    #[serde(default)]
    pub match_all: Vec<String>,
    /// Tie-break boost for on-disk formula auto-selection.
    #[serde(default)]
    pub priority: i32,
    /// After act, require a detached server from the audit log to be
    /// listening (TCP). Used by site-serve. When this check passes,
    /// LLM verify is skipped — the port is objective evidence.
    #[serde(default)]
    pub require_detached_server: bool,
    /// Optional chat model override for this formula's act stage
    /// (e.g. a larger coder). Empty = use config chat_model. If the
    /// named model isn't pulled, harness falls back to config.
    #[serde(default)]
    pub chat_model: String,
    /// Minimum bytes for each HTML file written this act (0 = skip).
    #[serde(default)]
    pub min_html_bytes: u64,
    /// Minimum bytes for CSS files written this act (0 = skip).
    #[serde(default)]
    pub min_css_bytes: u64,
}
fn default_retries() -> usize {
    2
}

#[derive(Debug, Clone, Deserialize)]
pub struct Stage {
    pub name: String,
    /// Role used for model selection. Broodlink uses `agent_role`.
    #[serde(default = "default_role", alias = "agent_role")]
    pub role: String,
    /// Optional prompt template with `{{task}}`, `{{plan}}`, `{{memory}}`, …
    #[serde(default)]
    pub prompt: Option<String>,
    /// Prior stage output key(s) to inject into the prompt context.
    #[serde(default)]
    pub input: Option<StageInput>,
    /// Name under which this stage's result is stored for later stages.
    #[serde(default)]
    pub output: Option<String>,
    /// Fail-closed skip condition (see `formula::evaluate_condition`).
    #[serde(default)]
    pub when: Option<String>,
    /// Parallel group id — same id runs concurrently in a future formula
    /// runner. Reserved; the default verify pipeline is sequential.
    #[serde(default)]
    #[allow(dead_code)]
    pub group: Option<u32>,
    /// Few-shot examples appended to the stage prompt when present.
    #[serde(default)]
    pub examples: Option<String>,
}
fn default_role() -> String {
    "worker".into()
}

impl Formula {
    /// Look up a stage by name (plan / act / verify / …).
    pub fn stage(&self, name: &str) -> Option<&Stage> {
        self.stages.iter().find(|s| s.name == name)
    }
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

/// Prefer formula.chat_model when set and installed; else optionally the
/// largest installed chat-capable model by parameter tag (`prefer_capable_act`);
/// else config chat_model. Family-agnostic — Qwen, Llama, Mistral, etc.
async fn resolve_act_model(cfg: &Config, formula: &Formula) -> String {
    let fallback = cfg.ollama.chat_model.clone();
    let wanted = formula.formula.chat_model.trim();
    let client = OllamaClient::new(&cfg.ollama);
    let installed = match client.health().await {
        Ok(m) => m,
        Err(_) => return fallback,
    };
    let is_in = |name: &str| {
        installed
            .iter()
            .any(|m| m == name || m == &format!("{name}:latest"))
    };

    if !wanted.is_empty() && wanted != fallback {
        if is_in(wanted) {
            eprintln!("  · act model override: {wanted}");
            return wanted.to_string();
        }
        eprintln!(
            "  · act model `{wanted}` not installed — falling back \
             (ollama pull {wanted})"
        );
    }

    if cfg.harness.prefer_capable_act {
        let exclude = [
            cfg.ollama.embed_model.as_str(),
            cfg.ollama.vision_model.as_str(),
        ];
        if let Some(pick) = pick_capable_act_model(&installed, &fallback, &exclude) {
            if pick != fallback {
                eprintln!("  · act model (largest installed): {pick}");
            }
            return pick;
        }
    }
    fallback
}

/// Parse a rough parameter score from an Ollama model name.
/// Handles `:7b`, `:32b-instruct-q4_K_M`, and MoE tags like `:8x7b`.
/// Returns milliparam units (7b → 7000) so integer compares stay simple.
pub(crate) fn model_param_score(name: &str) -> u32 {
    let lower = name.to_ascii_lowercase();
    let mut best = 0u32;

    // MoE: 8x7b ≈ 56B active-params proxy (good enough for ranking).
    let moe = regex::Regex::new(r"([0-9]+)x([0-9]+(?:\.[0-9]+)?)b\b").ok();
    if let Some(re) = &moe {
        for caps in re.captures_iter(&lower) {
            let a: u32 = caps[1].parse().unwrap_or(0);
            let b_str = &caps[2];
            let b_whole = b_str.split('.').next().unwrap_or("0");
            let b: u32 = b_whole.parse().unwrap_or(0);
            best = best.max(a.saturating_mul(b).saturating_mul(1000));
        }
    }

    let dense = regex::Regex::new(r":([0-9]+(?:\.[0-9]+)?)b\b").ok();
    if let Some(re) = &dense {
        for caps in re.captures_iter(&lower) {
            let n_str = &caps[1];
            // Prefer whole billions; fractional (1.5b) → floor * 1000 + fraction hint.
            if let Some((whole, frac)) = n_str.split_once('.') {
                let w: u32 = whole.parse().unwrap_or(0);
                let f: u32 = frac.chars().next().and_then(|c| c.to_digit(10)).unwrap_or(0);
                best = best.max(w.saturating_mul(1000).saturating_add(f * 100));
            } else {
                let w: u32 = n_str.parse().unwrap_or(0);
                best = best.max(w.saturating_mul(1000));
            }
        }
    }

    best
}

fn is_unlikely_chat_model(name: &str) -> bool {
    let l = name.to_ascii_lowercase();
    l.contains("embed")
        || l.contains("rerank")
        || l.contains("minilm")
        || l.starts_with("bge-")
        || l.contains("nomic-embed")
        || l.contains("mxbai-embed")
        || l.contains("snowflake-arctic-embed")
}

/// Pick the largest installed chat-capable model that beats `fallback`.
/// Scans every Ollama tag — not a fixed vendor list.
fn pick_capable_act_model(
    installed: &[String],
    fallback: &str,
    exclude: &[&str],
) -> Option<String> {
    let excluded = |name: &str| {
        exclude.iter().any(|e| {
            let e = e.trim();
            if e.is_empty() {
                return false;
            }
            name == e || name == format!("{e}:latest") || e == format!("{name}:latest")
        })
    };

    let fb_score = model_param_score(fallback);
    let mut best: Option<(u32, String)> = None;

    for name in installed {
        if is_unlikely_chat_model(name) || excluded(name) {
            continue;
        }
        let score = model_param_score(name);
        if score == 0 {
            continue;
        }
        // Upgrade only when strictly larger than the configured chat model.
        if score <= fb_score {
            continue;
        }
        match &best {
            Some((s, prev)) if *s > score || (*s == score && prev.as_str() <= name.as_str()) => {}
            _ => best = Some((score, name.clone())),
        }
    }

    if let Some((_, m)) = best {
        return Some(m);
    }

    // No larger model — keep fallback when present, else largest scorable chat model.
    let fallback_present = installed.iter().any(|m| {
        m == fallback || m == &format!("{fallback}:latest") || fallback == &format!("{m}:latest")
    });
    if fallback_present {
        return Some(fallback.to_string());
    }

    let mut largest: Option<(u32, String)> = None;
    for name in installed {
        if is_unlikely_chat_model(name) || excluded(name) {
            continue;
        }
        let score = model_param_score(name);
        if score == 0 {
            continue;
        }
        match &largest {
            Some((s, prev)) if *s > score || (*s == score && prev.as_str() <= name.as_str()) => {}
            _ => largest = Some((score, name.clone())),
        }
    }
    largest.map(|(_, m)| m)
}

#[cfg(test)]
mod capable_act_tests {
    use super::*;

    #[test]
    fn picks_any_family_larger_than_fallback() {
        let installed = vec![
            "qwen2.5-coder:7b".into(),
            "llama3.3:70b".into(),
            "nomic-embed-text".into(),
            "mistral:7b".into(),
        ];
        assert_eq!(
            pick_capable_act_model(&installed, "qwen2.5-coder:7b", &["nomic-embed-text"])
                .as_deref(),
            Some("llama3.3:70b")
        );
    }

    #[test]
    fn picks_32b_over_7b_same_or_other_family() {
        let installed = vec![
            "gemma2:9b".into(),
            "qwen2.5-coder:32b".into(),
            "nomic-embed-text".into(),
        ];
        assert_eq!(
            pick_capable_act_model(&installed, "gemma2:9b", &[]).as_deref(),
            Some("qwen2.5-coder:32b")
        );
    }

    #[test]
    fn ignores_embed_and_excluded_vision() {
        let installed = vec![
            "qwen2.5-coder:7b".into(),
            "nomic-embed-text:latest".into(),
            "llava:34b".into(),
        ];
        // llava excluded as vision → stay on 7b
        assert_eq!(
            pick_capable_act_model(&installed, "qwen2.5-coder:7b", &["llava:34b"]).as_deref(),
            Some("qwen2.5-coder:7b")
        );
    }

    #[test]
    fn keeps_fallback_when_already_largest() {
        let installed = vec!["mistral-small:24b".into(), "phi3:3.8b".into()];
        assert_eq!(
            pick_capable_act_model(&installed, "mistral-small:24b", &[]).as_deref(),
            Some("mistral-small:24b")
        );
    }

    #[test]
    fn moe_tag_outranks_dense_7b() {
        assert!(model_param_score("mixtral:8x7b") > model_param_score("qwen2.5-coder:7b"));
        assert!(model_param_score("llama3.3:70b-instruct-q4_K_M") > model_param_score("codestral:22b"));
    }

    #[test]
    fn no_upgrade_when_capable_missing() {
        let installed = vec!["qwen2.5-coder:7b".into()];
        assert_eq!(
            pick_capable_act_model(&installed, "qwen2.5-coder:7b", &[]).as_deref(),
            Some("qwen2.5-coder:7b")
        );
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

    let client = OllamaClient::new(&cfg_mut.ollama);
    let policy = QuorumPolicy::parse(&cfg_mut.harness.quorum_policy);
    let mut max_retries = formula
        .formula
        .max_verify_retries
        .max(cfg_mut.harness.max_retries);
    let plan_review = opts.plan_review.unwrap_or(cfg_mut.harness.plan_review);
    let test_command = opts
        .test_command
        .as_deref()
        .unwrap_or(cfg_mut.harness.test_command.as_str())
        .to_string();
    let verifiers = verifier_models(&cfg_mut);

    // Adaptive retries from recent harness_runs — low pass rate → one more try.
    if cfg_mut.harness.adaptive_retries {
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

    if cfg_mut.harness.require_distinct_models && verifiers.len() < cfg_mut.harness.quorum_min {
        if verifiers.is_empty() {
            anyhow::bail!(
                "quorum requires {} distinct verifier models, none available. \
                 Set [harness].verify_models or pull a chat model.",
                cfg_mut.harness.quorum_min
            );
        }
        // Single-model installs are the common case — auto-relax instead of
        // failing the run before plan even starts.
        eprintln!(
            "  · quorum relaxed: only {} verifier model(s) available {:?}; \
             wanted {} (set [harness].verify_models or quorum_min to silence)",
            verifiers.len(),
            verifiers,
            cfg_mut.harness.quorum_min
        );
        cfg_mut.harness.quorum_min = verifiers.len();
        cfg_mut.harness.require_distinct_models = false;
    }

    let cfg = Arc::new(cfg_mut);

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
        require_detached_server: formula.formula.require_detached_server,
        min_html_bytes: formula.formula.min_html_bytes,
        min_css_bytes: formula.formula.min_css_bytes,
    };

    // Formula may request a larger act model (e.g. site-serve → 32b).
    let act_model = resolve_act_model(&cfg, &formula).await;

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
    let mut step_results = serde_json::Map::new();
    step_results.insert("task".into(), serde_json::Value::String(task.to_string()));
    step_results.insert(
        "memory".into(),
        serde_json::Value::String(enriched_primer.clone()),
    );
    step_results.insert(
        "scout".into(),
        serde_json::Value::String(scout_findings.clone()),
    );

    let plan = if let Some(tmpl) = formula
        .stage("plan")
        .and_then(|s| s.prompt.as_deref())
        .filter(|t| !t.trim().is_empty())
    {
        let params = serde_json::Value::Object(step_results.clone());
        let mut user = render_prompt(tmpl, &params);
        if let Some(ex) = formula.stage("plan").and_then(|s| s.examples.as_deref()) {
            user.push_str("\n\nEXAMPLES:\n");
            user.push_str(ex);
        }
        let msgs = vec![
            crate::llm::types::ChatMessage::system(
                "You are the planner stage of a local multi-model harness. \
                 Plan for COMPLETE, accurate, high-quality delivery — not stubs. \
                 Follow the user instructions. Bulleted steps, max 250 words.",
            ),
            crate::llm::types::ChatMessage::user(user),
        ];
        client
            .chat_on(&msgs, None, false, plan_model.as_deref())
            .await
            .context("plan stage")?
            .content
            .trim()
            .to_string()
    } else {
        client
            .plan_task(task, &enriched_primer, plan_model.as_deref())
            .await
            .context("plan stage")?
    };
    let plan_key = formula
        .stage("plan")
        .and_then(|s| s.output.clone())
        .unwrap_or_else(|| "plan".into());
    step_results.insert(plan_key, serde_json::Value::String(plan.clone()));
    step_results.insert("plan".into(), serde_json::Value::String(plan.clone()));

    // ---- PLAN REVIEW QUORUM --------------------------------------------
    let review_when_ok = formula
        .stage("plan_review")
        .and_then(|s| s.when.as_deref())
        .map(|expr| evaluate_condition(expr, &serde_json::Value::Object(step_results.clone())))
        .unwrap_or(true);
    if plan_review && review_when_ok {
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

    // One AgentRun for all act retries. Use a fresh harness-scoped session —
    // never resume the cwd REPL history. Prior failed site-serve turns poison
    // small models into emitting ghost calls like `listening_ports()` as prose
    // instead of real tool_calls.
    let mut act_agent =
        AgentRun::new_with_mode(cfg.clone(), store.clone(), mode, true, true, true)?;
    // Large site builds need many tool rounds (dir + 3 files + ports + serve).
    act_agent.max_tool_iterations = 24;
    // Stream so the TTFB watchdog applies (not a hard wall on the whole
    // non-streaming generation). Large write_file bodies on 32B can take
    // several minutes — streaming keeps those alive under timeout_secs.
    act_agent.set_token_sink(std::sync::Arc::new(|chunk: &str| {
        use std::io::Write;
        let mut out = std::io::stdout().lock();
        let _ = write!(out, "{chunk}");
        let _ = out.flush();
    }));
    const SESSION_MAX_AGE_SECS: i64 = 7 * 86400;
    let harness_scope = format!(
        "harness:{}:{}",
        formula.formula.name,
        crate::util::now_ts()
    );
    if let Ok((session_id, _)) = store
        .session_get_or_create(&harness_scope, SESSION_MAX_AGE_SECS)
        .await
    {
        act_agent.set_session(session_id);
    }

    // Record harness start so retries can reuse earlier detached_server /
    // file writes as evidence (TCP still up + on-disk sizes), while
    // require_tool_use still looks at this attempt only.
    let harness_started_ts = crate::util::now_ts();

    for attempt in 0..=max_retries {
        attempts = attempt + 1;
        let act_started_ts = crate::util::now_ts();
        step_results.insert(
            "feedback".into(),
            serde_json::Value::String(feedback.clone()),
        );
        // Honour act.when — skip (treat as failed attempt) when false.
        if let Some(expr) = formula.stage("act").and_then(|s| s.when.as_deref()) {
            if !evaluate_condition(expr, &serde_json::Value::Object(step_results.clone())) {
                feedback = format!("act stage skipped: when `{expr}` was false");
                eprintln!("  · act skipped ({feedback})");
                break;
            }
        }
        // Inject declared inputs into the step map under short aliases.
        if let Some(inp) = formula.stage("act").and_then(|s| s.input.as_ref()) {
            let keys: Vec<String> = match inp {
                StageInput::Single(k) => vec![k.clone()],
                StageInput::Multiple(ks) => ks.clone(),
            };
            for k in keys {
                if let Some(v) = step_results.get(&k).cloned() {
                    step_results.insert(format!("input_{k}"), v);
                }
            }
        }
        let act_prompt = if let Some(tmpl) = formula
            .stage("act")
            .and_then(|s| s.prompt.as_deref())
            .filter(|t| !t.trim().is_empty())
        {
            let params = serde_json::Value::Object(step_results.clone());
            let mut p = render_prompt(tmpl, &params);
            if let Some(ex) = formula.stage("act").and_then(|s| s.examples.as_deref()) {
                p.push_str("\n\nEXAMPLES:\n");
                p.push_str(ex);
            }
            p
        } else if feedback.is_empty() {
            format!(
                "You are the ACT stage of a verified local harness.\n\
                 Produce COMPLETE, accurate, high-quality work for ANY task \
                 type (code, config, research, admin, writing) — never stubs \
                 or placeholders. Use real tool calls. When done, summarise \
                 what you did.\n\nTASK:\n{task}\n\nPLAN:\n{plan}"
            )
        } else {
            format!(
                "You are the ACT stage of a verified local harness.\n\
                 Previous attempt failed quality/accuracy verification. Fix \
                 every issue in the feedback with real tools — raise quality, \
                 do not re-ship stubs — then summarise what you changed.\n\n\
                 TASK:\n{task}\n\nPLAN:\n{plan}\n\n\
                 VERIFIER FEEDBACK:\n{feedback}"
            )
        };

        eprintln!("  · act attempt {attempts} ({act_model})");
        act_agent.force_next_model(&act_model);
        result = act_agent.turn(&act_prompt).await.context("act stage")?;
        act_agent.persist_new_messages().await;
        let act_key = formula
            .stage("act")
            .and_then(|s| s.output.clone())
            .unwrap_or_else(|| "result".into());
        step_results.insert(act_key, serde_json::Value::String(result.clone()));
        step_results.insert("result".into(), serde_json::Value::String(result.clone()));
        // Prefer skills the act agent actually primed over plan-only hits.
        let act_skills = act_agent.take_primed_skill_ids();
        if !act_skills.is_empty() {
            outcome_skill_ids = act_skills;
        }

        let act_audit =
            checks::read_audit_tail_since(audit.path(), act_started_ts, 80).unwrap_or_default();
        let evidence_audit = checks::read_audit_tail_since(audit.path(), harness_started_ts, 200)
            .unwrap_or_else(|_| act_audit.clone());
        let check_report =
            checks::run_checks_with_evidence(&check_opts, &act_audit, &evidence_audit).await;
        checks_passed = check_report.all_passed();
        checks_json = serde_json::to_string(&check_report.results).unwrap_or_else(|_| "[]".into());
        eprintln!(
            "  · ground checks: {}",
            if checks_passed { "PASS" } else { "FAIL" }
        );
        if !checks_passed {
            let claimed_up = result.to_ascii_lowercase().contains("site is up")
                || result.to_ascii_lowercase().contains("http://127.0.0.1");
            if claimed_up {
                eprintln!(
                    "      · note: model prose claimed success; ground checks disagree — not a pass"
                );
            }
            for line in check_report.summary().lines() {
                if !line.is_empty() {
                    eprintln!("      {line}");
                }
            }
            feedback = checks::retry_guidance(&check_report, &check_opts);
            if attempt == max_retries {
                break;
            }
            continue;
        }

        // Objective serve + substance checks already passed — don't let a
        // chatty verifier veto a real site that is accepting TCP.
        let server_up = check_report
            .results
            .iter()
            .any(|r| r.name == "detached_server" && r.passed);
        let substance_ok = !check_report.results.iter().any(|r| {
            (r.name == "html_substance" || r.name == "css_substance") && !r.passed
        });
        if (check_opts.require_detached_server || server_up)
            && substance_ok
            && (check_opts.min_html_bytes == 0
                || check_report
                    .results
                    .iter()
                    .any(|r| r.name == "html_substance" && r.passed))
        {
            eprintln!(
                "  · verify: PASS (detached_server + substance — skipping LLM quorum)"
            );
            passed = true;
            // Prefer a clean user-facing result with the URL when the act
            // prose was just a tool name / fragment.
            if result.trim().len() < 80
                || result.contains("listening_ports()")
                || !result.to_ascii_lowercase().contains("http://")
            {
                if let Some(detail) = check_report
                    .results
                    .iter()
                    .find(|r| r.name == "detached_server" && r.passed)
                {
                    result = format!("Site is up. {}", detail.detail);
                }
            }
            break;
        }

        let evidence = format!(
            "{}\n\nAUDIT TAIL:\n{}",
            check_report.summary(),
            crate::util::truncate(&evidence_audit, 3000)
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
        let mut fail_extra = String::new();
        if let Some(fail_stage) = &formula.on_failure {
            if let Some(tmpl) = fail_stage.prompt.as_deref() {
                let params = serde_json::Value::Object(step_results.clone());
                fail_extra = render_prompt(tmpl, &params);
            }
        }
        let fail_title = format!("harness fail: {}", crate::util::truncate(task, 60));
        let fail_content = format!(
            "When: {when}\nFAILED TASK: {task}\nPlan:\n{}\nResult:\n{}\nAttempts: {attempts}\nChecks passed: {checks_passed}\nVerifier feedback:\n{}\n{}\n\nAvoid repeating this approach unless the feedback is addressed.",
            crate::util::truncate(&plan, 1200),
            crate::util::truncate(&result, 1200),
            crate::util::truncate(&feedback, 800),
            if fail_extra.is_empty() {
                String::new()
            } else {
                format!("On-failure notes:\n{}", crate::util::truncate(&fail_extra, 800))
            },
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

    #[test]
    fn accepts_broodlink_style_aliases() {
        let raw = r#"
[formula]
name = "custom"
description = "broodlink-shaped"

[[steps]]
name = "plan"
agent_role = "planner"
prompt = "Plan {{task}}"
output = "plan"

[[steps]]
name = "act"
agent_role = "worker"
prompt = "Do {{plan}}"
input = "plan"
output = "result"
when = "plan.exists"
"#;
        let f = Formula::parse(raw).expect("parse");
        assert_eq!(f.stages.len(), 2);
        assert_eq!(f.stage("plan").unwrap().role, "planner");
        assert!(f
            .stage("plan")
            .unwrap()
            .prompt
            .as_deref()
            .unwrap()
            .contains("{{task}}"));
        assert_eq!(f.stage("act").unwrap().when.as_deref(), Some("plan.exists"));
        let ctx = serde_json::json!({"plan": "x"});
        assert!(evaluate_condition("plan.exists", &ctx));
        assert_eq!(
            render_prompt("Plan {{task}}", &serde_json::json!({"task": "hi"})),
            "Plan hi"
        );
    }
}
