//! Auto-select a harness formula from the user's task text.
//!
//! `llm run "…"` should not require `--formula` for common intents —
//! build+serve a static site, light coding, etc. Explicit `--formula`
//! always wins; otherwise we score built-ins (and optional `./formulas/*.toml`)
//! and fall back to the default verify formula.

use super::{Formula, DEFAULT_FORMULA};
use anyhow::Result;
use std::path::Path;

/// Bundled formulas available without a path on disk.
pub const SITE_SERVE_FORMULA: &str = include_str!("../../formulas/site-serve.toml");
pub const CODING_LITE_FORMULA: &str = include_str!("../../formulas/coding-lite.toml");

/// Resolve which formula to run. Returns the formula and a short human
/// reason printed to stderr (e.g. `site-serve (auto)`).
pub fn resolve(task: &str, explicit: Option<&Path>) -> Result<(Formula, String)> {
    if let Some(path) = explicit {
        let f = Formula::load_path(path)?;
        let label = format!("{} ({})", f.formula.name, path.display());
        return Ok((f, label));
    }

    if let Some((f, why)) = pick_best(task)? {
        return Ok((f, why));
    }

    let f = Formula::parse(DEFAULT_FORMULA)?;
    Ok((f, "verify (default)".into()))
}

fn pick_best(task: &str) -> Result<Option<(Formula, String)>> {
    let mut best: Option<(i32, Formula, String)> = None;

    // Built-ins with intent heuristics (more reliable than keyword lists alone).
    consider(
        &mut best,
        site_serve_score(task),
        SITE_SERVE_FORMULA,
        "site-serve (auto: build + serve static site)",
    )?;
    consider(
        &mut best,
        coding_lite_score(task),
        CODING_LITE_FORMULA,
        "coding-lite (auto: coding task)",
    )?;

    // User / project formulas in ./formulas — opt-in via match_any / match_all.
    if let Ok(entries) = std::fs::read_dir("formulas") {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("toml") {
                continue;
            }
            let Ok(f) = Formula::load_path(&path) else {
                continue;
            };
            let Some(score) = score_match_fields(task, &f) else {
                continue;
            };
            // Prefer on-disk overrides slightly over equal-scoring builtins.
            let score = score + 1;
            let why = format!("{} (auto: ./{})", f.formula.name, path.display());
            if best.as_ref().map(|(s, _, _)| score > *s).unwrap_or(true) {
                best = Some((score, f, why));
            }
        }
    }

    Ok(best.map(|(_, f, why)| (f, why)))
}

fn consider(
    best: &mut Option<(i32, Formula, String)>,
    score: Option<i32>,
    raw: &str,
    why: &str,
) -> Result<()> {
    let Some(score) = score else {
        return Ok(());
    };
    if best.as_ref().map(|(s, _, _)| score > *s).unwrap_or(true) {
        let f = Formula::parse(raw)?;
        *best = Some((score, f, why.to_string()));
    }
    Ok(())
}

/// Build / create a static site AND serve / run it locally.
pub fn site_serve_score(task: &str) -> Option<i32> {
    let t = task.to_ascii_lowercase();
    let site = [
        "website",
        "web site",
        "webpage",
        "web page",
        "landing page",
        "static site",
        "html site",
        "html page",
        "2-page",
        "2 page",
        "two-page",
        "two page",
        "5-page",
        "5 page",
        "multi-page",
        "multipage",
    ]
    .iter()
    .any(|k| t.contains(k))
        || (t.contains("site")
            && ["build", "create", "make", "scaffold", "generate"]
                .iter()
                .any(|v| t.contains(v)));

    let serve = [
        "serve",
        "run locally",
        "run it locally",
        "host it",
        "host locally",
        "localhost",
        "http.server",
        "local server",
        "unused port",
        "open locally",
        "run it on",
        "and run",
    ]
    .iter()
    .any(|k| t.contains(k));

    if site && serve {
        Some(100)
    } else {
        None
    }
}

/// Light coding / implement / fix tasks without an explicit serve intent.
pub fn coding_lite_score(task: &str) -> Option<i32> {
    let t = task.to_ascii_lowercase();
    // Don't steal site-serve tasks.
    if site_serve_score(task).is_some() {
        return None;
    }
    let coding = [
        "implement",
        "refactor",
        "fix the",
        "add a ",
        "add an ",
        "write a function",
        "write a test",
        "unit test",
        "cargo ",
        "binary",
        "cli ",
        "bug",
        "compile",
        "type error",
    ]
    .iter()
    .any(|k| t.contains(k));
    if coding {
        Some(50)
    } else {
        None
    }
}

fn score_match_fields(task: &str, f: &Formula) -> Option<i32> {
    let t = task.to_ascii_lowercase();
    let any = &f.formula.match_any;
    let all = &f.formula.match_all;
    if any.is_empty() && all.is_empty() {
        return None;
    }
    if !all.is_empty()
        && !all
            .iter()
            .all(|k| t.contains(&k.to_ascii_lowercase()))
    {
        return None;
    }
    let hits = any
        .iter()
        .filter(|k| t.contains(&k.to_ascii_lowercase()))
        .count();
    if hits == 0 && all.is_empty() {
        return None;
    }
    if hits == 0 && !all.is_empty() {
        return Some(20 + f.formula.priority);
    }
    Some((hits as i32) * 10 + f.formula.priority)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picks_site_serve_for_build_and_serve() {
        let (f, why) =
            resolve("build a 2-page site and serve it", None).expect("resolve");
        assert_eq!(f.formula.name, "site-serve");
        assert!(why.contains("site-serve"), "{why}");
    }

    #[test]
    fn picks_verify_for_generic_task() {
        let (f, why) = resolve("summarise what we decided last week", None).expect("resolve");
        assert_eq!(f.formula.name, "verify");
        assert!(why.contains("default"), "{why}");
    }

    #[test]
    fn picks_coding_lite_for_implement() {
        let (f, _) = resolve("implement a hello world binary", None).expect("resolve");
        assert_eq!(f.formula.name, "coding-lite");
    }

    #[test]
    fn explicit_path_wins() {
        // Point at the in-repo verify formula.
        let path = Path::new("formulas/verify.toml");
        if !path.exists() {
            return; // skip if cwd isn't the repo root
        }
        let (f, why) =
            resolve("build a 2-page site and serve it", Some(path)).expect("resolve");
        assert_eq!(f.formula.name, "verify");
        assert!(why.contains("verify"));
    }

    #[test]
    fn site_without_serve_is_not_site_serve() {
        assert!(site_serve_score("improve the castagna website design").is_none());
    }
}
