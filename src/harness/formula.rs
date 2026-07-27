//! Formula helpers inspired by broodlink's formula engine:
//! prompt templating (`{{key}}`), fail-closed `when` conditions, and
//! richer optional stage fields. The default verify formula still runs
//! the hardcoded plan→act→verify pipeline; custom formulas can override
//! stage prompts via these fields.

use serde::Deserialize;
use serde_json::Value;

/// Optional input wiring: one prior step name, or several.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum StageInput {
    Single(String),
    Multiple(Vec<String>),
}

/// Replace `{{key}}` placeholders from a flat JSON object of string values.
/// Unknown keys are left intact so typos stay visible in prompts.
pub fn render_prompt(template: &str, params: &Value) -> String {
    let Some(obj) = params.as_object() else {
        return template.to_string();
    };
    let mut out = template.to_string();
    for (key, val) in obj {
        let needle = format!("{{{{{key}}}}}");
        let replacement = match val {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        out = out.replace(&needle, &replacement);
    }
    out
}

/// Fail-closed condition evaluator for stage `when` clauses.
/// Supported forms (whitespace-tolerant):
///   - `key.exists`
///   - `key.count > N` / `>=` / `<` / `<=` / `==`
///   - `key == "literal"` / `key != "literal"`
///   - bare `key` (truthy string / non-empty array / true bool)
pub fn evaluate_condition(expr: &str, step_results: &Value) -> bool {
    let expr = expr.trim();
    if expr.is_empty() {
        return true;
    }

    if let Some(key) = expr.strip_suffix(".exists") {
        return lookup(step_results, key.trim()).is_some();
    }

    for op in [">=", "<=", "==", "!=", ">", "<"] {
        if let Some((left, right)) = split_once_op(expr, op) {
            let left = left.trim();
            let right = right.trim();
            if let Some(key) = left.strip_suffix(".count") {
                let n = count_of(step_results, key.trim());
                let Ok(threshold) = right.parse::<usize>() else {
                    return false;
                };
                return cmp_usize(n, threshold, op);
            }
            if (op == "==" || op == "!=") && right.starts_with('"') && right.ends_with('"') {
                let literal = &right[1..right.len() - 1];
                let actual = lookup(step_results, left)
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                return if op == "==" {
                    actual == literal
                } else {
                    actual != literal
                };
            }
            return false;
        }
    }

    // Bare key — truthy check.
    match lookup(step_results, expr) {
        None => false,
        Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::String(s)) => !s.trim().is_empty(),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Number(n)) => n.as_f64().unwrap_or(0.0) != 0.0,
        Some(Value::Object(o)) => !o.is_empty(),
    }
}

fn split_once_op<'a>(expr: &'a str, op: &str) -> Option<(&'a str, &'a str)> {
    let idx = expr.find(op)?;
    // Avoid matching `=` inside `==` when searching for `=` — we only pass
    // multi-char ops that don't have that ambiguity, plus `>` / `<`.
    Some((&expr[..idx], &expr[idx + op.len()..]))
}

fn lookup<'a>(step_results: &'a Value, key: &str) -> Option<&'a Value> {
    step_results.as_object()?.get(key)
}

fn count_of(step_results: &Value, key: &str) -> usize {
    match lookup(step_results, key) {
        Some(Value::Array(a)) => a.len(),
        Some(Value::String(s)) if !s.is_empty() => 1,
        Some(Value::Object(o)) => o.len(),
        Some(Value::Bool(true)) => 1,
        Some(Value::Number(_)) => 1,
        _ => 0,
    }
}

fn cmp_usize(a: usize, b: usize, op: &str) -> bool {
    match op {
        ">" => a > b,
        ">=" => a >= b,
        "<" => a < b,
        "<=" => a <= b,
        "==" => a == b,
        "!=" => a != b,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn render_prompt_replaces_placeholders() {
        let params = json!({"task": "fix deploy", "plan": "step 1"});
        let out = render_prompt("Do {{task}} per {{plan}}", &params);
        assert_eq!(out, "Do fix deploy per step 1");
        assert!(render_prompt("keep {{missing}}", &params).contains("{{missing}}"));
    }

    #[test]
    fn evaluate_condition_exists_and_count() {
        let ctx = json!({
            "plan": "do thing",
            "tests": ["a", "b"],
        });
        assert!(evaluate_condition("plan.exists", &ctx));
        assert!(!evaluate_condition("missing.exists", &ctx));
        assert!(evaluate_condition("tests.count > 1", &ctx));
        assert!(!evaluate_condition("tests.count > 5", &ctx));
        assert!(evaluate_condition("plan == \"do thing\"", &ctx));
        assert!(!evaluate_condition("plan == \"other\"", &ctx));
        assert!(evaluate_condition("plan", &ctx));
        assert!(!evaluate_condition("missing", &ctx));
    }
}
