//! Memory tools the model can call directly.

use crate::llm::ollama::OllamaClient as Ollama;
use crate::memory::{search::hybrid_search, NewMemory};
use crate::tools::registry::ToolContext;
use anyhow::{anyhow, Result};
use serde_json::Value;

pub async fn search_memory(ctx: &ToolContext, args: &Value) -> Result<String> {
    let query = args
        .get("query")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("missing query"))?;
    let top_k = args.get("top_k").and_then(|v| v.as_u64()).unwrap_or(8) as usize;
    let client = Ollama::new(&ctx.cfg.ollama);
    let hits = hybrid_search(&ctx.store, &client, &ctx.cfg, query, top_k).await?;
    let mut out = String::new();
    for (i, h) in hits.iter().enumerate() {
        out.push_str(&format!(
            "[#{} score={:.3} kind={} id={}]\n{}\n{}\n\n",
            i + 1,
            h.score,
            h.memory.kind,
            h.memory.id,
            h.memory.title,
            crate::util::truncate(&h.memory.content, 600)
        ));
    }
    if out.is_empty() {
        out.push_str("(no matching memories)");
    }
    Ok(out)
}

pub async fn store_memory(ctx: &ToolContext, args: &Value) -> Result<String> {
    let title = args
        .get("title")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("missing title"))?
        .to_string();
    let content = args
        .get("content")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("missing content"))?
        .to_string();
    let kind = args
        .get("kind")
        .and_then(|v| v.as_str())
        .unwrap_or("note")
        .to_string();
    let tags: Vec<String> = args
        .get("tags")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    let importance = args
        .get("importance")
        .and_then(|v| v.as_f64())
        .unwrap_or(if kind == "skill" { 0.85 } else { 0.5 }) as f32;
    let id = ctx
        .store
        .insert_memory(&NewMemory {
            kind,
            title,
            content,
            source: "agent".into(),
            tags,
            importance,
            trust_tier: None,
        })
        .await?;
    Ok(format!("stored memory {id}"))
}

pub async fn log_decision(ctx: &ToolContext, args: &Value) -> Result<String> {
    let decision = args
        .get("decision")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("missing decision"))?;
    let reasoning = args.get("reasoning").and_then(|v| v.as_str()).unwrap_or("");
    let alternatives = args
        .get("alternatives")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let outcome = args.get("outcome").and_then(|v| v.as_str()).unwrap_or("");
    let id = ctx
        .store
        .insert_decision(decision, reasoning, alternatives, outcome, "agent")
        .await?;
    // Also mirror into searchable memories so hybrid recall surfaces it.
    let _ = ctx
        .store
        .insert_memory(&NewMemory {
            kind: "decision".into(),
            title: decision.chars().take(80).collect(),
            content: format!(
                "Decision: {decision}\nReasoning: {reasoning}\nAlternatives: {alternatives}\nOutcome: {outcome}"
            ),
            source: "decision-ledger".into(),
            tags: vec!["decision".into()],
            importance: 0.75,
            trust_tier: Some("auto".into()),
        })
        .await;
    Ok(format!("logged decision {id}"))
}

pub async fn list_decisions(ctx: &ToolContext, args: &Value) -> Result<String> {
    let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
    let rows = ctx.store.list_decisions(limit).await?;
    if rows.is_empty() {
        return Ok("(no decisions logged yet)".into());
    }
    let mut out = String::new();
    for (i, d) in rows.iter().enumerate() {
        out.push_str(&format!(
            "[#{} id={} @ {}]\n{}\n  reasoning: {}\n  outcome: {}\n\n",
            i + 1,
            &d.id[..8.min(d.id.len())],
            crate::util::format_ts(d.created_at),
            d.decision,
            crate::util::truncate(&d.reasoning, 200),
            crate::util::truncate(&d.outcome, 120),
        ));
    }
    Ok(out)
}

/// What the agent did recently — audit log (when/how tool calls) plus
/// recent auto-persist / harness LTM notes. Use when the user asks
/// "what did you do?", "how did we fix X?", or similar.
pub async fn list_recent_actions(ctx: &ToolContext, args: &Value) -> Result<String> {
    let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(20) as usize;
    let limit = limit.clamp(1, 100);
    let mut out = String::new();

    let audit_path = ctx.audit.path();
    match ctx.audit.read_tail(limit) {
        Ok(raw) if !raw.trim().is_empty() => {
            out.push_str("## Recent tool calls (audit log)\n");
            for line in raw.lines() {
                if let Ok(v) = serde_json::from_str::<Value>(line) {
                    let ts = v.get("ts").and_then(|t| t.as_i64()).unwrap_or(0);
                    let tool = v.get("tool").and_then(|t| t.as_str()).unwrap_or("?");
                    let decision = v.get("decision").and_then(|t| t.as_str()).unwrap_or("?");
                    let summary = v
                        .get("result_summary")
                        .and_then(|t| t.as_str())
                        .unwrap_or("");
                    let args_s = v
                        .get("args")
                        .map(|a| crate::util::truncate(&a.to_string(), 160))
                        .unwrap_or_default();
                    out.push_str(&format!(
                        "- [{}] {tool} ({decision}) args={args_s} → {}\n",
                        crate::util::format_ts(ts),
                        crate::util::truncate(summary, 160)
                    ));
                } else {
                    out.push_str(&format!("- {line}\n"));
                }
            }
            out.push('\n');
        }
        _ => out.push_str("## Recent tool calls (audit log)\n(no audit entries yet)\n\n"),
    }

    match ctx.store.list_recent_work(limit.min(15)).await {
        Ok(rows) if !rows.is_empty() => {
            out.push_str("## Recent work notes (long-term memory)\n");
            for m in rows {
                out.push_str(&format!(
                    "- [{} @ {} | {}] {}: {}\n",
                    m.kind,
                    crate::util::format_ts(m.created_at),
                    m.source,
                    m.title,
                    crate::util::truncate(m.content.trim(), 280)
                ));
            }
        }
        _ => out.push_str("## Recent work notes (long-term memory)\n(none yet)\n"),
    }
    Ok(out)
}

pub async fn kg_link(ctx: &ToolContext, args: &Value) -> Result<String> {
    let s_n = args
        .get("src_name")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("src_name"))?;
    let s_t = args
        .get("src_type")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("src_type"))?;
    let d_n = args
        .get("dst_name")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("dst_name"))?;
    let d_t = args
        .get("dst_type")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("dst_type"))?;
    let r = args
        .get("relation")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("relation"))?;
    ctx.store.upsert_edge(s_n, s_t, d_n, d_t, r).await?;
    Ok(format!("linked {s_n}:{s_t} --{r}--> {d_n}:{d_t}"))
}
