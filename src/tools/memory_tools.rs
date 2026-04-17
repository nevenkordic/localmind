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
        .unwrap_or(0.5) as f32;
    let id = ctx
        .store
        .insert_memory(&NewMemory {
            kind,
            title,
            content,
            source: "agent".into(),
            tags,
            importance,
        })
        .await?;
    Ok(format!("stored memory {id}"))
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
