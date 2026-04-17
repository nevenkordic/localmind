//! PDF text extraction via the pdf-extract crate.

use crate::tools::registry::{resolve_path, ToolContext};
use anyhow::{anyhow, Result};
use serde_json::Value;

pub async fn read_pdf(ctx: &ToolContext, args: &Value) -> Result<String> {
    let path = args
        .get("path")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("missing path"))?;
    let resolved = resolve_path(ctx, path);
    let text = tokio::task::spawn_blocking(move || -> Result<String> {
        let bytes = std::fs::read(&resolved)?;
        Ok(pdf_extract::extract_text_from_mem(&bytes).map_err(|e| anyhow!("pdf extract: {e}"))?)
    })
    .await??;
    Ok(text)
}
