//! Image "reading" — send the image to Ollama's vision model for description.

use crate::llm::{ollama::OllamaClient as Ollama, types::ChatMessage};
use crate::tools::registry::{resolve_path, ToolContext};
use anyhow::{anyhow, Result};
use base64::Engine;
use serde_json::Value;

pub async fn read_image(ctx: &ToolContext, args: &Value) -> Result<String> {
    let path = args
        .get("path")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("missing path"))?;
    let user_prompt = args.get("prompt").and_then(|v| v.as_str()).unwrap_or(
        "Describe this image in detail. Transcribe any visible text. \
         Call out diagrams, UI screenshots, charts, or code snippets.",
    );
    let resolved = resolve_path(ctx, path);
    let bytes = tokio::task::spawn_blocking(move || std::fs::read(&resolved)).await??;
    let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
    let client = Ollama::new(&ctx.cfg.ollama);
    let msgs = vec![
        ChatMessage::system(
            "You are a careful visual analyst. Be concrete and faithful to the image.",
        ),
        ChatMessage::user_with_images(user_prompt, vec![b64]),
    ];
    let reply = client.chat(&msgs, None, true).await?;
    Ok(reply.content)
}
