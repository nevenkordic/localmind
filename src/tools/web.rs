//! Brave Search + simple fetch. Outbound network ONLY to api.search.brave.com
//! (for search) and the URL the user has approved for fetch. Honours
//! HTTPS_PROXY/HTTP_PROXY env vars so traffic can flow through your
//! corporate egress if required.

use crate::tools::registry::ToolContext;
use anyhow::{anyhow, Result};
use reqwest::Client;
use serde::Deserialize;
use serde_json::Value;
use std::time::Duration;

fn http_client() -> Client {
    Client::builder()
        .timeout(Duration::from_secs(20))
        .user_agent(format!(
            "localmind/{} (work-tool)",
            env!("CARGO_PKG_VERSION")
        ))
        .build()
        .expect("reqwest")
}

pub async fn web_search(ctx: &ToolContext, args: &Value) -> Result<String> {
    if ctx.cfg.web.brave_api_key.is_empty() {
        return Err(anyhow!("no brave_api_key configured"));
    }
    let query = args
        .get("query")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("missing query"))?;
    let count = args
        .get("count")
        .and_then(|v| v.as_u64())
        .unwrap_or(ctx.cfg.web.max_results as u64)
        .min(20);

    #[derive(Deserialize)]
    struct Resp {
        web: Option<WebBlock>,
    }
    #[derive(Deserialize)]
    struct WebBlock {
        results: Vec<Hit>,
    }
    #[derive(Deserialize)]
    struct Hit {
        title: String,
        url: String,
        description: Option<String>,
    }

    let resp: Resp = http_client()
        .get(&ctx.cfg.web.brave_endpoint)
        .query(&[("q", query), ("count", &count.to_string())])
        .header("X-Subscription-Token", &ctx.cfg.web.brave_api_key)
        .header("Accept", "application/json")
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;

    let mut out = String::new();
    if let Some(w) = resp.web {
        for (i, h) in w.results.iter().enumerate() {
            out.push_str(&format!(
                "{}. {}\n   {}\n   {}\n\n",
                i + 1,
                h.title,
                h.url,
                h.description.as_deref().unwrap_or("")
            ));
        }
    }
    if out.is_empty() {
        out.push_str("(no results)");
    }
    Ok(out)
}

pub async fn web_fetch(ctx: &ToolContext, args: &Value) -> Result<String> {
    let url = args
        .get("url")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("missing url"))?;
    if !url.starts_with("http://") && !url.starts_with("https://") {
        return Err(anyhow!("only http(s) URLs allowed"));
    }
    if ctx.cfg.web.block_private_addrs {
        crate::net_safety::ensure_public_url(url).await?;
    }
    let resp = http_client().get(url).send().await?;
    let status = resp.status();
    // Cap input to strip_html — adversarial pages can be tens of MB and the
    // tag-stripper is O(n²) in the worst case.
    const MAX_INPUT: usize = 1_000_000;
    let bytes = resp.bytes().await.unwrap_or_default();
    let trimmed = &bytes[..bytes.len().min(MAX_INPUT)];
    let text = String::from_utf8_lossy(trimmed);
    let stripped = strip_html(&text);
    Ok(format!(
        "HTTP {status}\n\n{}",
        crate::util::truncate(&stripped, 40_000)
    ))
}

fn strip_html(html: &str) -> String {
    // Cheap HTML-to-text: drop <script>/<style> blocks, then strip tags.
    let mut s = html.to_string();
    for tag in ["script", "style", "noscript"] {
        let open = format!("<{tag}");
        let close = format!("</{tag}>");
        while let (Some(i), Some(j)) = (s.to_lowercase().find(&open), s.to_lowercase().find(&close))
        {
            if j > i {
                let end = j + close.len();
                s.replace_range(i..end, " ");
            } else {
                break;
            }
        }
    }
    // Strip remaining tags.
    let re = regex::Regex::new(r"<[^>]+>").unwrap();
    let no_tags = re.replace_all(&s, " ");
    // Collapse whitespace.
    let ws = regex::Regex::new(r"\s+").unwrap();
    ws.replace_all(&no_tags, " ").trim().to_string()
}
