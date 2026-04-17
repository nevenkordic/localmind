//! Cross-platform networking tools — native-Rust replacements for
//! `nc -zv`, `whois`, `curl`/`wget`, and `ss`/`netstat`, so the agent
//! behaves identically on Linux, macOS, and Windows.
//!
//! - `port_check`       — TCP connect reachability test
//! - `whois`            — WHOIS via TCP:43 with IANA referral chain
//! - `http_fetch`       — HTTP(S) GET/HEAD/POST with size cap and timeout
//! - `listening_ports`  — local TCP/UDP socket inventory
//! - `dns_lookup`       — A/AAAA/name resolution via the OS resolver

use crate::tools::registry::ToolContext;
use anyhow::{anyhow, Context, Result};
use serde_json::{json, Value};
use std::net::{SocketAddr, ToSocketAddrs};
use std::time::{Duration, Instant};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

// ---------------------------------------------------------------------------
// port_check
// ---------------------------------------------------------------------------

pub async fn port_check(_ctx: &ToolContext, args: &Value) -> Result<String> {
    let host = args
        .get("host")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("missing host"))?;
    let port = args
        .get("port")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| anyhow!("missing port"))? as u16;
    let timeout_ms = args
        .get("timeout_ms")
        .and_then(|v| v.as_u64())
        .unwrap_or(3000);

    let addr_string = format!("{host}:{port}");
    // DNS resolution is blocking — do it on a thread.
    let addrs: Vec<SocketAddr> = tokio::task::spawn_blocking(move || {
        addr_string
            .to_socket_addrs()
            .map(|it| it.collect::<Vec<_>>())
    })
    .await
    .context("resolver task panicked")?
    .context("DNS resolution failed")?;

    if addrs.is_empty() {
        return Ok(json!({
            "reachable": false,
            "error": "no addresses resolved",
        })
        .to_string());
    }

    let mut last_err: Option<String> = None;
    for a in &addrs {
        let started = Instant::now();
        match tokio::time::timeout(Duration::from_millis(timeout_ms), TcpStream::connect(a)).await {
            Ok(Ok(_)) => {
                return Ok(json!({
                    "reachable": true,
                    "host": host,
                    "port": port,
                    "resolved": a.to_string(),
                    "elapsed_ms": started.elapsed().as_millis() as u64,
                })
                .to_string());
            }
            Ok(Err(e)) => last_err = Some(format!("{}: {e}", a)),
            Err(_) => last_err = Some(format!("{}: timed out after {timeout_ms} ms", a)),
        }
    }
    Ok(json!({
        "reachable": false,
        "host": host,
        "port": port,
        "tried": addrs.iter().map(|a| a.to_string()).collect::<Vec<_>>(),
        "error": last_err.unwrap_or_else(|| "unknown".into()),
    })
    .to_string())
}

// ---------------------------------------------------------------------------
// whois
// ---------------------------------------------------------------------------

const WHOIS_IANA: &str = "whois.iana.org";
const WHOIS_PORT: u16 = 43;

pub async fn whois(ctx: &ToolContext, args: &Value) -> Result<String> {
    let query = args
        .get("query")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("missing query"))?;
    let max_hops = args.get("max_hops").and_then(|v| v.as_u64()).unwrap_or(3) as usize;
    let timeout_secs = args
        .get("timeout_secs")
        .and_then(|v| v.as_u64())
        .unwrap_or(10);

    let mut server = args
        .get("server")
        .and_then(|v| v.as_str())
        .unwrap_or(WHOIS_IANA)
        .to_string();
    let mut chain: Vec<(String, String)> = Vec::new();
    let block_private = ctx.cfg.web.block_private_addrs;

    for _ in 0..=max_hops {
        // Validate the server before each hop so a malicious referral can't
        // redirect us to an internal WHOIS-shaped service.
        if block_private {
            // Strip optional :port for the host check.
            let host_only = server.split(':').next().unwrap_or(&server);
            crate::net_safety::ensure_public_host(host_only).await?;
        }
        let resp = query_whois(&server, query, timeout_secs)
            .await
            .with_context(|| format!("querying {server}"))?;
        chain.push((server.clone(), resp.clone()));

        match parse_whois_referral(&resp, &server) {
            Some(s) => server = s,
            None => break,
        }
    }

    // Return the LAST response (the authoritative one), prefixed with the referral trail.
    let mut out = String::new();
    if chain.len() > 1 {
        out.push_str("-- referral chain --\n");
        for (i, (s, _)) in chain.iter().enumerate() {
            out.push_str(&format!("  {}. {s}\n", i + 1));
        }
        out.push('\n');
    }
    if let Some((s, r)) = chain.last() {
        out.push_str(&format!("-- response from {s} --\n"));
        out.push_str(r);
    }
    Ok(out)
}

/// Extract the next WHOIS server from a response body, following the common
/// referral conventions used across RIR registries. Returns None when there's
/// no referral (or the referral points back to the same server).
fn parse_whois_referral(body: &str, current_server: &str) -> Option<String> {
    body.lines().find_map(|l| {
        let t = l.trim();
        for prefix in ["refer:", "whois:", "ReferralServer:"] {
            if let Some(rest) = t.strip_prefix(prefix) {
                let v = rest.trim();
                // Some responses use "whois://<host>" — strip the scheme.
                let v = v.strip_prefix("whois://").unwrap_or(v);
                if !v.is_empty() && v != current_server {
                    return Some(v.to_string());
                }
            }
        }
        None
    })
}

async fn query_whois(server: &str, query: &str, timeout_secs: u64) -> Result<String> {
    let addr = format!("{server}:{WHOIS_PORT}");
    let mut stream =
        tokio::time::timeout(Duration::from_secs(timeout_secs), TcpStream::connect(&addr))
            .await
            .map_err(|_| anyhow!("connect timeout"))?
            .with_context(|| format!("connecting to {addr}"))?;

    let q = format!("{query}\r\n");
    stream.write_all(q.as_bytes()).await?;
    stream.shutdown().await.ok();

    let mut buf = Vec::new();
    tokio::time::timeout(
        Duration::from_secs(timeout_secs),
        stream.read_to_end(&mut buf),
    )
    .await
    .map_err(|_| anyhow!("read timeout"))??;
    Ok(String::from_utf8_lossy(&buf).to_string())
}

// ---------------------------------------------------------------------------
// http_fetch
// ---------------------------------------------------------------------------

pub async fn http_fetch(ctx: &ToolContext, args: &Value) -> Result<String> {
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
    let method = args
        .get("method")
        .and_then(|v| v.as_str())
        .unwrap_or("GET")
        .to_uppercase();
    let timeout_secs = args
        .get("timeout_secs")
        .and_then(|v| v.as_u64())
        .unwrap_or(30);
    let max_bytes = args
        .get("max_bytes")
        .and_then(|v| v.as_u64())
        .unwrap_or(1_000_000) as usize;
    let body = args.get("body").and_then(|v| v.as_str());

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(timeout_secs))
        .user_agent(format!(
            "localmind/{} (work-tool)",
            env!("CARGO_PKG_VERSION")
        ))
        .build()?;

    let mut req = match method.as_str() {
        "GET" => client.get(url),
        "HEAD" => client.head(url),
        "POST" => client.post(url),
        "PUT" => client.put(url),
        "DELETE" => client.delete(url),
        other => return Err(anyhow!("unsupported method: {other}")),
    };

    if let Some(hdrs) = args.get("headers").and_then(|v| v.as_object()) {
        for (k, v) in hdrs {
            if let Some(vs) = v.as_str() {
                req = req.header(k, vs);
            }
        }
    }
    if let Some(b) = body {
        req = req.body(b.to_string());
    }

    let resp = req.send().await.context("http request failed")?;
    let status = resp.status();
    let final_url = resp.url().clone();
    let mut hdrs = serde_json::Map::new();
    for (k, v) in resp.headers() {
        hdrs.insert(
            k.to_string(),
            Value::String(v.to_str().unwrap_or("").to_string()),
        );
    }

    let bytes = resp.bytes().await.context("reading response body")?;
    let truncated = bytes.len() > max_bytes;
    let slice = &bytes[..bytes.len().min(max_bytes)];
    let body_text = match std::str::from_utf8(slice) {
        Ok(s) => s.to_string(),
        Err(_) => format!("(binary {} bytes, not shown)", slice.len()),
    };

    Ok(json!({
        "url": final_url.to_string(),
        "status": status.as_u16(),
        "status_text": status.canonical_reason().unwrap_or(""),
        "headers": Value::Object(hdrs),
        "bytes": bytes.len(),
        "truncated": truncated,
        "body": body_text,
    })
    .to_string())
}

// ---------------------------------------------------------------------------
// listening_ports
// ---------------------------------------------------------------------------

pub async fn listening_ports(_ctx: &ToolContext, args: &Value) -> Result<String> {
    let protocol = args
        .get("protocol")
        .and_then(|v| v.as_str())
        .unwrap_or("tcp")
        .to_lowercase();
    let include_udp = protocol == "udp" || protocol == "both";
    let include_tcp = protocol == "tcp" || protocol == "both";

    let out = tokio::task::spawn_blocking(move || -> Result<Value> {
        use netstat2::{get_sockets_info, AddressFamilyFlags, ProtocolFlags, ProtocolSocketInfo};

        let af = AddressFamilyFlags::IPV4 | AddressFamilyFlags::IPV6;
        let mut pf = ProtocolFlags::empty();
        if include_tcp {
            pf |= ProtocolFlags::TCP;
        }
        if include_udp {
            pf |= ProtocolFlags::UDP;
        }

        let infos = get_sockets_info(af, pf).map_err(|e| anyhow!("socket query failed: {e}"))?;
        let mut rows = Vec::new();
        for si in infos {
            match si.protocol_socket_info {
                ProtocolSocketInfo::Tcp(t) => {
                    // Only report listeners by default; add remote-connected
                    // entries later if the user wants them.
                    if format!("{:?}", t.state) == "Listen" {
                        rows.push(json!({
                            "proto": "tcp",
                            "local": format!("{}:{}", t.local_addr, t.local_port),
                            "state": format!("{:?}", t.state),
                            "pids": si.associated_pids,
                        }));
                    }
                }
                ProtocolSocketInfo::Udp(u) => {
                    rows.push(json!({
                        "proto": "udp",
                        "local": format!("{}:{}", u.local_addr, u.local_port),
                        "pids": si.associated_pids,
                    }));
                }
            }
        }
        Ok(Value::Array(rows))
    })
    .await
    .context("netstat2 task panicked")??;

    Ok(out.to_string())
}

// ---------------------------------------------------------------------------
// dns_lookup
// ---------------------------------------------------------------------------

pub async fn dns_lookup(_ctx: &ToolContext, args: &Value) -> Result<String> {
    let name = args
        .get("name")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("missing name"))?;
    let port = args.get("port").and_then(|v| v.as_u64()).unwrap_or(0) as u16;
    let name_owned = name.to_string();

    let addrs = tokio::task::spawn_blocking(move || {
        format!("{name_owned}:{port}")
            .to_socket_addrs()
            .map(|it| it.map(|a| a.ip().to_string()).collect::<Vec<_>>())
    })
    .await
    .context("resolver task panicked")?
    .context("DNS resolution failed")?;

    let mut v4 = Vec::new();
    let mut v6 = Vec::new();
    for a in addrs {
        if a.contains(':') {
            v6.push(a);
        } else {
            v4.push(a);
        }
    }
    v4.sort();
    v4.dedup();
    v6.sort();
    v6.dedup();

    Ok(json!({
        "name": name,
        "a": v4,
        "aaaa": v6,
    })
    .to_string())
}

// ---------------------------------------------------------------------------
// Tests — parser-only, no network, no ctx. The wrapper fns above are thin
// shells around std / tokio / reqwest / netstat2 and are exercised by the
// offline smoke tests in the README.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::parse_whois_referral;

    #[test]
    fn referral_refer_line() {
        let body = "% IANA WHOIS server\n\nrefer:        whois.verisign-grs.com\n";
        assert_eq!(
            parse_whois_referral(body, "whois.iana.org").as_deref(),
            Some("whois.verisign-grs.com")
        );
    }

    #[test]
    fn referral_whois_scheme_stripped() {
        let body = "whois:        whois://whois.arin.net\n";
        assert_eq!(
            parse_whois_referral(body, "whois.iana.org").as_deref(),
            Some("whois.arin.net")
        );
    }

    #[test]
    fn referral_server_variant() {
        let body = "ReferralServer: whois://rwhois.example.net:4321\n";
        assert_eq!(
            parse_whois_referral(body, "whois.arin.net").as_deref(),
            Some("rwhois.example.net:4321")
        );
    }

    #[test]
    fn referral_none_when_absent() {
        let body = "domain: example.com\nregistrar: Example Registrar\n";
        assert_eq!(parse_whois_referral(body, "whois.iana.org"), None);
    }

    #[test]
    fn referral_ignores_self_reference() {
        let body = "refer: whois.iana.org\n";
        assert_eq!(parse_whois_referral(body, "whois.iana.org"), None);
    }
}
