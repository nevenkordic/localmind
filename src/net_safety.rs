//! Outbound host validator. Used by `web_fetch`, `http_fetch`, and the WHOIS
//! referral chain to refuse connections to private, loopback, link-local
//! (incl. cloud-metadata IPs at 169.254.169.254), and multicast addresses.
//!
//! The check happens AFTER DNS resolution so a hostname like
//! `metadata.google.internal` (which resolves to 169.254.169.254) is also
//! blocked. Without this, prompt injection could exfiltrate cloud
//! credentials by asking the agent to fetch the metadata endpoint.

use anyhow::{anyhow, Result};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, ToSocketAddrs};

/// Returns true when the address is one we will not connect to without an
/// explicit opt-out. Private / loopback / link-local / multicast / reserved.
pub fn is_blocked(addr: &IpAddr) -> bool {
    match addr {
        IpAddr::V4(v4) => is_blocked_v4(v4),
        IpAddr::V6(v6) => is_blocked_v6(v6),
    }
}

fn is_blocked_v4(a: &Ipv4Addr) -> bool {
    if a.is_loopback()
        || a.is_private()
        || a.is_link_local()
        || a.is_multicast()
        || a.is_unspecified()
        || a.is_broadcast()
    {
        return true;
    }
    let oct = a.octets();
    // Carrier-grade NAT — 100.64.0.0/10
    if oct[0] == 100 && (oct[1] & 0b1100_0000) == 64 {
        return true;
    }
    // Reserved future-use — 240.0.0.0/4 (also catches 255/8 broadcast).
    if oct[0] >= 240 {
        return true;
    }
    false
}

fn is_blocked_v6(a: &Ipv6Addr) -> bool {
    if a.is_loopback() || a.is_unspecified() || a.is_multicast() {
        return true;
    }
    let segs = a.segments();
    // Link-local fe80::/10
    if (segs[0] & 0xffc0) == 0xfe80 {
        return true;
    }
    // Unique local addresses fc00::/7
    if (segs[0] & 0xfe00) == 0xfc00 {
        return true;
    }
    // IPv4-mapped (::ffff:0:0/96) — apply the IPv4 rules to the embedded address.
    // We can't use to_ipv4_mapped (unstable as of MSRV); inline the check.
    if segs[0] == 0
        && segs[1] == 0
        && segs[2] == 0
        && segs[3] == 0
        && segs[4] == 0
        && segs[5] == 0xffff
    {
        let v4 = Ipv4Addr::new(
            (segs[6] >> 8) as u8,
            (segs[6] & 0xff) as u8,
            (segs[7] >> 8) as u8,
            (segs[7] & 0xff) as u8,
        );
        return is_blocked_v4(&v4);
    }
    false
}

/// Resolve `host` (may be a literal IP or a hostname) and refuse if any
/// resolved address is blocked. Runs the DNS lookup on a blocking thread so
/// the calling tokio task isn't pinned during slow resolvers.
pub async fn ensure_public_host(host: &str) -> Result<()> {
    let host = host.to_string();
    let host_for_closure = host.clone();
    let addrs: Vec<IpAddr> = tokio::task::spawn_blocking(move || -> Result<Vec<IpAddr>> {
        // `to_socket_addrs` requires a port — port 0 is fine, we only want IPs.
        let addrs = format!("{host_for_closure}:0")
            .to_socket_addrs()
            .map_err(|e| anyhow!("DNS resolution for '{host_for_closure}' failed: {e}"))?
            .map(|sa| sa.ip())
            .collect::<Vec<_>>();
        Ok(addrs)
    })
    .await
    .map_err(|e| anyhow!("resolver task panicked: {e}"))??;

    if addrs.is_empty() {
        return Err(anyhow!("host '{host}' did not resolve to any address"));
    }
    for a in &addrs {
        if is_blocked(a) {
            return Err(anyhow!(
                "refusing to connect to '{host}' — resolves to {a}, \
                 which is private/loopback/metadata. Set [web].block_private_addrs = false \
                 to override (use only on trusted networks)."
            ));
        }
    }
    Ok(())
}

/// Convenience wrapper — extract host from a URL string and validate.
pub async fn ensure_public_url(url: &str) -> Result<()> {
    let host = extract_host(url).ok_or_else(|| anyhow!("could not parse host from URL: {url}"))?;
    ensure_public_host(host).await
}

/// Cheap URL → host extractor. Handles user-info (`user:pass@`), explicit
/// port (`:8080`), and IPv6 bracket notation (`[::1]`). Returns the bare host.
pub(crate) fn extract_host(url: &str) -> Option<&str> {
    let after_scheme = url.split_once("://")?.1;
    let authority = after_scheme
        .split('/')
        .next()?
        .split('?')
        .next()?
        .split('#')
        .next()?;
    // Strip optional user:pass@ prefix.
    let host_port = authority
        .rsplit_once('@')
        .map(|(_, h)| h)
        .unwrap_or(authority);
    // IPv6 literals: [::1]:8080 → ::1
    if let Some(rest) = host_port.strip_prefix('[') {
        return rest.split_once(']').map(|(h, _)| h);
    }
    Some(host_port.split(':').next().unwrap_or(host_port))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ip(s: &str) -> IpAddr {
        s.parse().unwrap()
    }

    #[test]
    fn blocks_v4_loopback() {
        assert!(is_blocked(&ip("127.0.0.1")));
        assert!(is_blocked(&ip("127.255.255.254")));
    }

    #[test]
    fn blocks_v4_private() {
        for s in ["10.0.0.1", "172.16.0.5", "192.168.1.1"] {
            assert!(is_blocked(&ip(s)), "expected blocked: {s}");
        }
    }

    #[test]
    fn blocks_aws_metadata() {
        // 169.254.169.254 — used by AWS / GCP / Azure for instance metadata.
        assert!(is_blocked(&ip("169.254.169.254")));
    }

    #[test]
    fn blocks_v4_carrier_grade_nat() {
        assert!(is_blocked(&ip("100.64.0.1")));
        assert!(is_blocked(&ip("100.127.255.255")));
        // 100.128.0.0 is OUTSIDE the CGN range — must NOT be blocked.
        assert!(!is_blocked(&ip("100.128.0.0")));
    }

    #[test]
    fn allows_public_v4() {
        for s in ["1.1.1.1", "8.8.8.8", "151.101.1.69"] {
            assert!(!is_blocked(&ip(s)), "expected allowed: {s}");
        }
    }

    #[test]
    fn blocks_v6_loopback_and_link_local() {
        assert!(is_blocked(&ip("::1")));
        assert!(is_blocked(&ip("fe80::1")));
        assert!(is_blocked(&ip("fc00::1"))); // ULA
    }

    #[test]
    fn blocks_v6_mapped_v4_loopback() {
        // ::ffff:127.0.0.1 — the IPv4-mapped form of localhost. Must block.
        assert!(is_blocked(&ip("::ffff:127.0.0.1")));
        // ::ffff:169.254.169.254 — IPv4-mapped metadata. Must block.
        assert!(is_blocked(&ip("::ffff:169.254.169.254")));
    }

    #[test]
    fn allows_public_v6() {
        // 2606:4700:4700::1111 — Cloudflare DNS. Must allow.
        assert!(!is_blocked(&ip("2606:4700:4700::1111")));
    }

    #[test]
    fn extract_host_handles_basic_url() {
        assert_eq!(extract_host("http://example.com/path"), Some("example.com"));
        assert_eq!(
            extract_host("https://example.com:8080/x?y=1"),
            Some("example.com")
        );
    }

    #[test]
    fn extract_host_handles_userinfo() {
        assert_eq!(
            extract_host("http://user:pass@example.com/p"),
            Some("example.com")
        );
    }

    #[test]
    fn extract_host_handles_ipv6() {
        assert_eq!(extract_host("http://[::1]:8080/"), Some("::1"));
        assert_eq!(
            extract_host("https://[2606:4700::1]/"),
            Some("2606:4700::1")
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn ensure_public_host_blocks_localhost() {
        let r = ensure_public_host("127.0.0.1").await;
        assert!(r.is_err(), "expected localhost to be blocked");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn ensure_public_host_blocks_metadata_ip() {
        let r = ensure_public_host("169.254.169.254").await;
        assert!(r.is_err(), "expected metadata IP to be blocked");
    }
}
