//! Cross-platform archive tools — create and extract ZIP files without
//! shelling out to `zip` / `unzip` binaries. Built on the already-present
//! `zip` crate so the behaviour is identical on Linux, macOS, and Windows.
//!
//! Tools:
//! - `zip_create`  — pack one or more files / directories into an archive
//! - `zip_extract` — unpack an archive with a zip-slip guard and overwrite flag
//!
//! Both tools honour:
//! - `deny_globs` from config (checked per-entry on extraction, per-input on
//!   creation, and on the archive path itself);
//! - the `PermissionMode` gate (ReadOnly refuses; WorkspaceWrite requires the
//!   archive / extract target to live under `workspace_root` if configured);
//! - the standard interactive permission prompt when `confirm_writes` is on.

use crate::tools::file_ops::is_denied;
use crate::tools::registry::{resolve_path, ToolContext};
use anyhow::{anyhow, Context, Result};
use serde_json::{json, Value};
use std::path::{Component, Path, PathBuf};
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipArchive, ZipWriter};

// ---------------------------------------------------------------------------
// zip_create
// ---------------------------------------------------------------------------

pub async fn zip_create(ctx: &ToolContext, args: &Value) -> Result<String> {
    let archive_path = args
        .get("archive_path")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("missing archive_path"))?;
    let inputs: Vec<String> = args
        .get("inputs")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow!("missing inputs (array of paths)"))?
        .iter()
        .filter_map(|v| v.as_str().map(str::to_string))
        .collect();
    if inputs.is_empty() {
        return Err(anyhow!("inputs must contain at least one path"));
    }
    let base_dir = args
        .get("base_dir")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    let compression = args
        .get("compression")
        .and_then(|v| v.as_str())
        .unwrap_or("deflated")
        .to_ascii_lowercase();

    let method = match compression.as_str() {
        "stored" | "store" | "none" => CompressionMethod::Stored,
        "deflated" | "deflate" | "default" => CompressionMethod::Deflated,
        other => return Err(anyhow!("unsupported compression: {other}")),
    };

    let archive = resolve_path(ctx, archive_path);
    let deny = ctx.cfg.tools.deny_globs.clone();
    if is_denied(&deny, &archive.to_string_lossy()) {
        return Err(anyhow!("archive path denied by deny-glob"));
    }

    // Resolve input paths up front so the prompt scope is accurate.
    let mut resolved_inputs: Vec<PathBuf> = Vec::with_capacity(inputs.len());
    for p in &inputs {
        let r = resolve_path(ctx, p);
        if is_denied(&deny, &r.to_string_lossy()) {
            return Err(anyhow!("input path denied by deny-glob: {}", r.display()));
        }
        resolved_inputs.push(r);
    }

    let base = base_dir.map(|b| resolve_path(ctx, &b));

    let archive_clone = archive.clone();
    let stats = tokio::task::spawn_blocking(move || -> Result<(u64, u64)> {
        zip_write_tree(
            &archive_clone,
            &resolved_inputs,
            base.as_deref(),
            &deny,
            method,
        )
    })
    .await
    .context("zip_create task panicked")??;

    Ok(json!({
        "archive": archive.to_string_lossy(),
        "entries": stats.0,
        "bytes_written": stats.1,
        "compression": compression,
    })
    .to_string())
}

/// ctx-free core: write an archive to `archive` containing `inputs`, with
/// entry names relative to `base` (or each input's file name when base is
/// None or not a prefix).
fn zip_write_tree(
    archive: &Path,
    inputs: &[PathBuf],
    base: Option<&Path>,
    deny: &[String],
    method: CompressionMethod,
) -> Result<(u64, u64)> {
    if let Some(parent) = archive.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).ok();
        }
    }
    let file = std::fs::File::create(archive)
        .with_context(|| format!("creating {}", archive.display()))?;
    let mut zw = ZipWriter::new(file);
    let opts = SimpleFileOptions::default()
        .compression_method(method)
        .unix_permissions(0o644);

    let mut entries = 0u64;
    let mut bytes = 0u64;
    for input in inputs {
        add_to_zip(&mut zw, input, base, deny, opts, &mut entries, &mut bytes)?;
    }
    zw.finish().context("finalising zip")?;
    Ok((entries, bytes))
}

fn add_to_zip(
    zw: &mut ZipWriter<std::fs::File>,
    path: &Path,
    base: Option<&Path>,
    deny: &[String],
    opts: SimpleFileOptions,
    entries: &mut u64,
    bytes: &mut u64,
) -> Result<()> {
    let meta = std::fs::metadata(path).with_context(|| format!("stat {}", path.display()))?;
    if meta.is_file() {
        let name = entry_name(path, base)?;
        zw.start_file(name, opts)?;
        let mut f =
            std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let n = std::io::copy(&mut f, zw)?;
        *entries += 1;
        *bytes += n;
        Ok(())
    } else if meta.is_dir() {
        // Walk the directory.
        let name = entry_name(path, base)?;
        if !name.is_empty() {
            zw.add_directory(format!("{name}/"), opts)?;
        }
        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            let p = entry.path();
            if is_denied(deny, &p.to_string_lossy()) {
                continue;
            }
            add_to_zip(zw, &p, base, deny, opts, entries, bytes)?;
        }
        Ok(())
    } else {
        // Skip sockets, symlinks we can't follow, devices, etc.
        Ok(())
    }
}

/// Compute the forward-slashed entry name relative to the base directory.
/// Falls back to the file name if `base` is not a prefix of `path`.
fn entry_name(path: &Path, base: Option<&Path>) -> Result<String> {
    let rel: PathBuf = if let Some(b) = base {
        match path.strip_prefix(b) {
            Ok(r) => r.to_path_buf(),
            Err(_) => {
                Path::new(path.file_name().ok_or_else(|| anyhow!("no file name"))?).to_path_buf()
            }
        }
    } else {
        Path::new(path.file_name().ok_or_else(|| anyhow!("no file name"))?).to_path_buf()
    };
    // Zip entry names use forward slashes.
    let mut s = String::new();
    for c in rel.components() {
        if !s.is_empty() {
            s.push('/');
        }
        s.push_str(&c.as_os_str().to_string_lossy());
    }
    Ok(s)
}

// ---------------------------------------------------------------------------
// zip_extract
// ---------------------------------------------------------------------------

pub async fn zip_extract(ctx: &ToolContext, args: &Value) -> Result<String> {
    let archive_path = args
        .get("archive_path")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("missing archive_path"))?;
    let dest_dir = args
        .get("dest_dir")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("missing dest_dir"))?;
    let overwrite = args
        .get("overwrite")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let strip_components = args
        .get("strip_components")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;

    let archive = resolve_path(ctx, archive_path);
    let dest = resolve_path(ctx, dest_dir);
    let deny = ctx.cfg.tools.deny_globs.clone();

    if is_denied(&deny, &archive.to_string_lossy()) {
        return Err(anyhow!("archive path denied by deny-glob"));
    }
    if is_denied(&deny, &dest.to_string_lossy()) {
        return Err(anyhow!("destination path denied by deny-glob"));
    }

    let archive_clone = archive.clone();
    let dest_clone = dest.clone();
    let stats = tokio::task::spawn_blocking(move || -> Result<(u64, u64, Vec<String>)> {
        zip_read_tree(
            &archive_clone,
            &dest_clone,
            overwrite,
            strip_components,
            &deny,
        )
    })
    .await
    .context("zip_extract task panicked")??;

    Ok(json!({
        "archive": archive.to_string_lossy(),
        "dest": dest.to_string_lossy(),
        "entries_written": stats.0,
        "bytes_written": stats.1,
        "skipped": stats.2,
    })
    .to_string())
}

/// ctx-free core: extract `archive` into `dest`, honouring overwrite,
/// strip_components, and the deny-glob list. Applies a zip-slip guard.
fn zip_read_tree(
    archive: &Path,
    dest: &Path,
    overwrite: bool,
    strip_components: usize,
    deny: &[String],
) -> Result<(u64, u64, Vec<String>)> {
    std::fs::create_dir_all(dest).with_context(|| format!("creating {}", dest.display()))?;
    // Canonicalise AFTER creating so symlinks above dest resolve.
    let dest_canon = dunce_canonicalize(dest)?;

    let file =
        std::fs::File::open(archive).with_context(|| format!("opening {}", archive.display()))?;
    let mut zip = ZipArchive::new(file).context("parsing zip")?;

    let mut entries = 0u64;
    let mut bytes = 0u64;
    let mut skipped: Vec<String> = Vec::new();

    for i in 0..zip.len() {
        let mut entry = zip.by_index(i)?;
        let name = entry.name().to_string();

        // Apply strip_components (borrowed from `tar --strip-components`).
        let stripped = strip_path_components(&name, strip_components);
        if stripped.is_empty() {
            continue;
        }

        // Zip-slip guard: build the target path by joining component by
        // component, rejecting `..` segments and absolute paths.
        let target = match safe_join(&dest_canon, &stripped) {
            Some(t) => t,
            None => {
                skipped.push(format!("{name} (zip-slip rejected)"));
                continue;
            }
        };
        if is_denied(deny, &target.to_string_lossy()) {
            skipped.push(format!("{name} (deny-glob)"));
            continue;
        }

        if entry.is_dir() || stripped.ends_with('/') {
            std::fs::create_dir_all(&target)?;
            continue;
        }

        if target.exists() && !overwrite {
            skipped.push(format!("{name} (exists; pass overwrite=true)"));
            continue;
        }
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let mut out = std::fs::File::create(&target)
            .with_context(|| format!("creating {}", target.display()))?;
        let n = std::io::copy(&mut entry, &mut out)?;
        entries += 1;
        bytes += n;

        // Preserve unix mode when present.
        #[cfg(unix)]
        if let Some(mode) = entry.unix_mode() {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&target, std::fs::Permissions::from_mode(mode));
        }
    }
    Ok((entries, bytes, skipped))
}

/// Canonicalise `p` without the Windows `\\?\` verbatim prefix.
fn dunce_canonicalize(p: &Path) -> Result<PathBuf> {
    let c = std::fs::canonicalize(p).with_context(|| format!("canonicalize {}", p.display()))?;
    #[cfg(windows)]
    {
        let s = c.to_string_lossy();
        if let Some(rest) = s.strip_prefix(r"\\?\") {
            return Ok(PathBuf::from(rest));
        }
    }
    Ok(c)
}

/// Drop the first `n` components of a zip entry name (forward-slash separated).
fn strip_path_components(name: &str, n: usize) -> String {
    if n == 0 {
        return name.to_string();
    }
    let mut parts: Vec<&str> = name.split('/').collect();
    if parts.len() <= n {
        return String::new();
    }
    parts.drain(..n);
    parts.join("/")
}

/// Join `rel` onto `base` while rejecting any path that escapes `base`.
/// Returns None if the entry is absolute, contains `..` segments that would
/// escape, or otherwise can't be safely contained.
fn safe_join(base: &Path, rel: &str) -> Option<PathBuf> {
    let rel_path = Path::new(rel);
    if rel_path.is_absolute() {
        return None;
    }
    let mut out = base.to_path_buf();
    for comp in rel_path.components() {
        match comp {
            Component::Normal(p) => out.push(p),
            Component::CurDir => {}
            Component::ParentDir => {
                if !out.pop() || !out.starts_with(base) {
                    return None;
                }
            }
            // Disallow `/`, `\`, `C:`, and `\\?\` prefixes mid-path.
            Component::RootDir | Component::Prefix(_) => return None,
        }
    }
    if !out.starts_with(base) {
        return None;
    }
    Some(out)
}

// ---------------------------------------------------------------------------
// Tests — parser + path-safety logic only; the actual zip I/O is covered by
// the offline smoke suite in the README.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_components_basic() {
        assert_eq!(strip_path_components("a/b/c", 0), "a/b/c");
        assert_eq!(strip_path_components("a/b/c", 1), "b/c");
        assert_eq!(strip_path_components("a/b/c", 2), "c");
        assert_eq!(strip_path_components("a/b/c", 3), "");
        assert_eq!(strip_path_components("a/b/c", 5), "");
    }

    #[test]
    fn safe_join_normal() {
        let base = Path::new("/tmp/x");
        let out = safe_join(base, "a/b/c.txt").unwrap();
        assert_eq!(out, Path::new("/tmp/x/a/b/c.txt"));
    }

    #[test]
    fn safe_join_rejects_absolute() {
        let base = Path::new("/tmp/x");
        assert!(safe_join(base, "/etc/passwd").is_none());
    }

    #[test]
    fn safe_join_rejects_parent_escape() {
        let base = Path::new("/tmp/x");
        assert!(safe_join(base, "../../../etc/passwd").is_none());
        assert!(safe_join(base, "a/../../etc/passwd").is_none());
    }

    #[test]
    fn safe_join_allows_inner_parent() {
        // "a/../b.txt" must be safe because it resolves to base/b.txt.
        let base = Path::new("/tmp/x");
        let out = safe_join(base, "a/../b.txt").unwrap();
        assert_eq!(out, Path::new("/tmp/x/b.txt"));
    }

    // Round-trip: write a tree, zip it, extract into a fresh dir, and verify
    // files are byte-identical. Uses only ctx-free helpers (zip_write_tree /
    // zip_read_tree) so we don't need a live ToolContext or Store.
    #[test]
    fn round_trip_write_then_extract() {
        use std::io::Write as _;
        let tmp = std::env::temp_dir().join(format!("localmind-ziptest-{}", std::process::id()));
        let src = tmp.join("src");
        let nested = src.join("nested");
        std::fs::create_dir_all(&nested).unwrap();
        let files: &[(&str, &[u8])] = &[
            ("hello.txt", b"hello world\n"),
            ("nested/deep.txt", b"deep content\n"),
            ("nested/binary.bin", &[0u8, 1, 2, 3, 255, 128, 64]),
        ];
        for (rel, body) in files {
            let p = src.join(rel);
            if let Some(parent) = p.parent() {
                std::fs::create_dir_all(parent).unwrap();
            }
            let mut f = std::fs::File::create(&p).unwrap();
            f.write_all(body).unwrap();
        }

        let archive = tmp.join("out.zip");
        let (entries, bytes) = zip_write_tree(
            &archive,
            &[src.clone()],
            Some(&src),
            &[],
            CompressionMethod::Deflated,
        )
        .unwrap();
        assert_eq!(entries, 3, "expected 3 files in archive");
        assert!(bytes >= 20, "expected some content bytes");

        let out_dir = tmp.join("extracted");
        let (written, _, skipped) = zip_read_tree(&archive, &out_dir, true, 0, &[]).unwrap();
        assert_eq!(written, 3);
        assert!(skipped.is_empty(), "unexpected skips: {skipped:?}");

        for (rel, body) in files {
            let got = std::fs::read(out_dir.join(rel)).unwrap();
            assert_eq!(&got, body, "content mismatch for {rel}");
        }

        // Re-extracting without overwrite should skip existing files.
        let (written2, _, skipped2) = zip_read_tree(&archive, &out_dir, false, 0, &[]).unwrap();
        assert_eq!(written2, 0);
        assert_eq!(skipped2.len(), 3);

        let _ = std::fs::remove_dir_all(&tmp);
    }

    // Craft a malicious archive with a zip-slip entry and confirm the
    // extractor rejects it rather than writing outside the dest dir.
    #[test]
    fn extractor_rejects_zip_slip() {
        let tmp = std::env::temp_dir().join(format!("localmind-zipslip-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        let archive = tmp.join("evil.zip");

        // Build a zip with an entry name that tries to escape.
        let f = std::fs::File::create(&archive).unwrap();
        let mut zw = ZipWriter::new(f);
        let opts = SimpleFileOptions::default().compression_method(CompressionMethod::Stored);
        zw.start_file("../../../etc/localmind-pwn", opts).unwrap();
        use std::io::Write as _;
        zw.write_all(b"pwn\n").unwrap();
        zw.start_file("safe.txt", opts).unwrap();
        zw.write_all(b"safe\n").unwrap();
        zw.finish().unwrap();

        let out = tmp.join("out");
        let (written, _, skipped) = zip_read_tree(&archive, &out, true, 0, &[]).unwrap();
        assert_eq!(written, 1, "only the safe entry should be written");
        assert_eq!(skipped.len(), 1);
        assert!(
            skipped[0].contains("zip-slip"),
            "expected zip-slip rejection, got {:?}",
            skipped[0]
        );
        assert!(out.join("safe.txt").exists());

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
