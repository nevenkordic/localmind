//! DOCX and XLSX text extraction. DOCX we do ourselves (unzip + word/document.xml),
//! which avoids pulling in a heavy word-processing dep.

use crate::tools::registry::{resolve_path, ToolContext};
use anyhow::{anyhow, Result};
use serde_json::Value;
use std::io::Read;

pub async fn read_docx(ctx: &ToolContext, args: &Value) -> Result<String> {
    let path = args
        .get("path")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("missing path"))?;
    let resolved = resolve_path(ctx, path);
    let text = tokio::task::spawn_blocking(move || -> Result<String> {
        let file = std::fs::File::open(&resolved)?;
        let mut archive = zip::ZipArchive::new(file).map_err(|e| anyhow!("unzip: {e}"))?;
        let mut xml = String::new();
        {
            let mut entry = archive
                .by_name("word/document.xml")
                .map_err(|_| anyhow!("no word/document.xml in .docx"))?;
            entry.read_to_string(&mut xml)?;
        }
        // Extract <w:t>...</w:t> runs.
        let mut out = String::new();
        let mut reader = quick_xml::Reader::from_str(&xml);
        reader.config_mut().trim_text(false);
        use quick_xml::events::Event;
        let mut in_text = false;
        let mut in_para_end = false;
        loop {
            match reader.read_event() {
                Ok(Event::Start(e)) => {
                    let name = e.name();
                    if name.as_ref() == b"w:t" || name.as_ref() == b"t" {
                        in_text = true;
                    }
                }
                Ok(Event::End(e)) => {
                    let name = e.name();
                    if name.as_ref() == b"w:t" || name.as_ref() == b"t" {
                        in_text = false;
                    }
                    if name.as_ref() == b"w:p" || name.as_ref() == b"p" {
                        in_para_end = true;
                    }
                }
                Ok(Event::Text(t)) => {
                    if in_text {
                        out.push_str(&t.unescape().unwrap_or_default());
                    }
                }
                Ok(Event::Eof) => break,
                Err(e) => return Err(anyhow!("xml: {e}")),
                _ => {}
            }
            if in_para_end {
                out.push('\n');
                in_para_end = false;
            }
        }
        Ok(out)
    })
    .await??;
    Ok(text)
}

pub async fn read_xlsx(ctx: &ToolContext, args: &Value) -> Result<String> {
    use calamine::{open_workbook_auto, Reader};
    let path = args
        .get("path")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("missing path"))?;
    let sheet_opt = args
        .get("sheet")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    let resolved = resolve_path(ctx, path);
    let out = tokio::task::spawn_blocking(move || -> Result<String> {
        let mut wb = open_workbook_auto(&resolved).map_err(|e| anyhow!("open xlsx: {e}"))?;
        let sheet_names = wb.sheet_names().to_vec();
        let names: Vec<String> = match &sheet_opt {
            Some(s) => vec![s.clone()],
            None => sheet_names.clone(),
        };
        let mut out = String::new();
        for name in names {
            match wb.worksheet_range(&name) {
                Ok(range) => {
                    out.push_str(&format!("--- sheet: {name} ---\n"));
                    for row in range.rows() {
                        let cells: Vec<String> = row.iter().map(|c| format!("{}", c)).collect();
                        out.push_str(&cells.join("\t"));
                        out.push('\n');
                    }
                    out.push('\n');
                }
                Err(e) => return Err(anyhow!("sheet {name}: {e}")),
            }
        }
        Ok(out)
    })
    .await??;
    Ok(out)
}
