//! Terminal UI helpers shared across the agent loop and the REPL.
//!
//! Two concerns live here:
//!
//!   * `color_enabled` / `paint` — small ANSI helpers honouring `NO_COLOR`
//!     and `TERM=dumb`.
//!   * `Spinner` — a braille "thinking" indicator drawn to stderr while the
//!     agent waits on the model. Suppressed automatically when stderr isn't
//!     a TTY so piped output stays clean.

use std::io::{IsTerminal, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::task::JoinHandle;

pub fn color_enabled(want: bool) -> bool {
    want && std::env::var_os("NO_COLOR").is_none()
        && std::env::var("TERM").map(|t| t != "dumb").unwrap_or(true)
}

/// Print a single dim "tool starting" line — replaces the old verbose
/// `→ tool_name({...big json...})` echo so call args don't leak into the chat.
/// Detailed call info still goes to the audit log.
pub fn print_tool_step(color: bool, name: &str) {
    let line = if color {
        format!("  \x1b[38;5;238m·\x1b[0m \x1b[38;5;111m{name}\x1b[0m")
    } else {
        format!("  · {name}")
    };
    let _ = writeln!(std::io::stderr(), "{line}");
}

/// Braille "thinking" spinner. Tied to a tokio task that ticks every 80ms.
/// `stop()` (or Drop) signals shutdown; the task clears its line on exit.
pub struct Spinner {
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl Spinner {
    pub fn start(label: &'static str, color: bool) -> Self {
        // Skip animation when stderr isn't a TTY — piped output would just
        // collect carriage returns and erase-line escapes.
        if !std::io::stderr().is_terminal() {
            return Self {
                stop: Arc::new(AtomicBool::new(true)),
                handle: None,
            };
        }
        let stop = Arc::new(AtomicBool::new(false));
        let stop_c = stop.clone();
        let handle = tokio::spawn(async move {
            const FRAMES: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
            // Short hold-off — fast calls (cache hits, errors) skip the spinner
            // entirely so it doesn't flicker on / off in 200ms.
            tokio::time::sleep(Duration::from_millis(120)).await;
            let mut i = 0usize;
            while !stop_c.load(Ordering::Relaxed) {
                let frame = FRAMES[i % FRAMES.len()];
                let line = if color {
                    format!("\r  \x1b[38;5;199m{frame}\x1b[0m \x1b[38;5;245m{label}…\x1b[0m")
                } else {
                    format!("\r  {frame} {label}…")
                };
                let _ = write!(std::io::stderr(), "{line}");
                let _ = std::io::stderr().flush();
                tokio::time::sleep(Duration::from_millis(80)).await;
                i += 1;
            }
            // Clear the spinner line so subsequent output starts at column 0.
            let _ = write!(std::io::stderr(), "\r\x1b[2K");
            let _ = std::io::stderr().flush();
        });
        Self {
            stop,
            handle: Some(handle),
        }
    }

    /// Cooperatively stop the spinner and wait for the cleanup write to land.
    pub async fn stop(mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.await;
        }
    }

    /// Same as `stop`, but synchronous — clears the line inline so a caller
    /// in a non-async context (e.g. a streaming token callback) can print
    /// the next byte on column 0 without waiting for the spinner task to
    /// notice the stop flag on its next tick. The background task is left
    /// to exit on its own time; double-clearing is harmless.
    pub fn stop_sync(mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if self.handle.is_some() {
            let _ = write!(std::io::stderr(), "\r\x1b[2K");
            let _ = std::io::stderr().flush();
        }
        self.handle.take();
    }
}

impl Drop for Spinner {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}
