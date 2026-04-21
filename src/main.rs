//! localmind — run any local LLM with persistent memory and context.
//! CLI agent over Ollama, with a SQLite-backed hybrid (BM25 + vector) memory
//! store. Runs entirely offline.

mod agent;
mod config;
mod health;
mod llm;
mod memory;
mod models;
mod net_safety;
mod preflight;
mod repl;
mod tools;
mod ui;
mod update;
mod util;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(
    name = "llm",
    version,
    about = "Run any local LLM with persistent memory and context. No cloud.",
    long_about = None
)]
struct Cli {
    /// Path to config file. Defaults to ./config/local.toml then $XDG/localmind/config.toml.
    #[arg(short, long)]
    config: Option<PathBuf>,

    /// Verbose logging (repeat for more: -v, -vv).
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Start an interactive REPL session (default).
    Chat {
        /// Resume a session by id.
        #[arg(long)]
        session: Option<String>,
        /// Permission mode: read-only | workspace-write | unrestricted.
        #[arg(long)]
        mode: Option<String>,
    },
    /// Run a single prompt and exit.
    Ask {
        /// The prompt.
        prompt: String,
        /// Permission mode: read-only | workspace-write | unrestricted.
        #[arg(long)]
        mode: Option<String>,
    },
    /// Memory management.
    Memory {
        #[command(subcommand)]
        cmd: MemoryCmd,
    },
    /// Print the effective configuration and exit.
    ConfigShow,
    /// Initialise the memory database and verify connectivity.
    Init,
    /// Print a one-shot health report (DB stats, embedder outbox, Ollama).
    Health,
    /// Check for a newer release and re-run the installer if one is available.
    /// Shells out to the same curl-pipe-sh one-liner you used to install.
    Update {
        /// Upgrade even if the remote version isn't newer than the current one.
        #[arg(long)]
        force: bool,
    },
    /// Write a portable copy of the memory database to a file. Default path:
    /// ~/localmind-backup-YYYYMMDD-HHMMSS.db. Safe to run while localmind is
    /// in use (uses SQLite VACUUM INTO).
    Backup {
        /// Destination file. Omit for a timestamped filename in $HOME.
        path: Option<PathBuf>,
    },
    /// Replace the memory database with the contents of a backup file. Keeps
    /// a .bak-before-restore copy of your current DB in case you want it back.
    Restore {
        /// Path to a backup .db file produced by `llm backup`.
        path: PathBuf,
        /// Skip the confirmation prompt (required in non-interactive sessions).
        #[arg(long)]
        yes: bool,
    },
    /// Pick which Ollama models to use (interactive picker, or set via flags).
    Models {
        /// Just list installed models and current selections, then exit.
        #[arg(long)]
        list: bool,
        /// Non-interactively set the chat model.
        #[arg(long, value_name = "NAME")]
        chat: Option<String>,
        /// Non-interactively set the vision model.
        #[arg(long, value_name = "NAME")]
        vision: Option<String>,
        /// Non-interactively set the embedding model.
        #[arg(long, value_name = "NAME")]
        embed: Option<String>,
    },
}

#[derive(Subcommand, Debug)]
enum MemoryCmd {
    /// Search memory and print hits.
    Search {
        /// Query string.
        query: String,
        /// How many results.
        #[arg(short = 'k', long, default_value_t = 10)]
        top_k: usize,
        /// Pure BM25 (no embedding round-trip). Sub-millisecond — use this
        /// for fast inspection when Ollama is slow or down.
        #[arg(long)]
        bm25: bool,
    },
    /// Store a memory manually.
    Add {
        /// Title.
        #[arg(short, long)]
        title: String,
        /// Content (if omitted, reads from stdin).
        content: Option<String>,
        /// Kind: fact | decision | project | preference | note.
        #[arg(long, default_value = "note")]
        kind: String,
        /// Importance 0.0 - 1.0
        #[arg(long, default_value_t = 0.5)]
        importance: f32,
    },
    /// Delete a memory by id.
    Delete { id: String },
    /// Print stats (counts, embedding coverage).
    Stats,
    /// Rebuild the vec0 ANN index from the portable blob table.
    Reindex,
    /// Re-embed every memory from scratch. Slower than Reindex (one
    /// Ollama embed call per row, plus one contextualize call per row
    /// when `contextual_embed = true`). Run this after enabling
    /// `contextual_embed`, switching `embed_model`, or restoring from a
    /// backup that lost its vectors.
    Reembed {
        /// Cap the number of rows processed. Useful for smoke-testing
        /// the flow before committing to a full re-embed on a large db.
        #[arg(long)]
        limit: Option<usize>,
    },
}

fn parse_mode_flag(s: Option<&str>) -> Result<Option<tools::permissions::PermissionMode>> {
    match s {
        None => Ok(None),
        Some(v) => tools::permissions::PermissionMode::parse(v)
            .map(Some)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "invalid --mode '{v}'. Use: read-only | workspace-write | unrestricted"
                )
            }),
    }
}

fn init_logging(verbosity: u8) {
    let level = match verbosity {
        0 => "localmind=info,warn",
        1 => "localmind=debug,info",
        _ => "localmind=trace,debug",
    };
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(level));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .with_level(true)
        .compact()
        .init();
}

#[tokio::main]
async fn main() -> Result<()> {
    let _ = dotenvy::dotenv();
    let cli = Cli::parse();
    init_logging(cli.verbose);

    let mut cfg = config::Config::load(cli.config.as_deref()).context("loading configuration")?;
    tracing::debug!("loaded config: {:?}", cfg.summary());

    // Non-blocking: prints a banner if a previously-cached release is newer,
    // and spawns a 3-second background refresh if the cache is older than
    // `updates.check_interval_hours`. Skipped for `llm update` since that
    // command does its own synchronous check.
    let skip_update_banner = matches!(cli.command, Some(Command::Update { .. }));
    if !skip_update_banner {
        update::maybe_check_and_print(&cfg);
    }

    // Register sqlite-vec before any connection is opened.
    memory::register_vec_extension();

    let store = memory::Store::open(&cfg)
        .await
        .context("opening memory store")?;
    store.migrate().await.context("running migrations")?;

    // Start the background embedding worker. Restore is the one command
    // where the worker could race with us (it writes outbox rows to the
    // old DB mid-copy), so the dispatcher below aborts the handle before
    // running the restore itself.
    let embedder_handle = memory::embedder::spawn(store.clone(), cfg.clone());

    let command = cli.command.unwrap_or(Command::Chat {
        session: None,
        mode: None,
    });

    // Preflight: check Ollama reachability + configured models are pulled
    // before we open the REPL or handle a one-shot ask. When stdin is a
    // TTY, prompts to switch chat model to an installed alternative and
    // writes the choice back to the config file — so the REPL banner and
    // first turn both see the new value. Never fails hard; an `llm ask`
    // inside a script still runs and surfaces the 404 itself.
    if matches!(command, Command::Chat { .. } | Command::Ask { .. }) {
        preflight::check(&mut cfg, cli.config.as_deref()).await;
    }

    let result = match command {
        Command::Chat { session, mode } => {
            let m = parse_mode_flag(mode.as_deref())?;
            repl::run(cfg.clone(), store.clone(), session, m).await
        }
        Command::Ask { prompt, mode } => {
            let m = parse_mode_flag(mode.as_deref())?;
            agent::one_shot(cfg.clone(), store.clone(), &prompt, m).await
        }
        Command::Memory { cmd } => memory_cmd(cmd, &cfg, &store).await,
        Command::ConfigShow => {
            println!("{}", cfg.pretty()?);
            Ok(())
        }
        Command::Health => {
            print!("{}", health::report(&cfg, &store).await);
            Ok(())
        }
        Command::Models {
            list,
            chat,
            vision,
            embed,
        } => {
            models::run(
                &cfg,
                cli.config.as_deref(),
                models::CliArgs {
                    list_only: list,
                    chat,
                    vision,
                    embed,
                },
            )
            .await
        }
        Command::Backup { path } => backup_cmd(&cfg, &store, path).await,
        Command::Restore { path, yes } => {
            // Stop the background embedder before swapping the file — it
            // holds an open handle to the current DB and would otherwise
            // write outbox rows into the about-to-be-renamed file. Abort
            // is cooperative; give the worker a small grace period to
            // drop its Connection before the copy starts.
            embedder_handle.abort();
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            restore_cmd(&cfg, &store, &path, yes).await
        }
        Command::Update { force } => {
            println!("current version: v{}", update::current_version());
            match update::fetch_and_cache().await {
                Ok(c) => {
                    println!("latest release: {}", c.latest_tag);
                    if !force && !update::is_newer(update::current_version(), &c.latest_tag) {
                        println!("already on the latest version. Pass --force to reinstall.");
                        return Ok(());
                    }
                }
                Err(e) => {
                    // Network is down or GitHub API is rate-limiting. Don't
                    // refuse the upgrade — the user explicitly asked for it.
                    // They'll see the installer's own error if the release
                    // fetch also fails.
                    eprintln!("warning: could not check GitHub releases ({e:#}). Running installer anyway.");
                }
            }
            update::run_installer()
        }
        Command::Init => {
            println!("memory db:   {}", cfg.memory.db_path_resolved().display());
            println!("ollama host: {}", cfg.ollama.host);
            let client = llm::ollama::OllamaClient::new(&cfg.ollama);
            match client.health().await {
                Ok(models) => {
                    println!("ollama:      OK ({} models available)", models.len());
                    for m in models.iter().take(10) {
                        println!("  - {}", m);
                    }
                }
                Err(e) => println!("ollama:      UNREACHABLE ({e})"),
            }
            Ok(())
        }
    };

    embedder_handle.abort();
    result
}

async fn backup_cmd(
    cfg: &config::Config,
    store: &memory::Store,
    dest: Option<PathBuf>,
) -> Result<()> {
    let dest = dest.unwrap_or_else(default_backup_path);
    if dest.exists() {
        anyhow::bail!(
            "destination already exists: {} — pick another path or remove it first",
            dest.display()
        );
    }
    println!("source:      {}", cfg.memory.db_path_resolved().display());
    println!("destination: {}", dest.display());
    let bytes = store.backup(&dest).await.context("running VACUUM INTO")?;
    println!("wrote {} bytes", bytes);
    println!();
    println!("Ship it:");
    println!("  scp {} user@host:", dest.display());
    println!("  llm restore {}", dest.display());
    Ok(())
}

async fn restore_cmd(
    cfg: &config::Config,
    store: &memory::Store,
    src: &std::path::Path,
    yes: bool,
) -> Result<()> {
    use std::io::IsTerminal;

    if !src.exists() {
        anyhow::bail!("backup file not found: {}", src.display());
    }
    validate_localmind_backup(src)?;
    let dest = cfg.memory.db_path_resolved();

    if !yes {
        if !std::io::stdin().is_terminal() {
            anyhow::bail!(
                "refusing to replace {} in non-interactive mode — pass --yes to skip the prompt",
                dest.display()
            );
        }
        use std::io::Write;
        eprint!(
            "Replace \x1b[1m{}\x1b[0m with \x1b[1m{}\x1b[0m?\nThis overwrites your current memory. A .bak-before-restore copy will be kept. [y/N] ",
            dest.display(),
            src.display()
        );
        let _ = std::io::stderr().flush();
        let mut line = String::new();
        std::io::stdin().read_line(&mut line)?;
        if !matches!(line.trim().to_lowercase().as_str(), "y" | "yes") {
            println!("cancelled");
            return Ok(());
        }
    }

    // Note: other localmind processes could still have the DB open — this
    // command only races safely with itself. Close other sessions before
    // running. The in-process embedder is aborted in main before we get
    // here (see the Restore guard there).
    let _ = store;

    if dest.exists() {
        // Timestamp the bak so consecutive restores keep independent rollback
        // points — without this, restoring twice would silently erase your
        // original pre-restore state, which is exactly the data someone might
        // want to get back if they regret the restore.
        let ts = chrono::Local::now().format("%Y%m%d-%H%M%S");
        let bak = dest.with_file_name(format!(
            "{}.bak-{ts}",
            dest.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("memory.db")
        ));
        std::fs::rename(&dest, &bak)
            .with_context(|| format!("moving current DB aside to {}", bak.display()))?;
        println!("kept previous DB at {}", bak.display());
    } else if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).ok();
    }

    std::fs::copy(src, &dest)
        .with_context(|| format!("copying {} -> {}", src.display(), dest.display()))?;
    // Also wipe any leftover WAL / SHM from the replaced DB so SQLite
    // doesn't try to replay journals that don't match the new file.
    for suffix in ["-wal", "-shm", "-journal"] {
        let side = dest.with_file_name(format!(
            "{}{suffix}",
            dest.file_name().and_then(|n| n.to_str()).unwrap_or("")
        ));
        let _ = std::fs::remove_file(side);
    }

    let bytes = std::fs::metadata(&dest)?.len();
    println!("✓ restored {} bytes to {}", bytes, dest.display());
    Ok(())
}

fn default_backup_path() -> PathBuf {
    let home = directories::UserDirs::new()
        .and_then(|u| Some(u.home_dir().to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."));
    let ts = chrono::Local::now().format("%Y%m%d-%H%M%S");
    home.join(format!("localmind-backup-{ts}.db"))
}

/// Minimal sanity check that a candidate file is actually a localmind
/// backup — opens it read-only and confirms the `memories` table is there.
/// Catches the case where a user fat-fingers the path and points at a
/// random SQLite file (or a totally unrelated file).
fn validate_localmind_backup(path: &std::path::Path) -> Result<()> {
    use rusqlite::{Connection, OpenFlags, OptionalExtension};
    let conn = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .with_context(|| format!("opening {} as SQLite", path.display()))?;
    let mut stmt = conn
        .prepare("SELECT name FROM sqlite_master WHERE type='table' AND name='memories'")
        .context("querying sqlite_master")?;
    let found = stmt
        .query_row([], |_| Ok(()))
        .optional()
        .context("checking for memories table")?
        .is_some();
    if !found {
        anyhow::bail!(
            "{} is a SQLite file but not a localmind backup — no 'memories' table",
            path.display()
        );
    }
    Ok(())
}

async fn memory_cmd(cmd: MemoryCmd, cfg: &config::Config, store: &memory::Store) -> Result<()> {
    match cmd {
        MemoryCmd::Search { query, top_k, bm25 } => {
            if bm25 {
                // Pure BM25 — no Ollama round-trip, sub-millisecond.
                let hits = store.bm25_search(&query, top_k).await?;
                for (i, (m, score)) in hits.iter().enumerate() {
                    println!(
                        "#{} [bm25={:.3}] {}  ({})\n  {}\n",
                        i + 1,
                        score,
                        m.title,
                        m.kind,
                        util::truncate(&m.content, 200),
                    );
                }
            } else {
                let client = llm::ollama::OllamaClient::new(&cfg.ollama);
                let hits =
                    memory::search::hybrid_search(store, &client, cfg, &query, top_k).await?;
                for (i, h) in hits.iter().enumerate() {
                    println!(
                        "#{} [{:.3}] {}  ({})\n  {}\n",
                        i + 1,
                        h.score,
                        h.memory.title,
                        h.memory.kind,
                        util::truncate(&h.memory.content, 200),
                    );
                }
            }
            Ok(())
        }
        MemoryCmd::Add {
            title,
            content,
            kind,
            importance,
        } => {
            let content = match content {
                Some(c) => c,
                None => {
                    use std::io::Read;
                    let mut buf = String::new();
                    std::io::stdin().read_to_string(&mut buf)?;
                    buf
                }
            };
            let id = store
                .insert_memory(&memory::NewMemory {
                    kind,
                    title,
                    content,
                    source: "cli".into(),
                    tags: vec![],
                    importance,
                })
                .await?;
            println!("stored {id}");
            Ok(())
        }
        MemoryCmd::Delete { id } => {
            store.delete_memory(&id).await?;
            println!("deleted {id}");
            Ok(())
        }
        MemoryCmd::Stats => {
            let s = store.stats().await?;
            println!("memories:          {}", s.memory_count);
            println!("with embedding:    {}", s.vector_count);
            println!("kg entities:       {}", s.entity_count);
            println!("kg edges:          {}", s.edge_count);
            println!("outbox pending:    {}", s.outbox_pending);
            println!("db size (bytes):   {}", s.db_size);
            Ok(())
        }
        MemoryCmd::Reindex => {
            let n = store.reindex_vectors().await?;
            println!("rebuilt ANN index over {n} vectors");
            Ok(())
        }
        MemoryCmd::Reembed { limit } => {
            let client = llm::ollama::OllamaClient::new(&cfg.ollama);
            let cap = limit.unwrap_or(10_000);
            let rows = store.list_all(cap).await?;
            let total = rows.len();
            if total == 0 {
                println!("no memories to re-embed");
                return Ok(());
            }
            let contextual = cfg.memory.contextual_embed;
            let entity_extraction = cfg.memory.entity_extraction;
            println!(
                "re-embedding {total} memor{plural} (contextual={ctx}, entities={ents}, model={model})",
                plural = if total == 1 { "y" } else { "ies" },
                ctx = contextual,
                ents = entity_extraction,
                model = cfg.ollama.embed_model,
            );
            let mut ok = 0usize;
            let mut failed = 0usize;
            for (i, mem) in rows.iter().enumerate() {
                // Same format the outbox worker uses — title\ncontent — so
                // the embedding space stays consistent regardless of path.
                let chunk = format!("{}\n{}", mem.title, mem.content);
                let to_embed = if contextual {
                    match client.contextualize(&mem.title, &mem.content).await {
                        Ok(ctx) if !ctx.is_empty() => format!("{ctx}\n\n{chunk}"),
                        _ => chunk.clone(),
                    }
                } else {
                    chunk.clone()
                };
                match client.embed(&to_embed).await {
                    Ok(vec) => {
                        match store
                            .upsert_embedding(&mem.id, &vec, &cfg.ollama.embed_model)
                            .await
                        {
                            Ok(_) => {
                                ok += 1;
                                // Entity extraction piggybacks on re-embed
                                // so the kg stays in sync when the user
                                // first enables it on an existing corpus.
                                let (ents, edges) = if entity_extraction {
                                    match client.extract_entities(&mem.title, &mem.content).await {
                                        Ok((es, eds)) => {
                                            let en = es.len();
                                            let ed = eds.len();
                                            for e in es.into_iter().take(8) {
                                                if let Ok(id) = store
                                                    .upsert_entity(&e.name, &e.etype, &mem.title)
                                                    .await
                                                {
                                                    let _ = store
                                                        .link_entity_memory(&id, &mem.id)
                                                        .await;
                                                }
                                            }
                                            for edge in eds.into_iter().take(12) {
                                                let _ = store
                                                    .upsert_edge(
                                                        &edge.src,
                                                        &edge.src_type,
                                                        &edge.dst,
                                                        &edge.dst_type,
                                                        &edge.relation,
                                                    )
                                                    .await;
                                            }
                                            (en, ed)
                                        }
                                        Err(_) => (0, 0),
                                    }
                                } else {
                                    (0, 0)
                                };
                                let ents_suffix = if entity_extraction {
                                    format!(" ({ents} entities, {edges} edges)")
                                } else {
                                    String::new()
                                };
                                println!(
                                    "  [{}/{}] {} — {}{}",
                                    i + 1,
                                    total,
                                    &mem.id[..8.min(mem.id.len())],
                                    util::truncate(&mem.title, 60),
                                    ents_suffix,
                                );
                            }
                            Err(e) => {
                                failed += 1;
                                eprintln!("  [{}/{}] upsert failed for {}: {e}", i + 1, total, mem.id);
                            }
                        }
                    }
                    Err(e) => {
                        failed += 1;
                        eprintln!("  [{}/{}] embed failed for {}: {e}", i + 1, total, mem.id);
                    }
                }
            }
            println!("done — {ok} re-embedded, {failed} failed");
            Ok(())
        }
    }
}
