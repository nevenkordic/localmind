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
mod repl;
mod tools;
mod ui;
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

    let cfg = config::Config::load(cli.config.as_deref()).context("loading configuration")?;
    tracing::debug!("loaded config: {:?}", cfg.summary());

    // Register sqlite-vec before any connection is opened.
    memory::register_vec_extension();

    let store = memory::Store::open(&cfg)
        .await
        .context("opening memory store")?;
    store.migrate().await.context("running migrations")?;

    // Start the background embedding worker.
    let embedder_handle = memory::embedder::spawn(store.clone(), cfg.clone());

    let result = match cli.command.unwrap_or(Command::Chat {
        session: None,
        mode: None,
    }) {
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
    }
}
