//! Tool subsystem — the agent's hands. Every tool goes through the Registry,
//! which enforces:
//!   * deny-glob checks (credential paths, private keys, etc.)
//!   * interactive permission prompts (Yes / Yes for session / No)
//!   * audit logging (JSONL to audit.log)

pub mod archive;
pub mod audit;
pub mod docs;
pub mod file_ops;
pub mod image;
pub mod memory_tools;
pub mod net;
pub mod pdf;
pub mod permissions;
pub mod registry;
pub mod shell;
pub mod shell_validation;
pub mod web;

pub use registry::{Registry, ToolContext};
