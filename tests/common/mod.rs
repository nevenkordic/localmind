//! Shared helpers for integration tests.
//!
//! `cargo test` automatically excludes files in `tests/common/` from being
//! their own test target as long as they're declared via `mod common;` from
//! the actual test file. Each test binary that needs the mock Ollama can do:
//!
//!     mod common;
//!     use common::mock_ollama::*;

#![allow(dead_code)] // each test binary uses a different subset

pub mod mock_ollama;
pub mod testenv;
