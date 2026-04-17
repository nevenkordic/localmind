//! Small utilities shared across modules.

use chrono::Utc;
use sha2::{Digest, Sha256};

pub fn now_ts() -> i64 {
    Utc::now().timestamp()
}

pub fn new_uuid() -> String {
    uuid::Uuid::new_v4().to_string()
}

pub fn sha256_hex(s: &str) -> String {
    let mut h = Sha256::new();
    h.update(s.as_bytes());
    format!("{:x}", h.finalize())
}

pub fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max).collect();
    out.push_str("…");
    out
}

/// Convert a Vec<f32> to a little-endian byte blob suitable for SQLite BLOB
/// storage and for sqlite-vec's vec0 virtual table input.
pub fn f32_to_blob(v: &[f32]) -> Vec<u8> {
    bytemuck::cast_slice(v).to_vec()
}

#[allow(dead_code)]
pub fn blob_to_f32(b: &[u8]) -> Vec<f32> {
    // Guard: blob length must be a multiple of 4.
    if b.len() % 4 != 0 {
        return vec![];
    }
    bytemuck::cast_slice(b).to_vec()
}

pub fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Cosine similarity. Assumes both vectors are non-empty and same length.
#[allow(dead_code)]
pub fn cosine(a: &[f32], b: &[f32], norm_a: f32, norm_b: f32) -> f32 {
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    dot / (norm_a * norm_b)
}

/// Temporal decay factor using exponential half-life.
/// Returns a value in (0, 1]. `age_days = 0` -> 1.0, `age_days = half_life` -> 0.5.
pub fn temporal_decay(age_days: f32, half_life_days: f32) -> f32 {
    if half_life_days <= 0.0 {
        return 1.0;
    }
    0.5_f32.powf(age_days / half_life_days)
}
