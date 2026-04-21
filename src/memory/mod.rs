//! Memory subsystem — SQLite-backed multi-layer store.

pub mod embedder;
pub mod kg;
pub mod search;
pub mod store;

#[allow(unused_imports)]
pub use store::{NewMemory, Stats, Store, StoredMemory};

/// Statically register the sqlite-vec extension with every connection the
/// process opens. Called once, early in main(), BEFORE any Connection::open.
///
/// The extension is linked directly into this binary via the `sqlite-vec`
/// crate — no file is loaded from disk at runtime, and the rusqlite
/// `load_extension` feature is deliberately not enabled.
pub fn register_vec_extension() {
    // sqlite_vec::sqlite3_vec_init has an `extern "C"` signature. We cast it
    // to the function-pointer type that sqlite3_auto_extension expects and
    // register it. This is the same pattern shown in sqlite-vec's docs.
    // Target signature expected by rusqlite's sqlite3_auto_extension:
    //     unsafe extern "C" fn(
    //         *mut sqlite3, *mut *const c_char, *const sqlite3_api_routines
    //     ) -> c_int
    type AutoExtInit = unsafe extern "C" fn(
        *mut rusqlite::ffi::sqlite3,
        *mut *const std::os::raw::c_char,
        *const rusqlite::ffi::sqlite3_api_routines,
    ) -> std::os::raw::c_int;
    unsafe {
        let init: AutoExtInit = std::mem::transmute(sqlite_vec::sqlite3_vec_init as *const ());
        rusqlite::ffi::sqlite3_auto_extension(Some(init));
    }
}
