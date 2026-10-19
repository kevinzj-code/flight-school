//! The small file helpers every module used to carry its own copy of.

use serde_json::Value;
use std::path::Path;

/// Write through a temp file and rename, so a crash mid-write leaves the
/// previous file rather than half of this one.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let tmp = path.with_extension("tmp-write");
    std::fs::write(&tmp, bytes).map_err(|e| e.to_string())?;
    std::fs::rename(&tmp, path).map_err(|e| e.to_string())
}

/// A JSON file as a Value, or None when it is missing or not JSON.
pub fn read_json(path: &Path) -> Option<Value> {
    std::fs::read_to_string(path).ok().and_then(|s| serde_json::from_str(&s).ok())
}
