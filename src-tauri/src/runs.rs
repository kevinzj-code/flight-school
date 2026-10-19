//! Shared time stamp for run folders and state files, written by the backend
//! so a UI reload never loses when something happened.

/// Milliseconds since the epoch. The frontend formats it; storing a formatted
/// string would bake one machine's locale into a file another machine reads.
pub fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn now_ms_is_sane() {
        // After 2020, before 2100 — catches unit mistakes (seconds vs millis).
        let t = now_ms();
        assert!(t > 1_577_836_800_000, "looks like seconds, not ms: {t}");
        assert!(t < 4_102_444_800_000, "implausibly far future: {t}");
    }
}
