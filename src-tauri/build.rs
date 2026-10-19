use std::path::Path;
use std::process::Command;

/// The window recorder (`recorder/Recorder.swift`) is Swift, because on
/// macOS 15 and later only ScreenCaptureKit can capture a window. It is built
/// into `binaries/` beside llama-server — Tauri's sidecar folder, which
/// `externalBin` requires to exist before `tauri_build` runs — whenever the
/// source is newer than the binary. The binary is not committed: every
/// machine compiles it, so it always matches the source beside it.
fn build_recorder() {
    let src = Path::new("recorder/Recorder.swift");
    let out = Path::new("binaries/fs-recorder-aarch64-apple-darwin");
    println!("cargo:rerun-if-changed=recorder/Recorder.swift");
    let newer = |a: &Path, b: &Path| match (a.metadata().and_then(|m| m.modified()), b.metadata().and_then(|m| m.modified())) {
        (Ok(x), Ok(y)) => x > y,
        (Ok(_), Err(_)) => true,
        _ => false,
    };
    if !newer(src, out) {
        return;
    }
    let status = Command::new("xcrun")
        .args(["swiftc", "-O", "-target", "arm64-apple-macosx12.3", "-o"])
        .arg(out)
        .arg(src)
        .status();
    match status {
        Ok(s) if s.success() => {}
        _ if out.exists() => println!("cargo:warning=could not rebuild fs-recorder; using the one already built"),
        _ => panic!("fs-recorder could not be built: install Xcode's command line tools (xcode-select --install)"),
    }
}

fn main() {
    build_recorder();
    tauri_build::build()
}
