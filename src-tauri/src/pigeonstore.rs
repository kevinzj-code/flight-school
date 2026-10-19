//! Models live in Pigeon. Flight School borrows them.
//!
//! Every model this Mac keeps — the GGUFs Pigeon runs, the teacher and drafter
//! Flight School writes with, the projectors, the MLX copies training reads —
//! sits in ONE place: Pigeon's `models/` folder. Flight School reads and writes
//! there and keeps nothing of its own. Decided 2026-09-21: "if you are using
//! Flight School, you should download Pigeon anyway", and a model downloaded
//! once should never be downloaded again by the other app.
//!
//! ```text
//! ~/Library/Application Support/mode.airoplane.pigeon/models/
//!   *.gguf                    running copies, any app's
//!   projectors/*.gguf         vision projectors (not models; never offered as one)
//!   training/                 HF_HOME for the MLX copies training reads
//!   borrowers/flight-school.json   which of these Flight School relies on
//! ```
//!
//! `borrowers/` is how Pigeon knows not to delete something out from under
//! Flight School: each borrowing app writes its OWN file there, so two apps
//! never write the same file.
//!
//! The move from Flight School's old folders happens once, at launch: a rename
//! on the same disk, so 38 GB moves in well under a second. Each old folder is
//! left as a symlink to its new home, so run folders whose config.toml names an
//! old path — and Airoplane Mode's own links into Flight School's folder — keep
//! resolving. The same trick moved these models out of Airoplane Mode's folder
//! on 2026-09-17.
//!
//! Nothing is ever deleted. A file whose name is already taken in the store is
//! left where it was and reported, never overwritten and never removed.

use serde::Serialize;
use std::path::{Path, PathBuf};

/// Pigeon's bundle identifier — its data folder's name.
pub const PIGEON_APP_ID: &str = "mode.airoplane.pigeon";
const BORROWER: &str = "flight-school";

/// Pigeon's data folder, whether or not it exists.
pub fn pigeon_root() -> Option<PathBuf> {
    let home = std::env::var_os("HOME").map(PathBuf::from)?;
    Some(home.join("Library").join("Application Support").join(PIGEON_APP_ID))
}

/// Pigeon is on this Mac when its data folder exists — it is created the first
/// time Pigeon opens. Flight School never creates it: an app that makes
/// another app's folder is an app pretending the other one is installed.
pub fn pigeon_present() -> bool {
    pigeon_root().is_some_and(|p| p.is_dir())
}

pub const NO_PIGEON: &str = "Flight School keeps its models in Pigeon, and Pigeon is not on this Mac yet. \
Install Pigeon and open it once, then reopen Flight School.";

fn store_in(root: &Path) -> PathBuf { root.join("models") }

/// The store, created on demand — only when Pigeon is present.
pub fn store() -> Result<PathBuf, String> {
    let root = pigeon_root().ok_or("no home folder")?;
    if !root.is_dir() {
        return Err(NO_PIGEON.into());
    }
    let dir = store_in(&root);
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    Ok(dir)
}

fn sub(name: &str) -> Result<PathBuf, String> {
    let dir = store()?.join(name);
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    Ok(dir)
}

pub fn projectors() -> Result<PathBuf, String> { sub("projectors") }
pub fn training() -> Result<PathBuf, String> { sub("training") }

/* ── the one-time move ────────────────────────────────────────────────────── */

#[derive(Serialize, Default, Debug, Clone, PartialEq, Eq)]
pub struct MoveReport {
    /// Store-relative paths that were moved in.
    pub moved: Vec<String>,
    pub bytes: u64,
    /// Names already taken in the store; the old copy was left where it was.
    pub kept_in_place: Vec<String>,
    /// Old folders now a symlink to their new home.
    pub linked: Vec<String>,
    /// Anything that could not be moved, and why.
    pub problems: Vec<String>,
}

fn size_of(p: &Path) -> u64 {
    let Ok(m) = std::fs::symlink_metadata(p) else { return 0 };
    if m.is_file() { return m.len(); }
    if m.is_dir() {
        return std::fs::read_dir(p).map(|rd| rd.flatten().map(|e| size_of(&e.path())).sum()).unwrap_or(0);
    }
    0
}

/// Move everything in `old` into `new`, then leave `old` as a symlink to `new`.
/// `label` is what the report calls the destination ("", "projectors/", …).
/// Pure filesystem work, testable with temp folders.
pub fn move_folder(old: &Path, new: &Path, label: &str, report: &mut MoveReport) {
    let Ok(meta) = std::fs::symlink_metadata(old) else { return };
    if meta.file_type().is_symlink() || !meta.is_dir() {
        return; // already moved, or nothing to move
    }
    if let Err(e) = std::fs::create_dir_all(new) {
        report.problems.push(format!("{}: {e}", new.display()));
        return;
    }
    let Ok(entries) = std::fs::read_dir(old) else { return };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name_s = name.to_string_lossy().into_owned();
        if name_s == ".DS_Store" { let _ = std::fs::remove_file(entry.path()); continue; }
        let dest = new.join(&name);
        if std::fs::symlink_metadata(&dest).is_ok() {
            report.kept_in_place.push(format!("{label}{name_s}"));
            continue;
        }
        let bytes = size_of(&entry.path());
        match std::fs::rename(entry.path(), &dest) {
            Ok(()) => { report.moved.push(format!("{label}{name_s}")); report.bytes += bytes; }
            // A different disk would mean COPYING gigabytes; that is not done
            // silently at launch. Reported, and the file stays where it is.
            Err(e) => report.problems.push(format!("{label}{name_s} could not be moved: {e}")),
        }
    }
    let empty = std::fs::read_dir(old).map(|mut rd| rd.next().is_none()).unwrap_or(false);
    if empty && std::fs::remove_dir(old).is_ok() {
        #[cfg(unix)]
        match std::os::unix::fs::symlink(new, old) {
            Ok(()) => report.linked.push(old.display().to_string()),
            Err(e) => report.problems.push(format!("could not leave a link at {}: {e}", old.display())),
        }
    }
}

/// Record what Flight School relies on, in its own file under `borrowers/`.
/// The list only grows: a file stays borrowed until Flight School says
/// otherwise, which is the safe direction for a delete guard.
pub fn record_borrowed(store: &Path, files: &[String]) -> Result<(), String> {
    let dir = store.join("borrowers");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let path = dir.join(format!("{BORROWER}.json"));
    let mut all: Vec<String> = std::fs::read_to_string(&path).ok()
        .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
        .and_then(|v| v.get("files").and_then(|f| f.as_array()).map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect()))
        .unwrap_or_default();
    for f in files { if !all.contains(f) { all.push(f.clone()); } }
    all.sort();
    let when = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let body = serde_json::json!({
        "app": "Flight School",
        "why": "Flight School trains, drafts and checks with these. Ask before removing one.",
        "updated_unix": when,
        "files": all,
    });
    let tmp = path.with_extension("json.part");
    std::fs::write(&tmp, serde_json::to_vec_pretty(&body).unwrap()).map_err(|e| e.to_string())?;
    std::fs::rename(&tmp, &path).map_err(|e| e.to_string())
}

/// Moves Flight School's three old folders into the store. Idempotent: once
/// the old folders are links, it does nothing.
pub fn migrate(fs_data: &Path, store: &Path) -> MoveReport {
    let mut r = MoveReport::default();
    move_folder(&fs_data.join("models"), store, "", &mut r);
    move_folder(&fs_data.join("mmproj"), &store.join("projectors"), "projectors/", &mut r);
    move_folder(&fs_data.join("hf_cache"), &store.join("training"), "training/", &mut r);
    if !r.moved.is_empty() {
        if let Err(e) = record_borrowed(store, &r.moved) {
            r.problems.push(format!("could not record what Flight School uses: {e}"));
        }
    }
    r
}

/* ── status for the page ─────────────────────────────────────────────────── */

#[derive(Serialize, Clone, Default)]
pub struct StoreStatus {
    pub pigeon_present: bool,
    pub store: Option<String>,
    pub files: usize,
    pub gb: f32,
    /// What this launch moved in, when it moved anything.
    pub last_move: Option<MoveReport>,
    pub message: Option<String>,
}

pub struct LastMove(pub std::sync::Mutex<Option<MoveReport>>);

#[tauri::command]
pub fn models_home_status(state: tauri::State<LastMove>) -> StoreStatus {
    if !pigeon_present() {
        return StoreStatus { pigeon_present: false, message: Some(NO_PIGEON.into()), ..Default::default() };
    }
    let Ok(dir) = store() else { return StoreStatus::default() };
    let files = std::fs::read_dir(&dir).map(|rd| rd.flatten()
        .filter(|e| e.path().extension().is_some_and(|x| x == "gguf")).count()).unwrap_or(0);
    StoreStatus {
        pigeon_present: true,
        store: Some(dir.display().to_string()),
        files,
        gb: size_of(&dir) as f32 / 1e9,
        last_move: state.0.lock().ok().and_then(|g| g.clone()),
        message: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("fs-pigeonstore-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn moves_everything_and_leaves_a_link_behind() {
        let root = tmp("move");
        let (fs, store) = (root.join("fs"), root.join("pigeon/models"));
        std::fs::create_dir_all(fs.join("models")).unwrap();
        std::fs::write(fs.join("models/a.gguf"), b"aaaa").unwrap();
        std::fs::create_dir_all(fs.join("hf_cache/hub/models--x")).unwrap();
        std::fs::write(fs.join("hf_cache/hub/models--x/w"), b"12").unwrap();
        let r = migrate(&fs, &store);
        assert_eq!(r.moved, vec!["a.gguf".to_string(), "training/hub".to_string()]);
        assert_eq!(r.bytes, 6);
        assert!(r.problems.is_empty(), "{:?}", r.problems);
        assert!(std::fs::symlink_metadata(fs.join("models")).unwrap().file_type().is_symlink());
        // An old path still reads, through the link.
        assert_eq!(std::fs::read(fs.join("models/a.gguf")).unwrap(), b"aaaa");
        assert_eq!(std::fs::read(store.join("training/hub/models--x/w")).unwrap(), b"12");
        // And a second launch does nothing.
        assert_eq!(migrate(&fs, &store), MoveReport::default());
    }

    #[test]
    fn a_name_already_taken_is_kept_in_place_never_overwritten() {
        let root = tmp("clash");
        let (fs, store) = (root.join("fs"), root.join("pigeon/models"));
        std::fs::create_dir_all(fs.join("models")).unwrap();
        std::fs::create_dir_all(&store).unwrap();
        std::fs::write(fs.join("models/a.gguf"), b"flight school's").unwrap();
        std::fs::write(store.join("a.gguf"), b"pigeon's").unwrap();
        let r = migrate(&fs, &store);
        assert_eq!(r.kept_in_place, vec!["a.gguf".to_string()]);
        assert_eq!(std::fs::read(store.join("a.gguf")).unwrap(), b"pigeon's");
        assert_eq!(std::fs::read(fs.join("models/a.gguf")).unwrap(), b"flight school's");
        // The old folder is not empty, so it stays a folder rather than a link.
        assert!(!std::fs::symlink_metadata(fs.join("models")).unwrap().file_type().is_symlink());
    }

    #[test]
    fn the_borrowed_list_only_grows() {
        let store = tmp("borrow");
        record_borrowed(&store, &["a.gguf".into()]).unwrap();
        record_borrowed(&store, &["b.gguf".into(), "a.gguf".into()]).unwrap();
        let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(store.join("borrowers/flight-school.json")).unwrap()).unwrap();
        assert_eq!(v["files"], serde_json::json!(["a.gguf", "b.gguf"]));
    }
}

/// The real move, on this Mac's real folders — the same `migrate` the app's
/// launch runs. Ignored so the suite never moves anyone's models by accident:
///   cargo test -- --ignored move_this_macs_models --nocapture
#[cfg(test)]
mod real {
    #[test]
    #[ignore]
    fn move_this_macs_models() {
        assert!(super::pigeon_present(), "{}", super::NO_PIGEON);
        let home = std::path::PathBuf::from(std::env::var_os("HOME").unwrap());
        let fs = home.join("Library/Application Support/mode.airoplane.flightschool");
        let store = super::store().unwrap();
        let r = super::migrate(&fs, &store);
        println!("{}", serde_json::to_string_pretty(&r).unwrap());
        assert!(r.problems.is_empty(), "{:?}", r.problems);
    }
}
