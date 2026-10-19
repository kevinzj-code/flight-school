//! The two-column table (Milestone 5): what the new screens (`src/`) keep on
//! disk, and the doors they reach the pipeline through.
//!
//! A run is a folder under `quickbuild/runs/`, like every Quick Build run, and
//! its source of truth is `table.json` — the run object the screens draw from
//! (`src/state.js`), saved whole after every change. The rules about what a
//! table means (what trains, what the exam is, what blocks) live in the page,
//! where they are drawn and tested; Rust stores, reads files the person
//! picked, fingerprints pictures, and starts the Python pipeline with the
//! rows the page decided should train.
//!
//! Row inputs name files two ways: relative to the run folder (recorded
//! frames), or absolute (a folder the person imported, left where it is). The
//! page can read back only a file its own table names.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde_json::{json, Value};
use tauri::{AppHandle, Manager, State};

use crate::paths;
use crate::pyproc::RunningJob;
use crate::quickbuild::{run_dir, runs_root};

const TABLE: &str = "table.json";
const SUMMARY: &str = "summary.json";
const IMAGE_EXTS: [&str; 6] = ["jpg", "jpeg", "png", "webp", "bmp", "gif"];

fn is_image(p: &Path) -> bool {
    p.extension().and_then(|e| e.to_str()).is_some_and(|e| IMAGE_EXTS.contains(&e.to_lowercase().as_str()))
}

use crate::fsutil::{read_json, write_atomic};

/* ── which files a table names ─────────────────────────────────────────── */

/// Per run, the image paths its table names — the only files `table_image`
/// and `table_fingerprints` will read. Refilled on every load and save.
#[derive(Default)]
pub struct Named(Mutex<HashMap<String, HashSet<String>>>);

fn paths_in(table: &Value) -> HashSet<String> {
    let mut out = HashSet::new();
    for r in table["rows"].as_array().into_iter().flatten() {
        if let Some(p) = r["input"]["path"].as_str() {
            out.insert(p.to_string());
        }
    }
    for s in table["strays"].as_array().into_iter().flatten() {
        if let Some(p) = s["path"].as_str() {
            out.insert(p.to_string());
        }
    }
    out
}

fn remember(app: &AppHandle, id: &str, table: &Value) {
    if let Ok(mut g) = app.state::<Named>().0.lock() {
        g.insert(id.to_string(), paths_in(table));
    }
}

/// A path the table names, resolved: relative ones inside the run folder,
/// absolute ones as they are. None for anything the table does not name.
fn resolve(app: &AppHandle, id: &str, path: &str) -> Result<PathBuf, String> {
    let p = Path::new(path);
    if !p.is_absolute() {
        /* A path inside the run folder cannot reach anything else, so it needs
           no allow-list — which also means a picture can be drawn the moment
           its row exists, before the table is saved. */
        if p.components().any(|c| !matches!(c, std::path::Component::Normal(_))) {
            return Err("not a path inside the run".into());
        }
        return Ok(run_dir(app, id)?.join(p));
    }
    /* An absolute path — an imported folder — only if a saved table names it. */
    let named = app.state::<Named>().0.lock().map(|g| g.get(id).is_some_and(|s| s.contains(path))).unwrap_or(false);
    if !named {
        return Err("that file is not in this table".into());
    }
    Ok(p.to_path_buf())
}

/* ── runs ──────────────────────────────────────────────────────────────── */

/// Every run on this Mac, for the runs list: a table run's whole table, or a
/// run recorded before tables existed (Milestone 4), as its recording, for
/// the page to turn into a table once. A folder from before either (the
/// screens that made them are gone) is left out.
#[tauri::command(async)]
pub fn table_list(app: AppHandle) -> Result<Value, String> {
    let root = runs_root(&app)?;
    let mut out = Vec::new();
    for (id, dir) in crate::quickbuild::run_folders(&root) {
        /* A summary beside the table (written on every save) is all the runs
           list needs: the whole table is read when the run is opened. Runs
           saved before summaries existed come whole, once. */
        if let Some(s) = read_json(&dir.join(SUMMARY)) {
            out.push(json!({ "id": id, "summary": s }));
        } else if let Some(t) = read_json(&dir.join(TABLE)) {
            remember(&app, &id, &t);
            out.push(json!({ "id": id, "table": t }));
        } else if dir.join("recording.json").is_file() {
            out.push(json!({ "id": id, "recording": crate::recorder::table_json(&dir),
                             "name": read_json(&dir.join("plan.json")).and_then(|p| p["job_name"].as_str().map(String::from)) }));
        }
    }
    Ok(json!({ "root": root.to_string_lossy(), "runs": out }))
}

/// Move a run's folder to the Trash — never a hard delete, so it can be put
/// back by hand from the Trash. Same-volume rename, so a run with a gigabyte
/// of adapters moves in an instant.
#[tauri::command]
pub fn table_trash(app: AppHandle, id: String) -> Result<(), String> {
    let dir = run_dir(&app, &id)?;
    if !dir.is_dir() {
        return Ok(());
    }
    let home = std::env::var_os("HOME").map(PathBuf::from).ok_or("no home folder")?;
    let trash = home.join(".Trash");
    if !trash.is_dir() {
        return Err("This Mac has no Trash folder to move the run into.".into());
    }
    let mut dest = trash.join(&id);
    let mut n = 1;
    while dest.exists() {
        dest = trash.join(format!("{id}-{n}"));
        n += 1;
    }
    std::fs::rename(&dir, &dest).map_err(|e| format!("could not move the run to the Trash: {e}"))?;
    if let Ok(mut g) = app.state::<Named>().0.lock() {
        g.remove(&id);
    }
    Ok(())
}

/// What a duplicate carries: the table and its pictures, and the caches that
/// describe them. Not the training, the exports, or the trainer's own files —
/// those belong to one teaching of one table.
const DUPLICATE_KEEP: [&str; 6] = ["table.json", "recording.json", "recording", "fingerprints.json", "signatures.json", ".thumbs"];

/// A new run with this run's table. Reply: the new id. The page rewrites its
/// table.json straight after, with the training cleared.
#[tauri::command]
pub fn table_duplicate(app: AppHandle, id: String) -> Result<String, String> {
    let src = run_dir(&app, &id)?;
    if !src.join(TABLE).is_file() {
        return Err("That run has no table to copy.".into());
    }
    let base = crate::runs::now_ms();
    let mut k = 0;
    let (nid, dst) = loop {
        let nid = format!("qb-{}", base + k);
        let dst = run_dir(&app, &nid)?;
        if !dst.exists() {
            break (nid, dst);
        }
        k += 1;
    };
    std::fs::create_dir_all(&dst).map_err(|e| e.to_string())?;
    for name in DUPLICATE_KEEP {
        let from = src.join(name);
        if from.is_file() {
            std::fs::copy(&from, dst.join(name)).map_err(|e| e.to_string())?;
        } else if from.is_dir() {
            copy_dir(&from, &dst.join(name))?;
        }
    }
    Ok(nid)
}

fn copy_dir(from: &Path, to: &Path) -> Result<(), String> {
    std::fs::create_dir_all(to).map_err(|e| e.to_string())?;
    for e in std::fs::read_dir(from).map_err(|e| e.to_string())? {
        let e = e.map_err(|e| e.to_string())?;
        let p = e.path();
        let t = to.join(e.file_name());
        if p.is_dir() {
            copy_dir(&p, &t)?;
        } else {
            std::fs::copy(&p, &t).map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

/// The whole table, as the page holds it. Written atomically, so a crash
/// mid-save leaves the previous table rather than half of this one.
#[tauri::command(async)]
pub fn table_save(app: AppHandle, id: String, table: Value) -> Result<(), String> {
    let dir = run_dir(&app, &id)?;
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    write_atomic(&dir.join(TABLE), serde_json::to_string(&table).map_err(|e| e.to_string())?.as_bytes())?;
    if let Some(s) = table.get("summary") {
        write_atomic(&dir.join(SUMMARY), serde_json::to_string(s).map_err(|e| e.to_string())?.as_bytes())?;
    }
    remember(&app, &id, &table);
    Ok(())
}

/// One run's whole table, for opening it.
#[tauri::command(async)]
pub fn table_get(app: AppHandle, id: String) -> Result<Value, String> {
    let dir = run_dir(&app, &id)?;
    let t = read_json(&dir.join(TABLE)).ok_or("That run has no table.")?;
    remember(&app, &id, &t);
    Ok(t)
}

/* ── pictures ──────────────────────────────────────────────────────────── */

/// Serves `pic://localhost/<run>/<path>[?full=1]` — the page's `<img src>`
/// for every picture, so WebKit fetches, caches and decodes them itself
/// instead of each one crossing the bridge as a base64 string that then
/// lives in the page's memory for good. Registered in lib.rs.
pub fn serve_picture(app: &AppHandle, request: &tauri::http::Request<Vec<u8>>) -> tauri::http::Response<Vec<u8>> {
    use percent_encoding::percent_decode_str;
    let uri = request.uri();
    let mut segs = uri.path().trim_start_matches('/').splitn(2, '/');
    let id = segs.next().map(|s| percent_decode_str(s).decode_utf8_lossy().into_owned()).unwrap_or_default();
    let path = segs.next().map(|s| percent_decode_str(s).decode_utf8_lossy().into_owned()).unwrap_or_default();
    let full = uri.query().is_some_and(|q| q.split('&').any(|kv| kv == "full=1"));
    match image_bytes(app, &id, &path, full) {
        Ok((bytes, mime)) => tauri::http::Response::builder()
            .status(200)
            .header("Content-Type", format!("image/{mime}"))
            .header("Cache-Control", "max-age=31536000, immutable")
            .body(bytes)
            .unwrap_or_else(|_| tauri::http::Response::new(Vec::new())),
        Err(e) => tauri::http::Response::builder().status(404).header("Content-Type", "text/plain").body(e.into_bytes())
            .unwrap_or_else(|_| tauri::http::Response::new(Vec::new())),
    }
}

/// A picture the table names, as bytes and a MIME subtype. A large photo is
/// shrunk first (by macOS's own `sips`, into the run's `.thumbs/`), because
/// the table shows it at a few hundred pixels — unless `full` asks for the
/// file as it is, for the picture viewer.
pub(crate) fn image_bytes(app: &AppHandle, id: &str, path: &str, full: bool) -> Result<(Vec<u8>, &'static str), String> {
    let file = resolve(app, id, path)?;
    if !is_image(&file) {
        return Err("not a picture".into());
    }
    let size = std::fs::metadata(&file).map_err(|e| e.to_string())?.len();
    let shrink = size > 400_000 && !full;
    let bytes = if shrink {
        let thumbs = run_dir(app, id)?.join(".thumbs");
        std::fs::create_dir_all(&thumbs).map_err(|e| e.to_string())?;
        let key = format!("{:016x}.jpg", fxhash(path.as_bytes()));
        let thumb = thumbs.join(key);
        if !thumb.is_file() {
            let ok = std::process::Command::new("sips")
                .args(["-Z", "640", "-s", "format", "jpeg"])
                .arg(&file).arg("--out").arg(&thumb)
                .stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null())
                .status().map(|s| s.success()).unwrap_or(false);
            if !ok {
                return Err("that picture could not be read".into());
            }
        }
        std::fs::read(&thumb).map_err(|e| e.to_string())?
    } else {
        std::fs::read(&file).map_err(|e| e.to_string())?
    };
    let mime = match file.extension().and_then(|e| e.to_str()).map(|e| e.to_lowercase()).as_deref() {
        _ if shrink => "jpeg",
        Some("png") => "png",
        Some("webp") => "webp",
        Some("gif") => "gif",
        Some("bmp") => "bmp",
        _ => "jpeg",
    };
    Ok((bytes, mime))
}

fn fxhash(b: &[u8]) -> u64 {
    b.iter().fold(0xcbf29ce484222325u64, |h, x| (h ^ *x as u64).wrapping_mul(0x100000001b3))
}

/// Each picture's fingerprint and its signature — the picture at 32 × 32 grey
/// levels, as base64 — so the page can say whether two pictures are the same
/// to the pixel or only close. Kept in `fingerprints.json` and
/// `signatures.json`.
#[tauri::command(async)]
pub fn table_signatures(app: AppHandle, id: String, paths: Vec<String>) -> Result<HashMap<String, Value>, String> {
    let (hashes, sigs) = picture_prints(&app, &id, &paths)?;
    Ok(paths.into_iter().filter_map(|p| {
        let h = hashes.get(&p)?;
        let s = sigs.get(&p)?;
        Some((p, serde_json::json!({ "hash": h, "sig": s })))
    }).collect())
}

/// The recorder is asked once, for every picture either cache is missing.
fn picture_prints(app: &AppHandle, id: &str, paths: &[String]) -> Result<(HashMap<String, String>, HashMap<String, String>), String> {
    let dir = run_dir(app, id)?;
    let hash_path = dir.join("fingerprints.json");
    let sig_path = dir.join("signatures.json");
    let load = |p: &Path| -> HashMap<String, String> {
        read_json(p).and_then(|v| serde_json::from_value(v).ok()).unwrap_or_default()
    };
    let mut hashes = load(&hash_path);
    let mut sigs = load(&sig_path);
    let want: Vec<(String, PathBuf)> = paths.iter()
        .filter(|p| !hashes.contains_key(*p) || !sigs.contains_key(*p))
        .filter_map(|p| resolve(app, id, p).ok().map(|f| (p.clone(), f)))
        .collect();
    if !want.is_empty() {
        let bin = crate::recorder::recorder_path()?;
        let mut child = std::process::Command::new(bin).arg("hash")
            .stdin(std::process::Stdio::piped()).stdout(std::process::Stdio::piped()).spawn()
            .map_err(|e| e.to_string())?;
        {
            use std::io::Write;
            let mut stdin = child.stdin.take().ok_or("no stdin")?;
            for (_, f) in &want {
                writeln!(stdin, "{}", f.to_string_lossy()).map_err(|e| e.to_string())?;
            }
        }
        let out = child.wait_with_output().map_err(|e| e.to_string())?;
        let line: Value = serde_json::from_slice(out.stdout.split(|b| *b == b'\n').next().unwrap_or_default())
            .map_err(|e| format!("the fingerprints could not be read: {e}"))?;
        for (key, f) in &want {
            let at = f.to_string_lossy();
            if let Some(h) = line["hashes"][at.as_ref()].as_str() {
                hashes.insert(key.clone(), h.to_string());
            }
            if let Some(g) = line["sigs"][at.as_ref()].as_str() {
                sigs.insert(key.clone(), g.to_string());
            }
        }
        write_atomic(&hash_path, serde_json::to_string(&hashes).unwrap().as_bytes())?;
        write_atomic(&sig_path, serde_json::to_string(&sigs).unwrap().as_bytes())?;
    }
    let pick = |m: &HashMap<String, String>| paths.iter().filter_map(|p| m.get(p).map(|v| (p.clone(), v.clone()))).collect();
    Ok((pick(&hashes), pick(&sigs)))
}

/* ── ways in ───────────────────────────────────────────────────────────── */

fn pick_folder(app: &AppHandle, title: &str) -> Option<PathBuf> {
    use tauri_plugin_dialog::DialogExt;
    app.dialog().file().set_title(title).blocking_pick_folder().and_then(|f| f.into_path().ok())
}

fn read_text(p: &Path) -> Option<String> {
    std::fs::read_to_string(p).ok().map(|s| s.trim().to_string()).filter(|s| !s.is_empty())
}

fn files_in(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir).map(|rd| rd.filter_map(|e| e.ok()).map(|e| e.path())
        .filter(|p| p.is_file() && !p.file_name().is_some_and(|n| n.to_string_lossy().starts_with('.'))).collect())
        .unwrap_or_default();
    v.sort();
    v
}

/// Rows from a folder, in either of the two layouts Flight School has always
/// read: `labels.json` naming each file's answer, or one subfolder per answer.
/// A picture is kept where it is (its path is absolute); a text file's words
/// become the input.
pub fn rows_from_folder(root: &Path) -> (Vec<Value>, Vec<String>) {
    let mut rows = Vec::new();
    let mut problems = Vec::new();
    let mut push = |file: &Path, answer: &str| {
        if is_image(file) {
            rows.push(json!({ "input": { "kind": "image", "path": file.to_string_lossy() }, "answer": answer }));
        } else if let Some(text) = read_text(file) {
            rows.push(json!({ "input": { "kind": "text", "text": text }, "answer": answer }));
        }
    };
    if let Some(v) = read_json(&root.join("labels.json")) {
        let entries = v.get("images").cloned().unwrap_or_else(|| if v.is_array() { v.clone() } else { json!([]) });
        for e in entries.as_array().cloned().unwrap_or_default() {
            let (file, label) = (e["file"].as_str().unwrap_or(""), e["label"].as_str().unwrap_or(""));
            if file.is_empty() || label.is_empty() {
                continue;
            }
            let f = root.join(file);
            if f.is_file() { push(&f, label) } else { problems.push(format!("labels.json names a missing file: {file}")) }
        }
        return (rows, problems);
    }
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(root).map(|rd| rd.filter_map(|e| e.ok()).map(|e| e.path())
        .filter(|p| p.is_dir() && !p.file_name().is_some_and(|n| n.to_string_lossy().starts_with('.'))).collect())
        .unwrap_or_default();
    dirs.sort();
    for d in dirs {
        let label = d.file_name().unwrap().to_string_lossy().into_owned();
        for f in files_in(&d) {
            push(&f, &label);
        }
    }
    if rows.is_empty() {
        problems.push("no labels.json and no subfolders of examples — one folder per answer, or a labels.json".into());
    }
    (rows, problems)
}

#[tauri::command]
pub async fn table_import_folder(app: AppHandle) -> Result<Option<Value>, String> {
    let app2 = app.clone();
    let picked = tauri::async_runtime::spawn_blocking(move || pick_folder(&app2, "Choose a folder of examples"))
        .await.map_err(|e| e.to_string())?;
    let Some(root) = picked else { return Ok(None) };
    let (rows, problems) = rows_from_folder(&root);
    Ok(Some(json!({ "folder": root.file_name().map(|n| n.to_string_lossy().into_owned()), "rows": rows, "problems": problems })))
}

/// Two folders matched by file name (without extension): what went in on
/// one side, what was written from it on the other. A file with no partner
/// is named, never guessed at.
pub fn rows_from_pair(inputs: &Path, answers: &Path) -> Value {
    let stem = |p: &Path| p.file_stem().map(|s| s.to_string_lossy().to_lowercase()).unwrap_or_default();
    let right: HashMap<String, PathBuf> = files_in(answers).into_iter().map(|p| (stem(&p), p)).collect();
    let mut rows = Vec::new();
    let mut alone = Vec::new();
    let mut used = HashSet::new();
    for f in files_in(inputs) {
        let Some(a) = right.get(&stem(&f)) else { alone.push(f.file_name().unwrap().to_string_lossy().into_owned()); continue };
        let Some(answer) = read_text(a) else { alone.push(a.file_name().unwrap().to_string_lossy().into_owned()); continue };
        used.insert(stem(&f));
        let input = if is_image(&f) {
            json!({ "kind": "image", "path": f.to_string_lossy() })
        } else if let Some(t) = read_text(&f) {
            json!({ "kind": "text", "text": t })
        } else {
            alone.push(f.file_name().unwrap().to_string_lossy().into_owned());
            continue;
        };
        rows.push(json!({ "input": input, "answer": answer }));
    }
    for (s, p) in &right {
        if !used.contains(s) {
            alone.push(p.file_name().unwrap().to_string_lossy().into_owned());
        }
    }
    alone.sort();
    json!({ "rows": rows, "alone": alone })
}

#[tauri::command]
pub async fn table_pair_files(app: AppHandle) -> Result<Option<Value>, String> {
    let app2 = app.clone();
    let picked = tauri::async_runtime::spawn_blocking(move || {
        let a = pick_folder(&app2, "First: the folder of what goes in")?;
        let b = pick_folder(&app2, "Then: the folder of what you wrote from each")?;
        Some((a, b))
    }).await.map_err(|e| e.to_string())?;
    let Some((a, b)) = picked else { return Ok(None) };
    Ok(Some(rows_from_pair(&a, &b)))
}

/* ── a play, back into the table ───────────────────────────────────────── */

/// A Pigeon's play log (Pigeon's `plays/<pigeon>/<started>/`: `play.json`,
/// `log.jsonl` and each turn's frame) as rows for the table: every turn the
/// Pigeon answered, with the frame it saw and what it said. The frames are
/// COPIED into this run (`plays/<started>/`), so the table does not depend on
/// Pigeon keeping them. The page marks the rows as drafts — a moment nobody
/// looked at is only the Pigeon's guess.
pub fn rows_from_play(log_dir: &Path, run: &Path) -> Result<Value, String> {
    let play = read_json(&log_dir.join("play.json")).ok_or("that folder is not a Pigeon play log — it has no play.json")?;
    let log = std::fs::read_to_string(log_dir.join("log.jsonl")).map_err(|_| "the play log has no log.jsonl".to_string())?;
    let stamp = log_dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "play".into());
    if stamp.contains(['/', '\\']) || stamp.starts_with('.') {
        return Err("not a play log".into());
    }
    let into = run.join("plays").join(&stamp);
    std::fs::create_dir_all(&into).map_err(|e| e.to_string())?;
    let mut rows = Vec::new();
    for line in log.lines() {
        let Ok(t) = serde_json::from_str::<Value>(line) else { continue };
        let (Some(frame), Some(answer)) = (t["frame"].as_str(), t["answer"].as_str()) else { continue };
        if frame.contains(['/', '\\']) || frame.starts_with('.') {
            continue;
        }
        let from = log_dir.join(frame);
        if !from.is_file() {
            continue;
        }
        std::fs::copy(&from, into.join(frame)).map_err(|e| e.to_string())?;
        rows.push(json!({ "input": { "kind": "image", "path": format!("plays/{stamp}/{frame}") },
                          "answer": answer.trim(), "turn": t["turn"], "on_list": t["on_list"] }));
    }
    Ok(json!({ "rows": rows, "pigeon": play["name"], "started_ms": play["started_ms"], "watch_only": play["watch_only"] }))
}

#[tauri::command]
pub async fn table_import_play(app: AppHandle, id: String) -> Result<Option<Value>, String> {
    use tauri_plugin_dialog::DialogExt;
    let run = run_dir(&app, &id)?;
    // Pigeon keeps its play logs in its own data folder; start there.
    let start = crate::pigeonstore::pigeon_root().map(|r| r.join("plays"));
    let app2 = app.clone();
    let picked = tauri::async_runtime::spawn_blocking(move || {
        let mut d = app2.dialog().file().set_title("Choose a play log — a folder in Pigeon's plays");
        if let Some(s) = start.filter(|p| p.is_dir()) {
            d = d.set_directory(s);
        }
        d.blocking_pick_folder().and_then(|f| f.into_path().ok())
    }).await.map_err(|e| e.to_string())?;
    let Some(dir) = picked else { return Ok(None) };
    rows_from_play(&dir, &run).map(Some)
}

/* ── teaching ──────────────────────────────────────────────────────────── */

/// The base a table trains on: Qwen3-VL-2B Instruct for text and pictures
/// alike — trained on its bf16 weights, served from the Q4_K_M GGUF, and for
/// pictures with the Q8_0 projector. One base, so one download serves every
/// Pigeon (spec registry: qwen3-vl-2b-instruct). Text moved 2026-10-02; on the
/// Constitution Reader it was not worse than Qwen2.5-1.5B on any check that
/// needs no judge (right section 41 vs 35 of 48, cut-off answers 2 vs 11).
///
/// bf16 for the reason text always used it (quickbuild.rs,
/// `precision_rank`): an adapter fitted on a 4-bit base absorbs that base's
/// rounding and loses it when served. Measured on Seed Run, 200 unseen boards
/// (base bake-off, 2026-10-02): trained 4-bit, 169 right in MLX and 139 served
/// from Q4_K_M; trained bf16, 165 in MLX and 164–166 served from F16, Q8_0 and
/// Q4_K_M alike. Hence the smallest serving pair: the base 1.1 GB, the projector
/// 0.45 GB. A reasoning Pigeon trained 4-bit fell from 166 to 4 when served.
fn bases(app: &AppHandle, vision: bool) -> Result<(String, PathBuf, Option<PathBuf>), String> {
    let models = paths::models_dir(app)?;
    if vision {
        let gguf = models.join("Qwen3VL-2B-Instruct-Q4_K_M.gguf");
        let proj = crate::pigeonstore::projectors().ok()
            .map(|d| d.join("mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf"))
            .filter(|p| p.is_file());
        if !gguf.is_file() || proj.is_none() {
            return Err("picture jobs need Qwen3-VL-2B and its projector in Pigeon's models folder".into());
        }
        return Ok(("mlx-community/Qwen3-VL-2B-Instruct-bf16".into(), gguf, proj));
    }
    let gguf = models.join("Qwen3VL-2B-Instruct-Q4_K_M.gguf");
    if !gguf.is_file() {
        return Err("text jobs need Qwen3-VL-2B in Pigeon's models folder".into());
    }
    Ok(("mlx-community/Qwen3-VL-2B-Instruct-bf16".into(), gguf, None))
}

/* ── what a picture Pigeon presses ─────────────────────────────────────── */

/// A key's macOS key code from the name the recorder gave it — for keys
/// recorded before codes were kept. The US layout; a key it does not know is
/// left without a code, and play will not press it.
pub fn key_code(name: &str) -> Option<i64> {
    const NAMED: [(&str, i64); 29] = [
        ("Return", 36), ("Enter", 76), ("Tab", 48), ("Space", 49), ("Backspace", 51), ("Delete", 117), ("Escape", 53),
        ("Left", 123), ("Right", 124), ("Down", 125), ("Up", 126), ("Home", 115), ("End", 119), ("PageUp", 116), ("PageDown", 121),
        ("F1", 122), ("F2", 120), ("F3", 99), ("F4", 118), ("F5", 96), ("F6", 97), ("F7", 98), ("F8", 100), ("F9", 101),
        ("F10", 109), ("F11", 103), ("F12", 111), ("-", 27), ("=", 24),
    ];
    if let Some((_, c)) = NAMED.iter().find(|(n, _)| *n == name) {
        return Some(*c);
    }
    let table: [(char, i64); 36] = [
        ('A', 0), ('S', 1), ('D', 2), ('F', 3), ('H', 4), ('G', 5), ('Z', 6), ('X', 7), ('C', 8), ('V', 9), ('B', 11),
        ('Q', 12), ('W', 13), ('E', 14), ('R', 15), ('Y', 16), ('T', 17), ('1', 18), ('2', 19), ('3', 20), ('4', 21),
        ('6', 22), ('5', 23), ('9', 25), ('7', 26), ('8', 28), ('0', 29), ('O', 31), ('U', 32), ('I', 34), ('P', 35),
        ('L', 37), ('J', 38), ('K', 40), ('N', 45), ('M', 46),
    ];
    let mut it = name.chars();
    match (it.next(), it.next()) {
        (Some(c), None) => table.iter().find(|(k, _)| *k == c.to_ascii_uppercase()).map(|(_, v)| *v),
        _ => None,
    }
}

/// `actions.json`, the part of a picture Pigeon that says what to PRESS
/// (Pigeon's docs/PLAY-LOOP.md): each key answer's key code, the key's name,
/// and how long you usually held it (the median of your own presses). A key
/// recorded with ⌘, Ctrl or Option is never packed as something to press —
/// ⌘Q would quit the game — and is listed under `not_pressed` instead.
pub fn actions_file(a: &Value) -> Value {
    let mut keys = serde_json::Map::new();
    let mut not_pressed = Vec::new();
    for k in a["keys"].as_array().into_iter().flatten() {
        let answer = k["answer"].as_str().unwrap_or("");
        let Some(inner) = answer.strip_prefix("KEY(").and_then(|x| x.strip_suffix(')')) else { continue };
        let raw = k["raw"].as_str().unwrap_or(inner);
        if raw.contains("Cmd+") || raw.contains("Ctrl+") || raw.contains("Option+") {
            not_pressed.push(json!({ "answer": answer, "key": raw, "why": "held with ⌘, Ctrl or Option" }));
            continue;
        }
        let Some(code) = k["code"].as_i64().or_else(|| key_code(raw)) else {
            not_pressed.push(json!({ "answer": answer, "key": raw, "why": "no key code was recorded for it" }));
            continue;
        };
        let mut holds: Vec<u64> = k["holds"].as_array().into_iter().flatten().filter_map(|h| h.as_u64()).collect();
        holds.sort_unstable();
        let hold = holds.get(holds.len() / 2).copied();
        keys.insert(inner.to_string(), json!({ "code": code, "name": raw, "hold_ms": hold }));
    }
    json!({
        "format_version": 1,
        "keys": keys,
        "not_pressed": not_pressed,
        "frames": a["frames"].clone(),
        "question": a["question"].clone(),
    })
}

fn write_actions(dir: &Path, a: Option<&Value>) -> Result<(), String> {
    let path = dir.join("actions.json");
    match a.filter(|a| !a.is_null()) {
        Some(a) => write_atomic(&path, serde_json::to_string_pretty(&actions_file(a)).unwrap().as_bytes()),
        None => {
            let _ = std::fs::remove_file(path);
            Ok(())
        }
    }
}

/// Copies of the pictures with their longer side at most `side` pixels, in
/// `.sized/<side>/` under the run — the exam's too, so the check ride grades
/// the size Pigeon will see. Made in the recorder's own order (shrink, then
/// cut) and saved as PNG, so a JPEG is never compressed a second time. A copy
/// keeps its picture's last folders and its whole name, with `.png` added
/// (`board.jpg` → `board.jpg.png`), so `board.jpg` and `board.png` in one
/// folder never share a copy, and the check ride finds the original by taking
/// the `.png` off (engine.js rideMatcher). Made once; a copy newer than its
/// picture is reused.
fn shrink_pictures(dir: &Path, side: u32, crop: Option<&str>, pictures: Vec<PathBuf>) -> Result<HashMap<PathBuf, PathBuf>, String> {
    let name = match crop { Some(c) => format!("{side}-crop-{}", c.replace(',', "_")), None => side.to_string() };
    let root = dir.join(".sized").join(name);
    let mut map = HashMap::new();
    let mut todo = String::new();
    for p in pictures {
        if map.contains_key(&p) { continue; }
        let inside = p.strip_prefix(dir).map(Path::to_path_buf)
            .unwrap_or_else(|_| p.components().filter(|c| matches!(c, std::path::Component::Normal(_))).collect());
        let mut to = root.join(inside).into_os_string();
        to.push(".png");
        let to = PathBuf::from(to);
        let fresh = |f: &Path| std::fs::metadata(f).and_then(|m| m.modified()).ok();
        if !(to.is_file() && fresh(&to) >= fresh(&p)) {
            todo.push_str(&format!("{}\t{}\n", p.to_string_lossy(), to.to_string_lossy()));
        }
        map.insert(p, to);
    }
    if !todo.is_empty() {
        let mut args = vec!["resize".to_string(), "--max-side".to_string(), side.to_string()];
        if let Some(c) = crop {
            args.extend(["--crop".to_string(), c.to_string()]);
        }
        let mut child = std::process::Command::new(crate::recorder::recorder_path()?)
            .args(&args)
            .stdin(std::process::Stdio::piped()).stdout(std::process::Stdio::piped()).spawn()
            .map_err(|e| e.to_string())?;
        {
            use std::io::Write;
            child.stdin.take().ok_or("no stdin")?.write_all(todo.as_bytes()).map_err(|e| e.to_string())?;
        }
        let out = child.wait_with_output().map_err(|e| e.to_string())?;
        let line: Value = serde_json::from_slice(out.stdout.split(|b| *b == b'\n').next().unwrap_or_default())
            .map_err(|e| format!("the pictures could not be made smaller: {e}"))?;
        let failed: Vec<&str> = line["failed"].as_array().into_iter().flatten().filter_map(|f| f.as_str()).collect();
        if let Some(first) = failed.first() {
            return Err(format!("{} {} could not be made smaller, starting with {first}", failed.len(), if failed.len() == 1 { "picture" } else { "pictures" }));
        }
    }
    Ok(map)
}

/// Everything a table run needs on disk to train, written from what the page
/// decided: `rows.jsonl` (what trains), `exam.jsonl` (the exam, pinned),
/// `plan.json`, `base.json`, `config.toml`, `labels.txt`.
pub fn write_teach_files(dir: &Path, t: &Value, repo: &str, gguf: &Path) -> Result<Value, String> {
    let vision = t["input"] == "image";
    let rows = t["rows"].as_array().ok_or("no rows")?;
    let exam = t["exam"].as_array().ok_or("no exam")?;
    let labels: Vec<String> = t["labels"].as_array().into_iter().flatten().filter_map(|l| l.as_str().map(String::from)).collect();
    let reasoned = t["reasoning"].as_bool().unwrap_or(false);
    let prose = t["output"] == "prose";
    let full = |img: &str| if Path::new(img).is_absolute() { PathBuf::from(img) } else { dir.join(img) };
    // A crop (fractions of each picture: x, y, w, h) and a size, when the page
    // asks to train on pictures cut or shrunk from the ones in the table.
    let crop: Option<String> = t["crop"].as_array().map(|c| c.iter().filter_map(|v| v.as_f64()).collect::<Vec<_>>())
        .filter(|c| c.len() == 4).map(|c| c.iter().map(|v| format!("{v:.4}")).collect::<Vec<_>>().join(","));
    let side = t["max_side"].as_u64().map(|s| s as u32);
    // A recorded table trains on its pictures as recorded: the recorder made
    // them in one pass, the way Pigeon makes the ones it plays from (spec
    // §13.2). Cut or shrunk again here they would be pictures Pigeon never sees.
    if vision && t["recorded"].as_bool() == Some(true) && (side.is_some() || crop.is_some()) {
        return Err("these pictures were recorded, so they train as recorded — record a new table for another part or size".into());
    }
    let sized = if vision && (side.is_some() || crop.is_some()) {
        shrink_pictures(dir, side.unwrap_or(4096), crop.as_deref(), rows.iter().chain(exam.iter()).filter_map(|r| r["image"].as_str()).map(full).collect())?
    } else {
        HashMap::new()
    };
    let line = |r: &Value| -> Result<String, String> {
        let mut o = json!({ "question": r["question"], "answer": r["answer"], "group": r["group"] });
        // A Reader's rows say whether the passage answers the question, so the
        // check ride can count refusals apart and skip them for the fact swap.
        if let Some(b) = r["refusal"].as_bool() { o["refusal"] = json!(b); }
        if r["shown"].is_array() { o["shown"] = r["shown"].clone(); }
        if let Some(img) = r["image"].as_str() {
            let p = full(img);
            o["image"] = json!(sized.get(&p).unwrap_or(&p).to_string_lossy());
        }
        Ok(o.to_string() + "\n")
    };
    let body = |list: &[Value]| -> Result<String, String> { list.iter().map(line).collect() };
    write_atomic(&dir.join("rows.jsonl"), body(rows)?.as_bytes())?;
    write_atomic(&dir.join("exam.jsonl"), body(exam)?.as_bytes())?;
    if !labels.is_empty() {
        write_atomic(&dir.join("labels.txt"), format!("{}\n", labels.join("\n")).as_bytes())?;
    }

    let mut groups: Vec<(String, u64)> = Vec::new();
    for r in rows.iter().chain(exam.iter()) {
        let g = r["group"].as_str().unwrap_or("rows").to_string();
        match groups.iter_mut().find(|(n, _)| *n == g) { Some(x) => x.1 += 1, None => groups.push((g, 1)) }
    }
    let longest = rows.iter().chain(exam.iter()).map(|r| r["answer"].as_str().unwrap_or("").len()).max().unwrap_or(0);
    let name = t["name"].as_str().unwrap_or("job");
    let mut plan = json!({
        "job_name": name,
        "one_line": t["does"].as_str().filter(|s| !s.trim().is_empty()).unwrap_or("Learned from a table of examples."),
        "cartridge_kind": "behavior",
        "knowledge_file": null,
        "modality": if vision { "vision" } else { "text" },
        "image_corpus": if vision { Value::String(dir.to_string_lossy().into_owned()) } else { Value::Null },
        "input_shape": if vision { "one picture" } else { "one piece of text" },
        "output_shape": if prose || reasoned { "the answer, written the way the table writes it".to_string() }
                        else { format!("exactly one of: {}", labels.join(", ")) },
        "style_rules": ["Answer exactly the way the table does."],
        "question_groups": groups.iter().map(|(n, c)| json!({ "name": n, "count": c, "purpose": "from the table" })).collect::<Vec<_>>(),
        "example_questions": [rows.first().and_then(|r| r["question"].as_str()).unwrap_or("")],
        "grading_rubric": ["Matches the answer in the table."],
        "holdout_count": exam.len(),
        "max_answer_tokens": (longest / 3 + 8).clamp(8, 2048),
        "preset": t["preset"].as_str().unwrap_or("standard"),
        // A Reader: the ride adds the fact swap (quickbuild.py stage_fly).
        "grounded": t["grounded"].as_bool().unwrap_or(false),
        "kind": t["kind"],
    });
    if reasoned {
        plan["grader"] = json!({ "how": "rules", "rule": "instructor.rules_table:reasoned" });
    }
    write_atomic(&dir.join("plan.json"), serde_json::to_string_pretty(&plan).unwrap().as_bytes())?;
    // Only a recorded table plays (spec §13.2, D71): imported pictures were not
    // made by a capture pass, so nothing could make a matching picture live.
    write_actions(dir, t.get("actions").filter(|_| t["recorded"].as_bool() == Some(true)))?;
    write_atomic(&dir.join("base.json"), serde_json::to_string_pretty(&json!({
        "student_repo": repo, "student_gguf": gguf.to_string_lossy(), "teacher_gguf": null })).unwrap().as_bytes())?;
    Ok(plan)
}

/// Train from the table and give it the check ride — one job, in the
/// background, reporting through `qb-event` like every Quick Build run.
/// `t` is what the page decided: name, input, output, reasoning, labels,
/// preset, and the rows and exam as `{question, answer, group, image?}`.
#[tauri::command]
pub async fn table_teach(app: AppHandle, running: State<'_, RunningJob>, id: String, t: Value) -> Result<(), String> {
    let dir = run_dir(&app, &id)?;
    let vision = t["input"] == "image";
    if vision && !crate::quickbuild::vision_ready(&app) {
        return Err("Picture training needs its own tools. Set them up once from Teach, or from the This Mac row in the rail.".into());
    }
    let (repo, gguf, projector) = bases(&app, vision)?;
    write_teach_files(&dir, &t, &repo, &gguf)?;
    let planner = crate::quickbuild::planner_for_judge(&app).unwrap_or_else(|_| gguf.clone());
    crate::quickbuild::write_run_config(&app, &dir, &planner, &gguf, &repo, projector.as_deref())?;
    let _ = std::fs::remove_file(dir.join("export").join("check-ride.json"));
    crate::quickbuild::drive_run(app, running, &id, "table").await
}

/// The last check ride's results, as the pipeline wrote them.
#[tauri::command]
pub fn table_ride(app: AppHandle, id: String) -> Result<Option<Value>, String> {
    Ok(read_json(&run_dir(&app, &id)?.join("export").join("check-ride.json")))
}

/* ── sending ───────────────────────────────────────────────────────────── */

/// Where a PNG the person chose for the icon is kept: copied into the run, so
/// it is still there if the original moves, and replaced by the next one chosen.
fn chosen_icon(dir: &Path) -> PathBuf {
    dir.join("export").join("icon-chosen.png")
}

/// What the Send screen draws the icon from: the name's random look, the parts
/// table to draw any look with, and the PNG chosen for this run, if any.
#[tauri::command]
pub async fn table_look_kit(app: AppHandle, id: String, name: String) -> Result<Value, String> {
    let dir = run_dir(&app, &id)?;
    let mut args = vec!["look-kit".to_string(), "--name".into(), name];
    let icon = chosen_icon(&dir);
    if icon.is_file() {
        args.push("--icon".into());
        args.push(icon.to_string_lossy().into_owned());
    }
    crate::quickbuild::pigeon_cli(&app, &args).await
}

/// "Choose a PNG…": the icon the person would rather have than a drawn one.
/// Checked to be a PNG and not huge, then copied into the run. Answers with
/// the same kit as `table_look_kit`, so the page gets the picture to show.
#[tauri::command]
pub async fn table_pick_icon(app: AppHandle, id: String, name: String) -> Result<Option<Value>, String> {
    use tauri_plugin_dialog::DialogExt;
    let dir = run_dir(&app, &id)?;
    let app2 = app.clone();
    let picked = tauri::async_runtime::spawn_blocking(move || app2.dialog().file().set_title("Choose an icon (PNG)")
        .add_filter("PNG image", &["png"]).blocking_pick_file())
        .await.map_err(|e| e.to_string())?;
    let Some(path) = picked.and_then(|f| f.into_path().ok()) else { return Ok(None) };
    let bytes = std::fs::read(&path).map_err(|e| format!("Could not read {}: {e}", path.display()))?;
    if !bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        return Err("That file is not a PNG. Choose a .png — a square one shows best.".into());
    }
    if bytes.len() > 512 * 1024 {
        return Err("That PNG is over 512 KB. A 512 × 512 icon is plenty — export a smaller one and choose it again.".into());
    }
    std::fs::create_dir_all(dir.join("export")).map_err(|e| e.to_string())?;
    std::fs::write(chosen_icon(&dir), &bytes).map_err(|e| e.to_string())?;
    table_look_kit(app, id, name).await.map(Some)
}

/// What the `.pigeon` would hold, measured from the files that would go in,
/// by the packer's own rule for the projector (python/pigeon/pack.py
/// `file_parts`): the page and the packed file never disagree about whether
/// the projector is carried. Also the base the run trained against, by name.
#[tauri::command]
pub async fn table_parts(app: AppHandle, id: String) -> Result<Value, String> {
    let dir = run_dir(&app, &id)?;
    let args = vec![
        "parts".to_string(),
        "--run-dir".into(), dir.to_string_lossy().into_owned(),
        "--models".into(), paths::models_dir(&app)?.to_string_lossy().into_owned(),
    ];
    crate::quickbuild::pigeon_cli(&app, &args).await
}

/// Every copy of Pigeon on this Mac, by its bundle id, as Spotlight knows
/// them, plus the usual place in case Spotlight is off.
fn pigeon_apps() -> Vec<PathBuf> {
    let mut apps: Vec<PathBuf> = std::process::Command::new("mdfind")
        .arg(format!("kMDItemCFBundleIdentifier == '{}'", crate::pigeonstore::PIGEON_APP_ID))
        .output().ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).lines().map(PathBuf::from).collect())
        .unwrap_or_default();
    let usual = PathBuf::from("/Applications/Pigeon.app");
    if usual.is_dir() && !apps.contains(&usual) {
        apps.push(usual);
    }
    apps.into_iter().filter(|a| a.is_dir()).collect()
}

/// When an app was built: the newest file in its Contents/MacOS.
fn built_at(app: &Path) -> Option<std::time::SystemTime> {
    std::fs::read_dir(app.join("Contents").join("MacOS")).ok()?
        .filter_map(|e| e.ok()?.metadata().ok()?.modified().ok())
        .max()
}

/// The newest-built of several copies. Opening by bundle id lets macOS pick
/// any of them, and it once picked a stale /Applications copy that refused a
/// file the current Pigeon reads.
fn newest(apps: Vec<(PathBuf, Option<std::time::SystemTime>)>) -> Option<PathBuf> {
    apps.into_iter().max_by_key(|(_, t)| *t).map(|(p, _)| p)
}

/// Open `file` in the newest Pigeon on this Mac, and say which one.
fn open_in_pigeon(file: &Path) -> Result<PathBuf, String> {
    let app = newest(pigeon_apps().into_iter().map(|a| { let t = built_at(&a); (a, t) }).collect())
        .ok_or_else(|| format!("Packed, but Pigeon is not on this Mac to open it. The file is at {}", file.display()))?;
    let ok = std::process::Command::new("open").arg("-a").arg(&app).arg(file)
        .status().map(|s| s.success()).unwrap_or(false);
    if !ok {
        return Err(format!("Packed, but {} would not open it. The file is at {}", app.display(), file.display()));
    }
    Ok(app)
}

/// What happens after a Pigeon is packed, by button. "Save file…" only writes
/// the file where the person chose; "Try it in Pigeon" also adds it to the
/// lineup and opens it in Pigeon.
#[derive(Debug, PartialEq)]
enum AfterPack { AddToLineup, OpenInPigeon }

fn after_pack(to_pigeon: bool) -> &'static [AfterPack] {
    if to_pigeon { &[AfterPack::AddToLineup, AfterPack::OpenInPigeon] } else { &[] }
}

/// Pack the taught Pigeon. `to_pigeon` writes it into the run, adds it to the
/// lineup and opens it in Pigeon; otherwise a save dialog chooses where it
/// goes, and that is all (`after_pack`).
#[tauri::command]
pub async fn table_send(app: AppHandle, id: String, name: String, does: String, to_pigeon: bool, actions: Option<Value>,
                        look: Option<String>, icon: Option<bool>) -> Result<Option<Value>, String> {
    use tauri_plugin_dialog::DialogExt;
    let dir = run_dir(&app, &id)?;
    // What it presses is written as the table says now — renaming a key after
    // teaching renames the answer, not what it learned — so it is refreshed here.
    write_actions(&dir, actions.as_ref())?;
    let name = name.trim().to_string();
    if name.is_empty() {
        return Err("it needs a name".into());
    }
    let slug: String = name.to_lowercase().chars().map(|c| if c.is_alphanumeric() { c } else { '-' }).collect::<String>()
        .trim_matches('-').to_string();
    let slug = if slug.is_empty() { id.clone() } else { slug };
    let out = if to_pigeon {
        dir.join("export").join(format!("{slug}.pigeon"))
    } else {
        let home = app.path().home_dir().map(|h| h.join("Downloads")).unwrap_or(dir.clone());
        let app2 = app.clone();
        let file = format!("{slug}.pigeon");
        let chosen = tauri::async_runtime::spawn_blocking(move || app2.dialog().file().set_title("Save the Pigeon")
            .set_directory(&home).set_file_name(&file).add_filter("Pigeon", &["pigeon"]).blocking_save_file())
            .await.map_err(|e| e.to_string())?;
        let Some(p) = chosen.and_then(|f| f.into_path().ok()) else { return Ok(None) };
        p
    };
    let readme = dir.join("export").join("README.draft.md");
    std::fs::write(&readme, format!("# {name}\n\n{does}\n\nMade in Flight School from a table of examples.\n")).map_err(|e| e.to_string())?;
    let models = paths::models_dir(&app)?;
    let mut args = vec![
        "pack".to_string(),
        "--run-dir".into(), dir.to_string_lossy().into_owned(),
        "--out".into(), out.to_string_lossy().into_owned(),
        "--name".into(), name.clone(),
        "--readme-file".into(), readme.to_string_lossy().into_owned(),
        "--example-seed".into(), "11".into(),
        "--example-count".into(), "4".into(),
        "--trained-with".into(), format!("Flight School {}", app.package_info().version),
        "--models".into(), models.to_string_lossy().into_owned(),
    ];
    crate::quickbuild::push_card_args(&mut args, Some(does.clone()).filter(|d| !d.trim().is_empty()), None, None);
    // The icon the Send screen showed: the chosen PNG, or the look it drew.
    // Neither — an older page — and the packer draws the name's random look.
    let chosen = chosen_icon(&dir);
    if icon.unwrap_or(false) && chosen.is_file() {
        args.push("--icon".into()); args.push(chosen.to_string_lossy().into_owned());
    } else if let Some(look) = look.filter(|l| !l.trim().is_empty()) {
        args.push("--look".into()); args.push(look);
    }
    args.push("--store".into());
    args.push(paths::pigeons_dir(&app)?.to_string_lossy().into_owned());
    // A Reader's material is embedded at pack time (pigeon/embed.py) through
    // the app's own llama-server; without it the lexical search is packed and
    // the reply's warnings say so.
    if let Ok(server) = crate::quickbuild::llama_server_path() {
        args.push("--llama-server".into());
        args.push(server.to_string_lossy().into_owned());
    }
    let reply = crate::quickbuild::pigeon_cli(&app, &args).await?;
    if !reply["validation"]["ok"].as_bool().unwrap_or(false) {
        let _ = std::fs::remove_file(&out);
        let why = reply["validation"]["errors"].as_array()
            .map(|a| a.iter().filter_map(Value::as_str).collect::<Vec<_>>().join(" · ")).unwrap_or_default();
        return Err(format!("The packed file failed its own import check, so it was not kept: {why}"));
    }
    let mut opened_with: Option<PathBuf> = None;
    for step in after_pack(to_pigeon) {
        match step {
            AfterPack::AddToLineup => { let _ = crate::pigeons::add_to_store(&app, &out, "run", Some(&id)).await; }
            AfterPack::OpenInPigeon => {
                // Pigeon opens it the way Finder would: its own import, its own
                // meeting card, its own check that the base is there.
                opened_with = Some(open_in_pigeon(&out)?);
            }
        }
    }
    Ok(Some(json!({ "path": out.to_string_lossy(), "bytes": reply["bytes"], "manifest": reply["manifest"], "warnings": reply["warnings"],
                    "opened_with": opened_with.map(|p| p.to_string_lossy().into_owned()) })))
}

/* ── a Prompted Pigeon (spec D83): a base with a system prompt ──────────── */

/// The test picture a Prompted Pigeon asks about, copied into the run so the
/// table names a file it owns. Answers with the copy's path, or None.
#[tauri::command]
pub async fn table_base_picture(app: AppHandle, id: String) -> Result<Option<String>, String> {
    use tauri_plugin_dialog::DialogExt;
    let dir = run_dir(&app, &id)?.join("prompted").join("pictures");
    let app2 = app.clone();
    let picked = tauri::async_runtime::spawn_blocking(move || app2.dialog().file().set_title("Choose a test picture")
        .add_filter("Picture", &["png", "jpg", "jpeg"]).blocking_pick_file())
        .await.map_err(|e| e.to_string())?;
    let Some(path) = picked.and_then(|f| f.into_path().ok()) else { return Ok(None) };
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("png").to_lowercase();
    let n = std::fs::read_dir(&dir).map(|d| d.count()).unwrap_or(0) + 1;
    let copy = dir.join(format!("{n:03}.{ext}"));
    std::fs::copy(&path, &copy).map_err(|e| format!("Could not copy {}: {e}", path.display()))?;
    Ok(Some(copy.to_string_lossy().into_owned()))
}

/// Ride and pack a Prompted Pigeon: write its base table (python/pigeon/
/// basepack.py), ask its test questions through llama.cpp as a Runner would,
/// and pack the file with no adapter. `table` is what the page decided, less
/// the base, which is decided here. Then, as for a taught Pigeon, "Try it in
/// Pigeon" adds it to the lineup and opens it; "Save file…" only writes it.
#[tauri::command]
pub async fn table_base_pack(app: AppHandle, id: String, name: String, mut table: Value, rows: Vec<Value>,
                             to_pigeon: bool) -> Result<Option<Value>, String> {
    use tauri_plugin_dialog::DialogExt;
    let dir = run_dir(&app, &id)?;
    let tdir = dir.join("prompted");
    std::fs::create_dir_all(&tdir).map_err(|e| e.to_string())?;
    let name = name.trim().to_string();
    if name.is_empty() {
        return Err("it needs a name".into());
    }
    let looks = table["prompt_template"].as_str().unwrap_or("").contains("{image}");
    let (repo, gguf, projector) = bases(&app, looks)?;
    table["base_gguf"] = json!(gguf.to_string_lossy());
    if let Some(p) = &projector {
        table["mmproj"] = json!(p.to_string_lossy());
    }
    table["name"] = json!(name);
    write_atomic(&tdir.join("base-table.json"), serde_json::to_string_pretty(&table).unwrap().as_bytes())?;
    let lines: String = rows.iter().map(|r| format!("{}\n", r)).collect();
    write_atomic(&tdir.join("rows.jsonl"), lines.as_bytes())?;
    let planner = crate::quickbuild::planner_for_judge(&app).unwrap_or_else(|_| gguf.clone());
    crate::quickbuild::write_run_config(&app, &tdir, &planner, &gguf, &repo, projector.as_deref())?;
    let slug: String = name.to_lowercase().chars().map(|c| if c.is_alphanumeric() { c } else { '-' }).collect::<String>()
        .trim_matches('-').to_string();
    let slug = if slug.is_empty() { id.clone() } else { slug };
    let out = if to_pigeon {
        dir.join("export").join(format!("{slug}.pigeon"))
    } else {
        let home = app.path().home_dir().map(|h| h.join("Downloads")).unwrap_or(dir.clone());
        let app2 = app.clone();
        let file = format!("{slug}.pigeon");
        let chosen = tauri::async_runtime::spawn_blocking(move || app2.dialog().file().set_title("Save the Pigeon")
            .set_directory(&home).set_file_name(&file).add_filter("Pigeon", &["pigeon"]).blocking_save_file())
            .await.map_err(|e| e.to_string())?;
        let Some(p) = chosen.and_then(|f| f.into_path().ok()) else { return Ok(None) };
        p
    };
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let args = vec![
        "base-pack".to_string(),
        "--table".into(), tdir.to_string_lossy().into_owned(),
        "--out".into(), out.to_string_lossy().into_owned(),
        "--store".into(), paths::pigeons_dir(&app)?.to_string_lossy().into_owned(),
        "--models".into(), paths::models_dir(&app)?.to_string_lossy().into_owned(),
    ];
    let reply = crate::quickbuild::pigeon_cli(&app, &args).await?;
    if let Some(e) = reply["error"].as_str() {
        return Err(e.to_string());
    }
    if !reply["validation"]["ok"].as_bool().unwrap_or(false) {
        let _ = std::fs::remove_file(&out);
        let why = reply["validation"]["errors"].as_array()
            .map(|a| a.iter().filter_map(Value::as_str).collect::<Vec<_>>().join(" · ")).unwrap_or_default();
        return Err(format!("The packed file failed its own import check, so it was not kept: {why}"));
    }
    let mut opened_with: Option<PathBuf> = None;
    for step in after_pack(to_pigeon) {
        match step {
            AfterPack::AddToLineup => { let _ = crate::pigeons::add_to_store(&app, &out, "run", Some(&id)).await; }
            AfterPack::OpenInPigeon => { opened_with = Some(open_in_pigeon(&out)?); }
        }
    }
    Ok(Some(json!({ "path": out.to_string_lossy(), "bytes": reply["bytes"], "manifest": reply["manifest"],
                    "check_ride": reply["check_ride"], "opened_with": opened_with.map(|p| p.to_string_lossy().into_owned()) })))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("fs-table-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn the_newest_built_pigeon_is_the_one_opened() {
        use std::time::{Duration, UNIX_EPOCH};
        let at = |s| Some(UNIX_EPOCH + Duration::from_secs(s));
        let picked = newest(vec![(PathBuf::from("/Applications/Pigeon.app"), at(100)),
                                 (PathBuf::from("/dev/Pigeon.app"), at(300)),
                                 (PathBuf::from("/Downloads/Pigeon-old.app"), at(200)),
                                 (PathBuf::from("/broken/Pigeon.app"), None)]);
        assert_eq!(picked, Some(PathBuf::from("/dev/Pigeon.app")));
        assert_eq!(newest(vec![]), None);
    }

    #[test]
    fn a_folder_of_subfolders_is_one_answer_per_subfolder() {
        let d = tmp("folder");
        std::fs::create_dir_all(d.join("boxer")).unwrap();
        std::fs::create_dir_all(d.join("beagle")).unwrap();
        std::fs::write(d.join("boxer/a.jpg"), b"x").unwrap();
        std::fs::write(d.join("beagle/b.txt"), b"  a note  ").unwrap();
        std::fs::write(d.join("beagle/.DS_Store"), b"x").unwrap();
        let (rows, problems) = rows_from_folder(&d);
        assert!(problems.is_empty());
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0]["answer"], "beagle");
        assert_eq!(rows[0]["input"], json!({ "kind": "text", "text": "a note" }));
        assert_eq!(rows[1]["input"]["kind"], "image");
        assert!(Path::new(rows[1]["input"]["path"].as_str().unwrap()).is_absolute());
    }

    #[test]
    fn labels_json_names_each_files_answer_and_a_missing_file_is_named() {
        let d = tmp("labels");
        std::fs::write(d.join("x.png"), b"x").unwrap();
        std::fs::write(d.join("labels.json"), br#"{"images":[{"file":"x.png","label":"cat"},{"file":"gone.png","label":"dog"}]}"#).unwrap();
        let (rows, problems) = rows_from_folder(&d);
        assert_eq!(rows.len(), 1);
        assert_eq!(problems, vec!["labels.json names a missing file: gone.png".to_string()]);
    }

    #[test]
    fn paired_files_match_by_name_and_a_file_alone_is_named() {
        let d = tmp("pair");
        let (a, b) = (d.join("statements"), d.join("articles"));
        std::fs::create_dir_all(&a).unwrap();
        std::fs::create_dir_all(&b).unwrap();
        std::fs::write(a.join("q2.txt"), "the statement").unwrap();
        std::fs::write(b.join("q2.md"), "the article").unwrap();
        std::fs::write(a.join("q3.txt"), "no partner").unwrap();
        std::fs::write(b.join("q4.md"), "no partner either").unwrap();
        let v = rows_from_pair(&a, &b);
        assert_eq!(v["rows"], json!([{ "input": { "kind": "text", "text": "the statement" }, "answer": "the article" }]));
        assert_eq!(v["alone"], json!(["q3.txt", "q4.md"]));
    }

    #[test]
    fn the_teach_files_carry_the_page_s_rows_exam_and_answer_list() {
        let d = tmp("teach");
        let t = json!({
            "name": "dog-breeds", "input": "image", "output": "function", "reasoning": false,
            "labels": ["KEY(Z)", "NONE"], "preset": "quick",
            "rows": [{ "question": "What do you do next?", "answer": "KEY(Z)", "group": "KEY(Z)", "image": "recording/s1/f1.jpg" }],
            "exam": [{ "question": "What do you do next?", "answer": "NONE", "group": "NONE", "image": "/abs/f2.jpg" }]
        });
        let plan = write_teach_files(&d, &t, "repo", Path::new("/m/base.gguf")).unwrap();
        assert_eq!(plan["output_shape"], "exactly one of: KEY(Z), NONE");
        assert_eq!(plan["holdout_count"], 1);
        assert_eq!(plan["modality"], "vision");
        let rows: Value = serde_json::from_str(std::fs::read_to_string(d.join("rows.jsonl")).unwrap().trim()).unwrap();
        assert_eq!(rows["image"], json!(d.join("recording/s1/f1.jpg").to_string_lossy()), "relative frames become absolute");
        let exam: Value = serde_json::from_str(std::fs::read_to_string(d.join("exam.jsonl")).unwrap().trim()).unwrap();
        assert_eq!(exam["image"], "/abs/f2.jpg");
        assert_eq!(std::fs::read_to_string(d.join("labels.txt")).unwrap(), "KEY(Z)\nNONE\n");
        let reasoned = write_teach_files(&d, &json!({ "name": "n", "input": "text", "output": "function", "reasoning": true,
            "labels": ["A", "B"], "rows": [], "exam": [] }), "r", Path::new("/g")).unwrap();
        assert_eq!(reasoned["grader"]["rule"], "instructor.rules_table:reasoned");
    }

    #[test]
    fn a_smaller_picture_size_trains_and_grades_on_shrunk_copies() {
        let d = tmp("sized");
        std::fs::create_dir_all(d.join("recording/s1")).unwrap();
        let icon = Path::new(env!("CARGO_MANIFEST_DIR")).join("icons/128x128@2x.png");
        std::fs::copy(&icon, d.join("recording/s1/f1.png")).unwrap();
        let t = json!({
            "name": "n", "input": "image", "output": "function", "labels": ["KEY(Z)"], "max_side": 64,
            "rows": [{ "question": "q", "answer": "KEY(Z)", "group": "KEY(Z)", "image": "recording/s1/f1.png" }],
            "exam": [{ "question": "q", "answer": "KEY(Z)", "group": "KEY(Z)", "image": icon.to_string_lossy() }]
        });
        write_teach_files(&d, &t, "repo", Path::new("/m/base.gguf")).unwrap();
        let row: Value = serde_json::from_str(std::fs::read_to_string(d.join("rows.jsonl")).unwrap().trim()).unwrap();
        let shrunk = d.join(".sized/64/recording/s1/f1.png.png");
        assert!(std::fs::read(&shrunk).unwrap().starts_with(b"\x89PNG"), "a copy is a PNG");
        assert_eq!(row["image"], json!(shrunk.to_string_lossy()), "the row trains on the copy");
        let exam: Value = serde_json::from_str(std::fs::read_to_string(d.join("exam.jsonl")).unwrap().trim()).unwrap();
        let exam_img = exam["image"].as_str().unwrap();
        assert!(exam_img.starts_with(&d.join(".sized/64").to_string_lossy().to_string()), "the exam is graded at the same size");
        assert!(exam_img.ends_with("src-tauri/icons/128x128@2x.png.png"), "a copy keeps its picture's last folders and name");
        // board.jpg and board.png in one folder get a copy each.
        std::fs::copy(d.join("recording/s1/f1.png"), d.join("recording/s1/f1.jpg")).unwrap();
        let t2 = json!({ "name": "n", "input": "image", "output": "function", "labels": ["KEY(Z)"], "max_side": 64,
            "rows": [{ "question": "q", "answer": "KEY(Z)", "group": "KEY(Z)", "image": "recording/s1/f1.jpg" },
                     { "question": "q", "answer": "KEY(Z)", "group": "KEY(Z)", "image": "recording/s1/f1.png" }], "exam": [] });
        write_teach_files(&d, &t2, "repo", Path::new("/m/base.gguf")).unwrap();
        let imgs: Vec<String> = std::fs::read_to_string(d.join("rows.jsonl")).unwrap().lines()
            .map(|l| serde_json::from_str::<Value>(l).unwrap()["image"].as_str().unwrap().to_string()).collect();
        assert!(imgs[0].ends_with("/recording/s1/f1.jpg.png") && imgs[1].ends_with("/recording/s1/f1.png.png"), "{imgs:?}");
        assert!(Path::new(&imgs[0]).is_file() && Path::new(&imgs[1]).is_file());
        let size = |p: &Path| { let b = std::fs::read(p).unwrap(); (u32::from_be_bytes(b[16..20].try_into().unwrap()), u32::from_be_bytes(b[20..24].try_into().unwrap())) };
        assert_eq!(size(&shrunk), (64, 64));

        // A crop cuts first: the left half of a 256-pixel square, at no size limit.
        let t = json!({ "name": "n", "input": "image", "output": "function", "labels": ["KEY(Z)"], "crop": [0.0, 0.0, 0.5, 1.0],
            "rows": [{ "question": "q", "answer": "KEY(Z)", "group": "KEY(Z)", "image": "recording/s1/f1.png" }], "exam": [] });
        write_teach_files(&d, &t, "repo", Path::new("/m/base.gguf")).unwrap();
        let row: Value = serde_json::from_str(std::fs::read_to_string(d.join("rows.jsonl")).unwrap().trim()).unwrap();
        let cut = PathBuf::from(row["image"].as_str().unwrap());
        assert!(cut.to_string_lossy().contains(".sized/4096-crop-"), "{cut:?}");
        assert_eq!(size(&cut), (128, 256));

        // Both: the whole picture shrinks so the kept half's longer side is 64
        // (256 → 64 across), then the left half is cut: 32 × 64.
        let t = json!({ "name": "n", "input": "image", "output": "function", "labels": ["KEY(Z)"], "crop": [0.0, 0.0, 0.5, 1.0],
            "max_side": 64, "rows": [{ "question": "q", "answer": "KEY(Z)", "group": "KEY(Z)", "image": "recording/s1/f1.png" }], "exam": [] });
        write_teach_files(&d, &t, "repo", Path::new("/m/base.gguf")).unwrap();
        let row: Value = serde_json::from_str(std::fs::read_to_string(d.join("rows.jsonl")).unwrap().trim()).unwrap();
        assert_eq!(size(Path::new(row["image"].as_str().unwrap())), (32, 64));
    }

    #[test]
    fn save_file_only_writes_the_file() {
        assert_eq!(after_pack(false), &[] as &[AfterPack], "Save file… neither adds to the lineup nor opens Pigeon");
    }

    #[test]
    fn try_it_in_pigeon_adds_to_the_lineup_then_opens_pigeon() {
        assert_eq!(after_pack(true), &[AfterPack::AddToLineup, AfterPack::OpenInPigeon]);
    }

    #[test]
    fn only_a_recorded_table_gets_an_actions_json() {
        let d = tmp("actions-d71");
        let actions = json!({ "keys": [], "frames": { "stack": 1, "gap_ms": 250, "max_side": 768 }, "question": "q" });
        let t = |recorded: bool| json!({ "name": "n", "input": "image", "output": "function", "labels": ["KEY(Up)"],
            "recorded": recorded, "rows": [], "exam": [], "actions": actions.clone() });
        write_teach_files(&d, &t(false), "repo", Path::new("/m/base.gguf")).unwrap();
        assert!(!d.join("actions.json").exists(), "imported pictures are not played (D71)");
        write_teach_files(&d, &t(true), "repo", Path::new("/m/base.gguf")).unwrap();
        assert!(d.join("actions.json").is_file());
        write_teach_files(&d, &t(false), "repo", Path::new("/m/base.gguf")).unwrap();
        assert!(!d.join("actions.json").exists(), "and an old one is removed");
    }

    #[test]
    fn a_recorded_table_is_never_cut_or_shrunk_at_teach() {
        let d = tmp("recorded");
        let row = json!({ "question": "q", "answer": "KEY(Z)", "group": "KEY(Z)", "image": "recording/s1/f1.jpg" });
        for extra in [json!({ "max_side": 64 }), json!({ "crop": [0.0, 0.0, 0.5, 1.0] })] {
            let mut t = json!({ "name": "n", "input": "image", "output": "function", "labels": ["KEY(Z)"], "recorded": true,
                "rows": [row.clone()], "exam": [] });
            t.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
            let err = write_teach_files(&d, &t, "repo", Path::new("/m/base.gguf")).unwrap_err();
            assert!(err.contains("train as recorded"), "{err}");
            assert!(!d.join(".sized").exists(), "nothing was shrunk");
        }
        // As recorded, it writes its rows from the recording itself.
        let t = json!({ "name": "n", "input": "image", "output": "function", "labels": ["KEY(Z)"], "recorded": true,
            "rows": [row], "exam": [] });
        write_teach_files(&d, &t, "repo", Path::new("/m/base.gguf")).unwrap();
        assert!(std::fs::read_to_string(d.join("rows.jsonl")).unwrap().contains("recording/s1/f1.jpg\""));
    }

    /// Writes a run's actions.json from what the page would send — for
    /// re-packing a Pigeon from the command line:
    ///   FS_ACTIONS_IN=<page actions json> FS_ACTIONS_OUT=<run>/actions.json \
    ///   cargo test --lib -- --ignored write_actions_for_a_run
    #[test]
    #[ignore]
    fn write_actions_for_a_run() {
        let (Some(inp), Some(out)) = (std::env::var_os("FS_ACTIONS_IN"), std::env::var_os("FS_ACTIONS_OUT")) else { return };
        let a: Value = serde_json::from_str(&std::fs::read_to_string(inp).unwrap()).unwrap();
        std::fs::write(&out, serde_json::to_string_pretty(&actions_file(&a)).unwrap()).unwrap();
    }

    #[test]
    fn a_play_log_becomes_rows_with_its_frames_copied_in() {
        let d = tmp("play");
        let (log, run) = (d.join("plays/fire-red/1790000000000"), d.join("run"));
        std::fs::create_dir_all(&log).unwrap();
        std::fs::create_dir_all(&run).unwrap();
        std::fs::write(log.join("play.json"), r#"{"name":"Fire Red","started_ms":1790000000000,"watch_only":true}"#).unwrap();
        std::fs::write(log.join("f000001.jpg"), b"a").unwrap();
        std::fs::write(log.join("f000002.jpg"), b"b").unwrap();
        std::fs::write(log.join("log.jsonl"), concat!(
            r#"{"turn":1,"frame":"f000001.jpg","answer":"KEY(A) ","on_list":true}"#, "\n",
            r#"{"turn":2,"frame":"../escape.jpg","answer":"NONE","on_list":true}"#, "\n",
            r#"{"turn":3,"frame":"f000002.jpg","answer":"NONE","on_list":true}"#, "\n")).unwrap();
        let v = rows_from_play(&log, &run).unwrap();
        let rows = v["rows"].as_array().unwrap();
        assert_eq!(rows.len(), 2, "a frame outside the log is never read");
        assert_eq!(rows[0]["answer"], "KEY(A)");
        assert_eq!(rows[0]["input"]["path"], "plays/1790000000000/f000001.jpg");
        assert!(run.join("plays/1790000000000/f000002.jpg").is_file(), "frames are copied into the run");
        assert!(rows_from_play(&d, &run).is_err(), "a folder that is not a play log says so");
    }

    #[test]
    fn actions_say_what_each_key_answer_presses_and_never_a_modifier() {
        let a = json!({
            "keys": [
                { "answer": "KEY(Confirm)", "raw": "X", "code": 7, "holds": [80, 120, 95] },
                { "answer": "KEY(Up)", "raw": "Up", "code": null, "holds": [] },
                { "answer": "KEY(Save)", "raw": "Cmd+S", "code": 1, "holds": [60] },
                { "answer": "KEY(Odd)", "raw": "Key200", "code": null, "holds": [] }
            ],
            "frames": { "stack": 1, "gap_ms": 250, "max_side": 768 }, "question": "What do you do next?"
        });
        let f = actions_file(&a);
        assert_eq!(f["keys"]["Confirm"], json!({ "code": 7, "name": "X", "hold_ms": 95 }), "the median hold");
        assert_eq!(f["keys"]["Up"], json!({ "code": 126, "name": "Up", "hold_ms": null }), "a code from the name, when none was recorded");
        assert!(f["keys"].get("Save").is_none() && f["keys"].get("Odd").is_none());
        assert_eq!(f["not_pressed"].as_array().unwrap().len(), 2);
        assert_eq!((key_code("z"), key_code("Return"), key_code("Key200")), (Some(6), Some(36), None));
    }
}
