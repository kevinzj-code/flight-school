//! Recording (Milestone 4): what a person DOES in a window becomes rows of a
//! run's table — the window as it was just before each input, and the input
//! as a named action.
//!
//! The seeing is `recorder/Recorder.swift` (the `fs-recorder` sidecar): one
//! window through ScreenCaptureKit, keys and clicks through a listen-only
//! event tap, the frame from just before each input written to the session
//! folder, one JSON line per input. This file decides what those inputs MEAN,
//! and it knows nothing about what app was recorded:
//!
//! - a key is an action from the moment it is pressed, named after the key
//!   (`KEY(Z)`) until someone renames it (`KEY(Confirm)`);
//! - clicks that land together are gathered into one proposed place, which
//!   has no name — and a click with no name is not an answer, so a place
//!   clicked more than once blocks training until it is named (`CLICK(Send)`);
//!   one clicked once is a stray (usually a click to focus) and is left out;
//! - a quiet stretch is a `NONE` row, capped at 20% of what trains.
//!
//! A run's recording is one file, `recording.json`, beside the run's other
//! files. `rows.jsonl` — the table the trainer reads — is written FROM it
//! after every change, with `labels.txt` and the plan's answer list, so the
//! two can never disagree.

use std::collections::BTreeMap;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tauri::{AppHandle, Emitter, Manager};

/// The words every recorded row is asked with. One sentence for every row, so
/// the Pigeon learns from the picture alone and Pigeon can ask it the same
/// way when a picture arrives with nothing typed.
pub const QUESTION: &str = "What do you do next?";
/// NONE may be at most this share of what trains (decided 2026-09-21).
pub const NONE_CAP: f64 = 0.20;
/// Clicks this close to a place (in the picture's pixels) are the same place.
const NEAR_PX: i32 = 16;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Action {
    pub id: String,
    /// "key" or "click".
    pub kind: String,
    /// The key as pressed ("Z", "Cmd+S"). Keys only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Where the clicks landed, `[x, y, w, h]` in the picture's pixels. Clicks only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rect: Option<[i32; 4]>,
    pub uses: u32,
    /// The key's code, as macOS reported it — what play presses. Keys only;
    /// absent on keys recorded before codes were kept.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<i64>,
}

impl Action {
    /// The answer it trains as, or None when it cannot be one yet.
    pub fn label(&self) -> Option<String> {
        match self.kind.as_str() {
            "key" => Some(format!("KEY({})", self.name.as_deref().or(self.raw.as_deref()).unwrap_or("?"))),
            _ => self.name.as_ref().map(|n| format!("CLICK({n})")),
        }
    }
    fn stray(&self) -> bool {
        self.kind == "click" && self.name.is_none() && self.uses < 2
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct RecRow {
    pub id: u32,
    pub session: String,
    /// The frame, relative to the run folder.
    pub image: String,
    /// The action's id; None is NONE.
    #[serde(default)]
    pub action: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub click: Option<[i32; 2]>,
    /// How long the key was held, when it was a key and it came up.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hold_ms: Option<u32>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Default)]
pub struct SessionInfo {
    pub name: String,
    pub started_ms: u64,
    pub seconds: u64,
    pub kept: u32,
    #[serde(default)]
    pub stack: u32,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Default)]
pub struct Book {
    pub question: String,
    /// The recorded window, as the last session found it.
    #[serde(default)]
    pub window: Value,
    #[serde(default)]
    pub sessions: Vec<SessionInfo>,
    #[serde(default)]
    pub actions: Vec<Action>,
    #[serde(default)]
    pub rows: Vec<RecRow>,
}

impl Book {
    pub fn new() -> Self {
        Book { question: QUESTION.into(), ..Default::default() }
    }

    fn next_row(&self) -> u32 {
        self.rows.iter().map(|r| r.id).max().unwrap_or(0) + 1
    }

    fn new_id(&self, prefix: char) -> String {
        let n = self.actions.iter().filter(|a| a.id.starts_with(prefix)).count() + 1;
        let mut id = format!("{prefix}{n}");
        let mut k = n;
        while self.actions.iter().any(|a| a.id == id) {
            k += 1;
            id = format!("{prefix}{k}");
        }
        id
    }

    /// One line from the recorder. Returns the row it added, if any.
    pub fn ingest(&mut self, line: &Value, session: &str, dir_rel: &str) -> Option<RecRow> {
        let frame = line["frame"].as_str()?;
        /* A release names the frame its press was kept with: the hold goes on
           that row. It adds no row of its own. */
        if line["type"] == "release" {
            let image = format!("{dir_rel}/{frame}");
            if let Some(r) = self.rows.iter_mut().rev().find(|r| r.image == image) {
                r.hold_ms = line["held_ms"].as_u64().map(|v| v.min(60_000) as u32);
            }
            return None;
        }
        if frame.contains(['/', '\\']) || frame.starts_with('.') {
            return None; // a file name, never a path
        }
        let image = format!("{dir_rel}/{frame}");
        let (action, click) = match (line["type"].as_str()?, line["kind"].as_str()) {
            ("idle", _) => (None, None),
            ("input", Some("key")) => {
                let raw = line["key"].as_str()?.trim();
                if raw.is_empty() {
                    return None;
                }
                let code = line["code"].as_i64().filter(|c| *c >= 0);
                let id = match self.actions.iter_mut().find(|a| a.kind == "key" && a.raw.as_deref() == Some(raw)) {
                    Some(a) => {
                        a.uses += 1;
                        if a.code.is_none() {
                            a.code = code;
                        }
                        a.id.clone()
                    }
                    None => {
                        let id = self.new_id('k');
                        self.actions.push(Action { id: id.clone(), kind: "key".into(), raw: Some(raw.into()), name: None, rect: None, uses: 1, code });
                        id
                    }
                };
                (Some(id), None)
            }
            ("input", Some("click")) => {
                let (x, y) = (line["x"].as_i64()? as i32, line["y"].as_i64()? as i32);
                (Some(self.place_click(x, y)), Some([x, y]))
            }
            _ => return None,
        };
        let row = RecRow { id: self.next_row(), session: session.into(), image, action, click, hold_ms: None };
        self.rows.push(row.clone());
        Some(row)
    }

    /// The place a click belongs to: one it lands in or near, grown to take
    /// it; otherwise a new, unnamed place around it.
    fn place_click(&mut self, x: i32, y: i32) -> String {
        let near = |r: &[i32; 4]| x >= r[0] - NEAR_PX && x <= r[0] + r[2] + NEAR_PX && y >= r[1] - NEAR_PX && y <= r[1] + r[3] + NEAR_PX;
        if let Some(a) = self.actions.iter_mut().find(|a| a.kind == "click" && a.rect.as_ref().is_some_and(near)) {
            let r = a.rect.unwrap();
            let (x0, y0) = (r[0].min(x - 8), r[1].min(y - 8));
            let (x1, y1) = ((r[0] + r[2]).max(x + 8), (r[1] + r[3]).max(y + 8));
            a.rect = Some([x0, y0, x1 - x0, y1 - y0]);
            a.uses += 1;
            return a.id.clone();
        }
        let id = self.new_id('c');
        self.actions.push(Action { id: id.clone(), kind: "click".into(), raw: None, name: None, rect: Some([x - 8, y - 8, 16, 16]), uses: 1, code: None });
        id
    }

    fn action(&self, id: &str) -> Option<&Action> {
        self.actions.iter().find(|a| a.id == id)
    }

    /// What a row trains as: its answer, or why it does not train.
    pub fn row_answer(&self, row: &RecRow) -> Result<String, &'static str> {
        match &row.action {
            None => Ok("NONE".into()),
            Some(id) => match self.action(id) {
                None => Err("gone"),
                Some(a) if a.stray() => Err("stray"),
                Some(a) => a.label().ok_or("unnamed"),
            },
        }
    }

    /// The table as it will train, and what is left out and why.
    pub fn training(&self) -> Training {
        let mut rows: Vec<(u32, String)> = Vec::new();
        let (mut unnamed, mut strays) = (0, 0);
        for r in &self.rows {
            match self.row_answer(r) {
                Ok(a) => rows.push((r.id, a)),
                Err("unnamed") => unnamed += 1,
                Err(_) => strays += 1,
            }
        }
        /* NONE is sampled down to its cap, evenly across the recording, so
           the rows kept are spread through every session. */
        let none: Vec<u32> = rows.iter().filter(|(_, a)| a == "NONE").map(|(id, _)| *id).collect();
        let rest = rows.len() - none.len();
        let allowed = ((rest as f64) * NONE_CAP / (1.0 - NONE_CAP)).floor() as usize;
        let mut none_dropped = 0;
        if none.len() > allowed {
            let keep: std::collections::HashSet<u32> =
                (0..allowed).map(|i| none[i * none.len() / allowed.max(1)]).collect();
            none_dropped = none.len() - keep.len();
            rows.retain(|(id, a)| a != "NONE" || keep.contains(id));
        }
        let mut labels: Vec<String> = Vec::new();
        for a in &self.actions {
            if a.stray() {
                continue;
            }
            if let Some(l) = a.label() {
                if !labels.contains(&l) {
                    labels.push(l);
                }
            }
        }
        if self.rows.iter().any(|r| r.action.is_none()) {
            labels.push("NONE".into());
        }
        let blocking: Vec<String> = self.actions.iter().filter(|a| a.kind == "click" && a.name.is_none() && !a.stray()).map(|a| a.id.clone()).collect();
        Training { rows, labels, unnamed_rows: unnamed, stray_rows: strays, none_dropped, blocking }
    }

    /// Write the trainer's table from the recording: `rows.jsonl`, `labels.txt`,
    /// and the plan's answer list. Image paths become absolute, as every other
    /// vision row's are.
    pub fn write_table(&self, run: &Path) -> Result<Training, String> {
        let t = self.training();
        let by_id: BTreeMap<u32, &RecRow> = self.rows.iter().map(|r| (r.id, r)).collect();
        let mut text = String::new();
        for (id, answer) in &t.rows {
            let r = by_id[id];
            let o = json!({ "question": self.question, "answer": answer, "group": answer,
                            "image": run.join(&r.image).to_string_lossy() });
            text.push_str(&o.to_string());
            text.push('\n');
        }
        write_atomic(&run.join("rows.jsonl"), text.as_bytes())?;
        write_atomic(&run.join("labels.txt"), format!("{}\n", t.labels.join("\n")).as_bytes())?;
        let plan_path = run.join("plan.json");
        let mut plan: Value = std::fs::read_to_string(&plan_path).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_else(|| json!({}));
        plan["modality"] = json!("vision");
        plan["input_shape"] = json!("one window, as it was just before each input");
        plan["output_shape"] = json!(format!("exactly one of: {}", t.labels.join(", ")));
        plan["example_questions"] = json!([self.question]);
        write_atomic(&plan_path, serde_json::to_string_pretty(&plan).unwrap().as_bytes())?;
        Ok(t)
    }
}

#[derive(Serialize, Debug, PartialEq)]
pub struct Training {
    /// (row id, answer) for every row that trains.
    pub rows: Vec<(u32, String)>,
    /// `labels.txt`, generated — never typed.
    pub labels: Vec<String>,
    /// Rows whose click has no name yet. They block.
    pub unnamed_rows: usize,
    /// Rows from a place clicked once — left out.
    pub stray_rows: usize,
    pub none_dropped: usize,
    /// The places that must be named before training.
    pub blocking: Vec<String>,
}

use crate::fsutil::write_atomic;

fn book_path(run: &Path) -> PathBuf {
    run.join("recording.json")
}

pub fn load(run: &Path) -> Book {
    std::fs::read_to_string(book_path(run)).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_else(Book::new)
}

pub fn save(run: &Path, book: &Book) -> Result<Training, String> {
    write_atomic(&book_path(run), serde_json::to_string_pretty(book).unwrap().as_bytes())?;
    book.write_table(run)
}

/* ── the sidecar ────────────────────────────────────────────────────────── */

/// The recorder: `FS_RECORDER_BIN` (tests), beside the executable (bundled),
/// or the checkout's `binaries/` (dev) — the same places llama-server is found.
pub(crate) fn recorder_path() -> Result<PathBuf, String> {
    if let Some(p) = std::env::var_os("FS_RECORDER_BIN").map(PathBuf::from).filter(|p| p.is_file()) {
        return Ok(p);
    }
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            candidates.push(dir.join("fs-recorder"));
            candidates.push(dir.join("fs-recorder-aarch64-apple-darwin"));
        }
    }
    candidates.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("binaries/fs-recorder-aarch64-apple-darwin"));
    candidates.into_iter().find(|p| p.is_file()).ok_or_else(|| "the recorder is missing from this build of Flight School".into())
}

/// One run of the recorder that prints one line and exits.
fn ask_recorder(arg: &str) -> Result<Value, String> {
    let out = Command::new(recorder_path()?).arg(arg).output().map_err(|e| format!("the recorder would not start: {e}"))?;
    let text = String::from_utf8_lossy(&out.stdout);
    let line = text.lines().last().ok_or("the recorder said nothing")?;
    serde_json::from_str(line).map_err(|e| e.to_string())
}

/// What a recorder error means, in words a person can act on.
pub fn error_words(code: &str) -> &'static str {
    match code {
        "no-screen-permission" => "macOS is not letting Flight School see the screen. Turn it on in System Settings › Privacy & Security › Screen & System Audio Recording, then quit and reopen Flight School.",
        "no-input-permission" => "macOS is not letting Flight School see key presses and clicks. Turn it on in System Settings › Privacy & Security › Input Monitoring, then quit and reopen Flight School.",
        "window-gone" => "That window closed or moved off screen. Pick it again.",
        "capture-failed" => "macOS would not start recording that window.",
        "no-windows" => "No windows could be listed. Open the app you want to teach, then check again.",
        "no-frame" => "That window sent no picture. Make sure it is on screen and not minimised, then try again.",
        "crop-under-a-pixel" => "The part of the window chosen is under a pixel wide or tall at this size. Choose a larger part.",
        _ => "The recorder stopped unexpectedly.",
    }
}

struct Live {
    child: Child,
    run: String,
    session: String,
    started_ms: u64,
    stack: u32,
}

#[derive(Default)]
pub struct Recording(Mutex<Option<Live>>);

/// One writer of a run's `recording.json` at a time: the recorder's lines and
/// a rename from the page must not overwrite each other.
static BOOK: Mutex<()> = Mutex::new(());
fn book_lock() -> std::sync::MutexGuard<'static, ()> {
    BOOK.lock().unwrap_or_else(|p| p.into_inner())
}

fn run_path(app: &AppHandle, run: &str) -> Result<PathBuf, String> {
    let dir = crate::quickbuild::run_dir(app, run)?;
    if !dir.is_dir() {
        return Err("no such run".into());
    }
    Ok(dir)
}

/* The two permissions, asked about by Flight School ITSELF. macOS keeps them
 * against the app, and the recorder it starts inherits them — but the
 * recorder asking "am I allowed?" on its own behalf could answer no after the
 * person had said yes, and the screen stayed on "Recording is off". So the
 * question is asked here, in the app's own process, as Pigeon's capture.rs
 * does. */
#[cfg(target_os = "macos")]
#[link(name = "CoreGraphics", kind = "framework")]
extern "C" {
    fn CGPreflightScreenCaptureAccess() -> bool;
    fn CGRequestScreenCaptureAccess() -> bool;
    fn CGPreflightListenEventAccess() -> bool;
    fn CGRequestListenEventAccess() -> bool;
}

fn permission_now() -> Value {
    // SAFETY: parameterless CoreGraphics queries; they never prompt.
    let (screen, input) = unsafe { (CGPreflightScreenCaptureAccess(), CGPreflightListenEventAccess()) };
    json!({ "type": "permission", "screen": screen, "input": input })
}

/// One line per permission check, window listing or start, in the app's
/// data folder (`record-log.txt`): what macOS answered, so a Record screen
/// that will not open can be diagnosed from the file rather than described.
fn log_line(app: &AppHandle, what: &str, v: &Value) {
    use std::io::Write;
    let Ok(dir) = crate::paths::data_dir(app) else { return };
    let path = dir.join("record-log.txt");
    // Kept short: past ~200 KB it starts again.
    if std::fs::metadata(&path).map(|m| m.len() > 200_000).unwrap_or(false) {
        let _ = std::fs::remove_file(&path);
    }
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(f, "{} {what} {v}", crate::runs::now_ms());
    }
}

#[tauri::command]
pub fn rec_permission(app: AppHandle) -> Result<Value, String> {
    let v = permission_now();
    log_line(&app, "permission", &v);
    Ok(v)
}

/// Asks macOS for both permissions — the prompt appears only when the person
/// has pressed Ask macOS, never at launch.
#[tauri::command]
pub fn rec_request_permission(app: AppHandle) -> Result<Value, String> {
    // SAFETY: as above; these may show macOS's own prompt, once.
    unsafe {
        if !CGPreflightScreenCaptureAccess() {
            CGRequestScreenCaptureAccess();
        }
        if !CGPreflightListenEventAccess() {
            CGRequestListenEventAccess();
        }
    }
    let v = permission_now();
    log_line(&app, "asked", &v);
    Ok(v)
}

/// Opens the one System Settings pane the person needs; `which` is a word,
/// never a URL.
#[tauri::command]
pub fn rec_open_privacy(which: String) -> Result<(), String> {
    let pane = match which.as_str() {
        "screen" => "Privacy_ScreenCapture",
        "input" => "Privacy_ListenEvent",
        _ => return Err("no such setting".into()),
    };
    Command::new("open")
        .arg(format!("x-apple.systempreferences:com.apple.preference.security?{pane}"))
        .status()
        .map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command(async)]
pub fn rec_windows(app: AppHandle) -> Result<Value, String> {
    let v = ask_recorder("windows");
    /* Only the count is logged. The list itself carries every open window's
       title — browser tabs, document names — which is nobody's business. */
    log_line(&app, "windows", &match &v {
        Ok(v) => json!({ "count": v["windows"].as_array().map_or(0, |a| a.len()) }),
        Err(e) => json!({ "error": e }),
    });
    let v = v?;
    if let Some(code) = v["code"].as_str() {
        return Err(error_words(code).into());
    }
    Ok(v["windows"].clone())
}

/// One picture of a whole window, as a data URL, for choosing the part of it
/// to keep. Written into the run folder and read back.
#[tauri::command(async)]
pub fn rec_still(app: AppHandle, run: String, window: u32) -> Result<Value, String> {
    let dir = run_path(&app, &run)?;
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let file = dir.join(".still.jpg");
    let out = Command::new(recorder_path()?)
        .args(["still", "--window", &window.to_string(), "--out", &file.to_string_lossy(), "--max-side", "960"])
        .output().map_err(|e| format!("the recorder would not start: {e}"))?;
    let v: Value = String::from_utf8_lossy(&out.stdout).lines().last()
        .and_then(|l| serde_json::from_str(l).ok()).ok_or("the recorder said nothing")?;
    if let Some(code) = v["code"].as_str() {
        return Err(error_words(code).into());
    }
    let bytes = std::fs::read(&file).map_err(|e| e.to_string())?;
    let _ = std::fs::remove_file(&file);
    Ok(json!({ "url": format!("data:image/jpeg;base64,{}", base64(&bytes)), "w": v["w"], "h": v["h"] }))
}

/// Start recording `window` into `run`. Lines arrive as `rec-event` events;
/// each input is in `recording.json` and the table the moment it happens.
#[tauri::command]
pub fn rec_start(app: AppHandle, run: String, window: u32, keys: bool, clicks: bool, stack: u32, max_side: Option<u32>, crop: Option<Vec<f64>>) -> Result<String, String> {
    let dir = run_path(&app, &run)?;
    let state = app.state::<Recording>();
    let mut live = state.0.lock().map_err(|_| "busy")?;
    if live.is_some() {
        return Err("already recording".into());
    }
    let book = load(&dir);
    let session = format!("Session {}", book.sessions.len() + 1);
    let folder = format!("s{}", book.sessions.len() + 1);
    let dir_rel = format!("recording/{folder}");
    let stack = stack.clamp(1, 4);
    // The longer side of every frame, in pixels: the size it will train at
    // unless Teach shrinks it further, and the size Pigeon is told to see.
    let max_side = max_side.unwrap_or(768).clamp(64, 1280);
    let mut args: Vec<String> = ["record", "--window", &window.to_string(), "--out", &dir.join(&dir_rel).to_string_lossy(),
        "--keys", if keys { "1" } else { "0" }, "--clicks", if clicks { "1" } else { "0" },
        "--stack", &stack.to_string(), "--max-side", &max_side.to_string()].iter().map(|a| a.to_string()).collect();
    // The part of the window to keep, as fractions of it: x, y, w, h.
    if let Some(c) = crop.filter(|c| c.len() == 4) {
        args.extend(["--crop".to_string(), c.iter().map(|v| format!("{v:.4}")).collect::<Vec<_>>().join(",")]);
    }
    let mut child = Command::new(recorder_path()?)
        .args(&args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("the recorder would not start: {e}"))?;
    let stdout = child.stdout.take().ok_or("no recorder output")?;
    *live = Some(Live { child, run: run.clone(), session: session.clone(), started_ms: crate::runs::now_ms(), stack });
    drop(live);
    register_stop_key(&app);
    let app2 = app.clone();
    let session2 = session.clone();
    std::thread::spawn(move || follow(app2, dir, session2, dir_rel, std::io::BufReader::new(stdout)));
    Ok(session)
}

/// Reads the recorder's lines into the run until it stops.
fn follow(app: AppHandle, dir: PathBuf, session: String, dir_rel: String, reader: impl BufRead) {
    /* The book is held here for the whole session and written back on a
       timer, not on every key: a game at four keys a second used to re-read
       and rewrite a growing JSON, plus the trainer's three files, per key.
       recording.json goes to disk every two seconds and at the end; the
       trainer's files are written once, at the end. */
    let mut book = { let _g = book_lock(); load(&dir) };
    let mut flushed = std::time::Instant::now();
    let flush = |book: &Book| { let _g = book_lock(); let _ = write_atomic(&book_path(&dir), serde_json::to_string_pretty(book).unwrap().as_bytes()); };
    for line in reader.lines().map_while(Result::ok) {
        let Ok(v) = serde_json::from_str::<Value>(&line) else { continue };
        match v["type"].as_str() {
            Some("input") | Some("idle") | Some("release") => {
                if let Some(row) = book.ingest(&v, &session, &dir_rel) {
                    let label = book.row_answer(&row).unwrap_or_else(|_| "CLICK(…)".into());
                    let kept = book.rows.iter().filter(|r| r.session == session).count();
                    let _ = app.emit("rec-event", json!({ "type": "kept", "kept": kept, "last": label }));
                    if flushed.elapsed() >= std::time::Duration::from_secs(2) { flush(&book); flushed = std::time::Instant::now(); }
                }
            }
            Some("ready") => {
                book.window = v["window"].clone();
                flush(&book);
                let _ = app.emit("rec-event", json!({ "type": "ready", "window": v["window"] }));
            }
            Some("error") => {
                let code = v["code"].as_str().unwrap_or("");
                log_line(&app, "record-error", &v);
                let _ = app.emit("rec-event", json!({ "type": "error", "code": code, "words": error_words(code) }));
            }
            // A password field has focus somewhere: the recorder keeps no keys
            // until it closes, and the page says so while it lasts.
            Some("secure-input") => {
                log_line(&app, "record-secure-input", &v);
                let _ = app.emit("rec-event", json!({ "type": "secure-input", "on": v["on"].as_bool().unwrap_or(false) }));
            }
            _ => {}
        }
    }
    // The recorder has exited: write what it saw, then close the session.
    flush(&book);
    let state = app.state::<Recording>();
    let finished = state.0.lock().ok().and_then(|mut g| g.take());
    finish(&app, &dir, finished);
}

/// ⌘⇧R from any app stops the recording, so a full-screen game need not be
/// left to reach the Stop button. Registered for the session only.
const STOP_KEY: &str = "CmdOrCtrl+Shift+R";
fn register_stop_key(app: &AppHandle) {
    use tauri_plugin_global_shortcut::{GlobalShortcutExt, ShortcutState};
    let _ = app.global_shortcut().on_shortcut(STOP_KEY, |app, _shortcut, event| {
        if event.state() == ShortcutState::Pressed {
            let _ = rec_stop(app.clone());
            let _ = app.emit("rec-event", json!({ "type": "hotkey" }));
            /* Bring Flight School forward, so the session's rows are seen
               landing rather than left behind another app. */
            if let Some(w) = app.get_webview_window("main") {
                let _ = w.unminimize();
                let _ = w.set_focus();
            }
        }
    });
}
fn unregister_stop_key(app: &AppHandle) {
    use tauri_plugin_global_shortcut::GlobalShortcutExt;
    let _ = app.global_shortcut().unregister(STOP_KEY);
}

fn finish(app: &AppHandle, dir: &Path, live: Option<Live>) {
    unregister_stop_key(app);
    let Some(mut live) = live else { return };
    let _ = live.child.wait();
    let _g = book_lock();
    let mut book = load(dir);
    let kept = book.rows.iter().filter(|r| r.session == live.session).count() as u32;
    book.sessions.push(SessionInfo {
        name: live.session.clone(),
        started_ms: live.started_ms,
        seconds: crate::runs::now_ms().saturating_sub(live.started_ms) / 1000,
        kept,
        stack: live.stack,
    });
    let _ = save(dir, &book);
    let _ = app.emit("rec-event", json!({ "type": "stopped", "run": live.run, "session": live.session, "kept": kept }));
}

/// Stop: the recorder is told to stop and finishes its last frame; the
/// session is closed when its lines end.
#[tauri::command]
pub fn rec_stop(app: AppHandle) -> Result<(), String> {
    let state = app.state::<Recording>();
    let mut g = state.0.lock().map_err(|_| "busy")?;
    let Some(live) = g.as_mut() else { return Ok(()) };
    if let Some(mut stdin) = live.child.stdin.take() {
        let _ = stdin.write_all(b"stop\n");
    }
    Ok(())
}

/// The run's recorded table: every row with its answer (or why it has none),
/// the actions, and what blocks training.
#[tauri::command]
pub fn rec_table(app: AppHandle, run: String) -> Result<Value, String> {
    Ok(table_json(&run_path(&app, &run)?))
}

pub(crate) fn table_json(dir: &Path) -> Value {
    let book = load(dir);
    let t = book.training();
    let rows: Vec<Value> = book.rows.iter().map(|r| {
        let (answer, why) = match book.row_answer(r) { Ok(a) => (Some(a), None), Err(w) => (None, Some(w)) };
        json!({ "id": r.id, "session": r.session, "image": r.image, "action": r.action, "click": r.click, "hold_ms": r.hold_ms,
                "answer": answer, "left_out": why, "trains": t.rows.iter().any(|(id, _)| *id == r.id) })
    }).collect();
    json!({ "question": book.question, "window": book.window, "sessions": book.sessions,
            "actions": book.actions, "rows": rows, "labels": t.labels, "blocking": t.blocking,
            "unnamed_rows": t.unnamed_rows, "stray_rows": t.stray_rows, "none_dropped": t.none_dropped,
            "training_rows": t.rows.len() })
}

pub(crate) fn base64(bytes: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for c in bytes.chunks(3) {
        let n = (c[0] as u32) << 16 | (*c.get(1).unwrap_or(&0) as u32) << 8 | *c.get(2).unwrap_or(&0) as u32;
        out.push(T[(n >> 18) as usize & 63] as char);
        out.push(T[(n >> 12) as usize & 63] as char);
        out.push(if c.len() > 1 { T[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if c.len() > 2 { T[n as usize & 63] as char } else { '=' });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(k: &str, f: &str) -> Value { json!({"type":"input","kind":"key","key":k,"frame":f}) }
    fn click(x: i32, y: i32, f: &str) -> Value { json!({"type":"input","kind":"click","x":x,"y":y,"frame":f}) }
    fn idle(f: &str) -> Value { json!({"type":"idle","frame":f}) }

    #[test]
    fn a_key_is_an_answer_named_after_itself() {
        let mut b = Book::new();
        b.ingest(&key("Z", "f1.jpg"), "Session 1", "recording/s1");
        b.ingest(&key("Z", "f2.jpg"), "Session 1", "recording/s1");
        assert_eq!(b.actions.len(), 1);
        assert_eq!(b.actions[0].uses, 2);
        assert_eq!(b.row_answer(&b.rows[0]).unwrap(), "KEY(Z)");
        assert_eq!(b.row_answer(&b.rows[1]).unwrap(), "KEY(Z)");
    }

    #[test]
    fn clicks_that_land_together_are_one_place_and_block_until_named() {
        let mut b = Book::new();
        b.ingest(&click(100, 100, "f1.jpg"), "S", "recording/s1");
        b.ingest(&click(108, 104, "f2.jpg"), "S", "recording/s1");
        b.ingest(&click(400, 300, "f3.jpg"), "S", "recording/s1"); // once: a stray
        let clicks: Vec<_> = b.actions.iter().filter(|a| a.kind == "click").collect();
        assert_eq!(clicks.len(), 2);
        assert_eq!(clicks[0].uses, 2);
        let t = b.training();
        assert_eq!(t.blocking, vec!["c1".to_string()]);
        assert_eq!((t.unnamed_rows, t.stray_rows, t.rows.len()), (2, 1, 0));
    }

    #[test]
    fn none_is_capped_at_a_fifth_of_what_trains() {
        let mut b = Book::new();
        for i in 0..8 { b.ingest(&key("Z", &format!("k{i}.jpg")), "S", "r/s1"); }
        for i in 0..10 { b.ingest(&idle(&format!("i{i}.jpg")), "S", "r/s1"); }
        let t = b.training();
        let none = t.rows.iter().filter(|(_, a)| a == "NONE").count();
        assert_eq!(none, 2, "8 actions allow 2 NONE rows (20%)");
        assert_eq!(t.none_dropped, 8);
        assert_eq!(t.labels.last().map(String::as_str), Some("NONE"));
    }

    #[test]
    fn a_frame_is_a_file_name_never_a_path() {
        let mut b = Book::new();
        assert!(b.ingest(&key("Z", "../x.jpg"), "S", "r/s1").is_none());
        assert!(b.ingest(&key("Z", "/etc/x.jpg"), "S", "r/s1").is_none());
    }

    #[test]
    fn the_table_is_written_from_the_recording_with_its_labels_and_plan() {
        let run = std::env::temp_dir().join(format!("fs-rec-table-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&run);
        std::fs::create_dir_all(&run).unwrap();
        let mut b = Book::new();
        b.ingest(&key("Z", "f1.jpg"), "S", "recording/s1");
        b.ingest(&idle("f2.jpg"), "S", "recording/s1");
        b.ingest(&key("X", "f3.jpg"), "S", "recording/s1");
        b.ingest(&key("Z", "f4.jpg"), "S", "recording/s1");
        b.ingest(&key("X", "f5.jpg"), "S", "recording/s1");
        save(&run, &b).unwrap();
        let rows: Vec<Value> = std::fs::read_to_string(run.join("rows.jsonl")).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect();
        assert_eq!(rows.len(), 5);
        assert_eq!(rows[0]["question"], QUESTION);
        assert_eq!(rows[0]["answer"], "KEY(Z)");
        assert!(rows[0]["image"].as_str().unwrap().ends_with("recording/s1/f1.jpg"));
        assert!(Path::new(rows[0]["image"].as_str().unwrap()).is_absolute());
        assert_eq!(std::fs::read_to_string(run.join("labels.txt")).unwrap(), "KEY(Z)\nKEY(X)\nNONE\n");
        let plan: Value = serde_json::from_str(&std::fs::read_to_string(run.join("plan.json")).unwrap()).unwrap();
        assert_eq!(plan["output_shape"], "exactly one of: KEY(Z), KEY(X), NONE");
        assert_eq!(load(&run), b);
        let _ = std::fs::remove_dir_all(&run);
    }

    #[test]
    fn base64_matches_the_standard() {
        assert_eq!(base64(b"Man"), "TWFu");
        assert_eq!(base64(b"Ma"), "TWE=");
        assert_eq!(base64(b"M"), "TQ==");
    }

    /// The Swift recorder's `synthetic` mode — drawn frames, scripted inputs,
    /// the same writer and lines as a live recording — read by this file's
    /// own `follow` logic into a run. Skips when the helper is not built.
    #[test]
    fn a_synthetic_session_from_the_real_recorder_lands_in_the_table() {
        let Ok(bin) = recorder_path() else { eprintln!("(skipped: no recorder built)"); return };
        let run = std::env::temp_dir().join(format!("fs-rec-synth-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&run);
        std::fs::create_dir_all(&run).unwrap();
        let mut child = Command::new(bin)
            .args(["synthetic", "--out", &run.join("recording/s1").to_string_lossy(), "--idle-s", "0.6"])
            .stdin(Stdio::piped()).stdout(Stdio::piped()).spawn().unwrap();
        let reader = std::io::BufReader::new(child.stdout.take().unwrap());
        let mut book = Book::new();
        for line in reader.lines().map_while(Result::ok) {
            if let Ok(v) = serde_json::from_str::<Value>(&line) {
                book.ingest(&v, "Session 1", "recording/s1");
            }
        }
        drop(child.stdin.take());
        let _ = child.wait();
        save(&run, &book).unwrap();
        // Z, W, click, W, the same place again, Cmd+S, (a click outside the
        // window: never an input), a click once elsewhere (a stray), Z, and a
        // quiet stretch.
        let mut answers: Vec<String> = book.rows.iter().map(|r| book.row_answer(r).unwrap_or_else(|w| w.to_string())).collect();
        // The quiet stretch at the end is one NONE or two, by the clock.
        let nones = answers.iter().rev().take_while(|a| *a == "NONE").count();
        assert!((1..=2).contains(&nones), "{answers:?}");
        answers.truncate(answers.len() - nones);
        assert_eq!(answers, vec!["KEY(Z)", "KEY(W)", "unnamed", "KEY(W)", "unnamed", "KEY(Cmd+S)", "stray", "KEY(Z)"]);
        assert!(book.rows.iter().all(|r| run.join(&r.image).is_file()), "every row's frame is on disk");
        // Every key press came up again, and its row says how long it was held.
        let keys: Vec<&RecRow> = book.rows.iter().filter(|r| r.action.as_deref().is_some_and(|a| a.starts_with('k'))).collect();
        assert!(keys.iter().all(|r| r.hold_ms.is_some_and(|h| (100..400).contains(&h))), "{:?}", keys.iter().map(|r| r.hold_ms).collect::<Vec<_>>());
        assert!(book.actions.iter().filter(|a| a.kind == "key").all(|a| a.code.is_some()), "every key has its code");
        let t = save(&run, &book).unwrap();
        assert_eq!(t.blocking, vec!["c1".to_string()], "an unnamed click blocks until Review names its rows");
        assert_eq!(t.labels, vec!["KEY(Z)", "KEY(W)", "KEY(Cmd+S)", "NONE"]);
        let _ = std::fs::remove_dir_all(&run);
    }
}
