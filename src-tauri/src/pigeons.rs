//! The lineup — adding, listing, removing and exporting Pigeons.
//!
//! Every archive decision lives in python/pigeon/{store,nest}.py; this file
//! picks files, names folders and relays one JSON line back to the webview.
//! The store is `pigeons/` under the app's data folder (paths::pigeons_dir):
//! a `.pigeon` added from a flash drive is COPIED there, so the drive can
//! be unplugged, and a run that packages drops a copy there too.

use serde_json::Value;
use std::sync::Mutex;
use tauri::{AppHandle, Emitter, Manager};

use crate::paths;
use crate::quickbuild::pigeon_cli;

fn store_arg(app: &AppHandle) -> Result<String, String> {
    Ok(paths::pigeons_dir(app)?.to_string_lossy().into_owned())
}

/// `cli add`: validate the file (a `.pigeon` or a `.nest`) against the
/// models Airoplane Mode has, and copy what is accepted into the lineup.
/// Reply: `{results: [{status: added|already|refused, id, name, base_needed,
/// errors, warnings, from_nest}], added, already, refused}`.
pub(crate) async fn add_to_store(app: &AppHandle, file: &std::path::Path, source: &str,
                                 run_id: Option<&str>) -> Result<Value, String> {
    let models = paths::models_dir(app)?;
    let mut args = vec![
        "add".to_string(),
        "--file".into(), file.to_string_lossy().into_owned(),
        "--store".into(), store_arg(app)?,
        "--models".into(), models.to_string_lossy().into_owned(),
        "--source".into(), source.to_string(),
    ];
    if let Some(r) = run_id {
        args.push("--run-id".into()); args.push(r.to_string());
    }
    pigeon_cli(app, &args).await
}

/// One combined reply for several files: results concatenated, counts summed.
async fn add_many(app: &AppHandle, files: &[String]) -> Result<Value, String> {
    let mut results = Vec::new();
    for f in files {
        let path = std::path::PathBuf::from(f);
        match add_to_store(app, &path, "added", None).await {
            Ok(r) => results.extend(r["results"].as_array().cloned().unwrap_or_default()),
            // A file the packager could not even open still gets a line, so
            // a nest of five with one broken member never hides the four.
            Err(e) => results.push(serde_json::json!({
                "status": "refused", "file": f,
                "name": path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
                "errors": [e], "warnings": [],
            })),
        }
    }
    let count = |s: &str| results.iter().filter(|r| r["status"] == s).count();
    Ok(serde_json::json!({
        "added": count("added"), "already": count("already"), "refused": count("refused"),
        "results": results,
    }))
}

/// Everything in the lineup, with what a card needs (name, kind, base, the
/// scorecard read out of the file's own report.json, when it was added).
#[tauri::command]
pub async fn pigeon_list(app: AppHandle) -> Result<Value, String> {
    let reply = pigeon_cli(&app, &["list".to_string(), "--store".into(), store_arg(&app)?]).await?;
    Ok(reply["pigeons"].clone())
}

/// "Add a Pigeon…": pick one or more `.pigeon` / `.nest` files. Async for
/// the dialog reason documented in quickbuild.rs — a sync command deadlocks
/// the native panel.
#[tauri::command]
pub async fn pigeon_add_pick(app: AppHandle) -> Result<Option<Value>, String> {
    use tauri_plugin_dialog::DialogExt;
    let picked = app.dialog().file()
        .set_title("Add a Pigeon (or a whole nest)")
        .add_filter("Pigeon or nest", &["pigeon", "nest"])
        .blocking_pick_files();
    let Some(files) = picked else { return Ok(None) };
    let paths: Vec<String> = files.into_iter()
        .filter_map(|f| f.into_path().ok())
        .map(|p| p.to_string_lossy().into_owned())
        .collect();
    if paths.is_empty() { return Ok(None) }
    add_many(&app, &paths).await.map(Some)
}

/// Out of the lineup and into the Trash — never deleted outright.
#[tauri::command]
pub async fn pigeon_remove(app: AppHandle, id: String) -> Result<Value, String> {
    pigeon_cli(&app, &["remove".to_string(), "--store".into(), store_arg(&app)?, "--id".into(), id]).await
}

/// Show the stored file in the Finder — the export story is still "copy
/// the file", so it has to be findable.
#[tauri::command]
pub async fn pigeon_reveal(app: AppHandle, id: String) -> Result<(), String> {
    if id.is_empty() || id.contains(['/', '\\']) || id.contains("..") {
        return Err("bad id".into());
    }
    let path = paths::pigeons_dir(&app)?.join(format!("{id}.pigeon"));
    if !path.exists() {
        return Err("That Pigeon is not in the lineup any more.".into());
    }
    std::process::Command::new("open").arg("-R").arg(&path).spawn().map_err(|e| e.to_string())?;
    Ok(())
}

/// "Export the flock…": the chosen Pigeons (all of them by default) into one
/// `.nest` where the person says. Reply: `{path, bytes, nest}`.
#[tauri::command]
pub async fn pigeon_export_nest(app: AppHandle, ids: Vec<String>, name: Option<String>) -> Result<Option<Value>, String> {
    use tauri_plugin_dialog::DialogExt;
    if ids.is_empty() {
        return Err("Pick at least one Pigeon to export.".into());
    }
    let store = paths::pigeons_dir(&app)?;
    let mut files = Vec::new();
    for id in &ids {
        if id.is_empty() || id.contains(['/', '\\']) || id.contains("..") {
            return Err("bad id".into());
        }
        let p = store.join(format!("{id}.pigeon"));
        if !p.exists() {
            return Err(format!("{id} is not in the lineup any more."));
        }
        files.push(p.to_string_lossy().into_owned());
    }
    let name = name.map(|n| n.trim().to_string()).filter(|n| !n.is_empty()).unwrap_or_else(|| "my-flock".into());
    let slug: String = name.to_lowercase().chars()
        .map(|c| if c.is_alphanumeric() { c } else { '-' })
        .collect::<String>().trim_matches('-').to_string();
    let default_dir = app.path().home_dir().map(|h| h.join("Downloads")).unwrap_or(store.clone());
    let chosen = app.dialog().file()
        .set_title("Export the flock")
        .set_directory(&default_dir)
        .set_file_name(&format!("{}.nest", if slug.is_empty() { "my-flock".into() } else { slug }))
        .add_filter("Nest", &["nest"])
        .blocking_save_file();
    let Some(out) = chosen.and_then(|f| f.into_path().ok()) else { return Ok(None) };
    let mut args = vec![
        "nest".to_string(), "pack".into(),
        "--out".into(), out.to_string_lossy().into_owned(),
        "--name".into(), name,
    ];
    args.extend(files);
    pigeon_cli(&app, &args).await.map(Some)
}

/// Results of files opened from the Finder that the webview has not shown
/// yet. A cold launch delivers RunEvent::Opened before any JS listener
/// exists, so the add is done HERE and the result parked until the webview
/// asks for it at boot (`pigeon_take_opens`); a warm app gets the same
/// result pushed as a `pigeon-added` event.
#[derive(Default)]
pub struct OpenedFiles(Mutex<Vec<Value>>);

/// Called from lib.rs on RunEvent::Opened. Adds in the background so the
/// event loop is never blocked on the packager.
pub fn handle_opened(app: &AppHandle, paths: Vec<String>) {
    if paths.is_empty() { return }
    let app = app.clone();
    tauri::async_runtime::spawn(async move {
        let result = match add_many(&app, &paths).await {
            Ok(v) => v,
            Err(e) => serde_json::json!({
                "added": 0, "already": 0, "refused": paths.len(),
                "results": paths.iter().map(|p| serde_json::json!({
                    "status": "refused", "file": p,
                    "name": std::path::Path::new(p).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
                    "errors": [e.clone()], "warnings": [],
                })).collect::<Vec<_>>(),
            }),
        };
        if let Ok(mut q) = app.state::<OpenedFiles>().0.lock() {
            q.push(result.clone());
        }
        let _ = app.emit("pigeon-added", result);
    });
}

/// The webview drains this at boot and whenever `pigeon-added` arrives, so
/// a result is shown exactly once whichever way it got there.
#[tauri::command]
pub fn pigeon_take_opens(app: AppHandle) -> Result<Vec<Value>, String> {
    let opened = app.state::<OpenedFiles>();
    let mut q = opened.0.lock().map_err(|e| e.to_string())?;
    Ok(std::mem::take(&mut *q))
}
