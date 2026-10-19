//! A Reader's material on disk: the files a person drops in, read to text,
//! chunked (`chunker.rs`) and kept as `<run>/material/chunks.json` — the
//! exact file `pigeon/manifest.py::material_dir` looks for, so a run with
//! this folder packs as `kind: grounded` with the material inside it.
//!
//! Text comes from three readers: the file itself for markdown, HTML and
//! plain text; PDFKit for a PDF's text layer; and Vision's text recognition
//! for a PDF page that has no text layer. The last two run through
//! `osascript`, which every Mac has, with `pdftext.js` beside this file —
//! no Python, nothing to download.
//!
//! `sources.json` beside the chunks is the panel's memory of each file (its
//! size, page count, pages read by recognition); the chunk index itself stays
//! exactly the shape Pigeon parses, with nothing extra in it.

use std::path::{Path, PathBuf};

use serde_json::{json, Value};
use tauri::AppHandle;

use crate::chunker::{chunk_file, kind_for_ext, Chunk, ChunkIndex, RefKind};
use crate::fsutil::{read_json, write_atomic};
use crate::quickbuild::run_dir;

/// Under this many estimated tokens the whole material is pasted into every
/// ask; over it, the section that best matches the question. The same number
/// as `WHOLE_MATERIAL_CEILING` in `pigeon/manifest.py`, which decides the
/// manifest's `retrieval`, and `RULES.wholeMaterial` in `src/state.js`.
pub const WHOLE_MATERIAL_CEILING: u32 = 3000;

const PDFTEXT: &str = include_str!("pdftext.js");

pub fn material_dir(run: &Path) -> PathBuf { run.join("material") }
fn chunks_path(run: &Path) -> PathBuf { material_dir(run).join("chunks.json") }
fn sources_path(run: &Path) -> PathBuf { material_dir(run).join("sources.json") }

/// What one file became: its text and, for a PDF, how it was read.
struct Extracted {
    kind: RefKind,
    text: String,
    pages: Option<u32>,
    ocr_pages: u32,
}

/// A PDF's pages as text, through PDFKit and Vision (pdftext.js).
fn pdf_text(path: &Path) -> Result<(Vec<String>, u32), String> {
    let out = std::process::Command::new("/usr/bin/osascript")
        .arg("-l").arg("JavaScript").arg("-e").arg(PDFTEXT).arg(path)
        .output()
        .map_err(|e| format!("could not start macOS's text reader: {e}"))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        return Err(format!("macOS could not read this PDF: {}", err.trim()));
    }
    let v: Value = serde_json::from_slice(&out.stdout).map_err(|e| format!("the PDF reader answered oddly: {e}"))?;
    if let Some(e) = v["error"].as_str() {
        return Err(e.to_string());
    }
    let pages: Vec<String> = v["pages"].as_array().into_iter().flatten().filter_map(|p| p.as_str().map(str::to_string)).collect();
    Ok((pages, v["ocr"].as_u64().unwrap_or(0) as u32))
}

fn extract(path: &Path) -> Result<Extracted, String> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    let kind = kind_for_ext(ext).ok_or_else(|| format!("{} is not a kind of file a Reader can read (.md, .txt, .html, .pdf).", name_of(path)))?;
    if kind == RefKind::Pdf {
        let (pages, ocr) = pdf_text(path)?;
        let n = pages.len() as u32;
        // A page break is a paragraph break; a line that PDFKit wrapped stays
        // a line, and the heading promotion in chunker.rs does the rest.
        let text = pages.iter().map(|p| p.trim()).filter(|p| !p.is_empty()).collect::<Vec<_>>().join("\n\n");
        if text.trim().is_empty() {
            return Err(format!("{} has no readable text, even by recognition.", name_of(path)));
        }
        return Ok(Extracted { kind, text, pages: Some(n), ocr_pages: ocr });
    }
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", name_of(path)))?;
    let text = String::from_utf8_lossy(&bytes).into_owned();
    if text.trim().is_empty() {
        return Err(format!("{} is empty.", name_of(path)));
    }
    Ok(Extracted { kind, text, pages: None, ocr_pages: 0 })
}

fn name_of(p: &Path) -> String { p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default() }

/// Every readable file among what was dropped: a folder is walked one level
/// deep, hidden files are skipped, and the order is by name so ordinals are
/// the same every time.
fn readable_files(paths: &[PathBuf]) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for p in paths {
        if p.is_dir() {
            let mut inner: Vec<PathBuf> = std::fs::read_dir(p).map(|rd| rd.filter_map(|e| e.ok()).map(|e| e.path())
                .filter(|f| f.is_file() && !name_of(f).starts_with('.') && kind_for_ext(f.extension().and_then(|e| e.to_str()).unwrap_or("")).is_some())
                .collect()).unwrap_or_default();
            inner.sort();
            out.extend(inner);
        } else if p.is_file() {
            out.push(p.clone());
        }
    }
    out
}

fn read_index(run: &Path) -> ChunkIndex {
    read_json(&chunks_path(run)).and_then(|v| serde_json::from_value(v).ok()).unwrap_or_default()
}

fn read_sources(run: &Path) -> Vec<Value> {
    read_json(&sources_path(run)).and_then(|v| v.as_array().cloned()).unwrap_or_default()
}

/// Ordinals run 1..n in file order, whatever was added or removed; the
/// citation label in a row is only right while it matches the index it was
/// drafted against, and `src/state.js` keeps a material version for that.
fn renumber(chunks: &mut [Chunk]) {
    for (i, c) in chunks.iter_mut().enumerate() {
        c.ordinal = i as u32 + 1;
    }
}

fn write_all(run: &Path, index: &ChunkIndex, sources: &[Value]) -> Result<(), String> {
    std::fs::create_dir_all(material_dir(run)).map_err(|e| e.to_string())?;
    write_atomic(&chunks_path(run), serde_json::to_vec_pretty(index).map_err(|e| e.to_string())?.as_slice())?;
    write_atomic(&sources_path(run), serde_json::to_vec_pretty(&Value::Array(sources.to_vec())).map_err(|e| e.to_string())?.as_slice())
}

/// What the page draws: each file with its section count and size, the
/// totals, and the sections themselves (title and text, for drafting).
fn summary(index: &ChunkIndex, sources: &[Value]) -> Value {
    let files: Vec<Value> = sources.iter().map(|s| {
        let file = s["file"].as_str().unwrap_or("");
        let mine: Vec<&Chunk> = index.chunks.iter().filter(|c| c.file == file).collect();
        json!({
            "file": file,
            "sections": mine.len(),
            "est_tokens": mine.iter().map(|c| c.est_tokens).sum::<u32>(),
            "pages": s["pages"],
            "ocr_pages": s["ocr_pages"],
            "bytes": s["bytes"],
        })
    }).collect();
    json!({
        "files": files,
        "sections": index.chunks.len(),
        "est_tokens": index.total_est_tokens,
        "whole": index.total_est_tokens <= WHOLE_MATERIAL_CEILING,
        "chunks": index.chunks,
    })
}

/// Add files to a run's material: chosen in a dialog when `paths` is empty,
/// or handed over by a drop. A file already in by that name is replaced. The
/// answer is the new summary, plus what could not be read, by name.
#[tauri::command]
pub async fn table_material_add(app: AppHandle, id: String, paths: Option<Vec<String>>) -> Result<Option<Value>, String> {
    let dir = run_dir(&app, &id)?;
    let picked: Vec<PathBuf> = match paths.filter(|p| !p.is_empty()) {
        Some(p) => p.into_iter().map(PathBuf::from).collect(),
        None => {
            use tauri_plugin_dialog::DialogExt;
            let app2 = app.clone();
            let chosen = tauri::async_runtime::spawn_blocking(move || {
                app2.dialog().file().set_title("The document the Reader will answer from")
                    .add_filter("Documents", &["md", "markdown", "txt", "text", "html", "htm", "pdf"])
                    .blocking_pick_files()
            }).await.map_err(|e| e.to_string())?;
            let Some(files) = chosen else { return Ok(None) };
            files.into_iter().filter_map(|f| f.into_path().ok()).collect()
        }
    };
    let files = readable_files(&picked);
    if files.is_empty() {
        return Err("Nothing there a Reader can read. It takes .md, .txt, .html and .pdf files, or a folder of them.".into());
    }
    let dir2 = dir.clone();
    tauri::async_runtime::spawn_blocking(move || add_files(&dir2, &files)).await.map_err(|e| e.to_string())?.map(Some)
}

fn add_files(run: &Path, files: &[PathBuf]) -> Result<Value, String> {
    let mut index = read_index(run);
    let mut sources = read_sources(run);
    let mut problems: Vec<String> = Vec::new();
    for f in files {
        let name = name_of(f);
        let got = match extract(f) { Ok(x) => x, Err(e) => { problems.push(e); continue } };
        index.chunks.retain(|c| c.file != name);
        sources.retain(|s| s["file"].as_str() != Some(&name));
        let mut fresh = chunk_file(&name, got.kind, &got.text, 1);
        if fresh.is_empty() {
            problems.push(format!("{name} has no text to make sections from."));
            continue;
        }
        index.chunks.append(&mut fresh);
        sources.push(json!({
            "file": name, "path": f.to_string_lossy(), "bytes": std::fs::metadata(f).map(|m| m.len()).unwrap_or(0),
            "pages": got.pages, "ocr_pages": got.ocr_pages, "added": chrono_now(),
        }));
    }
    if index.chunks.is_empty() {
        return Err(problems.first().cloned().unwrap_or_else(|| "Nothing could be read.".into()));
    }
    renumber(&mut index.chunks);
    index.total_est_tokens = index.chunks.iter().map(|c| c.est_tokens).sum();
    write_all(run, &index, &sources)?;
    let mut out = summary(&index, &sources);
    out["problems"] = json!(problems);
    Ok(out)
}

/// The time as the rest of the app writes it, without a date crate.
fn chrono_now() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    secs.to_string()
}

/// One file out of the material, by name; the rest is renumbered.
#[tauri::command]
pub fn table_material_remove(app: AppHandle, id: String, file: String) -> Result<Option<Value>, String> {
    let dir = run_dir(&app, &id)?;
    let mut index = read_index(&dir);
    let mut sources = read_sources(&dir);
    index.chunks.retain(|c| c.file != file);
    sources.retain(|s| s["file"].as_str() != Some(&file));
    if index.chunks.is_empty() {
        let _ = std::fs::remove_dir_all(material_dir(&dir));
        return Ok(None);
    }
    renumber(&mut index.chunks);
    index.total_est_tokens = index.chunks.iter().map(|c| c.est_tokens).sum();
    write_all(&dir, &index, &sources)?;
    Ok(Some(summary(&index, &sources)))
}

/// The material as it is on disk, or nothing.
#[tauri::command]
pub fn table_material_get(app: AppHandle, id: String) -> Result<Option<Value>, String> {
    let dir = run_dir(&app, &id)?;
    let index = read_index(&dir);
    if index.chunks.is_empty() { return Ok(None) }
    Ok(Some(summary(&index, &read_sources(&dir))))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunker::est_tokens;

    #[test]
    fn a_folder_is_walked_one_level_and_filtered_to_readable_files() {
        let dir = std::env::temp_dir().join(format!("fs-material-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("deeper")).unwrap();
        std::fs::write(dir.join("b.md"), "# B\nbody").unwrap();
        std::fs::write(dir.join("a.txt"), "text").unwrap();
        std::fs::write(dir.join("c.png"), "not text").unwrap();
        std::fs::write(dir.join(".hidden.md"), "# no").unwrap();
        std::fs::write(dir.join("deeper").join("d.md"), "# deep").unwrap();
        let got: Vec<String> = readable_files(&[dir.clone()]).iter().map(|p| name_of(p)).collect();
        assert_eq!(got, vec!["a.txt", "b.md"]);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn adding_replaces_a_file_of_the_same_name_and_renumbers() {
        let run = std::env::temp_dir().join(format!("fs-run-{}", std::process::id()));
        std::fs::create_dir_all(&run).unwrap();
        let a = run.join("a.md"); std::fs::write(&a, "# One\nx\n# Two\ny").unwrap();
        let b = run.join("b.md"); std::fs::write(&b, "# Three\nz").unwrap();
        let s = add_files(&run, &[a.clone(), b.clone()]).unwrap();
        assert_eq!(s["sections"], 3);
        std::fs::write(&a, "# Only\nx").unwrap();
        let s = add_files(&run, &[a]).unwrap();
        assert_eq!(s["sections"], 2);
        let ords: Vec<u64> = s["chunks"].as_array().unwrap().iter().map(|c| c["ordinal"].as_u64().unwrap()).collect();
        assert_eq!(ords, vec![1, 2]);
        assert_eq!(s["files"].as_array().unwrap().len(), 2);
        assert!(s["whole"].as_bool().unwrap());
        std::fs::remove_dir_all(&run).unwrap();
    }

    #[test]
    fn est_tokens_agrees_with_the_manifest_side() {
        // chars / 3.5, rounded up — manifest.py::est_tokens uses the same.
        assert_eq!(est_tokens("abcdefg"), 2);
        assert_eq!(est_tokens(""), 0);
    }

    /// PDFKit and Vision, through osascript: a PDF made on this Mac with a
    /// text layer reads back. Skipped where cupsfilter is missing.
    #[test]
    fn a_pdf_with_a_text_layer_reads_through_pdfkit() {
        let dir = std::env::temp_dir().join(format!("fs-pdf-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let txt = dir.join("s.txt");
        std::fs::write(&txt, "Article I\n\nSection 1\nAll legislative Powers herein granted shall be vested in a Congress.\n").unwrap();
        let pdf = dir.join("s.pdf");
        let made = std::process::Command::new("/usr/sbin/cupsfilter").arg(&txt).output();
        let Ok(out) = made else { return };
        if !out.status.success() || out.stdout.is_empty() { return }
        std::fs::write(&pdf, &out.stdout).unwrap();
        let got = extract(&pdf).unwrap();
        assert_eq!(got.kind, RefKind::Pdf);
        assert!(got.text.contains("legislative"), "got {:?}", got.text);
        assert_eq!(got.pages, Some(1));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
