//! What is left of Quick Build: the machinery Teach shares with it.
//!
//! The one-click brief→cartridge pipeline and its screens are gone; the
//! table flow (table.rs) trains through the same driver, `python/quickbuild.py
//! table`, and this module keeps what that needs — resolve real paths,
//! generate the config the vendored `python/instructor/` package reads, spawn
//! the driver through pyproc, and lift its `@@QB {json}` progress lines into
//! `qb-event`s for the UI. No pipeline logic lives on this side of the
//! boundary.
//!
//! One run is one folder under app-data/quickbuild/runs/<id> — the same
//! everything-in-one-folder rule everywhere — holding config.toml (paths
//! frozen at start), plan.json, rows.jsonl, exam.jsonl, the dataset,
//! adapters/, export/ (adapter + card + holdout + check-ride.json) and
//! state.json, which is what the UI reloads a run from.
//!
//! The commands still here are the ones the screens call: qb_options (which
//! model drafts), the vision environment, the Icon Maker, and Reveal in Finder.

use crate::paths;
use crate::pyproc::{self, RunningJob};
use serde::Serialize;
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tauri::{AppHandle, Emitter, Manager, State};

// ---------------------------------------------------------------------------
// locations

fn qb_root(app: &AppHandle) -> Result<PathBuf, String> {
    let dir = paths::data_dir(app)?.join("quickbuild");
    std::fs::create_dir_all(dir.join("runs")).map_err(|e| e.to_string())?;
    Ok(dir)
}

/// Where every run folder lives.
pub(crate) fn runs_root(app: &AppHandle) -> Result<PathBuf, String> {
    Ok(qb_root(app)?.join("runs"))
}

pub(crate) fn vision_ready(app: &AppHandle) -> bool {
    paths::vision_venv_python(app).map(|p| p.exists()).unwrap_or(false)
}

/// The model that judges prose at the check ride — the biggest on disk.
pub(crate) fn planner_for_judge(app: &AppHandle) -> Result<PathBuf, String> {
    planner_gguf(app)
}

pub(crate) fn write_run_config(app: &AppHandle, dir: &Path, planner: &Path, student_gguf: &Path,
                               student_repo: &str, mmproj: Option<&Path>) -> Result<PathBuf, String> {
    write_config(app, dir, planner, student_gguf, student_repo, mmproj)
}


pub(crate) fn run_dir(app: &AppHandle, id: &str) -> Result<PathBuf, String> {
    // App-generated ids (qb-<millis>), but guard against a path anyway.
    if id.is_empty() || id.contains(['/', '\\', '.']) {
        return Err("bad run id".into());
    }
    Ok(qb_root(app)?.join("runs").join(id))
}

/// The llama-server this app already bundles as a Tauri sidecar, as a real
/// filesystem path — the Python engine spawns it itself, so a shell-plugin
/// handle is no use. Bundled: beside the executable, its target-triple suffix
/// stripped. Dev: the checked-in binary under src-tauri/binaries.
pub(crate) fn llama_server_path() -> Result<PathBuf, String> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            candidates.push(dir.join("llama-server"));
            candidates.push(dir.join("llama-server-aarch64-apple-darwin"));
        }
    }
    candidates.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("binaries/llama-server-aarch64-apple-darwin"));
    candidates
        .into_iter()
        .find(|p| p.is_file())
        .ok_or_else(|| "llama-server binary not found beside the app or in src-tauri/binaries".into())
}

fn resource(app: &AppHandle, rel: &str) -> Result<PathBuf, String> {
    use tauri::path::BaseDirectory;
    app.path()
        .resolve(rel, BaseDirectory::Resource)
        .map_err(|e| format!("{rel} not bundled: {e}"))
}

// ---------------------------------------------------------------------------
// what this machine can run

/// The trainable pairs Quick Build offers. A pair is only real when BOTH
/// halves exist: the mlx repo trains, the GGUF serves the check ride and is
/// the base the packed .pigeon names. Both pairs were verified
/// end to end in the Instructor pilot (its RESULTS.md).
#[derive(Serialize, Clone)]
pub struct StudentOption {
    pub kind: String,        // "helper" | "answering"
    pub label: String,
    pub note: String,
    pub repo: String,
    pub gguf: Option<String>,
    pub available: bool,
}

/// How faithful a GGUF is to the original weights, from its file name. Lower is
/// closer to the source.
///
/// This exists because the choice is NOT arbitrary. An adapter is fitted against
/// a particular set of base weights, and a LoRA trained on a quantized base
/// absorbs that base's errors; serving it against a different quantization flips
/// exactly the tokens the model is least sure about. Measured (the baseline notes, "The
/// logit test"): the two runtimes' top-2 candidates matched and their
/// probabilities INVERTED — 0.82/0.18 one way, 0.35/0.65 the other — on a
/// coordinate token, and greedy decoding turned that into a different drawing.
///
/// A short-output job survives this; an exact-structure one does not. So when a
/// machine holds several quantizations of the same base, serve the most faithful
/// one rather than whichever sorts first.
fn precision_rank(name: &str) -> u8 {
    let n = name.to_ascii_lowercase();
    for (i, tag) in ["-f32", "-bf16", "-f16", "-fp16", "-q8", "-q6", "-q5", "-q4", "-q3", "-q2"]
        .iter()
        .enumerate()
    {
        if n.contains(tag) {
            return i as u8;
        }
    }
    // No recognisable tag: rank it below every named precision, above nothing.
    200
}

/// The one base the Pigeon app runs every imported adapter on: bartowski's
/// requant, sha256 1adf0b11065d8ad2e8123ea110d1ec956dab4ab038eab665614adba04b6c3370
/// (the reader brief §4).
///
/// It outranks `precision_rank` for a concrete, measured reason. The official
/// Qwen F16 and Q8_0 GGUFs label themselves `qwen2.5-1.5b-instruct` / `1.8B`;
/// this file labels itself `Qwen2.5` / `1.5B`. A Pigeon packed against the F16
/// therefore names a base whose SIZE differs from Pigeon's, so Pigeon refuses it
/// outright as a different model — the emoji designer did exactly that. And a
/// check ride flown on any other file measures a runtime Pigeon never uses.
/// Since 2026-10-02 the file is Qwen3-VL-2B's Q4_K_M, the base every Pigeon trains on.
const PIGEON_CATEGORY_BASE: &str = "Qwen3VL-2B-Instruct-Q4_K_M.gguf";

fn find_gguf(dir: &Path, needle: &str) -> Option<PathBuf> {
    let rd = std::fs::read_dir(dir).ok()?;
    let mut hits: Vec<PathBuf> = rd
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("gguf"))
        .filter(|p| p.file_name().and_then(|n| n.to_str()).map(|n| n.contains(needle)).unwrap_or(false))
        .collect();
    // Pigeon's own base first when it is here; otherwise most faithful first,
    // then by name so the pick is stable across machines.
    hits.sort_by_key(|p| {
        let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string();
        (name != PIGEON_CATEGORY_BASE, precision_rank(&name), name)
    });
    hits.into_iter().next()
}

fn student_options(app: &AppHandle) -> Result<Vec<StudentOption>, String> {
    let models = paths::models_dir(app)?;
    let helper = Some(models.join("Qwen3VL-2B-Instruct-Q4_K_M.gguf")).filter(|p| p.is_file());
    let answering = find_gguf(&models, "Qwen3.5-4B");
    Ok(vec![
        StudentOption {
            kind: "helper".into(),
            label: "Helper — Qwen3-VL 2B".into(),
            note: "For short, exact jobs: sorting, titles, recaps, charts. Pure attention, so it trains fast — a Standard run is a few minutes.".into(),
            // bf16 for the reason in `precision_rank`: an adapter fitted on a
            // quantized base breaks when served against another file.
            repo: "mlx-community/Qwen3-VL-2B-Instruct-bf16".into(),
            available: helper.is_some(),
            gguf: helper.map(|p| p.to_string_lossy().into_owned()),
        },
        StudentOption {
            kind: "answering".into(),
            label: "Answering — Qwen3.5 4B".into(),
            // The honest cost, stated where the choice is made. the Phase A results
            // measured Qwen3.5's hybrid layers at ~60x slower to LoRA-train
            // than a pure-attention model of similar size; a Standard run here
            // is hours, not minutes. Worth it for a persona that has to ride
            // the answering model — but never a surprise.
            note: "For personas and always-on styles, on a base that can hold a whole conversation. Hybrid attention, so training is slow — expect an hour or more, and start with the Quick preset.".into(),
            repo: "mlx-community/Qwen3.5-4B-4bit".into(),
            available: answering.is_some(),
            gguf: answering.map(|p| p.to_string_lossy().into_owned()),
        },
    ])
}

/// Every downloaded GGUF with its size, biggest first.
fn downloaded_ggufs(app: &AppHandle) -> Result<Vec<(u64, PathBuf)>, String> {
    let models = paths::models_dir(app)?;
    let rd = std::fs::read_dir(&models)
        .map_err(|_| "Flight School's models folder is missing — put a GGUF model in it first".to_string())?;
    let mut out: Vec<(u64, PathBuf)> = rd
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("gguf"))
        .map(|p| (std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0), p))
        .collect();
    out.sort_by(|a, b| b.0.cmp(&a.0));
    if out.is_empty() {
        return Err("No GGUF models in Flight School's models folder yet — put one there first".into());
    }
    Ok(out)
}

/// PLANNING is one call and the plan governs everything downstream, so it
/// takes the biggest model on disk.
fn planner_gguf(app: &AppHandle) -> Result<PathBuf, String> {
    Ok(downloaded_ggufs(app)?.remove(0).1)
}

/// Weights this app will read once per decode step, above which a model is
/// too slow to be asked a hundred and fifty times. Measured on the failing
/// run: the 5.3 GB 9B drafted at 7.6 tok/s across four slots — 600-token
/// answers at ~20 s each, ~49 minutes for 150 rows. A ~2.5 GB 4B decodes
/// about 2.5x faster for data the Instructor pilot measured as good enough
/// (19/20 and 25/25 check rides were drafted on the 4B).
const DRAFTING_MAX_BYTES: u64 = 4_000_000_000;

/// DRAFTING is a hundred and fifty calls, so it takes the biggest model that
/// is still quick to decode — the largest under DRAFTING_MAX_BYTES, and the
/// smallest thing available when everything on disk is above it (a slow
/// drafter is still better than none).
fn drafting_gguf(app: &AppHandle) -> Result<PathBuf, String> {
    let all = downloaded_ggufs(app)?;
    Ok(pick_drafter(&all).clone())
}

/// A file in the models folder that is not a model to draft with: a vision
/// base (the students — Qwen3-VL-2B in three precisions since 2026-10-02)
/// or an embedder. Before this, "the biggest under 4 GB" was
/// `Qwen3VL-2B-Instruct-F16.gguf` (3.45 GB): the student drafting for itself,
/// not the 4B the ceiling was measured on.
fn not_a_drafter(p: &Path) -> bool {
    let name = p.file_name().map(|n| n.to_string_lossy().to_lowercase()).unwrap_or_default();
    name.contains("vl-") || name.contains("vl2") || name.contains("vl3") || name.contains("qwen3vl") || name.contains("embed")
}

/// Parameters in billions, from the file name's "-4B-" / "-1.5B-" part, or 0.
fn params_b(p: &Path) -> f64 {
    let name = p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    name.split(['-', '_'])
        .filter_map(|part| part.strip_suffix('B').or_else(|| part.strip_suffix('b')).and_then(|n| n.parse::<f64>().ok()))
        .fold(0.0, f64::max)
}

/// The choice itself, over a biggest-first list — pure, so it can be tested.
/// Text models only, when there are any; under the ceiling, the most
/// parameters, then the most bytes. Bytes alone picked a 1.5B in F16 over a
/// 4B in Q4_K_M: a bigger file, a smaller model.
fn pick_drafter(all: &[(u64, PathBuf)]) -> &PathBuf {
    let text: Vec<&(u64, PathBuf)> = all.iter().filter(|(_, p)| !not_a_drafter(p)).collect();
    let pool: Vec<&(u64, PathBuf)> = if text.is_empty() { all.iter().collect() } else { text };
    pool.iter()
        .filter(|(size, _)| *size <= DRAFTING_MAX_BYTES)
        .max_by(|a, b| params_b(&a.1).total_cmp(&params_b(&b.1)).then(a.0.cmp(&b.0)))
        .map(|(_, p)| p)
        .unwrap_or_else(|| &pool.last().expect("caller guarantees non-empty").1)
}

// ---------------------------------------------------------------------------
// the generated config

/// config.toml for the vendored package, from this app's own real paths.
/// Written once per run so a run's paths are frozen with it — a later change
/// of teacher can never silently redefine what an old run meant.
/// `planner` becomes the config's `instructor_gguf`, which is what the check
/// ride's JUDGE loads — grading prose is the other job where quality beats
/// speed, and the pilot graded with the strongest model it had. The DRAFTER
/// never comes from here; it is passed explicitly as --teacher-gguf.
fn write_config(app: &AppHandle, dir: &Path, planner: &Path, student_gguf: &Path,
                student_repo: &str, mmproj: Option<&Path>) -> Result<PathBuf, String> {
    let llama = llama_server_path()?;
    let venv_train = paths::venv_python(app)?;
    let venv_convert = paths::convert_venv_python(app)?;
    let venv_vision = paths::vision_venv_python(app)?;
    // An absent projector is written as a path that cannot exist, which is
    // what builder.run_build checks before it spends a minute generating —
    // "this is a vision plan and there is no projector" is a refusal worth
    // getting early rather than at the check ride.
    let mmproj_path = mmproj
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|| dir.join("no-mmproj").to_string_lossy().into_owned());
    let mlx_to_peft = resource(app, "python/mlx_adapter_to_peft.py")?;
    let lora_to_gguf = resource(app, "python/vendor/convert_lora_to_gguf.py")?;
    let hf_home = paths::hf_cache_dir(app)?;
    let ram = paths::total_ram_gb();
    let toml = format!(
        r#"# Generated by Flight School's Quick Build — do not edit; regenerated per run.
[binaries]
llama_server = "{llama}"
llama_tokenize = "{llama}"

[models]
instructor_gguf = "{teacher}"
target_gguf = "{student}"
target_mmproj = "{mmproj}"

[mlx]
target_repo = "{repo}"
instructor_repo = "{repo}"
hf_home = "{hf}"

[venvs]
train = "{vt}"
convert = "{vc}"
vision = "{vv}"

[vendor]
mlx_to_peft = "{mp}"
lora_to_gguf = "{lg}"

[limits]
generation_budget_gb = {budget:.1}
total_ram_gb = {ram:.1}
"#,
        llama = llama.display(),
        teacher = planner.display(),
        student = student_gguf.display(),
        mmproj = mmproj_path,
        vv = venv_vision.display(),
        repo = student_repo,
        hf = hf_home.display(),
        vt = venv_train.display(),
        vc = venv_convert.display(),
        mp = mlx_to_peft.display(),
        lg = lora_to_gguf.display(),
        budget = (ram * 0.75 - 1.5).max(4.0),
        ram = ram,
    );
    let path = dir.join("config.toml");
    std::fs::write(&path, toml).map_err(|e| e.to_string())?;
    Ok(path)
}

// ---------------------------------------------------------------------------
// running the driver

/// Same memory refusal training.rs makes: a resident drafting model and this
/// pipeline (which runs its own llama-server AND an MLX training pass) do not
/// fit together on a 16 GB machine.
/// A run and a drafting server cannot share a 16 GB machine, so a run puts
/// the drafter away first. It comes back on the next draft, resident again.
fn refuse_if_drafting(app: &AppHandle) -> Result<(), String> {
    let gen = app.state::<crate::genserver::GenServer>();
    if gen.status.lock().map(|s| s.running).unwrap_or(false) {
        gen.kill();
        let _ = app.emit("gen-log", serde_json::json!({ "line": "drafting model put away for the run" }));
    }
    Ok(())
}

/// Run one driver subcommand for a run folder, streaming its lines.
pub(crate) async fn drive_run(app: AppHandle, running: State<'_, RunningJob>, id: &str, cmd: &str) -> Result<(), String> {
    refuse_if_drafting(&app)?;
    // The tools heal themselves before a run — see python_env::ensure_requirements.
    crate::python_env::ensure_requirements(&app, &running).await?;
    // And a run that trains will repack; the repacker's own environment is a
    // one-time ~750 MB build on a fresh Mac, done here rather than discovered
    // after ten minutes of drafting and training.
    crate::python_env::ensure_convert_env(&app, &running).await?;
    let root = qb_root(&app)?;
    let dir = run_dir(&app, id)?;
    let venv_python = paths::venv_python(&app)?;
    let driver = resource(&app, "python/quickbuild.py")?;
    let hf_home = paths::hf_cache_dir(&app)?;
    let args: Vec<String> = vec![
        driver.to_string_lossy().into_owned(),
        cmd.into(),
        "--config".into(), dir.join("config.toml").to_string_lossy().into_owned(),
        "--run-dir".into(), dir.to_string_lossy().into_owned(),
    ];
    let envs = vec![
        ("HF_HOME".to_string(), hf_home.to_string_lossy().into_owned()),
        ("INSTRUCTOR_ROOT".to_string(), root.to_string_lossy().into_owned()),
        ("PYTHONUNBUFFERED".to_string(), "1".to_string()),
    ];
    // Lift `@@QB {json}` lines into structured events the stepper renders;
    // every line still reaches the log console through py-log as usual.
    let app_for_events = app.clone();
    let run_id = id.to_string();
    let watcher: pyproc::LineWatcher = Arc::new(move |line: &str| {
        if let Some(rest) = line.strip_prefix("@@QB ") {
            if let Ok(mut v) = serde_json::from_str::<Value>(rest) {
                if let Some(obj) = v.as_object_mut() {
                    obj.insert("runId".into(), Value::String(run_id.clone()));
                }
                let _ = app_for_events.emit("qb-event", v);
            }
        }
    });
    pyproc::run_streamed(&app, &running, "quickbuild", &venv_python, &args, &envs, Some(watcher)).await
}

// ---------------------------------------------------------------------------
// commands

#[derive(Serialize)]
pub struct QbOptions {
    /// The model that writes the plan — the biggest on disk.
    pub teacher: Option<String>,
    /// The model that drafts the questions and answers. Usually smaller, and
    /// named separately because it is what the build's wall-clock is made of.
    pub drafter: Option<String>,
    pub teacher_error: Option<String>,
    pub students: Vec<StudentOption>,
}

#[tauri::command]
pub fn qb_options(app: AppHandle) -> Result<QbOptions, String> {
    let (teacher, teacher_error) = match planner_gguf(&app) {
        Ok(p) => (Some(p.to_string_lossy().into_owned()), None),
        Err(e) => (None, Some(e)),
    };
    let drafter = drafting_gguf(&app).ok().map(|p| p.to_string_lossy().into_owned());
    Ok(QbOptions { teacher, drafter, teacher_error, students: student_options(&app)? })
}

/// Is the mlx-vlm environment built? Vision needs its own, and it is created
/// on demand rather than during first-run setup.
#[tauri::command]
pub fn qb_vision_ready(app: AppHandle) -> Result<bool, String> {
    Ok(paths::vision_venv_python(&app)?.exists())
}

/// Build it. Streams pip through the same log console everything else uses.
#[tauri::command]
pub async fn qb_setup_vision(app: AppHandle, running: State<'_, RunningJob>) -> Result<(), String> {
    let venv = paths::vision_venv_dir(&app)?;
    let py = paths::vision_venv_python(&app)?;
    if !py.exists() {
        let sys_py = which_python().ok_or("no system python3 found")?;
        let out = std::process::Command::new(&sys_py)
            .args(["-m", "venv", &venv.to_string_lossy()])
            .output()
            .map_err(|e| format!("could not create the vision environment: {e}"))?;
        if !out.status.success() {
            return Err(format!("could not create the vision environment: {}",
                String::from_utf8_lossy(&out.stderr)));
        }
    }
    // jinja2 is NOT optional and NOT implied: mlx-vlm's dependency list omits
    // it while transformers' apply_chat_template needs it, which showed up as
    // an ImportError at training start rather than at install.
    // Pinned to the versions the picture gate was measured on (2026-09-20).
    let args: Vec<String> = ["-m", "pip", "install", "mlx-vlm[train]==0.7.0", "mlx==0.32.2", "jinja2==3.1.6"]
        .iter().map(|s| s.to_string()).collect();
    pyproc::run_streamed(&app, &running, "setup-vision", &py, &args, &[], None).await
}

fn which_python() -> Option<PathBuf> {
    for c in ["python3", "/opt/homebrew/bin/python3", "/usr/local/bin/python3", "/usr/bin/python3"] {
        if std::process::Command::new(c).arg("--version").output()
            .map(|o| o.status.success()).unwrap_or(false) {
            return Some(PathBuf::from(c));
        }
    }
    None
}


/// The run folders under `runs`, as (id, path). The check ride and the judge
/// keep their server logs beside the run folders (fly.py writes
/// cfg.runs_dir / "fly-server.log"): a file is never a run, and a dotted name
/// that reached run_dir would come back as "bad run id".
pub(crate) fn run_folders(runs: &std::path::Path) -> Vec<(String, PathBuf)> {
    let Ok(rd) = std::fs::read_dir(runs) else { return vec![] };
    rd.filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .filter_map(|p| {
            let id = p.file_name()?.to_str()?.to_string();
            (!id.is_empty() && !id.contains(['/', '\\', '.'])).then_some((id, p))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(sizes: &[u64]) -> Vec<(u64, PathBuf)> {
        let mut v: Vec<(u64, PathBuf)> =
            sizes.iter().map(|s| (*s, PathBuf::from(format!("{s}.gguf")))).collect();
        v.sort_by(|a, b| b.0.cmp(&a.0));
        v
    }

    fn gguf_dir(label: &str, names: &[&str]) -> PathBuf {
        let d = std::env::temp_dir().join(format!("fs-findgguf-{label}"));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        for n in names {
            std::fs::write(d.join(n), b"x").unwrap();
        }
        d
    }

    /// Server logs sit beside the run folders; clicking Runs used to open the
    /// newest entry, a log, and fail with "bad run id".
    #[test]
    fn run_folders_skip_the_server_logs_beside_them() {
        let d = std::env::temp_dir().join("fs-runfolders");
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(d.join("qb-1789395142924")).unwrap();
        std::fs::create_dir_all(d.join("qb-baseline-2026-09-09")).unwrap();
        std::fs::write(d.join("fly-server.log"), b"x").unwrap();
        std::fs::write(d.join("judge-server.log"), b"x").unwrap();
        let mut ids: Vec<String> = run_folders(&d).into_iter().map(|(id, _)| id).collect();
        ids.sort();
        assert_eq!(ids, vec!["qb-1789395142924", "qb-baseline-2026-09-09"]);
        let _ = std::fs::remove_dir_all(&d);
    }

    /// Pigeon runs every imported adapter on one file. A more faithful copy on
    /// disk must not win, or the packed manifest names a base Pigeon refuses —
    /// the official F16 labels itself 1.8B where Pigeon's labels itself 1.5B.
    #[test]
    fn pigeons_pinned_base_is_served_even_when_a_more_faithful_copy_exists() {
        let d = gguf_dir("pinned", &["Qwen3VL-2B-Instruct-F16.gguf",
                                     "Qwen3VL-2B-Instruct-Q8_0.gguf",
                                     "Qwen3VL-2B-Instruct-Q4_K_M.gguf"]);
        assert_eq!(find_gguf(&d, "Qwen3VL-2B").unwrap().file_name().unwrap(), PIGEON_CATEGORY_BASE);
        let _ = std::fs::remove_dir_all(&d);
    }

    /// Any base Pigeon does not pin still gets the most faithful copy, rather than
    /// whichever name happens to sort first.
    #[test]
    fn other_bases_still_get_the_most_faithful_quantization() {
        let d = gguf_dir("faithful", &["Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
                                       "Qwen3-4B-Instruct-2507-Q8_0.gguf"]);
        assert_eq!(find_gguf(&d, "Qwen3-4B").unwrap().file_name().unwrap(),
                   "Qwen3-4B-Instruct-2507-Q8_0.gguf");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn precision_rank_orders_the_tags_it_knows_and_puts_unknowns_last() {
        assert!(precision_rank("m-bf16.gguf") < precision_rank("m-Q8_0.gguf"));
        assert!(precision_rank("m-Q8_0.gguf") < precision_rank("m-Q4_K_M.gguf"));
        assert!(precision_rank("m-Q4_K_M.gguf") < precision_rank("m-plain.gguf"));
        // Case must not matter: converters disagree about it.
        assert_eq!(precision_rank("m-q4_k_m.gguf"), precision_rank("m-Q4_K_M.gguf"));
    }

    /// The failing run's own shape: a 9B, a 4B, a 2B and a 1.5B on disk.
    /// Planning takes the 9B; drafting must NOT — it decoded at 7.6 tok/s
    /// across four slots, ~49 minutes for 150 rows.
    #[test]
    fn drafting_prefers_the_biggest_quick_model_not_the_biggest_model() {
        let v = set(&[5_300_000_000, 2_500_000_000, 1_600_000_000, 1_050_000_000]);
        assert_eq!(v[0].0, 5_300_000_000, "planning still takes the biggest");
        assert_eq!(pick_drafter(&v), &PathBuf::from("2500000000.gguf"));
    }

    /// Biggest UNDER the ceiling, not smallest overall — drafting quality
    /// still matters, it just may not cost 2.5x the wall clock.
    #[test]
    fn drafting_takes_the_largest_under_the_ceiling() {
        let v = set(&[9_000_000_000, 3_900_000_000, 900_000_000]);
        assert_eq!(pick_drafter(&v), &PathBuf::from("3900000000.gguf"));
    }

    /// Everything on disk is huge: a slow drafter beats refusing to build,
    /// so the smallest available is used rather than erroring.
    #[test]
    fn all_models_above_the_ceiling_falls_back_to_the_smallest() {
        let v = set(&[9_000_000_000, 5_300_000_000]);
        assert_eq!(pick_drafter(&v), &PathBuf::from("5300000000.gguf"));
    }

    /// This Mac's folder on 2026-10-16: the biggest file under the ceiling was
    /// the 2B vision base in F16. Drafting must skip the vision bases and the
    /// embedder and take the 4B text model.
    #[test]
    fn drafting_skips_the_vision_bases_and_the_embedder() {
        let v: Vec<(u64, PathBuf)> = [
            (5_966_095_584, "Qwen3.5-9B-UD-Q4_K_XL.gguf"), (4_280_405_600, "Qwen3-4B-Instruct-2507-Q8_0.gguf"),
            (3_560_416_288, "Qwen2.5-1.5B-Instruct-F16.gguf"), (3_447_350_304, "Qwen3VL-2B-Instruct-F16.gguf"),
            (2_740_937_888, "Qwen3.5-4B-Q4_K_M.gguf"), (1_107_409_952, "Qwen3VL-2B-Instruct-Q4_K_M.gguf"),
            (1_834_427_424, "Qwen3VL-2B-Instruct-Q8_0.gguf"), (146_146_432, "nomic-embed-text-v1.5.Q8_0.gguf"),
        ].iter().map(|(s, n)| (*s, PathBuf::from(n))).collect();
        assert_eq!(pick_drafter(&v), &PathBuf::from("Qwen3.5-4B-Q4_K_M.gguf"),
                   "the 4B text model, not the 1.5B in F16 that is a bigger file");
        assert_eq!(params_b(Path::new("Qwen2.5-1.5B-Instruct-F16.gguf")), 1.5);
        assert_eq!(params_b(Path::new("Qwen3.5-9B-UD-Q4_K_XL.gguf")), 9.0);
        let only_vision = vec![(3_447_350_304, PathBuf::from("Qwen3VL-2B-Instruct-F16.gguf"))];
        assert_eq!(pick_drafter(&only_vision), &PathBuf::from("Qwen3VL-2B-Instruct-F16.gguf"), "a vision base beats no drafter");
    }

    /// A projector belongs to ONE base. Handing a 4B cartridge the 9B's
    /// projector is not a fallback, it is a wrong answer that loads.
    #[test]
    fn one_model_is_both_planner_and_drafter() {
        let v = set(&[5_300_000_000]);
        assert_eq!(pick_drafter(&v), &v[0].1);
    }
}

/// What the packager is told before every command (python/pigeon/host.py):
/// the sealed base registry this app ships (D73), the key list it trusts for
/// registries — separate from any Pigeon key list (§15.2) — and the app's one
/// cache of model hashes, kept in its data folder and never beside a model.
/// A part that cannot be found is left out; the packager then carries every
/// projector, as it did before the registry was sealed.
fn host_args(app: &AppHandle) -> Vec<String> {
    let mut out = Vec::new();
    for (flag, path) in [("--registry", resource(app, "registry/registry.zip")),
                         ("--registry-keys", resource(app, "keys/registry-keys.json")),
                         ("--hash-cache", paths::data_dir(app).map(|d| d.join("pigeon-hashes.json")))] {
        if let Ok(p) = path {
            out.push(flag.to_string());
            out.push(p.to_string_lossy().into_owned());
        }
    }
    out
}

/// Open the app's registry once at launch, only to say in the log whether it
/// is used. Every command opens it again for itself (it is 6.5 KB).
pub(crate) async fn log_registry(app: AppHandle) {
    match pigeon_cli(&app, &["registry".to_string()]).await {
        Ok(r) if r["opened"].as_bool() == Some(true) =>
            eprintln!("[registry] opened: version {}, bases {}", r["registry_version"], r["bases"]),
        Ok(_) => eprintln!("[registry] not used: every picture Pigeon packed now carries its projector"),
        Err(e) => eprintln!("[registry] not checked at launch: {e}"),
    }
}

/// Run the packager (python/pigeon/cli.py) and hand back its one-line JSON.
/// Not streamed: `draft` is instant and `pack` is a few seconds of hashing,
/// and both reply with a document the UI renders rather than a log.
pub(crate) async fn pigeon_cli(app: &AppHandle, args: &[String]) -> Result<Value, String> {
    let python = paths::venv_python(app)?;
    if !python.exists() {
        return Err("The training tools aren't set up yet — the packager runs inside them.".into());
    }
    let script = resource(app, "python/pigeon/cli.py")?;
    let out = tokio::process::Command::new(&python)
        .arg(script)
        .args(host_args(app))
        .args(args)
        // A caller that gives up (qb_pigeon_look's timeout) must not leave the
        // packager running on its own.
        .kill_on_drop(true)
        .output()
        .await
        .map_err(|e| e.to_string())?;
    // A registry that did not open is not fatal (D44), but the log says so.
    for line in String::from_utf8_lossy(&out.stderr).lines().filter(|l| l.starts_with("[registry]")) {
        eprintln!("{line}");
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    let reply: Value = serde_json::from_str(stdout.trim()).map_err(|_| {
        format!("the packager did not reply: {}",
                String::from_utf8_lossy(&out.stderr).lines().last().unwrap_or("no output"))
    })?;
    if let Some(err) = reply.get("error").and_then(Value::as_str) {
        return Err(err.to_string());
    }
    Ok(reply)
}

/// The card's words from the Package step. Empty means "use the run's own",
/// so only text someone actually typed is passed on.
pub fn push_card_args(args: &mut Vec<String>, does: Option<String>, input_hint: Option<String>, output_noun: Option<String>) {
    for (flag, v) in [("--does", does), ("--input-hint", input_hint), ("--output-noun", output_noun)] {
        if let Some(v) = v.map(|s| s.trim().to_string()).filter(|s| !s.is_empty()) {
            args.push(flag.into());
            args.push(v);
        }
    }
}

/// A look for the Pigeon being packaged, from the Icon Maker when the lineup
/// has one (a Pigeon carrying the `pigeon-look` skill), asked with its name
/// and one line about it. Never an error the page must handle: no Icon Maker,
/// a slow one, or an answer that is not a look each come back as the name's
/// random look with the reason. Python bounds the ask itself at 60 s so it can
/// stop its own engine; the 90 s here is only the backstop.
#[tauri::command]
pub async fn qb_pigeon_look(app: AppHandle, name: String, line: Option<String>) -> Result<Value, String> {
    let args = vec![
        "look".to_string(),
        "--store".into(), paths::pigeons_dir(&app)?.to_string_lossy().into_owned(),
        "--name".into(), name,
        "--line".into(), line.unwrap_or_default(),
        "--llama-server".into(), llama_server_path()?.to_string_lossy().into_owned(),
        "--models".into(), paths::models_dir(&app)?.to_string_lossy().into_owned(),
        "--timeout".into(), "60".into(),
    ];
    match tokio::time::timeout(std::time::Duration::from_secs(90), pigeon_cli(&app, &args)).await {
        Ok(reply) => reply,
        Err(_) => Err("the Icon Maker took longer than 90 seconds".into()),
    }
}

/// Show a path in the Finder. Only the app's own data folder: the webview
/// names what to reveal, and a path from there is the only kind it should
/// ever hold.
#[tauri::command]
pub async fn qb_reveal(app: AppHandle, path: String) -> Result<(), String> {
    let root = paths::data_dir(&app)?;
    let p = std::path::Path::new(&path);
    if !p.starts_with(&root) || !p.exists() {
        return Err("That file is not where it was saved any more.".into());
    }
    std::process::Command::new("open").arg("-R").arg(p).spawn().map_err(|e| e.to_string())?;
    Ok(())
}
