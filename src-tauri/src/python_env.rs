//! Detects and creates the managed Python virtualenv Flight School trains through.
//! No terminal use required from the user: "Set Up Environment" in the UI drives this
//! end to end.

use crate::paths;
use crate::pyproc::{self, RunningJob};
use serde::Serialize;
use std::process::Command;
use tauri::{AppHandle, Manager, State};

#[derive(Serialize, Clone)]
pub struct EnvStatus {
    pub system_python_found: bool,
    pub system_python_version: Option<String>,
    /// Below the floor mlx-lm needs (3.9). The rail says so instead of letting
    /// setup run for two minutes and fail inside pip.
    pub python_too_old: bool,
    pub venv_exists: bool,
    pub venv_ready: bool, // venv exists AND mlx-lm is importable in it
    /// Physical memory, GB — the page sizes its training budget from it.
    pub ram_gb: f32,
}

fn find_system_python() -> Option<(String, String)> {
    // Prefer python3 on PATH; macOS ships one via Xcode CLT at /usr/bin/python3, but a
    // Homebrew install is a safer minimum version-wise. Try a few common spots.
    for candidate in [
        "python3",
        "/opt/homebrew/bin/python3",
        "/usr/local/bin/python3",
        "/usr/bin/python3",
    ] {
        if let Ok(out) = Command::new(candidate).arg("--version").output() {
            if out.status.success() {
                let version = String::from_utf8_lossy(&out.stdout)
                    .trim()
                    .to_string();
                let version = if version.is_empty() {
                    String::from_utf8_lossy(&out.stderr).trim().to_string()
                } else {
                    version
                };
                return Some((candidate.to_string(), version));
            }
        }
    }
    None
}

#[tauri::command]
pub async fn python_env_status(app: AppHandle) -> Result<EnvStatus, String> {
    let (found, version) = match find_system_python() {
        Some((_, v)) => (true, Some(v)),
        None => (false, None),
    };
    // The floor is what mlx-lm itself needs. Below it, "set up" would spend two
    // minutes and then fail somewhere inside pip; the rail says it up front.
    let too_old = version.as_deref().map(python_too_old).unwrap_or(false);
    let venv_python = paths::venv_python(&app)?;
    let venv_exists = venv_python.exists();
    let venv_ready = if venv_exists {
        Command::new(&venv_python)
            .args(["-m", "pip", "show", "mlx-lm"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    } else {
        false
    };
    Ok(EnvStatus {
        system_python_found: found,
        system_python_version: version,
        python_too_old: too_old,
        venv_exists,
        venv_ready,
        ram_gb: paths::total_ram_gb(),
    })
}

/// "3.9.6" is fine (with tomli); anything below 3.9 cannot run mlx-lm at all.
///
/// The string comes from `python3 --version`, which prints "Python 3.9.6" —
/// with the word. The first version of this parsed "Python 3" as the major
/// version, got nothing, and told a Mac with 3.9.6 that it needed 3.9 or
/// newer. The number is the last whitespace-separated token.
pub fn python_too_old(version: &str) -> bool {
    let number = version.split_whitespace().last().unwrap_or("");
    let mut it = number.split('.').map(|p| p.trim().parse::<u32>().unwrap_or(0));
    let (major, minor) = (it.next().unwrap_or(0), it.next().unwrap_or(0));
    major < 3 || (major == 3 && minor < 9)
}

#[cfg(test)]
mod version_tests {
    use super::python_too_old;

    /// The exact string that produced the false refusal on the second Mac.
    #[test]
    fn the_xcode_tools_python_is_not_too_old() {
        assert!(!python_too_old("Python 3.9.6"));
        assert!(!python_too_old("3.9.6"));
        assert!(!python_too_old("Python 3.11.8"));
    }

    #[test]
    fn genuinely_old_pythons_are() {
        assert!(python_too_old("Python 3.8.9"));
        assert!(python_too_old("Python 2.7.18"));
        assert!(python_too_old(""));
    }
}

/// Creates a virtualenv at `venv_dir` and installs `requirements_resource` into it.
/// Shared by the training environment and the conversion environment, which are kept
/// separate because their dependency pins are mutually incompatible (see paths.rs).
pub async fn build_venv(
    app: &AppHandle,
    running: &RunningJob,
    job: &str,
    venv_dir: &std::path::Path,
    venv_python: &std::path::Path,
    requirements_resource: &str,
) -> Result<(), String> {
    let (python3, _) = find_system_python().ok_or(
        "No python3 found on this Mac. Install Python 3.10+ (e.g. via Homebrew: brew install python@3.11) and try again.",
    )?;

    // --clear so re-running setup genuinely rebuilds. Without it an existing venv is
    // reused and packages installed by an earlier, different requirements file survive
    // — which is exactly how a stale, incompatible dependency set goes unnoticed.
    pyproc::run_streamed(
        app,
        running,
        job,
        std::path::Path::new(&python3),
        &[
            "-m".into(),
            "venv".into(),
            "--clear".into(),
            venv_dir.to_string_lossy().into_owned(),
        ],
        &[],
        None,
    )
    .await?;

    pyproc::run_streamed(
        app,
        running,
        job,
        venv_python,
        &["-m".into(), "pip".into(), "install".into(), "--upgrade".into(), "pip".into()],
        &[],
        None,
    )
    .await?;

    let requirements = app
        .path()
        .resolve(requirements_resource, tauri::path::BaseDirectory::Resource)
        .map_err(|e| e.to_string())?;

    pyproc::run_streamed(
        app,
        running,
        job,
        venv_python,
        &[
            "-m".into(),
            "pip".into(),
            "install".into(),
            "-r".into(),
            requirements.to_string_lossy().into_owned(),
        ],
        &[],
        None,
    )
    .await
}

#[tauri::command]
pub async fn setup_environment(
    app: AppHandle,
    running: State<'_, RunningJob>,
) -> Result<(), String> {
    if let Some((_, v)) = find_system_python() {
        if python_too_old(&v) {
            return Err(format!(
                "This Mac's python3 is {v}; Flight School needs 3.9 or newer. \
                 Install a newer Python (python.org or Homebrew) and set up again."
            ));
        }
    }
    // Training only. The conversion environment is built on first export, so the path
    // to a first training run doesn't wait on a multi-gigabyte torch download.
    build_venv(
        &app,
        &running,
        "setup",
        &paths::venv_dir(&app)?,
        &paths::venv_python(&app)?,
        "python/requirements-train.txt",
    )
    .await
}

#[tauri::command]
pub async fn stop_job(running: State<'_, RunningJob>) -> Result<(), String> {
    pyproc::stop(&running).await
}

/// The person answered "close anyway" to the guard in lib.rs. Kill the running
/// job and quit for real.
///
/// The guard only holds the FIRST close request, deliberately: it exists to ask
/// the question, not to make the window unclosable. This is the answer, so it
/// does not consult the slot again — the drafting sidecar is killed by the
/// `RunEvent::Exit` arm on the way out.
#[tauri::command]
pub async fn force_quit(app: AppHandle, running: State<'_, RunningJob>) -> Result<(), String> {
    let _ = pyproc::stop(&running).await;
    app.exit(0);
    Ok(())
}

/// Bring an existing training environment up to the shipped requirements
/// before a run — no Terminal, no "set up again".
///
/// The first second-Mac run died on a missing package, and the fix was a
/// pip command relayed by hand to a person sitting at the other keyboard. That
/// is not a workflow. `pip install -r` is idempotent and takes a second when
/// everything is already there, so it runs whenever the shipped requirements
/// file differs from the one the environment was last built against — a
/// hash beside the venv remembers which.
pub async fn ensure_requirements(app: &AppHandle, running: &RunningJob) -> Result<(), String> {
    use std::io::Read;
    let venv_python = paths::venv_python(app)?;
    if !venv_python.exists() {
        return Ok(()); // nothing to heal; setup will say so
    }
    let req = app.path()
        .resolve("python/requirements-train.txt", tauri::path::BaseDirectory::Resource)
        .map_err(|e| e.to_string())?;
    let mut bytes = Vec::new();
    std::fs::File::open(&req).and_then(|mut f| f.read_to_end(&mut bytes)).map_err(|e| e.to_string())?;
    let digest = format!("{:x}", md5_like(&bytes));
    let stamp = paths::venv_dir(app)?.join(".requirements.stamp");
    if std::fs::read_to_string(&stamp).map(|s| s.trim() == digest).unwrap_or(false) {
        return Ok(());
    }
    pyproc::run_streamed(
        app, running, "tools", &venv_python,
        &["-m".into(), "pip".into(), "install".into(), "-q".into(),
          "-r".into(), req.to_string_lossy().into_owned()],
        &[], None,
    ).await?;
    let _ = std::fs::write(&stamp, digest);
    Ok(())
}

/// A stable fingerprint for a small text file. Not cryptographic — it only
/// has to change when the file does.
fn md5_like(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// The second environment — llama.cpp's converter, which pins a transformers
/// the trainer cannot share — built the first time anything needs it.
///
/// The lesson lane built it on first export, so on the machine everything
/// was written on it always existed. The second Mac's first Quick Build got
/// all the way through training and died reaching for it. A run needs both
/// environments; a run makes sure it has both.
pub async fn ensure_convert_env(app: &AppHandle, running: &RunningJob) -> Result<(), String> {
    let convert_python = paths::convert_venv_python(app)?;
    // "Exists" is not "works": the second Mac had a venv-convert whose pip
    // install had died halfway, and a python3 binary was there to find. The
    // test is whether the converter's imports load; if not, rebuild (--clear).
    if convert_python.exists() {
        let ok = Command::new(&convert_python)
            .args(["-c", "import torch, transformers, numpy"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        if ok {
            return Ok(());
        }
    }
    build_venv(
        app, running, "tools",
        &paths::convert_venv_dir(app)?,
        &convert_python,
        "python/requirements-convert.txt",
    ).await
}
