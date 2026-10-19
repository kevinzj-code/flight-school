use std::path::PathBuf;
use tauri::{AppHandle, Manager};

/// Root of everything this app owns on disk: `~/Library/Application Support/mode.airoplane.flightschool/`.
pub fn data_dir(app: &AppHandle) -> Result<PathBuf, String> {
    let dir = app.path().app_data_dir().map_err(|e| e.to_string())?;
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    Ok(dir)
}

/// The training virtualenv: mlx-lm and friends.
///
/// Training and GGUF conversion get SEPARATE environments on purpose. mlx-lm requires
/// transformers>=5.0.0; llama.cpp's convert_hf_to_gguf.py pins transformers==4.57.6.
/// In one environment pip resolves that conflict by quietly installing an older
/// mlx-lm with a different CLI. They never run in the same process, so two venvs cost
/// nothing and keep both at the versions they were verified against.
pub fn venv_dir(app: &AppHandle) -> Result<PathBuf, String> {
    Ok(data_dir(app)?.join("venv"))
}

/// Path to the training venv's python3. Does not check it exists — callers should
/// check `python_env::status` first and surface "run Set Up Environment" if it's missing.
pub fn venv_python(app: &AppHandle) -> Result<PathBuf, String> {
    Ok(venv_dir(app)?.join("bin").join("python3"))
}

/// Physical RAM, GB. The one place this machine's size is asked for.
pub fn total_ram_gb() -> f32 {
    let bytes = unsafe {
        let mut v: u64 = 0;
        let mut len = std::mem::size_of::<u64>();
        let name = std::ffi::CString::new("hw.memsize").unwrap();
        if libc::sysctlbyname(
            name.as_ptr(),
            &mut v as *mut u64 as *mut _,
            &mut len,
            std::ptr::null_mut(),
            0,
        ) == 0
        {
            v
        } else {
            17_179_869_184 // assume 16 GB when the machine won't say
        }
    };
    bytes as f32 / 1e9
}

/// What this app may use of it: the same 75% Metal rule the earlier Airoplane Mode app
/// applies, minus 1.5 GB for the OS and this app itself. Every memory verdict
/// in Flight School is measured against this one number.
pub fn memory_budget_gb() -> f32 {
    total_ram_gb() * 0.75 - 1.5
}

/// The conversion virtualenv: llama.cpp's converter and its pinned deps, including a
/// large CPU torch wheel. Created on first export rather than during initial setup, so
/// getting to a first training run stays fast.
pub fn convert_venv_dir(app: &AppHandle) -> Result<PathBuf, String> {
    Ok(data_dir(app)?.join("venv-convert"))
}

pub fn convert_venv_python(app: &AppHandle) -> Result<PathBuf, String> {
    Ok(convert_venv_dir(app)?.join("bin").join("python3"))
}

/// The VISION virtualenv: mlx-vlm and its own dependency tree.
///
/// A THIRD environment, and it has to be. mlx-lm cannot train a VLM at all —
/// its Qwen3.5 loader drops every `vision_tower.*` and `model.visual.*` key by
/// design — so vision goes through mlx-vlm, which LoRAs the language model
/// with the vision tower frozen. That is also the only shape a GGUF adapter
/// can represent, since llama.cpp handles vision through a separate mmproj
/// file. Kept apart from the training venv for the same reason venv-convert
/// is: pip resolving both dependency sets at once quietly downgrades one of
/// them, and the version each was verified against is the point.
/// Created on demand by the vision setup, never during first-run setup — most
/// people never train a vision cartridge and should not pay for the download.
pub fn vision_venv_dir(app: &AppHandle) -> Result<PathBuf, String> {
    Ok(data_dir(app)?.join("venv-vlm"))
}

pub fn vision_venv_python(app: &AppHandle) -> Result<PathBuf, String> {
    Ok(vision_venv_dir(app)?.join("bin").join("python3"))
}

/// The Hugging Face cache training reads the MLX copies from (via HF_HOME) —
/// in Pigeon's store, not the user's global ~/.cache/huggingface, and not a
/// folder of Flight School's own. See pigeonstore.rs.
pub fn hf_cache_dir(_app: &AppHandle) -> Result<PathBuf, String> {
    crate::pigeonstore::training()
}

/// The lineup: every Pigeon this Mac has, whether made here or added from a
/// file. `<id>.pigeon` beside `<id>.json` (when and where it came from). The
/// library reads this folder and nothing else. See python/pigeon/store.py.
pub fn pigeons_dir(app: &AppHandle) -> Result<PathBuf, String> {
    let dir = data_dir(app)?.join("pigeons");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    Ok(dir)
}

/// The GGUF models Flight School plans, drafts, tests and packs with: the
/// teacher, the drafter, and every base a Pigeon can name. Since 2026-09-21
/// they live in PIGEON's store, and Flight School borrows them — see
/// pigeonstore.rs. (History: first read out of Airoplane Mode's folder, moved
/// into Flight School's own on 2026-09-17, and into Pigeon's on 2026-09-21,
/// each time with a symlink left behind so old run folders still resolve.)
pub fn models_dir(_app: &AppHandle) -> Result<PathBuf, String> {
    crate::pigeonstore::store()
}
