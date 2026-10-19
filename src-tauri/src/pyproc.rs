//! Generic "spawn a process, stream its stdout/stderr as Tauri events" primitive.
//! No sidecar precedent exists for this in the Airoplane Mode codebase (the earlier Airoplane Mode
//! app's engine.rs uses a pre-bundled sidecar binary); this is the new pattern for a
//! runtime-created venv's python3. python_env.rs, training.rs, adapter.rs, and
//! generate.rs all funnel through here so the frontend needs one log-console
//! component, not one per feature.

use serde::Serialize;
use std::path::Path;
use std::process::Stdio;
use std::sync::Arc;
use tauri::{AppHandle, Emitter};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;

/// Called with every stdout line, in addition to it being forwarded as a `py-log`
/// event — lets a caller (training.rs) derive a lighter progress event without
/// re-parsing the log console client-side.
pub type LineWatcher = Arc<dyn Fn(&str) + Send + Sync>;

/// One line of output from a running job, forwarded to the frontend as a `py-log` event.
#[derive(Clone, Serialize)]
pub struct PyLogLine {
    pub job: String,
    pub stream: String, // "stdout" | "stderr"
    pub line: String,
}

/// Single-slot guard for whatever job is currently running (setup, train, fuse/convert,
/// or generate). Flight School only ever runs one job at a time, so a single Option is
/// enough — no generation counter needed the way the earlier Airoplane Mode app's Engine has.
///
/// "Only ever runs one job at a time" used to be a claim the frontend was
/// solely responsible for keeping, and it did not: nothing stopped a lesson's
/// "Start training" while a Quick Build run was live, because the two sit
/// behind separate busy flags that never consulted each other. The slot was
/// then OVERWRITTEN rather than defended, so the first child went untracked —
/// two trainers competing for the memory of a machine the app's own pre-flight
/// says fits one, and a Stop button that could only reach the second.
///
/// The slot now refuses. See `run_streamed`.
#[derive(Default)]
pub struct RunningJob(pub Mutex<Option<Child>>);

const TAIL_LINES_FOR_ERROR: usize = 20;

/// Spawn `program args...` with the given extra environment variables, stream every
/// line of stdout/stderr as a `py-log` event tagged with `job`, and wait for exit.
/// On non-zero exit, the error message includes the last ~20 lines of stderr so a pip
/// failure or Python traceback is actually visible in the UI, not just "exit code 1".
pub async fn run_streamed(
    app: &AppHandle,
    running: &RunningJob,
    job: &str,
    program: &Path,
    args: &[String],
    envs: &[(String, String)],
    on_stdout_line: Option<LineWatcher>,
) -> Result<(), String> {
    // Claim the slot BEFORE spawning. A busy slot is a hard refusal, not an
    // overwrite: whatever is already running keeps both its process and its
    // claim on Stop.
    //
    // This closes the window where two jobs overlap, not the narrower one
    // where a multi-step run (draft, then train, then fly) is between its own
    // steps and the slot is briefly empty. The frontend guards that case; this
    // is the backstop for when it doesn't.
    {
        let guard = running.0.lock().await;
        if guard.is_some() {
            return Err(format!(
                "Can't start {job} — another job is already running on this Mac. \
                 Wait for it to finish, or stop it first."
            ));
        }
    }

    let mut cmd = Command::new(program);
    cmd.args(args)
        .envs(envs.iter().cloned())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);

    let mut child = cmd.spawn().map_err(|e| {
        format!(
            "Failed to start {} {}: {e}",
            program.display(),
            args.join(" ")
        )
    })?;

    let stdout = child.stdout.take().expect("piped stdout");
    let stderr = child.stderr.take().expect("piped stderr");

    // A deque, not a Vec: this is the per-stderr-line path, and dropping the
    // oldest line with Vec::remove(0) shifts the whole buffer every time.
    let stderr_tail: std::sync::Arc<std::sync::Mutex<std::collections::VecDeque<String>>> =
        std::sync::Arc::new(std::sync::Mutex::new(std::collections::VecDeque::new()));

    let app_out = app.clone();
    let job_out = job.to_string();
    let out_task = tokio::spawn(async move {
        let mut lines = BufReader::new(stdout).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            if let Some(watcher) = &on_stdout_line {
                watcher(&line);
            }
            let _ = app_out.emit(
                "py-log",
                PyLogLine {
                    job: job_out.clone(),
                    stream: "stdout".into(),
                    line,
                },
            );
        }
    });

    let app_err = app.clone();
    let job_err = job.to_string();
    let tail_err = stderr_tail.clone();
    let err_task = tokio::spawn(async move {
        let mut lines = BufReader::new(stderr).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            if let Ok(mut tail) = tail_err.lock() {
                tail.push_back(line.clone());
                if tail.len() > TAIL_LINES_FOR_ERROR {
                    tail.pop_front();
                }
            }
            let _ = app_err.emit(
                "py-log",
                PyLogLine {
                    job: job_err.clone(),
                    stream: "stderr".into(),
                    line,
                },
            );
        }
    });

    // Park the child in the shared slot so a stop_* command can kill it concurrently,
    // then poll for exit without holding the lock across the wait (that would block
    // stop() from ever acquiring it).
    *running.0.lock().await = Some(child);
    let status = loop {
        let mut guard = running.0.lock().await;
        match guard.as_mut() {
            None => break None, // stop() took it out and killed it
            Some(child_ref) => match child_ref.try_wait() {
                Ok(Some(status)) => break Some(Ok(status)),
                Ok(None) => {
                    drop(guard);
                    tokio::time::sleep(std::time::Duration::from_millis(150)).await;
                }
                Err(e) => break Some(Err(e)),
            },
        }
    };
    *running.0.lock().await = None;

    let _ = out_task.await;
    let _ = err_task.await;

    match status {
        None => Err(format!("{job} was stopped")),
        Some(Ok(s)) if s.success() => Ok(()),
        Some(Ok(s)) => {
            let tail = stderr_tail
                .lock()
                .map(|t| t.iter().cloned().collect::<Vec<_>>().join("\n"))
                .unwrap_or_default();
            Err(format!(
                "{job} exited with {s}\n--- last output ---\n{tail}"
            ))
        }
        Some(Err(e)) => Err(format!("{job} failed: {e}")),
    }
}

/// Kill whatever job is currently running, if any. Returns Ok even if nothing was running.
pub async fn stop(running: &RunningJob) -> Result<(), String> {
    let mut guard = running.0.lock().await;
    if let Some(mut child) = guard.take() {
        child.kill().await.map_err(|e| e.to_string())?;
    }
    Ok(())
}
