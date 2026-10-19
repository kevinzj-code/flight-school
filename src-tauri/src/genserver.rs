//! The drafting engine: a llama-server sidecar Flight School runs itself, so
//! "have a model write some example rows for me" works with nothing else open.
//!
//! Invariants, mirrored from the earlier Airoplane Mode app's engine.rs because an orphaned
//! llama-server silently holding gigabytes is the worst failure mode either app
//! has:
//!
//!   1. AT MOST ONE child exists. Starting a second kills the first, and the
//!      kill happens before the spawn, never after.
//!   2. The child is killed on stop, on replace, and on Drop.
//!   3. The port is OS-assigned (bind :0, read it back, release), never a
//!      hardcoded one — Flight School and Airoplane Mode may both be open.
//!
//! It deliberately does NOT use pyproc's RunningJob slot. That slot is for
//! short-lived Python jobs and `stop_job` kills whatever is in it; a long-lived
//! inference server parked there would be killed by the Stop button on the
//! Train screen. The two are separate lifetimes, so they get separate state.
//!
//! What they are NOT is separate MEMORY. A resident 9B and an MLX training run
//! cannot coexist on a 16 GB machine, so start_training refuses while a
//! drafting server is up (see training.rs) rather than letting both thrash.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Mutex;
use tauri::{AppHandle, Emitter, Manager};
use tauri_plugin_shell::process::CommandChild;
use tauri_plugin_shell::ShellExt;

use crate::paths;

#[derive(Serialize, Clone, Default)]
pub struct GenStatus {
    pub running: bool,
    pub port: Option<u16>,
    pub model: Option<String>,
    /// How many requests this server can have in flight at once. The frontend
    /// reads it as the width of its drafting fan-out — one slot means the
    /// strictly-sequential behaviour this app had before, which is still what
    /// a machine without headroom gets.
    pub slots: u8,
}

#[derive(Default)]
pub struct GenServer {
    child: Mutex<Option<CommandChild>>,
    pub status: Mutex<GenStatus>,
    /// When it last answered a request — the idle timer reads it.
    pub last_used: Mutex<Option<std::time::Instant>>,
}

/// How long a drafting server may sit idle before it is put away. Loading a
/// 4B model costs 5–15 s, so a second draft inside this window is instant;
/// after it, the memory goes back to whatever else the Mac is doing.
const IDLE_AFTER: std::time::Duration = std::time::Duration::from_secs(5 * 60);

impl GenServer {
    pub fn touch(&self) {
        if let Ok(mut g) = self.last_used.lock() { *g = Some(std::time::Instant::now()); }
    }
}

impl GenServer {
    /// Kill the current child if there is one. Takes the handle out first so a
    /// concurrent caller can't see a half-dead server, and never panics — this
    /// runs from Drop, where a panic would orphan the very thing being killed.
    pub fn kill(&self) {
        let taken = self.child.lock().ok().and_then(|mut g| g.take());
        if let Some(child) = taken {
            let _ = child.kill();
        }
        if let Ok(mut s) = self.status.lock() {
            *s = GenStatus::default();
        }
    }
}

impl Drop for GenServer {
    fn drop(&mut self) {
        self.kill();
    }
}

/// One HTTP client for the whole process, built once.
///
/// `reqwest::Client` owns a connection pool; constructing one per request
/// throws that pool away and pays a fresh TCP handshake every time. Cloning
/// is cheap and is the documented way to share it, so every call below takes
/// a clone of this instead of building its own. It matters more now that the
/// drafting loop can have several requests in flight.
fn http() -> reqwest::Client {
    static CLIENT: std::sync::OnceLock<reqwest::Client> = std::sync::OnceLock::new();
    CLIENT.get_or_init(reqwest::Client::new).clone()
}

/// One random key for the life of the process. llama-server refuses requests
/// without it, so another local process that finds the port cannot use the
/// model or read its `/props`. `/health` stays open, by the server's design.
/// The bytes come from the OS's own generator (arc4random on macOS), not from
/// a hash of the clock.
fn api_key() -> &'static str {
    static KEY: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    KEY.get_or_init(|| {
        let mut bytes = [0u8; 32];
        // SAFETY: arc4random_buf fills exactly `len` bytes of a valid buffer.
        unsafe { libc::arc4random_buf(bytes.as_mut_ptr() as *mut libc::c_void, bytes.len()) };
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    })
}

/// Ask the OS for a free port, then immediately let it go. There is a race
/// between this and llama-server binding it, but it is the same approach the
/// earlier Airoplane Mode app uses and the alternative — a hardcoded port — collides with
/// the other app the moment both are open.
fn free_port() -> Result<u16, String> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").map_err(|e| e.to_string())?;
    let port = listener.local_addr().map_err(|e| e.to_string())?.port();
    drop(listener);
    Ok(port)
}

/// Context window per server slot. Drafting prompts in the lesson flow run
/// the system prompt plus the whole brief plus one reference section — a few
/// thousand tokens — so a slot narrower than this would evict mid-draft.
const CTX_PER_SLOT: u32 = 4096;

/// Headroom, in GB, that must exist before an extra slot is opened.
///
/// Deliberately an OVER-estimate of what a slot's KV actually costs at
/// CTX_PER_SLOT for the models this app serves. The two errors are not
/// symmetric: over-estimating costs a slot, under-estimating costs the
/// swap-thrash failure this whole app is arranged to avoid.
const GB_PER_EXTRA_SLOT: f32 = 0.75;

/// The ceiling below a 32 GB machine. Decode throughput on Metal stops
/// scaling well past a handful of sequences, and the point here is to stop
/// wasting a mostly-idle GPU, not to chase the last percent.
const MAX_SLOTS: u8 = 4;

/// Bigger Macs get a higher ceiling: (budget at or above, cap). 22 GB is a
/// 32 GB machine's budget, 46 GB a 64 GB machine's. Mirrors
/// instructor/engine.py::SLOT_CAPS — the two must agree about what fits.
const SLOT_CAPS: [(f32, u8); 2] = [(46.0, 16), (22.0, 8)];

fn max_slots(budget_gb: f32) -> u8 {
    SLOT_CAPS.iter().find(|(floor, _)| budget_gb >= *floor).map_or(MAX_SLOTS, |(_, cap)| *cap)
}

/// How many requests this machine can afford to have in flight at once against
/// a model of `weights_gb`.
///
/// Generation is bandwidth-bound on the weights, which are read once per decode
/// step no matter how many sequences share it — so two to four concurrent
/// requests cost roughly one request's time. What they DO cost is KV cache, and
/// that is the thing this app must never overcommit. So the answer is
/// arithmetic against the same budget every other memory verdict here uses, and
/// it falls back to 1 — today's behaviour exactly — the moment the headroom
/// isn't there.
///
/// Note this spends only memory that is genuinely free while it is spent: a
/// drafting server and a training run are already refused as a pair
/// (training.rs), so nothing else is holding the budget at this moment.
fn slots_for(weights_gb: f32) -> u8 {
    slots_within(paths::memory_budget_gb(), weights_gb)
}

/// The arithmetic on its own, so it can be tested against a budget instead of
/// against whatever machine happens to be running the tests.
fn slots_within(budget_gb: f32, weights_gb: f32) -> u8 {
    let headroom = budget_gb - weights_gb - 1.0;
    if !headroom.is_finite() || headroom <= 0.0 {
        return 1;
    }
    let extra = (headroom / GB_PER_EXTRA_SLOT).floor();
    (1.0 + extra.clamp(0.0, (max_slots(budget_gb) - 1) as f32)) as u8
}

#[tauri::command]
pub async fn gen_stop(app: AppHandle) -> Result<(), String> {
    app.state::<GenServer>().kill();
    let _ = app.emit("gen-log", serde_json::json!({"line": "drafting model stopped"}));
    Ok(())
}

/// Start llama-server on `model_path` and wait until it answers /health.
///
/// Flags mirror the earlier Airoplane Mode app's engine.rs: everything on Metal, flash
/// attention on, its own web UI off. `--jinja` matters most — without it the
/// model's own chat template is ignored and `chat_template_kwargs` (which is
/// how thinking gets turned off for drafting) has nothing to act on.
#[tauri::command]
pub async fn gen_start(
    app: AppHandle,
    model_path: String,
    // A LoRA adapter to load RESIDENT BUT INERT (--lora-init-without-apply):
    // base behaviour is untouched until a request names it via gen_chat's
    // `lora` array. This is what makes the check ride one server — the stock
    // side and the adapter side are the same weights, differing only in what
    // each request asks for. Verified against this exact binary in the
    // adapter-cache measurements, kept outside this repo.
    lora_path: Option<String>,
) -> Result<GenStatus, String> {
    if !std::path::Path::new(&model_path).is_file() {
        return Err(format!("{model_path} is not a file"));
    }
    if let Some(ref lp) = lora_path {
        if !std::path::Path::new(lp).is_file() {
            return Err(format!("adapter not found at {lp}"));
        }
    }
    /* Already serving this model with no adapter: keep it. The second draft
       of a session does not wait for the model to load again. */
    if lora_path.is_none() {
        let state = app.state::<GenServer>();
        let same = state.status.lock().map(|s| s.running && s.model.as_deref() == Some(model_path.as_str())).unwrap_or(false);
        let alive = state.child.lock().map(|c| c.is_some()).unwrap_or(false);
        if same && alive {
            state.touch();
            return state.status.lock().map(|s| s.clone()).map_err(|e| e.to_string());
        }
    }
    // Kill BEFORE spawn, always. The reverse ordering is how two servers end up
    // alive with only the second one's handle recorded.
    app.state::<GenServer>().kill();

    let port = free_port()?;
    // The GGUF's own size is the floor under what it will occupy resident, and
    // it is the only shape information available without parsing the file.
    let weights_gb = std::fs::metadata(&model_path)
        .map(|m| m.len() as f32 / 1e9)
        .unwrap_or(0.0);
    let slots = slots_for(weights_gb);
    let ctx = CTX_PER_SLOT * slots as u32;
    let mut args: Vec<String> = [
        "-m", &model_path,
        "--port", &port.to_string(),
        "--host", "127.0.0.1",   // localhost only. Never the network.
        "-ngl", "999",
        "-fa", "on",
        "-c", &ctx.to_string(),
        "-np", &slots.to_string(),
        "--no-webui",
        "--jinja",
        "--api-key", api_key(),
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    if let Some(lp) = &lora_path {
        args.extend(["--lora".into(), lp.clone(), "--lora-init-without-apply".into()]);
    }
    let (mut rx, child) = app
        .shell()
        .sidecar("llama-server")
        .map_err(|e| format!("llama-server sidecar not found ({e})"))?
        .args(&args)
        .spawn()
        .map_err(|e| format!("Couldn't start llama-server: {e}"))?;

    {
        let state = app.state::<GenServer>();
        *state.child.lock().map_err(|e| e.to_string())? = Some(child);
    }

    // Forward output so a failed load is visible instead of a silent timeout.
    let app_log = app.clone();
    tauri::async_runtime::spawn(async move {
        use tauri_plugin_shell::process::CommandEvent;
        while let Some(event) = rx.recv().await {
            let line = match event {
                CommandEvent::Stdout(b) | CommandEvent::Stderr(b) => {
                    String::from_utf8_lossy(&b).trim_end().to_string()
                }
                CommandEvent::Terminated(_) => {
                    // The child died on its own; drop the stale handle so
                    // gen_status stops claiming a server that isn't there.
                    app_log.state::<GenServer>().kill();
                    "drafting model exited".to_string()
                }
                _ => continue,
            };
            if !line.is_empty() {
                let _ = app_log.emit("gen-log", serde_json::json!({ "line": line }));
            }
        }
    });

    // Wait for /health. A big model takes a while to load, and answering
    // "ready" before it is would make the first draft request fail for a
    // reason the person cannot see.
    let client = http();
    let url = format!("http://127.0.0.1:{port}/health");
    let mut ok = false;
    for _ in 0..180 {
        if let Ok(resp) = client.get(&url).send().await {
            if resp.status().is_success() {
                ok = true;
                break;
            }
        }
        // Bail out early if the child already died — no point waiting 90s for
        // a server that will never come up.
        {
            let state = app.state::<GenServer>();
            let alive = state.child.lock().map(|g| g.is_some()).unwrap_or(false);
            if !alive {
                return Err("llama-server exited while starting — see the log for why.".into());
            }
        }
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }
    if !ok {
        app.state::<GenServer>().kill();
        return Err("llama-server did not become ready in 90 seconds.".into());
    }

    /* The idle timer: put the server away after IDLE_AFTER without a request. */
    {
        let state = app.state::<GenServer>();
        state.touch();
        let app_idle = app.clone();
        tauri::async_runtime::spawn(async move {
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(30)).await;
                let state = app_idle.state::<GenServer>();
                let running = state.status.lock().map(|s| s.running).unwrap_or(false);
                if !running { break; }
                let idle = state.last_used.lock().ok().and_then(|g| g.map(|t| t.elapsed())).unwrap_or_default();
                if idle >= IDLE_AFTER {
                    state.kill();
                    let _ = app_idle.emit("gen-log", serde_json::json!({ "line": "drafting model put away after five idle minutes" }));
                    break;
                }
            }
        });
    }
    let budget_gb = paths::memory_budget_gb();
    let plural = if slots == 1 { "" } else { "s" };
    let _ = app.emit("gen-log", serde_json::json!({
        "line": format!(
            "{slots} slot{plural} · {ctx} token context · {weights_gb:.1} GB of weights \
             against a {budget_gb:.1} GB budget"
        ),
    }));
    let status = GenStatus {
        running: true,
        port: Some(port),
        model: Some(model_path),
        slots,
    };
    {
        let state = app.state::<GenServer>();
        *state.status.lock().map_err(|e| e.to_string())? = status.clone();
    }
    Ok(status)
}

#[derive(Deserialize)]
pub struct ChatRequest {
    pub system: String,
    pub user: String,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f64>,
    /// Per-request adapter selection: `[{"id":0,"scale":1.0}]` applies the
    /// adapter gen_start loaded; `[]` is the explicit "base weights" request.
    /// Omitted (None) sends no field at all — only correct on a server
    /// started without an adapter.
    #[serde(default)]
    pub lora: Option<Value>,
    /// Whether llama-server may reuse the KV it already has for a matching
    /// prompt prefix. Defaults to TRUE — every drafting prompt in the lesson
    /// flow shares a constant system prompt plus the whole brief, and
    /// re-prefilling those ~2,000 tokens once per answer was the largest
    /// avoidable cost in the app.
    ///
    /// Caching is UNSAFE for one shape of request and one only: alternating
    /// adapter state over identical prompts, where reused KV can silently
    /// carry the other configuration's state (measured; the adapter-cache
    /// notes are kept outside this repo). That case is not left to a caller to
    /// remember: `gen_chat` forces caching off whenever a request names an
    /// adapter, so the unsafe combination cannot be constructed. This field
    /// exists for a caller that wants a cold prefill for some other reason.
    #[serde(default)]
    pub cache_prompt: Option<bool>,
}

/// One plain chat completion against the resident model — the call behind
/// the brief/questions/answers generation prompts AND both sides of the
/// check ride. Returns the reply text; parsing belongs to parsers.rs.
///
/// `cache_prompt` defaults ON, and is forced OFF for any request that names
/// an adapter — see the field's note on ChatRequest for the measurement that
/// rule comes from.
///
/// (The lesson flow's own check ride does not come through here at all: it
/// runs `mlx_lm.generate` per side, in generate.rs, so that it uses the same
/// runtime and chat template training did.)
#[tauri::command]
pub async fn gen_chat(app: AppHandle, req: ChatRequest) -> Result<Value, String> {
    let port = {
        let state = app.state::<GenServer>();
        let s = state.status.lock().map_err(|e| e.to_string())?;
        s.port.ok_or("The model isn't running — start it first.")?
    };
    // A request that selects an adapter never reuses KV, whatever it asked
    // for: that is the one combination measured to carry the other
    // configuration's state across. Everything else caches by default.
    let names_adapter = req.lora.as_ref().is_some_and(|l| l.is_array());
    let cache_prompt = !names_adapter && req.cache_prompt.unwrap_or(true);
    let mut body = serde_json::json!({
        "messages": [
            {"role": "system", "content": req.system},
            {"role": "user", "content": req.user}
        ],
        "temperature": req.temperature.unwrap_or(0.7),
        "max_tokens": req.max_tokens.unwrap_or(700),
        "stream": false,
        "cache_prompt": cache_prompt,
        // Non-negotiable for every drafting call: without this a reasoning model
        // spends the whole budget inside a think block.
        "chat_template_kwargs": {"enable_thinking": false}
    });
    if let Some(lora) = req.lora {
        if lora.is_array() {
            body["lora"] = lora;
        }
    }
    let started = std::time::Instant::now();
    app.state::<GenServer>().touch();
    let client = http();
    let resp = client
        .post(format!("http://127.0.0.1:{port}/v1/chat/completions"))
        .bearer_auth(api_key())
        .json(&body)
        .timeout(std::time::Duration::from_secs(600))
        .send()
        .await
        .map_err(|e| format!("The model didn't answer: {e}"))?;
    let json: Value = resp
        .json()
        .await
        .map_err(|e| format!("The model's reply wasn't JSON: {e}"))?;
    let text = json["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("")
        .to_string();
    if text.is_empty() {
        return Err("The model returned an empty reply.".into());
    }
    // Elapsed time and token count ride back for the check ride's
    // under-output stats line.
    let completion_tokens = json["usage"]["completion_tokens"].as_u64().unwrap_or(0);
    Ok(serde_json::json!({
        "text": text,
        "ms": started.elapsed().as_millis() as u64,
        "tokens": completion_tokens,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Caching defaults ON. The whole point of the field is that a caller who
    /// says nothing gets the fast path — before this, every drafted answer
    /// re-prefilled the system prompt and the entire brief.
    #[test]
    fn cache_prompt_defaults_to_on_when_absent() {
        let req: ChatRequest =
            serde_json::from_str(r#"{"system":"s","user":"u"}"#).unwrap();
        assert_eq!(req.cache_prompt, None);
        assert!(req.cache_prompt.unwrap_or(true));
        let off: ChatRequest =
            serde_json::from_str(r#"{"system":"s","user":"u","cache_prompt":false}"#).unwrap();
        assert_eq!(off.cache_prompt, Some(false));
    }

    /// A machine with no room to spare gets exactly the behaviour it had
    /// before slots existed: one request at a time. This is the fallback the
    /// whole change rests on, so it is pinned.
    #[test]
    fn no_headroom_means_one_slot() {
        assert_eq!(slots_within(10.5, 999.0), 1, "a model larger than the budget");
        assert_eq!(slots_within(10.5, 10.5), 1, "a model exactly filling it");
        assert_eq!(slots_within(4.0, 3.2), 1, "less than one slot of headroom");
        assert_eq!(slots_within(f32::NAN, 1.0), 1, "an unusable budget");
    }

    #[test]
    fn headroom_buys_slots_up_to_the_cap() {
        // 16 GB machine (budget 10.5) serving the 4B (~2.5 GB): 7 GB spare.
        assert_eq!(slots_within(10.5, 2.5), MAX_SLOTS);
        // 8 GB machine (budget 4.5) serving the same: 1 GB spare — one extra.
        assert_eq!(slots_within(4.5, 2.5), 2);
        // Under a 32 GB machine's budget the ceiling stays 4.
        assert_eq!(slots_within(16.5, 2.5), MAX_SLOTS, "24 GB machine");
        // 32 GB (budget 22.5) and 64 GB (budget 46.5) lift it.
        assert_eq!(slots_within(22.5, 2.5), 8);
        assert_eq!(slots_within(46.5, 2.5), 16);
        // Never past the top cap, however much room there is.
        assert_eq!(slots_within(512.0, 2.5), 16);
        // The cap is a ceiling, not a grant: a big model on a 32 GB machine
        // still gets only what its headroom pays for (22.5 - 18 - 1 = 3.5 GB).
        assert_eq!(slots_within(22.5, 18.0), 5);
    }

    #[test]
    fn the_api_key_is_64_hex_chars_from_the_os_generator() {
        let k = api_key();
        assert_eq!(k.len(), 64);
        assert!(k.chars().all(|c| c.is_ascii_hexdigit()));
        assert_ne!(k, "0".repeat(64));
        assert_eq!(k, api_key(), "one key for the life of the process");
    }
}
