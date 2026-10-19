/* engine.js — the seam, made real (Milestone 5).
 *
 * Every screen talks to this object and to nothing else below it: it saves a
 * run's table, reads the pictures it names, records a window, trains, grades
 * and packs — each through a named Rust command (src-tauri/src/table.rs,
 * recorder.rs, quickbuild.rs). It replaces prototype/engine.fake.js, keeping
 * its method names, so the screens above it did not change shape.
 *
 * Numbers on screen are of two kinds, and each is tagged where it is shown:
 *   MEASURED  taken on the M1 Pro 16 GB, used for estimates before a run.
 *   (none)    what the run itself reported: iterations, loss, speed, memory,
 *             every check-ride answer.
 */

import { S, trainingSet, exam, answerList, labelOf, tableRows, markHoldout, blankRun, legacyKind, setMaterial, materialBlock, readerRow, graftPlan, partnerKindsFor, PARTNER_KINDS, sampleSections, addRows, citedOrdinals, unsupportedCitations, annotateReaderRow, draftProblems, nextRowId, repeatGroups, conflictPairs, pictureSide, sourceSide, seenSide, recorded, nextStep, currentStage } from './state.js';

const tauri = window.__TAURI__;
const invoke = (cmd, args) => tauri.core.invoke(cmd, args);
const listen = (ev, fn) => tauri.event.listen(ev, fn);
const redraw = () => { if (typeof window !== 'undefined') window.dispatchEvent(new Event('proto:render')); };

export const MEASURED = {
  vision3b: {
    small: { tokensPerFrame: 57, peakGb: 4.03, itPerSec: 2.1, source: 'the earlier 3B picture base on 160×144 pictures, 2026-09-20' },
    photo: { tokensPerFrame: 597, peakGb: 9.90, itPerSec: 0.236, source: 'the earlier 3B picture base on 768×576 pictures, 450 iterations, 2026-09-20' },
    loss: [[0, 1.9], [450, 0.030]]
  },
  vision2b: { photo: { peakGb: 6.26, itPerSec: 0.345, source: 'Qwen2-VL-2B on 768×576 images, 2026-09-20' } },
  // The picture base since 2026-10-02. Only small pictures measured so far.
  qwen3vl2b: { small: { tokensPerFrame: 64, peakGb: 5.23, itPerSec: 1.69, source: 'Qwen3-VL-2B, bf16, on 256×256 Seed Run boards, 2026-10-02' } },
  text: {
    // Qwen3-VL-2B trained on bf16, 2026-10-02 (were 6 and 0.9 it/s on Qwen2.5-1.5B)
    short: { peakGb: 3.8, itPerSec: 8.2, source: 'a text job with short rows on Qwen3-VL-2B (bf16), 2026-10-02' },
    long: { peakGb: 5.1, itPerSec: 1.7, source: 'a Reader with passages of up to about 1,100 tokens on Qwen3-VL-2B (bf16), 2026-10-02' }
  },
  pilot: { peakGb: 10.808, itPerSec: 0.5, source: 'a photo job on Qwen3.5-4B, before MLX’s cache was capped' },
  budgetGb: 10.5,
  // vision: Qwen3-VL-2B trained on bf16, Seed Run, 2026-10-02 (was 30.0 on the earlier 3B base)
  adapterMb: { text: 21.3, vision: 17.5 },
  // mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf (was 844.8 on the earlier 3B base)
  projectorMb: 445.1,
  visionBaseMb: 1929.9,
  /* Frame stacking, Milestone 4 (the redesign milestones). */
  stacking: { small: [57, 87, 112], full: [1361, 2695, 4000] }
};

/* Iterations per preset (instructor/planspec.py). Standard unless chosen on
   Teach: a picture job on quick (200) was measured still learning when it
   stopped — loss 0.41, one answer for every picture (2026-09-21). */
export const ITERS = { quick: 200, standard: 600, thorough: 1200 };
export const itersFor = (r) => ITERS[r.preset || 'standard'];

/* What a picture of a given longer side costs the 3B, 4:3 assumed: tokens
   are its pixels over 28 × 28 plus the frame's own; speed and memory are
   drawn between the two measured runs (160 and 768 pixels). The ends are
   the measurements; everything between is an estimate, and says so. */
export function visionCost (side) {
  const { small, photo } = MEASURED.vision3b;
  if (side <= 160) return { ...small, measured: true };
  if (side >= 768) return { ...photo, measured: true };
  const tokens = Math.round(side * side * 0.75 / 784 + 28);
  const t = Math.log(tokens / small.tokensPerFrame) / Math.log(photo.tokensPerFrame / small.tokensPerFrame);
  return {
    tokensPerFrame: tokens,
    itPerSec: +(small.itPerSec * Math.pow(photo.itPerSec / small.itPerSec, t)).toFixed(3),
    peakGb: +(small.peakGb + (photo.peakGb - small.peakGb) * (tokens - small.tokensPerFrame) / (photo.tokensPerFrame - small.tokensPerFrame)).toFixed(2),
    source: 'an estimate between the 160-pixel and 768-pixel runs of the earlier 3B picture base',
    measured: false
  };
}

/* What goes in the file, from what the packer measured (table_parts →
   pack.py `file_parts`). Whether the projector is in it is the packer's own
   rule, so this never disagrees with the file: `projector` is 'carried',
   'stock' (named, not carried: it comes with the base), 'none' for a text
   Pigeon, 'pending' while the packer is asked, or 'unknown' when it could not
   answer — then the projector is shown as carried, the old estimate. `base`
   is the base the run trained against, never a default. */
export function fileParts (r, got) {
  const mb = (b, fallback) => (b ? b / 1e6 : fallback);
  const known = Boolean(got?.base);
  const base = known ? got.base : null;
  const projector = r.input !== 'image' ? 'none' : got === undefined ? 'pending' : !known ? 'unknown' : got.projector_carried ? 'carried' : 'stock';
  const rows = [];
  rows.push({ what: 'The adapter', detail: base ? `what it learned, on ${base.name}` : 'what it learned', mb: mb(got?.adapter, r.input === 'image' ? MEASURED.adapterMb.vision : MEASURED.adapterMb.text), tag: got?.adapter ? null : 'measured' });
  if (projector === 'carried' || projector === 'unknown') rows.push({ what: 'The projector', detail: 'how the base turns a picture into something it can read', mb: mb(got?.projector, MEASURED.projectorMb), tag: got?.projector ? null : 'measured' });
  if (known ? got.labels > 0 : r.output === 'function') rows.push({ what: 'labels.txt', detail: 'the answers it may give, generated', mb: known ? got.labels / 1e6 : 0.001, tag: null });
  rows.push({ what: 'The card', detail: 'its name, what it does, examples from the exam it passed', mb: known ? got.card / 1e6 : 0.01, tag: null });
  return { rows, total: rows.reduce((s, p) => s + p.mb, 0), base, projector };
}

export function cost (r, baseOverride) {
  const b = baseOverride || r.base;
  if (r.input === 'image') {
    if (b === 'Qwen2-VL-2B') return { ...MEASURED.vision2b.photo, measured: true };
    return visionCost(r.picture === 'small' && !r.maxSide ? 160 : seenSide(r));
  }
  return r.output === 'prose' && r.grounded ? MEASURED.text.long : MEASURED.text.short;
}

export const minutesFor = (r, iters = itersFor(r), base) => Math.max(1, Math.round(iters / cost(r, base).itPerSec / 60));

/* What training may use on THIS Mac: its memory less what macOS and the
   window keep (measured: 5.5 GB on the 16 GB M1 Pro the numbers come from,
   which is where 10.5 came from). Until the Mac has answered, the measured
   machine's budget stands in. */
export const ramGb = () => S.env?.py?.ram_gb || null;
export const budgetGb = () => { const g = ramGb(); return g ? Math.max(4, Math.round((g - 5.5) * 10) / 10) : MEASURED.budgetGb; };

/* The words every recorded row is asked with (recorder.rs QUESTION). */
export const RECORD_QUESTION = 'What do you do next?';

let changed = () => {};
const run = (id) => S.runs[id];

/* ── saving ────────────────────────────────────────────────────────────── */

const saving = new Map();
/* The table is written whole, a moment after the last change, so a burst of
   edits is one write. Nothing is kept only in the page. */
function persist (r) {
  if (!r || r.edge || r._stub) return;
  clearTimeout(saving.get(r.id));
  saving.set(r.id, setTimeout(() => { saving.delete(r.id); save(r); }, 350));
}

/* A save that fails is said on screen and stays said until one succeeds
   (S.ui.saveError, drawn by app.js). Nothing is kept only in the page, so a
   silent failure here is the one way to lose an hour of work. */
/* What the runs list shows for a run, worked out here and saved beside the
   table (summary.json), so opening the app reads one small file per run
   instead of every table whole. */
export function summaryOf (r) {
  const s = r.ride?.state === 'done' ? summarise(r.ride) : null;
  return {
    name: r.name, kind: r.kind, input: r.input, output: r.output, reasoning: r.reasoning, grounded: r.grounded,
    pigeon: r.pigeon, sent: r.sent ? { revision: r.sent.revision, at: r.sent.at } : null, base: r.base, updated: r.updated,
    rows: tableRows(r).length, next: nextStep(r), teaching: r.teach?.state === 'running', stage: currentStage(r), done: Boolean(r.sent) && currentStage(r) === 'send',
    score: !s ? null : s.kind === 'prose' ? { kind: 'prose', good: s.good, n: s.n } : { kind: 'function', accuracy: s.accuracy },
    /* The Bench's two columns. */
    bench: !s ? null : { passed: r.ride.passed, total: r.ride.total, control: r.ride.controlPassed ?? null, judged: (r.ride.gradedByModel || 0) > 0 }
  };
}

const lastSaved = new Map();   // run id → the JSON last written, so a click that changed nothing writes nothing
function save (r) {
  if (r._stub) return Promise.resolve();
  const { _parts, _stub, _chunks, ...table } = r;
  table.summary = summaryOf(r);
  const json = JSON.stringify(table);
  if (lastSaved.get(r.id) === json) return Promise.resolve();
  lastSaved.set(r.id, json);
  return invoke('table_save', { id: r.id, table })
    .then(() => { if (S.ui.saveError) { S.ui.saveError = null; changed(r); } })
    .catch(e => { lastSaved.delete(r.id); S.ui.saveError = { text: String(e), run: r.id, at: Date.now() }; console.error('[save]', e); changed(r); });
}

/* Rust only hands out a picture the saved table names (table.rs resolve), so
   anything that reads the pictures of rows just added saves first. */
function saveNow (r) {
  if (!r || r.edge) return Promise.resolve();
  clearTimeout(saving.get(r.id));
  saving.delete(r.id);
  return save(r);
}

/* ── pictures ──────────────────────────────────────────────────────────── */

/* A picture is a URL WebKit fetches itself — pic://localhost/<run>/<path>,
   served by table::serve_picture — so nothing crosses the bridge as base64
   and nothing is kept in the page's memory. The browser preview hands in its own resolver instead. A picture asked for
   before its table is saved gets a 404 and is asked again (app.js). */
const enc = (p) => String(p).split('/').map(encodeURIComponent).join('/');
function image (r, path, { full = false } = {}) {
  if (!r || !path) return null;
  if (typeof tauri.picUrl === 'function') return tauri.picUrl(r.id, path, full);
  return `pic://localhost/${encodeURIComponent(r.id)}/${enc(path)}${full ? '?full=1' : ''}`;
}

/* Prints kept for a run that is gone, or being left: forgotten. */
function forgetImages (id) {
  for (const k of [...prints.keys()]) if (k.startsWith(`${id}|`)) prints.delete(k);
}

/* ── loading ───────────────────────────────────────────────────────────── */

/* A run recorded before tables existed (Milestone 4) becomes a table once. */
function fromRecording (id, name, rec) {
  const r = blankRun(id);
  Object.assign(r, { name: name || 'Recorded job', input: 'image', output: 'function', answerSet: 'record', answers: ['NONE'], base: 'Qwen3-VL-2B', question: RECORD_QUESTION });
  addRecording(r, rec);
  return r;
}


/* A recording's sessions, into the table: every session not already in it.
   Its keys and places keep the recorder's ids, so a later session's clicks
   land on the place they joined — renamed or merged here or not. A place
   clicked only once is a stray: its frame is kept, and it becomes a row only
   if you make an action of it. */
export function addRecording (r, rec) {
  const fresh = (rec.sessions || []).map(s => s.name).filter(s => !r.sessionsIn.includes(s));
  if (!fresh.length) return 0;
  const via = (id) => { let x = id; for (let i = 0; i < 8 && r.alias[x]; i++) x = r.alias[x]; return x; };
  const byId = new Map(rec.actions.map(a => [a.id, a]));
  const strayIds = new Set(rec.actions.filter(a => a.kind === 'click' && !a.name && a.uses < 2).map(a => a.id));
  for (const a of rec.actions) {
    const id = via(a.id);
    const mine = r.actions.find(x => x.id === id);
    if (strayIds.has(a.id) && !mine) continue;
    if (mine) { mine.rect = a.rect || mine.rect; if (mine.code == null && a.code != null) mine.code = a.code; continue; }
    r.actions.push({ id: a.id, kind: a.kind, raw: a.raw || null, name: null, rect: a.rect || null, uses: 0, ...(a.code != null ? { code: a.code } : {}) });
  }
  let added = 0;
  for (const row of rec.rows) {
    if (!fresh.includes(row.session)) continue;
    const input = { kind: 'image', path: row.image, ...(row.click ? { click: row.click } : {}), ...(row.hold_ms != null ? { hold: row.hold_ms } : {}) };
    if (row.action && strayIds.has(row.action) && !r.actions.some(x => x.id === via(row.action))) {
      r.strays.push({ x: row.click?.[0] ?? 0, y: row.click?.[1] ?? 0, path: row.image, session: row.session, decision: null });
      continue;
    }
    const answer = row.action ? `@${via(row.action)}` : 'NONE';
    const act = row.action ? r.actions.find(x => x.id === via(row.action)) : null;
    if (act) act.uses++;
    r.rows.push({ id: nextRowId(r), input, answer, session: row.session, fresh: true });
    added++;
  }
  if (!r.answers.includes('NONE') && rec.rows.some(x => !x.action)) r.answers.push('NONE');
  const w = rec.window || {};
  if (w.w && w.h) r.picture = Math.max(w.w, w.h) <= 400 ? 'small' : 'full';
  r.window = { w: w.w, h: w.h, app: w.app, title: w.title };
  r.sessionsIn.push(...fresh);
  if (!r.teach) markHoldout(r, r.holdoutSeed);
  r.version++;
  r.updated = new Date().toISOString();
  return added;
}

/* ── recording ─────────────────────────────────────────────────────────── */

let recListening = false;
function listenRecorder () {
  if (recListening) return;
  recListening = true;
  listen('rec-event', async (e) => {
    const p = e.payload || {};
    const rec = S.ui.record;
    const r = rec?.runId ? run(rec.runId) : null;
    if (!rec || !r) return;
    if (p.type === 'ready' && p.window) rec.windowInfo = p.window;
    if (p.type === 'kept') { rec.kept = p.kept; rec.last = p.last; }
    if (p.type === 'error') { rec.error = p.words; if (p.code === 'no-screen-permission' || p.code === 'no-input-permission') rec.step = 'denied'; }
    if (p.type === 'hotkey') rec.stopping = Date.now();
    if (p.type === 'secure-input') rec.secure = !!p.on;
    if (p.type === 'stopped') {
      clearInterval(rec.tick);
      clearTimeout(rec.watchdog);
      const table = await invoke('rec_table', { run: r.id });
      const before = r.rows.length;
      addRecording(r, table);
      rec.added = r.rows.slice(before).map(x => x.id);
      rec.session = p.session;
      rec.seconds = Math.round((Date.now() - rec.started) / 1000);
      if (rec.step !== 'denied') rec.step = 'done';
      persist(r);
      findConflicts(r);
    }
    changed(r);
  });
}

async function recPermission () { try { return await invoke('rec_permission'); } catch { return { screen: false, input: false }; } }

/* ── conflicts ─────────────────────────────────────────────────────────── *
 * Two rows that show the same input and carry different answers. Text: the
 * same words. Pictures: the same pixels — each picture at 32 × 32 grey levels
 * (fs-recorder `hash`), no part more than REPEAT.exactTol apart. The 64-bit
 * fingerprint only picks which pairs to compare: on a game board it cannot
 * see a small sprite move at all (0 bits apart on 550 Seed Run pairs whose
 * right key differed), so on its own it called every board a conflict. A pair
 * already settled keeps its settlement. */
const MAX_PAIRS = 2000;

let conflictsBusy = null;
async function findConflicts (r) {
  if (conflictsBusy === r.id) return;
  conflictsBusy = r.id;
  try { await conflictsOf(r); } finally { conflictsBusy = null; }
}

/* ── comparing, off the page ───────────────────────────────────────────
   The pairwise loops run in a Web Worker (compare.js) with a slim copy of the
   rows; the page never waits on them. Where there is no Worker (the tests),
   the same functions run here. */
let worker = null, seq = 0;
const waiting = new Map();
const slimRows = (r) => ({ input: r.input, rows: r.rows.map(x => ({ id: x.id, answer: x.answer, holdout: !!x.holdout, left: x.left || null, draft: !!x.draft, input: { kind: x.input.kind, text: x.input.text, path: x.input.path } })) });
function compare (kind, r, prints, opts) {
  const run = slimRows(r);
  if (typeof Worker === 'undefined') {
    return Promise.resolve(kind === 'repeats'
      ? { groups: repeatGroups(run, prints, opts).map(g => ({ key: g.key, ids: g.rows.map(x => x.id), answer: g.answer, exam: g.exam })) }
      : { pairs: conflictPairs(run, prints, opts) });
  }
  if (!worker) {
    worker = new Worker(new URL('./compare.js', import.meta.url), { type: 'module' });
    worker.onmessage = (e) => { const w = waiting.get(e.data.id); if (!w) return; waiting.delete(e.data.id); e.data.error ? w.reject(new Error(e.data.error)) : w.resolve(e.data); };
    worker.onerror = (e) => { for (const w of waiting.values()) w.reject(new Error(e.message || 'the comparison could not run')); waiting.clear(); worker = null; };
  }
  return new Promise((resolve, reject) => { const id = ++seq; waiting.set(id, { resolve, reject }); worker.postMessage({ id, kind, run, prints, opts }); });
}

async function conflictsOf (r) {
  await saveNow(r);
  const rows = tableRows(r).filter(x => !x.holdout && x.answer != null);
  let got = {};
  if (r.input === 'image') {
    try { got = await picturePrints(r, rows.filter(x => x.input.path).map(x => x.input.path)); }
    catch (e) { S.ui.conflictError = String(e); console.error('[fingerprints]', e); changed(r); return; }
  }
  let found;
  try { ({ pairs: found } = await compare('conflicts', r, got, { maxPairs: MAX_PAIRS })); }
  catch (e) { S.ui.conflictError = String(e); console.error('[conflicts]', e); changed(r); return; }
  S.ui.conflictError = null;
  const old = new Map(r.conflicts.map(c => [`${c.a}|${c.b}`, c]));
  r.conflicts = found.map(([a, b, sim], i) => old.get(`${a}|${b}`) || { id: `c${i + 1}-${a}`, a, b, similarity: sim, resolution: null });
  /* A row left out for a conflict that is no longer one — the old
     fingerprint-only test's false alarm — comes back. */
  const still = new Set(r.conflicts.flatMap(c => [c.a, c.b]));
  let back = 0;
  for (const x of r.rows) if (x.left === 'conflict' && !still.has(x.id)) { x.left = null; back++; }
  if (back) r.version++;
  r.conflictsAt = r.version;
  persist(r);
  changed(r);
}

/* ── picture prints ────────────────────────────────────────────────────
   Each picture's fingerprint and 32 × 32 signature, asked of Rust once per
   picture and kept here. Conflicts and repeats both judge on them. */
const prints = new Map();   // `${run}|${path}` → { hi, lo, sig }
const unpack = ({ hash, sig }) => {
  const bin = atob(sig);
  const px = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) px[i] = bin.charCodeAt(i);
  return { hi: parseInt(hash.slice(0, 8), 16) | 0, lo: parseInt(hash.slice(8), 16) | 0, sig: px };
};

async function picturePrints (r, paths) {
  const want = [...new Set(paths)];
  const missing = want.filter(p => !prints.has(`${r.id}|${p}`));
  if (missing.length) {
    const fresh = await invoke('table_signatures', { id: r.id, paths: missing });
    for (const [p, v] of Object.entries(fresh || {})) prints.set(`${r.id}|${p}`, unpack(v));
  }
  const got = {};
  for (const p of want) { const v = prints.get(`${r.id}|${p}`); if (v) got[p] = v; }
  return got;
}

async function findRepeats (r) {
  const mode = S.ui.repeatMode, tol = S.ui.repeatTol;
  const ask = { state: 'working', mode, tol, version: r.version, run: r.id, groups: S.ui.repeats?.groups || [] };
  S.ui.repeats = ask;
  try {
    const got = r.input === 'image' ? await picturePrints(r, r.rows.filter(x => !x.left && !x.draft && x.input.path).map(x => x.input.path)) : {};
    if (S.ui.repeats !== ask) return;
    const { groups } = await compare('repeats', r, got, { mode, tol });
    if (S.ui.repeats !== ask) return;
    const byId = new Map(r.rows.map(x => [x.id, x]));
    S.ui.repeats = { ...ask, state: 'done', groups: groups.map(g => ({ ...g, rows: g.ids.map(id => byId.get(id)).filter(Boolean) })), unread: r.input === 'image' ? r.rows.filter(x => !x.left && !x.draft && x.input.path && !got[x.input.path]).length : 0 };
  } catch (e) {
    if (S.ui.repeats === ask) S.ui.repeats = { ...ask, state: 'failed', error: String(e) };
  }
  changed(r);
}

/* ── this Mac ──────────────────────────────────────────────────────────── *
 * What the Mac has: the training tools, the picture tools, and Pigeon with its
 * models. Asked once at launch and again after a setup — so a fresh Mac learns
 * on the first screen, not at Teach, what it still needs. */
async function refreshEnv () {
  const [py, models, vision] = await Promise.all([
    invoke('python_env_status').catch(e => ({ error: String(e) })),
    invoke('models_home_status').catch(e => ({ pigeon_present: false, message: String(e) })),
    invoke('qb_vision_ready').catch(() => false)
  ]);
  S.env = { ...(S.env || {}), py, models, vision, checked: true };
  redraw();
}

/* What stands between this run and teaching, in the order to fix it. Each
   item names what it is; the ones the app can fix itself carry `setup`. */
export function needs (r) {
  const e = S.env;
  if (!e?.checked) return [];
  const out = [];
  if (!e.models?.pigeon_present) out.push({ id: 'pigeon', text: e.models?.message || 'Flight School keeps its models in Pigeon, and Pigeon is not on this Mac yet. Install Pigeon and open it once, then reopen Flight School.' });
  if (e.py?.python_too_old) out.push({ id: 'python', text: `This Mac's Python is ${e.py.system_python_version || 'too old'} — Flight School needs 3.9 or newer, from python.org or Homebrew.` });
  else if (e.py?.system_python_found === false) out.push({ id: 'python', text: 'This Mac has no python3. Install the Xcode command-line tools (xcode-select --install in Terminal) or Python from python.org, then check again.' });
  else if (!e.py?.venv_ready) out.push({ id: 'env', text: 'The training tools are not set up on this Mac yet. Setting up happens once and takes about two minutes.', setup: 'env' });
  if (r?.input === 'image' && !e.vision) out.push({ id: 'vision', text: 'Picture training needs its own tools, set up once.', setup: 'vision' });
  return out;
}

let setupListening = false;
function listenSetup () {
  if (setupListening) return;
  setupListening = true;
  listen('py-log', (e) => {
    const p = e.payload || {};
    if (!S.env?.busy || (p.job !== 'setup' && p.job !== 'setup-vision')) return;
    const line = String(p.line || '').trim();
    if (!line) return;
    S.env.line = line.length > 96 ? `${line.slice(0, 94)}…` : line;
    for (const el of document.querySelectorAll('[data-live="setup-line"]')) el.textContent = S.env.line;
  });
}

async function setup (which) {
  if (S.env?.busy) return;
  S.env = { ...(S.env || {}), busy: which, line: 'Starting…', error: null };
  listenSetup();
  redraw();
  try { await invoke(which === 'vision' ? 'qb_setup_vision' : 'setup_environment'); }
  catch (e) { S.env.error = String(e); }
  S.env.busy = null;
  await refreshEnv();
}

/* ── teaching ──────────────────────────────────────────────────────────── */

const questionOf = (r, x) => x.input.kind === 'text' ? x.input.text : (r.question || RECORD_QUESTION);
function answerText (r, x) {
  if (r.output === 'prose') return x.answer;
  const label = labelOf(r, x.answer);
  return r.reasoning ? `<think>\n${x.think || ''}\n</think>\n<do>\n${label}\n</do>` : label;
}
const asPipelineRow = (r) => (x) => ({
  question: questionOf(r, x),
  answer: answerText(r, x),
  group: r.output === 'prose' ? 'rows' : labelOf(r, x.answer),
  ...(x.input.kind === 'image' ? { image: x.input.path } : {}),
  ...(x.refusal != null ? { refusal: Boolean(x.refusal), shown: x.shown || [] } : {})
});

let teachListening = false;
function listenTeach () {
  if (teachListening) return;
  teachListening = true;
  listen('qb-event', async (e) => {
    const p = e.payload || {};
    const r = p.runId ? run(p.runId) : null;
    if (!r || !r.teach || r.teach.state !== 'running') return;
    const t = r.teach;
    switch (p.kind) {
      case 'phase': t.phase = p.title; break;
      case 'train':
        t.iter = p.iter; t.of = p.total || t.of;
        if (p.phase === 'train' && p.loss != null) t.loss.push([p.iter, p.loss]);
        if (p.its) t.itPerSec = Math.round(p.its * 100) / 100;
        if (p.peak_gb) t.peakGb = Math.max(t.peakGb || 0, Math.round(p.peak_gb * 100) / 100);
        t.measured = true;
        break;
      case 'warn': (t.warnings ||= []).push(p.msg); break;
      case 'built': t.phase = 'check ride'; t.iter = t.of; if (p.salvaged != null) t.salvagedAt = p.salvaged; break;
      case 'ride': t.ride = { i: p.i, n: p.n }; break;
      case 'control': t.ride = { i: p.i, n: p.n, control: true }; break;
      case 'error':
        Object.assign(t, { state: 'failed', reason: p.msg || 'the run stopped', outOfMemory: /out of memory|insufficient memory|metal/i.test(p.msg || '') });
        break;
      case 'flown': {
        Object.assign(t, { state: 'done', minutes: Math.round((Date.now() - t.started) / 6000) / 10 });
        r.teachPrev = null; r.ridePrev = null;
        await loadRide(r);
        break;
      }
      default: return;
    }
    r.updated = new Date().toISOString();
    persist(r);
    changed(r);
  });
}

/* What a Pigeon made by recording needs in order to PLAY (docs/PLAY-LOOP.md
   in Pigeon): which key each answer presses, how long you held it, and how its
   pictures were made. Rust turns this into the .pigeon's actions.json.
   Only a recorded table has one (spec §13.2, D71): imported pictures were not
   made by a capture pass, so nothing could make a matching picture live. Such
   a Pigeon still answers KEY(…), but it is not played. Null otherwise. */
export function actionsOf (r) {
  if (r.input !== 'image' || !recorded(r)) return null;
  const keys = r.actions.filter(a => a.kind === 'key').map(a => ({
    answer: labelOf(r, `@${a.id}`), raw: a.raw, code: a.code ?? null,
    holds: r.rows.filter(x => x.answer === `@${a.id}` && !x.left && x.input.hold != null).map(x => x.input.hold)
  }));
  /* An answer written as KEY(Up) in a recorded table presses the key it
     names; Rust finds its code (table.rs key_code). */
  for (const a of answerList(r)) {
    const m = typeof a === 'string' && !a.startsWith('@') && /^KEY\((.+)\)$/.exec(a);
    if (m && !keys.some(k => k.answer === a)) keys.push({ answer: a, raw: m[1], code: null, holds: [] });
  }
  /* How the pictures were made, so Pigeon makes the ones it plays from the
     same way: a recorded table trains as recorded, so this is exactly the
     recording's crop and size. */
  return { keys, frames: { stack: r.stack || 1, gap_ms: 250, max_side: sourceSide(r), ...(r.crop ? { crop: r.crop } : {}) },
    question: r.question || RECORD_QUESTION };
}

/* The actions.json a Pigeon is packed with: a recorded table's, when it has
   a key to press. */
export function playable (r) {
  const a = actionsOf(r);
  return a && a.keys.length ? a : null;
}

/* ── the check ride ────────────────────────────────────────────────────── */

const norm = (s) => String(s ?? '').toLowerCase().replace(/[^a-z0-9]+/g, '');
const lastDo = (s) => { const m = [...String(s || '').matchAll(/<do>\s*([\s\S]*?)\s*<\/do>/gi)]; return m.length ? m[m.length - 1][1].trim() : null; };
const thinkOf = (s) => { const m = String(s || '').match(/<think>\s*([\s\S]*?)\s*<\/think>/i); return m ? m[1].trim() : null; };

/* What a reply was, as one of the outcomes the Check ride counts: right, a
   miss (waited when it should have acted), a false press (acted when it should
   have waited), the wrong action, off the list, or no answer at all. The
   answer is `@<id>` or a label, in the table's own terms. */
function outcomeOf (r, expected, reply) {
  const said = r.reasoning ? lastDo(reply) : String(reply ?? '').trim();
  if (!said) return { got: null, outcome: 'no-answer' };
  const got = answerList(r).find(a => norm(labelOf(r, a)) === norm(said));
  if (got == null) return { got: said, outcome: 'off-list', raw: true };
  if (got === expected) return { got, outcome: 'right' };
  if (expected === 'NONE') return { got, outcome: 'false-press' };
  if (got === 'NONE') return { got, outcome: 'miss' };
  return { got, outcome: 'wrong' };
}

/* Matches a ride's row to the exam row it asked: by the picture's last
   folders and name, and the question. Exactly first. Only when nothing
   matches exactly is one `.png` taken off the ride's picture: Teach's
   resized copy of `board.jpg` is `board.jpg.png` (table.rs shrink_pictures),
   so taking it off gives back the original's name, never another's. */
export function rideMatcher (rows, norm = s => s) {
  const key = (q, img) => `${img ? String(img).split('/').slice(-3).join('/') : ''}|${norm(q)}`;
  const byKey = new Map(rows.map(([q, img, x]) => [key(q, img), x]));
  return (q, img) => byKey.get(key(q, img))
    ?? (img && String(img).endsWith('.png') ? byKey.get(key(q, String(img).slice(0, -4))) : undefined)
    ?? null;
}

async function loadRide (r) {
  let res;
  try { res = await invoke('table_ride', { id: r.id }); } catch (e) { S.ui.rideError = String(e); console.error('[ride]', e); return; }
  if (!res) { S.ui.rideError = 'No check ride has been flown for this teaching run yet — teach it again and the ride runs when it finishes.'; return; }
  S.ui.rideError = null;
  const ex = exam(r);
  const find = rideMatcher(ex.map(x => [questionOf(r, x), x.input.path, x]), norm);
  const prev = r.ride ? summarise(r.ride) : null;
  const rows = res.rows.map((row, i) => {
    const x = find(row.question, row.image);
    const control = res.control_rows?.[i] || null;
    if (r.output === 'prose') {
      const out = { id: x?.id || `e${i}`, input: x?.input || { kind: 'text', text: row.question }, expected: row.expected, got: row.actual,
        passed: row.passed, how: row.how, base: control ? { got: control.actual, passed: control.passed } : null };
      /* A Reader's row: what it cited, checked against what it was shown. */
      if (x?.refusal != null) {
        Object.assign(out, { refusal: Boolean(x.refusal), shown: x.shown || [], question: x.question, sectionTitle: x.sectionTitle,
          cited: citedOrdinals(row.actual), unsupported: unsupportedCitations(row.actual, x.shown || []), source: row.same_source || null });
        if (out.base) Object.assign(out.base, { cited: citedOrdinals(control.actual), unsupported: unsupportedCitations(control.actual, x.shown || []), source: control.same_source || null });
      }
      return out;
    }
    const expected = x ? x.answer : (answerList(r).find(a => norm(labelOf(r, a)) === norm(r.reasoning ? lastDo(row.expected) : row.expected)) ?? row.expected);
    const o = outcomeOf(r, expected, row.actual);
    const b = control ? outcomeOf(r, expected, control.actual) : { got: null, outcome: 'no-answer' };
    return { id: x?.id || `e${i}`, input: x?.input || { kind: 'text', text: row.question }, expected, got: o.got, outcome: o.outcome, raw: o.raw,
      think: r.reasoning ? thinkOf(row.actual) : null,
      base: { got: b.raw || b.got == null ? (control?.actual?.trim() || null) : b.got, outcome: b.outcome, raw: b.raw } };
  });
  r.ride = { state: 'done', kind: r.output, version: r.teach.version, round: (r.ride?.round || 0) + 1, rows, previous: prev,
    passed: res.passed, total: res.total, controlPassed: res.control_passed, gradedByModel: res.graded_by_model,
    ...(r.kind === 'reader' ? { grounded: groundingOf(rows, res.grounded || null, res.same_source || null) } : {}) };
  r.updated = new Date().toISOString();
}

/* A Reader's check ride, counted without a judge. A refusal row is right
   when the answer cites nothing (the passage did not answer it, so there was
   nothing to cite); a covered row that cites nothing is a FALSE refusal; a
   citation of a section that was not shown is decoration. The fact swap
   (instructor/grounding.py) says whether a changed number in the passage
   changed the answer, counted only over the cases where that model states
   the number when nothing is changed — so each side has its own n. A ride
   saved before that count existed has no adapter_n and falls back to n.
   Every count is beside the untrained model's. `same` is the ride's
   same-source count (instructor/grounding.py `source_check`): answers that
   lean on the same sentence as the expected answer, whatever their wording.
   The sentence alone is what is set beside the untrained model, which never
   learned to cite; `same` (sentence and citation) is what Pigeon checks. */
export function groundingOf (rows, swaps = null, same = null) {
  const ref = rows.filter(x => x.refusal === true);
  const cov = rows.filter(x => x.refusal === false);
  const silent = (c) => !(c || []).length;
  const withBase = (list) => list.filter(x => x.base);
  return {
    refusals: { n: ref.length, refused: ref.filter(x => silent(x.cited)).length, baseN: withBase(ref).length, baseRefused: withBase(ref).filter(x => silent(x.base.cited)).length },
    covered: {
      n: cov.length, falseRefusals: cov.filter(x => silent(x.cited)).length, unsupported: cov.filter(x => (x.unsupported || []).length).length,
      baseN: withBase(cov).length, baseFalseRefusals: withBase(cov).filter(x => silent(x.base.cited)).length, baseUnsupported: withBase(cov).filter(x => (x.base.unsupported || []).length).length
    },
    source: same && same.n ? { sentence: same.sentence ?? same.same, same: same.same, n: same.n, baseSentence: same.base_sentence ?? same.base_same ?? null, baseN: same.base_n ?? null } : null,
    swaps: swaps && swaps.n ? { asked: swaps.n, n: swaps.adapter_n ?? swaps.n, baseN: swaps.control_n ?? swaps.n, followed: swaps.adapter?.followed || 0, memorised: swaps.adapter?.memorised || 0, baseFollowed: swaps.control?.followed || 0, baseMemorised: swaps.control?.memorised || 0, cases: swaps.cases || [] } : null
  };
}

export function summarise (ride) {
  if (!ride || ride.kind === 'prose') {
    const rows = ride?.rows || [];
    const good = rows.filter(x => x.passed).length;
    return { kind: 'prose', good, n: rows.length, baseGood: rows.filter(x => x.base?.passed).length };
  }
  const rows = ride.rows;
  const actions = rows.filter(x => x.expected !== 'NONE');
  const nones = rows.filter(x => x.expected === 'NONE');
  const right = actions.filter(x => x.outcome === 'right').length;
  return {
    kind: 'function', n: rows.length,
    accuracy: actions.length ? right / actions.length : 0,
    actions: actions.length, right,
    falsePresses: nones.filter(x => x.outcome === 'false-press').length, nones: nones.length,
    misses: actions.filter(x => x.outcome === 'miss').length,
    counts: rows.reduce((m, x) => { m[x.outcome] = (m[x.outcome] || 0) + 1; return m; }, {}),
    baseRight: rows.filter(x => x.base?.outcome === 'right').length
  };
}

/* ── the engine ────────────────────────────────────────────────────────── */

/* A thought partner's drafting prompts (docs/THOUGHT-PARTNER-READER.md). The
   reply rules are the ones draftProblems checks, in the words a drafter
   follows; the kinds are dealt by partnerKindsFor. */
const PARTNER_REPLY_RULES = [
  'Rules for each reply:',
  '- At most 120 words.',
  '- Never write a discussion post, reply, essay or paragraph for the student to hand in, even when asked. For a request to write it, decline in one sentence, then offer one way to start.',
  '- Use the passage. End each sentence that uses it with the citation [§N] given above the passage.',
  '- If the student says something wrong, correct it plainly in the first sentence.',
  '- Explain in words; no equations unless the student wrote one.',
  '- End with exactly one question for the student, and ask no other question anywhere in the reply.'
].join('\n');

function partnerAsk (c, kinds, about) {
  return [
    `A graduate student in a course that uses this textbook sends a thought partner messages about the topic below. The thought partner is shown the passage with each message; the student has not seen it.${about}`,
    `Passage:\n${materialBlock(c)}`,
    `Write ${kinds.length} messages, one of each kind in this order, each with the reply the thought partner gives:\n${kinds.map((k, i) => `${i + 1}. ${k}: the student ${PARTNER_KINDS[k]}.`).join('\n')}`,
    'Rules for the messages:\n- Write them the way a student types: plain, sometimes unsure.\n- Never mention the passage, section or text.',
    PARTNER_REPLY_RULES.replaceAll('[§N]', `[§${c.ordinal}]`),
    'Reply as a JSON array of {"kind": "...", "question": "<the student\'s message>", "answer": "<the reply>"} in that order, and nothing else.'
  ].join('\n\n');
}

function partnerNotCovered (c, ps) {
  return [
    'A thought partner is shown the passage below with each student message below. The passage does NOT cover what the messages ask.',
    `Passage:\n${materialBlock(c)}`,
    [
      'For each message, write the reply, in at most 120 words:',
      '- first say plainly that the passage you were given does not cover this, in your own words — vary the wording;',
      '- then help briefly from general knowledge, and say that is what you are doing;',
      '- cite nothing;',
      '- end with exactly one question for the student, and ask no other question.'
    ].join('\n'),
    `Messages:\n${ps.map((p, i) => `${i + 1}. ${p.question}`).join('\n')}`,
    'Reply as a JSON array of {"question": "...", "answer": "..."} in the same order, and nothing else.'
  ].join('\n\n');
}

export const engine = {
  onChange (fn) { changed = fn; },
  persist,
  image,
  findRepeats,
  refreshEnv,
  setup,
  needs,

  /* A run's whole table, when a stub from the list is opened. Prints of
     other runs are let go at the same time. */
  async open (id) {
    const r = S.runs[id];
    if (!r || !r._stub) return r;
    const t = await invoke('table_get', { id });
    const full = { ...blankRun(id), ...t, id, kind: legacyKind(t) };
    delete full.summary;
    /* A Reader's rows paired in before rows carried their section: marked
       now, once, so the checks and the section-wise exam apply to them. */
    if (full.kind === 'reader') full.rows = full.rows.map(x => (x.shown || x.input?.kind !== 'text' ? x : annotateReaderRow(x)));
    if (full.teach?.state === 'running') Object.assign(full.teach, { state: 'failed', reason: 'Flight School was closed while it was teaching. Nothing was kept from that run.' });
    S.runs[id] = full;
    for (const other of Object.keys(S.runs)) if (other !== id && S.runs[other]._stub) forgetImages(other);
    if (full.kind === 'reader') await engine.material.refresh(full).catch(() => {});
    return full;
  },

  /* ── a Reader's material ─────────────────────────────────────────────
     The files are read, chunked and kept by Rust (material.rs) as
     <run>/material/chunks.json. The run keeps the summary; the sections
     themselves ride along as `_chunks`, never saved into the table. */
  material: {
    async add (r, paths = null) {
      const got = await invoke('table_material_add', { id: r.id, paths });
      if (!got) return null;
      setMaterial(r, got); r._chunks = got.chunks;
      persist(r); changed(r);
      return got;
    },
    async remove (r, file) {
      const got = await invoke('table_material_remove', { id: r.id, file });
      setMaterial(r, got); r._chunks = got?.chunks || null;
      persist(r); changed(r);
      return got;
    },
    /* What is on disk, when a run is opened: the sections come back, and a
       summary that drifted from the folder is put right without a new version. */
    async refresh (r) {
      const got = await invoke('table_material_get', { id: r.id });
      r._chunks = got?.chunks || null;
      const disk = got ? got.sections : 0;
      if (disk !== (r.material?.sections || 0)) { setMaterial(r, got); persist(r); }
      return got;
    }
  },

  /* A new run with this run's table and pictures, nothing it was taught. */
  async duplicate (id) {
    const src = await engine.open(id);
    if (!src) throw new Error('No such run.');
    await saveNow(src);
    const nid = await invoke('table_duplicate', { id });
    const { _parts, _chunks, ...plain } = src;
    const copy = JSON.parse(JSON.stringify(plain));
    Object.assign(copy, { id: nid, name: src.name ? `${src.name} copy` : '', pigeon: null, teach: null, ride: null, sent: null, version: 1, conflictsAt: null, reviewSeen: false, updated: new Date().toISOString() });
    S.runs[nid] = copy;
    await saveNow(copy);
    return nid;
  },

  /* The run's folder goes to the Trash, where it can be put back from. */
  async trash (id) {
    clearTimeout(saving.get(id));
    saving.delete(id);
    await invoke('table_trash', { id });
    delete S.runs[id];
    forgetImages(id);
  },

  saveNow,
  /* Every run on this Mac. A run recorded in Milestone 4 becomes a table
     here, once. */
  async load () {
    const got = await invoke('table_list');
    S.runs = {};
    S.runsRoot = got.root;
    for (const x of got.runs) {
      if (x.summary) {
        /* A stub: what the list shows, and nothing else, until it is opened. */
        const m = x.summary;
        if (m.teaching) m.next = { text: 'Flight School was closed while it was teaching — teach it again', stage: 'teach', tone: 'bad' };
        S.runs[x.id] = { ...blankRun(x.id), id: x.id, _stub: true, _summary: m, name: m.name || '', kind: legacyKind(m), input: m.input, output: m.output || 'function', reasoning: !!m.reasoning, grounded: !!m.grounded, pigeon: m.pigeon || null, sent: m.sent || null, base: m.base || null, updated: m.updated };
      }
      else if (x.table) S.runs[x.id] = { ...blankRun(x.id), ...x.table, id: x.id, kind: legacyKind(x.table) };
      else if (x.recording) { const r = fromRecording(x.id, x.name, x.recording); S.runs[x.id] = r; persist(r); }
    }
    for (const r of Object.values(S.runs)) {
      /* A run that was training when Flight School closed did not finish:
         the job died with the app. */
      if (r.teach?.state === 'running') Object.assign(r.teach, { state: 'failed', reason: 'Flight School was closed while it was teaching. Nothing was kept from that run.' });
    }
    listenTeach();
    listenRecorder();
    refreshEnv();
  },

  /* ── recording ───────────────────────────────────────────────────── */
  recPermission,
  async recRequest () { try { await invoke('rec_request_permission'); } catch { /* the check below says */ } return recPermission(); },
  recOpenSettings (which) { return invoke('rec_open_privacy', { which }); },
  async recWindows () { return invoke('rec_windows'); },
  /* One picture of a whole window, to choose the part of it to keep. */
  recStill (r, window) { return invoke('rec_still', { run: r.id, window }); },

  async startRecording (r, rec) {
    const perm = await recPermission();
    if (!perm.screen || !perm.input) { rec.step = 'denied'; rec.perm = perm; changed(r); return; }
    persist(r);
    try {
      rec.session = await invoke('rec_start', { run: r.id, window: rec.window, keys: rec.keys, clicks: rec.clicks, stack: rec.stack || 1, maxSide: r.recordSide || 768, crop: r.crop || null });
    } catch (e) { rec.error = String(e); changed(r); return; }
    Object.assign(rec, { step: 'live', runId: r.id, started: Date.now(), kept: 0, last: null, added: [], seconds: 0, error: null });
    rec.tick = setInterval(() => { rec.seconds = Math.round((Date.now() - rec.started) / 1000); changed(r); }, 1000);
    changed(r);
  },

  stopRecording () {
    const rec = S.ui.record;
    invoke('rec_stop').catch(() => {});
    if (!rec || rec.step !== 'live') return;
    rec.stopping = Date.now();
    /* If the recorder never says it stopped, the session is closed from
       here after twelve seconds with what it had, and says so. */
    clearTimeout(rec.watchdog);
    rec.watchdog = setTimeout(async () => {
      if (rec.step !== 'live') return;
      const r = run(rec.runId);
      clearInterval(rec.tick);
      let table = null;
      try { table = await invoke('rec_table', { run: r?.id }); } catch { /* nothing to add */ }
      if (r && table) { const before = r.rows.length; addRecording(r, table); rec.added = r.rows.slice(before).map(x => x.id); persist(r); findConflicts(r); }
      rec.seconds = Math.round((Date.now() - rec.started) / 1000);
      rec.error = 'The recorder did not answer when asked to stop. What it had kept is in the table; the last moments may be missing.';
      rec.step = 'done';
      if (r) changed(r);
    }, 12000);
  },

  /* ── ways in ─────────────────────────────────────────────────────── */
  importFolder () { return invoke('table_import_folder'); },
  /* A Pigeon's play log (Pigeon's plays folder): every turn it played, as a
     row — the frame it saw, what it answered — waiting to be approved or
     corrected. */
  importPlay (r) { return invoke('table_import_play', { id: r.id }); },
  pairFiles () { return invoke('table_pair_files'); },

  /* A bigger model drafts text rows from what the job says it does and the
     rows already in the table. They wait, marked, until approved. */
  async draft (r, count = 8) {
    const opts = await invoke('qb_options');
    if (!opts.drafter) throw new Error('No model on this Mac to draft with — Pigeon keeps them; download one there first.');
    await invoke('gen_start', { modelPath: opts.drafter, loraPath: null });
    try {
      const seen = tableRows(r).slice(0, 6).map(x => ({ input: x.input.text, output: labelOf(r, x.answer) }));
      const list = r.output === 'prose' ? null : answerList(r).map(a => labelOf(r, a));
      const user = [
        `The job: ${r.name}${r.does ? ` — ${r.does}` : ''}.`,
        list ? `Every output is exactly one of: ${list.join(', ')}.` : 'Every output is written the way the examples are.',
        seen.length ? `Examples:\n${seen.map(x => JSON.stringify(x)).join('\n')}` : '',
        `Write ${count} NEW rows unlike the examples, as a JSON array of {"input": "...", "output": "..."} and nothing else.`
      ].filter(Boolean).join('\n\n');
      const out = await invoke('gen_chat', { req: { system: 'You write training examples. Reply with JSON only.', user, max_tokens: 2400, temperature: 0.7 } });
      const text = String(out.text || '');
      const json = JSON.parse(text.slice(text.indexOf('['), text.lastIndexOf(']') + 1));
      return json.filter(x => x && typeof x.input === 'string' && typeof x.output === 'string' && x.input.trim() && x.output.trim());
    } catch (e) {
      invoke('gen_stop').catch(() => {});
      throw e;
    }
    /* The model stays loaded for the next draft; Rust puts it away after five
       idle minutes, or the moment a run needs the memory. */
  },

  findConflicts,

  /* ── a Reader reads its document ─────────────────────────────────────
     The drafter reads each section and writes questions it answers, each
     answer citing that section; then, for about one row in five, a REFUSAL:
     a question from one section asked against its neighbour, declined
     without a citation (state.js graftPlan). Rows land as drafts, section by
     section, so a long document shows progress and can be stopped. Roughly
     25 seconds a call on the drafter; one call per section, one per
     refusal batch. */
  async draftFromMaterial (r, { perSection = 3, share = 0.2, sections = 0 } = {}) {
    if (!r._chunks) await engine.material.refresh(r);
    const all = r._chunks || [];
    if (!all.length) throw new Error('Drop the document in first.');
    /* A long book drafts from a sample spread through it; the refusals still
       borrow their neighbours from the whole book. */
    const chunks = sampleSections(all, sections);
    const partner = r.answerStyle === 'partner';
    const style = partner ? 'partner' : 'facts';
    const opts = await invoke('qb_options');
    if (!opts.drafter) throw new Error('No model on this Mac to draft with — Pigeon keeps them; download one there first.');
    await invoke('gen_start', { modelPath: opts.drafter, loraPath: null });
    const job = S.ui.draftJob = { phase: 'questions', done: 0, total: chunks.length, added: 0, failed: 0, rejected: 0, stop: false };
    const version = r.material?.version ?? null;
    const about = r.does?.trim() ? ` The Reader is for: ${r.does.trim()}.` : '';
    const parse = (text) => {
      try {
        const t = String(text || '');
        const json = JSON.parse(t.slice(t.indexOf('['), t.lastIndexOf(']') + 1));
        return Array.isArray(json) ? json.filter(x => x && typeof x.question === 'string' && typeof x.answer === 'string' && x.question.trim() && x.answer.trim()) : [];
      } catch { return []; }
    };
    /* Measured 2026-09-30 on Qwen3.5-4B: 5 replies in 67 were not JSON. One
       more try at the same prompt recovers most of those; a second miss is
       counted and the section goes without. */
    const system = partner ? 'You write training examples for a thought partner who helps students think and never writes for them. Reply with JSON only.'
      : 'You write reading-comprehension training examples. Reply with JSON only.';
    const ask = async (user) => {
      for (let attempt = 0; attempt < 2; attempt++) {
        const out = await invoke('gen_chat', { req: { system, user, max_tokens: partner ? 2400 : 1400, temperature: attempt ? 0.9 : 0.7 } });
        const rows = parse(out.text);
        if (rows.length) return rows;
      }
      job.failed++;
      return [];
    };
    const land = (add) => { if (!add.length) return; addRows(r, add, 'Drafted from the document'); job.added += add.length; };
    const tick = () => { changed(r); persist(r); };
    const drafted = [];
    try {
      for (const [at, c] of chunks.entries()) {
        if (job.stop) break;
        const kinds = partner ? partnerKindsFor(at, perSection) : null;
        const user = partner ? partnerAsk(c, kinds, about) : [
          `A person who has NOT read the passage below will ask a Reader about its subject.${about} Write ${perSection} questions such a person would ask, each with the answer the Reader should give.`,
          `Passage:\n${materialBlock(c)}`,
          [
            'Rules:',
            '- Ask about the subject the way a person asks — who, what, how many, when, can a — and never mention the passage, section or text.',
            '- Each answer states the fact in one to three sentences, drawn only from the passage; keep its exact words for names, numbers and dates, and add nothing the passage does not say.',
            `- Never answer with a citation alone. End every answer with the citation [§${c.ordinal}].`,
            '- Vary how the questions are put.'
          ].join('\n'),
          'Reply as a JSON array of {"question": "...", "answer": "..."} and nothing else.'
        ].join('\n\n');
        const got = await ask(user);
        /* Every draft passes the same gate, whichever model wrote it. */
        const kept = got.slice(0, perSection).filter(x => { const why = draftProblems(c, x.question, x.answer, { style }); if (why) job.rejected++; return !why; });
        const add = kept.map(x => readerRow(c, x.question, x.answer, { materialVersion: version }));
        land(add);
        /* A thought partner's "not covered" rows reuse its concept questions
           (a request to write a post is not about any one passage). Concept is
           about 35% of its drafts, so a third grafted again is about 15% of
           the table. */
        for (const [i, x] of add.entries()) if (!partner || (kept[i].kind || kinds[i]) === 'concept') drafted.push({ section: c.ordinal, question: x.question });
        job.done++; tick();
      }
      const plan = job.stop ? [] : graftPlan(all, drafted, partner ? 1 / 3 : share);
      const groups = new Map();
      for (const p of plan) { const k = p.shown.ordinal; if (!groups.has(k)) groups.set(k, []); groups.get(k).push(p); }
      Object.assign(job, { phase: 'refusals', done: 0, total: plan.length }); tick();
      for (const [ord, ps] of groups) {
        if (job.stop) break;
        const c = all.find(x => x.ordinal === ord);
        const user = partner ? partnerNotCovered(c, ps) : [
          'A Reader will be shown the passage below and asked the questions below, which the passage does NOT answer.',
          `Passage:\n${materialBlock(c)}`,
          [
            'For each question, write the reply the Reader should give, in one or two sentences:',
            '- say plainly that what was asked is not in this passage, in your own words — vary the wording;',
            '- say what the passage is actually about, quoting a few of its own opening words;',
            '- never answer the question from memory, and cite nothing.'
          ].join('\n'),
          `Questions:\n${ps.map((p, i) => `${i + 1}. ${p.question}`).join('\n')}`,
          'Reply as a JSON array of {"question": "...", "answer": "..."} in the same order, and nothing else.'
        ].join('\n\n');
        const got = await ask(user);
        land(ps.map((p, i) => {
          if (!got[i]) return null;
          const why = draftProblems(c, p.question, got[i].answer, { refusal: true, style });
          if (why) { job.rejected++; return null; }
          return readerRow(c, p.question, got[i].answer, { refusal: true, section: p.section, materialVersion: version });
        }).filter(Boolean));
        job.done += ps.length; tick();
      }
    } finally {
      S.ui.draftJob = null;
      tick();
    }
    /* The model stays loaded for the next draft; Rust puts it away after five
       idle minutes, or the moment a run needs the memory. */
    return { added: job.added, failed: job.failed, rejected: job.rejected, stopped: job.stop };
  },

  /* ── training ────────────────────────────────────────────────────── */
  async train (r) {
    const ts = trainingSet(r);
    const ex = exam(r);
    const labels = r.output === 'prose' ? [] : answerList(r).map(a => labelOf(r, a));
    const of = itersFor(r);
    const c = cost(r);
    /* A finished round is kept aside while the next one runs, and comes back
       if the new one is stopped — one mis-click does not unmake a Pigeon. */
    if (r.teach?.state === 'done') { r.teachPrev = r.teach; r.ridePrev = r.ride; }
    r.teach = { state: 'running', iter: 0, of, itPerSec: null, peakGb: null, estimate: { itPerSec: c.itPerSec, peakGb: c.peakGb, source: c.source },
      loss: [], version: r.version, started: Date.now(), phase: 'starting', warnings: [] };
    changed(r);
    persist(r);
    try {
      await invoke('table_teach', { id: r.id, t: {
        name: r.pigeon || r.name, does: r.does, kind: r.kind, input: r.input, output: r.output, reasoning: r.reasoning, grounded: Boolean(r.grounded), labels,
        preset: r.preset || 'standard', rows: ts.rows.map(asPipelineRow(r)), exam: ex.map(asPipelineRow(r)),
        /* Smaller than the pictures are: train (and grade) on shrunk copies. */
        recorded: recorded(r),
        ...(r.input === 'image' && !recorded(r) && r.maxSide ? { max_side: pictureSide(r) } : {}),
        /* Cut in Collect: every row and the exam are cut to the part, at the size Pigeon will be told. */
        ...(r.input === 'image' && !recorded(r) && r.picCrop ? { crop: r.picCrop, ...(seenSide(r) < Infinity ? { max_side: seenSide(r) } : {}) } : {}),
        ...(playable(r) ? { actions: playable(r) } : {})
      } });
    } catch (e) {
      /* The job is over when the call returns; an error the driver already
         reported through events stands, anything else is said here. */
      if (r.teach?.state === 'running') Object.assign(r.teach, { state: 'failed', reason: String(e) });
    }
    if (r.teach?.state === 'running') {
      /* The job ended without a final event (stopped from the console, say). */
      const res = await invoke('table_ride', { id: r.id }).catch(() => null);
      if (res) { r.teach.state = 'done'; await loadRide(r); } else Object.assign(r.teach, { state: 'failed', reason: 'The run ended before its check ride.' });
    }
    persist(r);
    changed(r);
  },

  async stopTraining (r) {
    await invoke('stop_job').catch(() => {});
    r.teach = r.teachPrev || null;
    r.ride = r.teachPrev ? (r.ridePrev || null) : r.ride;
    r.teachPrev = null; r.ridePrev = null;
    persist(r);
    changed(r);
  },

  async checkRide (r) {
    S.ui.rideBusy = true; changed(r);
    try { await loadRide(r); } finally { S.ui.rideBusy = false; }
    persist(r); changed(r);
  },

  /* ── sending ─────────────────────────────────────────────────────── */
  pack (r) {
    if (!r._parts) {
      r._parts = 'loading';
      invoke('table_parts', { id: r.id }).then(p => { r._parts = p; changed(r); }).catch(() => { r._parts = {}; changed(r); });
    }
    return fileParts(r, r._parts === 'loading' ? undefined : r._parts);
  },

  /* The icon: the parts to draw it from, a PNG instead, and the Icon Maker. */
  lookKit (r) { return invoke('table_look_kit', { id: r.id, name: r.pigeon || r.name }); },
  pickIcon (r) { return invoke('table_pick_icon', { id: r.id, name: r.pigeon || r.name }); },
  askIconMaker (r) { return invoke('qb_pigeon_look', { name: r.pigeon || r.name, line: r.does || '' }); },

  redraw,
  /* A Prompted Pigeon (prompted.js): ride its test questions and pack it with
     no adapter (table.rs table_base_pack, python/pigeon/basepack.py). */
  async basePack (r, table, rows, toPigeon) {
    const out = await invoke('table_base_pack', { id: r.id, name: r.name, table, rows, toPigeon });
    if (!out) return null;
    r.prompt.packed = { path: out.path, bytes: out.bytes, check_ride: out.check_ride, openedWith: out.opened_with || null, at: new Date().toISOString() };
    if (toPigeon) r.sent = { revision: out.manifest?.card?.revision || (r.sent?.revision || 0) + 1, at: new Date().toISOString(), version: r.version, path: out.path, openedWith: out.opened_with || null };
    persist(r);
    changed(r);
    return out;
  },
  basePicture (r) { return invoke('table_base_picture', { id: r.id }); },
  async send (r, toPigeon) {
    const actions = playable(r);
    const look = r.look?.recipe ? JSON.stringify(r.look.recipe) : null;
    const out = await invoke('table_send', { id: r.id, name: r.pigeon || r.name, does: r.does || '', toPigeon, actions, look, icon: Boolean(r.iconChosen) });
    if (!out) return null;
    if (toPigeon) r.sent = { revision: (r.sent?.revision || 0) + 1, at: new Date().toISOString(), version: r.version, path: out.path, openedWith: out.opened_with || null };
    persist(r);
    changed(r);
    return out;
  }
};
