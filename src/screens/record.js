/* record.js — Recording, inside Collect (decided 2026-09-21: recording lives
 * in Flight School, by screen capture). Made real in Milestone 5: the window
 * list, the permission, the live count and the session's rows all come from
 * the recorder (fs-recorder, src-tauri/src/recorder.rs) through engine.js.
 *
 * Four states: setup, live, done, denied.
 *   - A recording ends back in the TABLE, with its new rows marked, not in
 *     Review. People record in several sessions, and conflicts are between
 *     sessions — Review is worth more after the second one.
 *   - Each input keeps the frame from ~80 ms BEFORE it: the decision frame,
 *     not the result.
 *   - Permission is the failure that matters most on macOS, so it is a whole
 *     screen with steps. */

import { S, labelOf, SIDES } from '../state.js';
import { engine, MEASURED, visionCost, ITERS } from '../engine.js';
import { esc, ICON, n } from './html.js';
import { toast, toastError, tag } from './cells.js';

const redraw = () => window.dispatchEvent(new Event('proto:render'));

/* The windows and the permission are asked for once per visit, and again on
   "Check again". */
async function refresh (rec) {
  rec.checking = true;
  rec.perm = await engine.recPermission();
  if (rec.perm.screen && rec.perm.input) {
    try { rec.windows = await engine.recWindows(); rec.windowsError = null; } catch (e) { rec.windows = []; rec.windowsError = String(e); }
  } else rec.step = 'denied';
  rec.checking = false;
  redraw();
}

function setup (r, rec) {
  if (!rec.perm && !rec.checking) refresh(rec);
  if (!rec.perm) return '<p class="hint">Checking what macOS allows…</p>';
  const wins = rec.windows || [];
  const chosen = wins.find(w => w.id === rec.window);
  const small = chosen && Math.max(chosen.w, chosen.h) <= 400;
  return `<div class="record-setup">
    <h3>Which window?</h3>
    ${rec.windowsError ? `<p class="warn-line">${esc(rec.windowsError)}</p>` : ''}
    <div class="win-list" role="radiogroup" aria-label="Window to record">
      ${wins.length ? wins.map(w => `<label class="win${rec.window === w.id ? ' on' : ''}" for="win-${w.id}">
        <input type="radio" id="win-${w.id}" name="win" value="${w.id}" ${rec.window === w.id ? 'checked' : ''} data-change="rec-window">
        <span class="win-name">${esc(w.app)}</span><span class="win-title">${esc(w.title || 'untitled')}</span><span class="win-size">${n(w.w)} × ${n(w.h)}</span>
      </label>`).join('') : '<p class="hint">No windows to record. Open the app you want to teach, then check again.</p>'}
    </div>
    <p><button type="button" class="btn sm ghost" data-do="record-recheck">Check again</button></p>
    ${cropChoice(r, rec, chosen)}
    <h3>What to keep</h3>
    <div class="checks">
      <label class="check" for="rec-keys"><input type="checkbox" id="rec-keys" ${rec.keys ? 'checked' : ''} data-change="rec-keys"> Keys pressed</label>
      <label class="check" for="rec-clicks"><input type="checkbox" id="rec-clicks" ${rec.clicks ? 'checked' : ''} data-change="rec-clicks"> Clicks</label>
      ${small ? `<label class="check" for="rec-stack">Frames per row <select id="rec-stack" class="select" data-change="rec-stack">${[1, 2, 3].map(k => `<option value="${k}" ${k === (rec.stack || 1) ? 'selected' : ''}>${k}</option>`).join('')}</select></label>` : ''}
    </div>
    ${sizeChoice(r, rec, chosen)}
    <p class="hint">Each key or click is kept with the window as it was just before it — what you were looking at when you decided, not what it did. Keys become <span class="mono">KEY(…)</span> answers named after the key; places you click become <span class="mono">CLICK(…)</span> answers once you name them in Review. A quiet stretch is kept as <span class="mono">NONE</span>. Only this window is captured, and nothing leaves this Mac.</p>
    ${small ? `<p class="hint"><b>Frames per row</b> puts the last few moments side by side, for things you can only tell apart by what just happened. Three frames of a small window cost ${MEASURED.stacking.small[2]} tokens against ${MEASURED.stacking.small[0]} for one ${tag('measured')}.</p>` : ''}
    <div class="stage-foot">
      <button type="button" class="btn ghost" data-do="record-close">${ICON.back}Back to the table</button>
      <button type="button" class="btn primary" data-do="record-start" ${chosen && (rec.keys || rec.clicks) ? '' : 'aria-disabled="true"'}>${ICON.rec}Start recording</button>
    </div>
  </div>`;
}

/* The part of the window to keep: a sidebar, a toolbar, a score panel — what
   the Pigeon should not learn from — is cut away before a frame is kept, and
   Pigeon cuts the same part away when it plays. Chosen once per table, like
   the size; stored as fractions of the window, so it is the same part at any
   window size. */
const pct = (v) => `${(v * 100).toFixed(2)}%`;
function cropChoice (r, rec, chosen) {
  if (r.sessionsIn.length) {
    return r.crop ? `<p class="hint">This table keeps <b>part of the window</b> — ${Math.round(r.crop[2] * 100)}% of its width and ${Math.round(r.crop[3] * 100)}% of its height — like every session before.</p>` : '';
  }
  if (!chosen) return '';
  const c = rec.crop;
  const still = rec.still?.window === rec.window ? rec.still : null;
  const kept = still && c ? `${Math.round(still.w * c[2])} × ${Math.round(still.h * c[3])} of ${still.w} × ${still.h}` : '';
  return `<h3>Part of the window</h3>
    ${still ? `<p class="hint">Drag over the part the Pigeon should see. Everything outside it — a sidebar, a toolbar, a score — is cut away from every picture, and Pigeon cuts the same part away when it plays.</p>
      <div class="crop-pick" data-drag="crop" role="img" aria-label="${c ? `The window, with ${kept} pixels kept` : 'The window, all of it kept'}">
        <img src="${still.url}" alt="" draggable="false">
        ${c ? `<div class="crop-box" style="left:${pct(c[0])};top:${pct(c[1])};width:${pct(c[2])};height:${pct(c[3])}"></div>` : ''}
      </div>
      <div class="crop-bar"><span class="hint" data-live="crop-size">${c ? `Keeping ${kept}` : 'Keeping the whole window'}</span>
        <button type="button" class="btn sm ghost" data-do="crop-refresh">Take a new picture</button>
        ${c ? '<button type="button" class="btn sm ghost" data-do="crop-clear">Keep the whole window</button>' : ''}</div>`
    : `<p class="hint">${c ? 'Part of the window is chosen. ' : ''}Keep only part of it — leave out a sidebar or a toolbar that has nothing to do with the job.</p>
      <p><button type="button" class="btn sm" data-do="crop-open" ${rec.stillBusy ? 'aria-busy="true" disabled' : ''}>${rec.stillBusy ? 'Taking a picture…' : 'Choose part of the window…'}</button>${rec.stillError ? ` <span class="warn-line">${esc(rec.stillError)}</span>` : ''}</p>`}`;
}

/* The picture size is chosen once per table: every session is recorded the
   same, as frames per row is. Smaller pictures teach faster and use less
   memory; a job that needs fine detail — small text, a thin line — wants
   them larger. They train at this size: another size is a new recording. */
function sizeChoice (r, rec, chosen) {
  if (r.sessionsIn.length) {
    return `<p class="hint">Pictures are recorded at <b>${r.recordSide || 768} pixels</b> on the longer side, like the rest of this table, and train at that size.</p>`;
  }
  const win = chosen ? Math.max(chosen.w, chosen.h) : Infinity;
  const side = rec.side || 768;
  const opt = (k) => {
    const seen = Math.min(k, win);
    const c = visionCost(seen);
    const mins = Math.max(1, Math.round(ITERS.standard / c.itPerSec / 60));
    return `<option value="${k}" ${k === side ? 'selected' : ''}>${k} px${seen < k ? ` (the window is ${seen} px)` : ''} — about ${mins} min to teach, ~${c.tokensPerFrame} tokens a picture</option>`;
  };
  return `<h3>Picture size</h3>
    <label class="field-label small" for="rec-side">Longer side of each picture</label>
    <select id="rec-side" class="select" data-change="rec-side">${SIDES.map(opt).join('')}</select>
    <p class="hint">Smaller pictures teach faster and use less memory, but lose fine detail such as small text. They train at the size they are recorded at, so choose it here. Times are for a standard run on this Mac: ${tag('measured')} at 160 and 768 px, ${tag('estimate')} between.</p>`;
}

function live (r, rec) {
  const secs = rec.seconds || 0;
  return `<div class="record-live">
    <div class="hud" role="status" aria-live="off">
      <span class="rec-dot" aria-hidden="true"></span>
      <span class="hud-time">${Math.floor(secs / 60)}:${String(secs % 60).padStart(2, '0')}</span>
      <span class="hud-sep"></span>
      <span><b>${n(rec.kept || 0)}</b> kept</span>
      <span class="hud-sep"></span>
      <span class="hud-last">${esc(rec.last || 'waiting…')}</span>
      <button type="button" class="btn sm hud-stop" data-do="record-stop">${ICON.stop}Stop</button>
    </div>
    ${rec.error ? `<p class="warn-line">${esc(rec.error)}</p>` : ''}
    ${rec.secure ? `<p class="warn-line">A password field has focus, so keys are not being kept until it closes.</p>` : ''}
    <p class="hint">Use ${esc(rec.windowInfo?.app || 'the window')} as you normally would. Keys typed into other apps and clicks outside it are not kept. Press <kbd>⌘⇧R</kbd> from any app, or come back here and press Stop, when you are done.${rec.stopping ? ' Stopping…' : ''}</p>
  </div>`;
}

function done (r, rec) {
  const added = r.rows.filter(x => (rec.added || []).includes(x.id));
  const counts = new Map();
  for (const x of added) counts.set(x.answer, (counts.get(x.answer) || 0) + 1);
  const secs = rec.seconds || 0;
  return `<div class="record-done">
    <h3>${esc(rec.session || 'The session')}: ${n(added.length)} rows in ${secs < 60 ? `${secs} seconds` : `${Math.round(secs / 60)} ${secs < 90 ? 'minute' : 'minutes'}`}</h3>
    ${rec.error ? `<p class="warn-line">${esc(rec.error)}</p>` : ''}
    <div class="seen">${[...counts].sort((a, b) => b[1] - a[1]).map(([a, c]) => `<span class="fn">${esc(labelOf(r, a))} <b>${c}</b></span>`).join('')}</div>
    <p>They are in the table, marked <span class="badge fresh">new</span>. Name what you did in Review — keys already carry the key’s name. Recording another session is worth it: doing the same thing twice is where Review finds rows that disagree.</p>
    <div class="stage-foot">
      <button type="button" class="btn" data-do="record-again">${ICON.rec}Record another session</button>
      <button type="button" class="btn primary" data-do="record-close">See them in the table${ICON.arrow}</button>
    </div>
  </div>`;
}

function denied (rec) {
  const p = rec.perm || {};
  const screen = !p.screen, input = !p.input;
  return `<div class="record-denied">
    <p class="denied-head">${ICON.lock}<b>macOS is not letting Flight School ${screen && input ? 'see the screen or your key presses and clicks' : screen ? 'see the screen' : 'see your key presses and clicks'}.</b></p>
    <ol class="steps">
      <li>Press <b>Ask macOS</b>. If nothing appears, open the settings below.</li>
      ${screen ? '<li>Under <b>Privacy &amp; Security › Screen &amp; System Audio Recording</b>, turn on <b>Flight School</b>.</li>' : ''}
      ${input ? '<li>Under <b>Privacy &amp; Security › Input Monitoring</b>, turn on <b>Flight School</b>.</li>' : ''}
      <li>Quit Flight School and open it again. macOS only applies the change after a restart.</li>
    </ol>
    <p class="hint">Flight School only listens — it never presses a key or clicks for you. If Flight School is already on and you still see this, an older copy holds the permission: remove it from the list with the minus button, then add it again.</p>
    <div class="stage-foot">
      <button type="button" class="btn ghost" data-do="record-close">${ICON.back}Back to the table</button>
      ${screen ? '<button type="button" class="btn" data-do="open-settings" data-which="screen">Screen Recording settings</button>' : ''}
      ${input ? '<button type="button" class="btn" data-do="open-settings" data-which="input">Input Monitoring settings</button>' : ''}
      <button type="button" class="btn" data-do="record-ask">Ask macOS</button>
      <button type="button" class="btn primary" data-do="record-retry">Check again</button>
    </div>
  </div>`;
}

export function render (r) {
  const rec = S.ui.record || { step: 'setup', keys: true, clicks: true, stack: 1 };
  S.ui.record = rec;
  const body = rec.step === 'live' ? live(r, rec) : rec.step === 'done' ? done(r, rec) : rec.step === 'denied' ? denied(rec) : setup(r, rec);
  const title = { setup: 'Record', live: 'Recording', done: 'Recorded', denied: 'Recording is off' }[rec.step];
  return `<div class="record"><h2 class="sub-title">${ICON.rec}${title}</h2>${body}</div>`;
}

export const actions = {
  'record-start' (r) {
    const rec = S.ui.record;
    if (!rec.window) { toast('Pick the window to record first.'); return; }
    if (!(rec.keys || rec.clicks)) { toast('Keep keys, clicks, or both.'); return; }
    if (!r.answers.includes('NONE')) r.answers.push('NONE');
    const w = (rec.windows || []).find(x => x.id === rec.window);
    if (w) r.picture = Math.max(w.w, w.h) <= 400 ? 'small' : 'full';
    /* How many frames each row shows. Every session of a table must agree: a
       Pigeon is asked the way it was trained, and it trains on one shape. */
    if (r.rows.length && (r.stack || 1) !== (rec.stack || 1)) { toast(`This table's rows are ${r.stack || 1} frame${(r.stack || 1) > 1 ? 's' : ''} each — record the same, or start a new run.`); return; }
    r.stack = rec.stack || 1;
    if (!r.sessionsIn.length) { r.recordSide = rec.side || 768; r.crop = rec.crop || null; }
    engine.startRecording(r, rec);
    return false;
  },
  'record-stop' () { engine.stopRecording(); toast('Stopping — the last frames are being saved.'); },
  'record-again' () { const o = S.ui.record; S.ui.record = { step: 'setup', window: o.window, keys: o.keys, clicks: o.clicks, stack: o.stack, side: o.side, crop: o.crop, windows: o.windows, perm: o.perm }; },
  'crop-open' (r) {
    const rec = S.ui.record;
    rec.stillBusy = true; rec.stillError = null;
    engine.recStill(r, rec.window)
      .then(v => { rec.still = { ...v, window: rec.window }; })
      .catch(e => { rec.stillError = String(e); })
      .finally(() => { rec.stillBusy = false; redraw(); });
  },
  'crop-refresh' (r) { S.ui.record.still = null; return actions['crop-open'](r); },
  'crop-clear' () { S.ui.record.crop = null; },
  'record-close' () {
    if (S.ui.record?.step === 'live') { toast('Stop the recording first.'); return; }
    S.tab = null; S.ui.record = null; S.ui.filter = 'all'; S.ui.page = 1;
  },
  'record-recheck' () { const rec = S.ui.record; rec.perm = null; rec.windows = null; refresh(rec); return false; },
  'record-retry' () { S.ui.record = { step: 'setup', keys: true, clicks: true, stack: 1 }; },
  'record-ask' () {
    const rec = S.ui.record;
    engine.recRequest().then(p => {
      rec.perm = p;
      if (p.screen && p.input) { rec.step = 'setup'; refresh(rec); toast('Allowed. Pick a window.'); }
      else { toast('Not allowed yet. Turn Flight School on in the settings, then quit and reopen it.'); redraw(); }
    });
    return false;
  },
  'open-settings' (r, el) { engine.recOpenSettings(el.dataset.which).catch(e => toastError(String(e))); return false; }
};

export const changes = {
  'rec-window' (r, el) { S.ui.record.window = Number(el.value); S.ui.record.stillError = null; },
  'rec-keys' (r, el) { S.ui.record.keys = el.checked; },
  'rec-clicks' (r, el) { S.ui.record.clicks = el.checked; },
  'rec-stack' (r, el) { S.ui.record.stack = Number(el.value); },
  'rec-side' (r, el) { S.ui.record.side = Number(el.value); }
};

/* Dragging out the part to keep. The box follows the pointer without a
   redraw; letting go keeps it (a click without a drag keeps what was there). */
export const drags = {
  crop (r, el, e) {
    const rec = S.ui.record;
    const box = el.getBoundingClientRect();
    const at = (ev) => [Math.min(1, Math.max(0, (ev.clientX - box.left) / box.width)), Math.min(1, Math.max(0, (ev.clientY - box.top) / box.height))];
    const [x0, y0] = at(e);
    let div = el.querySelector('.crop-box');
    if (!div) { div = document.createElement('div'); div.className = 'crop-box'; el.appendChild(div); }
    let rect = null;
    const move = (ev) => {
      const [x1, y1] = at(ev);
      rect = [Math.min(x0, x1), Math.min(y0, y1), Math.abs(x1 - x0), Math.abs(y1 - y0)];
      Object.assign(div.style, { left: pct(rect[0]), top: pct(rect[1]), width: pct(rect[2]), height: pct(rect[3]) });
      const live = document.querySelector('[data-live="crop-size"]');
      if (live && rec.still) live.textContent = `Keeping ${Math.round(rec.still.w * rect[2])} × ${Math.round(rec.still.h * rect[3])} of ${rec.still.w} × ${rec.still.h}`;
    };
    const up = () => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
      if (rect && rect[2] > 0.03 && rect[3] > 0.03) rec.crop = rect.map(v => +v.toFixed(4));
      redraw();
    };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
  }
};
