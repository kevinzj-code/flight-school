/* teach.js — CARRIES FORWARD. Stage 4: training.
 *
 * Flow-review changes folded in:
 *   - Before it starts, it says what it will cost — time and memory — from
 *     measured numbers, and for the 3B on full-size pictures it says out loud
 *     that there is 0.6 GB to spare.
 *   - A table that changed after training makes the Pigeon stale, and this
 *     screen says so instead of quietly showing the old run as done.
 *   - Out of memory is a screen with choices, not an error: a smaller base,
 *     the capped 3B, or the last good checkpoint. */

import { S, trainingSet, exam, stale, gate, SIDES, sourceSide, pictureSide, seenSide, setPictureSide, recorded, cutAfterRecording, trainAsRecorded, recordAgain } from '../state.js';
import { engine, cost, minutesFor, itersFor, ITERS, visionCost, needs, budgetGb, ramGb } from '../engine.js';
import { esc, ICON, n, BUSY } from './html.js';
import { tag, toast, dialogHtml } from './cells.js';
import { patch } from './patch.js';

function memBar (peak, budget = budgetGb()) {
  const scale = Math.max(budget, peak) * 1.12;
  const over = peak > budget;
  return `<div class="mem" role="img" aria-label="Peak memory ${peak} GB against a budget of ${budget} GB">
    <div class="mem-track"><span class="mem-fill${over ? ' over' : peak > budget - 1 ? ' tight' : ''}" style="width:${(peak / scale) * 100}%"></span>
      <span class="mem-line" style="left:${(budget / scale) * 100}%"><span>budget ${budget} GB</span></span></div>
    <div class="mem-read"><b>${peak} GB</b> peak · ${over ? `<span class="bad">${(peak - budget).toFixed(1)} GB over</span>` : `${(budget - peak).toFixed(1)} GB to spare`}</div>
  </div>`;
}

function lossChart (t) {
  const pts = t.loss || [];
  if (pts.length < 2) return '';
  const W = 600, H = 120, pad = 6;
  const top = Math.max(2, ...pts.map(p => p[1]));
  const x = (i) => pad + (i / t.of) * (W - pad * 2);
  const y = (v) => pad + (1 - Math.min(top, v) / top) * (H - pad * 2);
  const d = pts.map(([i, v], k) => `${k ? 'L' : 'M'}${x(i).toFixed(1)} ${y(v).toFixed(1)}`).join(' ');
  const last = pts[pts.length - 1];
  return `<figure class="loss"><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Training loss, from ${pts[0][1].toFixed(2)} to ${last[1].toFixed(2)}">
    <path d="${d} L${x(last[0])} ${H - pad} L${x(pts[0][0])} ${H - pad} Z" fill="var(--aqua-tint)"/>
    <path d="${d}" fill="none" stroke="var(--aqua)" stroke-width="2"/>
    <circle cx="${x(last[0])}" cy="${y(last[1])}" r="4" fill="var(--aqua)"/>
  </svg><figcaption>Loss ${last[1].toFixed(3)} at iteration ${n(last[0])}, as the trainer reported it</figcaption></figure>`;
}

/* Train on smaller pictures than were recorded. Every row, and the exam,
   is shrunk the same way; the size goes into the Pigeon, and Pigeon shrinks
   what it sees to match, so it plays on what it was taught on. */
function sizeChoice (r) {
  if (r.input !== 'image' || r.base === 'Qwen2-VL-2B') return '';
  /* Recorded pictures train as recorded (state.js recorded): another size is
     another recording. Cut after recording, the choice is settled below. */
  if (recorded(r)) {
    if (cutAfterRecording(r)) return '';
    const src = sourceSide(r);
    const smaller = SIDES.some(k => k < src);
    return `<div class="panel flat"><h3>Picture size</h3><p>Recorded at <b>${src} px</b>.${smaller ? ' <button type="button" class="btn sm ghost" data-do="record-smaller">Record a new table at a smaller size…</button>' : ''}</p></div>`;
  }
  const src = sourceSide(r);
  const now = pictureSide(r);
  /* A size at or above the window's own is the same pictures as recorded:
     offered once, as recorded. */
  const sides = SIDES.filter(k => !src || k === src || (k < src && seenSide(r, k) < seenSide(r, src)));
  if (src && !sides.includes(src)) sides.unshift(src);
  const row = (k, on = k === now) => {
    const seen = seenSide(r, k);
    const c = visionCost(seen);
    const mins = Math.max(1, Math.round(itersFor(r) / c.itPerSec / 60));
    return `<label class="radio-row" for="side-${k}"><input type="radio" id="side-${k}" name="side" value="${k}" ${on ? 'checked' : ''} data-change="side">
      <span><b>${k} px</b>${src === k ? ' — as recorded' : ''}${seen < k ? ` <span class="hint">(the window is ${seen} px, so pictures stay ${seen} px)</span>` : ''}<br>
      <span class="hint">~${c.tokensPerFrame} tokens a picture · about ${mins} min · ${c.peakGb} GB peak ${tag(c.measured ? 'measured' : 'estimate')}</span></span></label>`;
  };
  const asIs = src ? '' : `<label class="radio-row" for="side-0"><input type="radio" id="side-0" name="side" value="0" ${!r.maxSide ? 'checked' : ''} data-change="side">
      <span><b>As they are</b><br><span class="hint">Every picture at its own size.</span></span></label>`;
  return `<fieldset class="choices">
    <legend class="field-label">Picture size — the longer side of each picture</legend>
    ${asIs}${sides.map(k => row(k, (src || r.maxSide) ? k === now : false)).join('')}
    <p class="hint">${src ? 'Smaller than recorded: every row and the exam are shrunk to this size before it trains, and the Pigeon is told to see at this size too.' : 'Pictures larger than this are shrunk before it trains; smaller ones stay as they are.'} Small text and thin lines are the first things lost — open a few pictures to check they still read.</p>
  </fieldset>`;
}

/* A recorded table cut or shrunk after recording, from before that stopped:
   shown in place of Start until it is trained as recorded or recorded again. */
function recutPanel (r) {
  const src = sourceSide(r);
  const what = r.picCrop
    ? `cut to ${Math.round(r.picCrop[2] * 100)}% of their width and ${Math.round(r.picCrop[3] * 100)}% of their height`
    : `shrunk from ${src} px to ${r.maxSide} px`;
  return `<div class="panel setup">
    <h3>Before it can teach</h3>
    <p class="warn-line">${ICON.warn}<span>These pictures were recorded at ${src} px, then ${what}. Your Pigeon would train on pictures it will never see when it plays.</span></p>
    <div class="stage-foot">
      <button type="button" class="btn" data-do="record-recut">Record a new table…</button>
      <button type="button" class="btn primary" data-do="train-as-recorded">Train as recorded</button>
    </div>
  </div>`;
}

/* What this Mac still needs before it can teach, with the one button that
   fixes it. Shown in place of Start, so nobody learns it from an error. */
function setupPanel (need) {
  const e = S.env;
  const first = need[0];
  const label = first.setup === 'vision' ? 'Set up picture training (once)' : 'Set up this Mac (once, about two minutes)';
  return `<div class="panel setup">
    <h3>Before it can teach</h3>
    <ul class="needs">${need.map(x => `<li>${ICON.warn}<span>${esc(x.text)}</span></li>`).join('')}</ul>
    ${e.busy ? `<p class="hint setup-live">${BUSY}<span>Setting up — <span data-live="setup-line">${esc(e.line || '')}</span></span></p>` : ''}
    ${e.error ? `<p class="warn-line" role="alert">${ICON.warn}${esc(e.error)}</p>` : ''}
    <div class="stage-foot">
      <button type="button" class="btn ghost" data-do="env-recheck" ${e.busy ? 'disabled' : ''}>Check again</button>
      ${first.setup ? `<button type="button" class="btn primary" data-do="setup-${first.setup}" ${e.busy ? 'disabled aria-busy="true"' : ''}>${e.busy ? 'Setting up…' : label}</button>` : ''}
    </div>
  </div>`;
}

function before (r) {
  const ts = trainingSet(r);
  const c = cost(r);
  const wasStale = stale(r);
  const need = needs(r);
  return `<div class="teach-before">
    ${wasStale ? `<p class="warn-line">${ICON.warn}The table changed since this was taught. Teaching again makes round ${(r.ride?.round || 1) + 1}, graded on the same exam as before.</p>` : ''}
    <div class="sum-grid">
      <div class="sum big"><b>${n(ts.rows.length)}</b><span>rows to practise on</span></div>
      <div class="sum"><b>${n(exam(r).length)}</b><span>held back for the check ride</span></div>
      <div class="sum"><b>${n(itersFor(r))}</b><span>iterations</span></div>
      <div class="sum"><b>~${minutesFor(r)} min</b><span>on this Mac ${tag(c.measured === false ? 'estimate' : 'measured')}</span></div>
    </div>
    <fieldset class="choices">
      <legend class="field-label">How long</legend>
      ${[['quick', 'Quick', 'A first look. Enough for an easy job; a harder one stops while it is still learning.'],
         ['standard', 'Standard', 'Right for most jobs.'],
         ['thorough', 'Thorough', 'For a hard job with plenty of rows.']].map(([id, name, line]) => `
      <label class="radio-row" for="preset-${id}"><input type="radio" id="preset-${id}" name="preset" value="${id}" ${(r.preset || 'standard') === id ? 'checked' : ''} data-change="preset">
        <span><b>${name}</b> — ${n(ITERS[id])} iterations, about ${minutesFor(r, ITERS[id])} min ${tag(c.measured === false ? 'estimate' : 'measured')}<br><span class="hint">${line}</span></span></label>`).join('')}
    </fieldset>
    ${sizeChoice(r)}
    <div class="panel flat">
      <h3>Memory</h3>
      ${memBar(c.peakGb)}
      <p class="hint">Expected from ${esc(c.source)} ${tag(c.measured === false ? 'estimate' : 'measured')}.${ramGb() ? ` This Mac has ${Math.round(ramGb())} GB.` : ''}${r.input === 'image' && c.peakGb > budgetGb() - 1 ? ' Close other apps before you start: a full-size picture job leaves little to spare.' : ''}</p>
    </div>
    <p class="hint">When it finishes it is converted for Pigeon and given its check ride straight away, on the ${n(exam(r).length)} exam rows.</p>
    ${cutAfterRecording(r) ? recutPanel(r) : need.length ? setupPanel(need) : `<div class="stage-foot">
      <p class="hint">You can leave this screen while it trains.</p>
      <button type="button" class="btn primary" data-do="train">${wasStale ? 'Teach it again' : 'Start teaching'}${ICON.arrow}</button>
    </div>`}
  </div>`;
}

function running (r) {
  const t = r.teach;
  const pct = Math.round(100 * (t.iter || 0) / (t.of || 1));
  const speed = t.itPerSec || t.estimate?.itPerSec;
  const left = speed ? Math.max(0, Math.round(((t.of || 0) - (t.iter || 0)) / speed / 60)) : null;
  const stage = t.phase === 'check ride'
    ? `Giving it the check ride${t.ride ? ` — ${t.ride.control ? 'the untrained model' : 'the taught Pigeon'}, row ${t.ride.i} of ${t.ride.n}` : ''}`
    : t.phase === 'convert' ? 'Converting it for Pigeon, and checking the conversion'
    : t.iter ? `Iteration ${n(t.iter)} of ${n(t.of)}${t.itPerSec ? ` · ${t.itPerSec} it/s` : ''}${left != null ? ` · about ${left} min left` : ''}`
    : `Getting ready — ${esc(t.phase || 'starting')}`;
  return `<div class="teach-run">
    <div class="patch-row">${patch('paint', { size: 72, play: true })}<div>
      <div class="prog-head"><b>${pct}%</b><span>${stage}</span></div>
      <div class="prog" role="progressbar" aria-valuemin="0" aria-valuemax="${t.of}" aria-valuenow="${t.iter || 0}" aria-label="Teaching progress"><span style="width:${pct}%"></span></div>
    </div></div>
    ${lossChart(t)}
    <div class="panel flat"><h3>Memory</h3>${memBar(t.peakGb || t.estimate?.peakGb || 0)}<p class="hint">${t.peakGb ? 'Peak so far, as the trainer reports it.' : `Expected ${tag('measured')} — this trainer does not report memory as it goes.`}</p></div>
    ${(t.warnings || []).slice(-3).map(w => `<p class="warn-line">${ICON.warn}${esc(w)}</p>`).join('')}
    <div class="stage-foot">
      <p class="hint">Leave this screen if you like — it keeps going, and the runs list shows how far it has got.</p>
      <button type="button" class="btn ghost" data-do="train-stop">Stop</button>
    </div>
  </div>`;
}

function done (r) {
  const t = r.teach;
  return `<div class="teach-done">
    <div class="patch-row">${patch('proud', { size: 72, frame: 10 })}<div>
      <p class="ready">${ICON.check}Taught${t.salvagedAt != null ? ` — from the checkpoint at iteration ${t.salvagedAt}, after the trainer stopped` : ''}.</p>
    </div></div>
    <div class="sum-grid">
      <div class="sum"><b>${t.minutes ?? '—'} min</b><span>on this Mac</span></div>
      <div class="sum"><b>${t.peakGb ? `${t.peakGb} GB` : '—'}</b><span>peak memory${t.peakGb ? '' : ' (not reported)'}</span></div>
      ${t.loss?.length ? `<div class="sum"><b>${t.loss[t.loss.length - 1][1].toFixed(3)}</b><span>final loss</span></div>` : ''}
    </div>
    ${lossChart(t)}
    <p class="hint">A low loss is not a grade. Flight School has measured, twice, that the lowest-loss checkpoint was the worse one. The check ride is the grade.</p>
    <div class="stage-foot">
      <button type="button" class="btn" data-do="train">Teach it again</button>
      <button type="button" class="btn primary" data-do="go" data-stage="ride">See the check ride${ICON.arrow}</button>
    </div>
  </div>`;
}

function failed (r) {
  const t = r.teach;
  return `<div class="teach-failed">
    <p class="bad-head">${ICON.warn}<b>${t.outOfMemory ? `It ran out of memory${t.iter ? ` at iteration ${n(t.iter)} of ${n(t.of)}` : ''}.` : 'It stopped before it finished.'}</b></p>
    <pre class="why">${esc(t.reason || '')}</pre>
    ${t.outOfMemory ? `<p>Close other apps and teach it again. ${r.input === 'image' ? 'A smaller window — or recording at a smaller size — needs less.' : 'Shorter rows need less.'}</p>` : ''}
    <div class="stage-foot">
      <button type="button" class="btn primary" data-do="train">Teach it again</button>
    </div>
  </div>`;
}

export function render (r) {
  const t = r.teach;
  if (t?.state === 'running') return running(r);
  if (t?.state === 'failed') return failed(r);
  if (t?.state === 'done' && !stale(r)) return done(r);
  return before(r);
}

export const actions = {
  train (r) {
    if (cutAfterRecording(r)) { toast('Train it as recorded, or record a new table, first.'); return; }
    const g = gate(r, 'teach');
    if (!g.open && !(r.teach)) { toast(g.why); return; }
    const need = needs(r);
    if (need.length) { toast(need[0].text); return; }   /* not false: the toast needs the redraw */
    /* Teaching again replaces a finished Pigeon: ask first, naming what goes. */
    if (r.teach?.state === 'done') { S.ui.dialog = { kind: 'teach-again' }; return; }
    start(r);
  },
  'train-start' (r) { S.ui.dialog = null; start(r); },
  'train-stop' () { S.ui.dialog = { kind: 'teach-stop' }; },
  'train-stop-confirm' (r) { S.ui.dialog = null; engine.stopTraining(r); toast('Teaching stopped. Nothing from this round was kept.'); },
  'train-as-recorded' (r) {
    const was = { picCrop: r.picCrop, maxSide: r.maxSide };
    trainAsRecorded(r);
    toast(`It trains on the pictures as recorded, at ${sourceSide(r)} px.`, () => Object.assign(r, was));
  },
  'record-smaller' (r) {
    const src = sourceSide(r);
    return openRecordAgain(r, { side: SIDES.find(k => k < src) }, 'Choose the smaller size, then record.');
  },
  'record-recut' (r) {
    return openRecordAgain(r, r.picCrop ? { part: r.picCrop } : { side: r.maxSide }, 'Record it again; this table stays as it is.');
  }
};

/* A new table, opened on Record with the part or size already chosen. The
   old table is untouched (state.js recordAgain). */
export function openRecordAgain (r, what, line) {
  const { id, record } = recordAgain(r, what);
  engine.persist(S.runs[id]);
  S.ui.record = { step: 'setup', window: null, keys: true, clicks: true, ...record, forRun: id };
  location.hash = `#/run/${encodeURIComponent(id)}/collect/record`;
  toast(`New table: ${S.runs[id].name}. ${line}`);
  return false;
}

function start (r) {
  /* Starting a run accepts Review's defaults, whether or not Review was opened. */
  r.reviewSeen = true;
  engine.train(r);
}

/* The two questions this stage asks before throwing work away. */
export function dialog (r, d) {
  if (d.kind === 'teach-stop') {
    const t = r.teach;
    const pct = t?.of ? Math.round(100 * (t.iter || 0) / t.of) : 0;
    return dialogHtml('Stop teaching?', `<p>It is ${pct}% through${t?.started ? `, ${Math.round((Date.now() - t.started) / 60000)} minutes in` : ''}. Stopping throws this round away; it cannot be resumed from here.${r.teachPrev ? ' The Pigeon it was taught before stays as it was.' : ''}</p>`,
      `<button type="button" class="btn primary" data-do="dialog-close">Keep teaching</button>
       <button type="button" class="btn danger" data-do="train-stop-confirm">Stop and lose this round</button>`, { ask: true });
  }
  if (d.kind === 'teach-again') {
    return dialogHtml('Teach it again?', `<p>This makes round ${(r.ride?.round || 1) + 1}, graded on the same exam. The Pigeon as it stands is kept until the new round finishes, and comes back if you stop early.</p>`,
      `<button type="button" class="btn ghost" data-do="dialog-close">Not now</button>
       <button type="button" class="btn primary" data-do="train-start">Teach it again</button>`, { ask: true });
  }
  return '';
}

export const changes = {
  preset (r, el) { r.preset = el.value; },
  side (r, el) { const k = Number(el.value); setPictureSide(r, !k || k === sourceSide(r) ? null : k); }
};
