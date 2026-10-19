/* app.js — CARRIES FORWARD. Routing, the one render, and event delegation.
 *
 * The same shape as src/index.html: every screen is a function of S, one
 * render() redraws, and events are handled once, at the root:
 *   data-do       a click     → a screen module's `actions`, or one below
 *   data-input    typing      → `inputs`  (returns 'none' to skip the redraw)
 *   data-change   a choice    → `changes`
 *   data-submit   a form      → `submits`
 *
 * Rules from Pigeon's UX audit enforced here, not left to each screen:
 *   - No redraw takes the keyboard: focus and caret survive every render.
 *   - Nothing fails in silence: a control that will not act says why.
 *   - Every dialog says it is one, and Escape closes it.
 *   - Background work (training, recording) redraws only what depends on it,
 *     so an open menu is not torn down by a progress tick.
 */

import { S, STAGES, reset, run, newRun, setKind, gate, currentStage, stale, tableRows } from './state.js';
import { engine } from './engine.js';
import { rail } from './screens/shell.js';
import { openMenu } from './screens/menu.js';
import { settingsDialog, actions as settingsActions, applyTheme } from './screens/settings.js';

/* The appearance chosen last time, before anything is drawn. */
applyTheme();
import { runsScreen } from './screens/runs.js';
import { kindScreen } from './screens/kind.js';
import * as pigeons from './screens/pigeons.js';
import * as bench from './screens/bench.js';
import { runScreen, runDialog, MODULES } from './screens/run.js';
import { toastHtml, toast, toastError, viewerKey, dialogHtml } from './screens/cells.js';
import { esc, ICON, n } from './screens/html.js';

const mods = Object.values(MODULES);
const pickAll = (key) => Object.assign({}, ...mods.map(m => m[key] || {}));
const ACTIONS = pickAll('actions');
const INPUTS = pickAll('inputs');
const CHANGES = pickAll('changes');
const SUBMITS = pickAll('submits');
const BLURS = pickAll('blurs');
const DRAGS = pickAll('drags');

const root = document.getElementById('app');
let moveFocus = false;
let hadDialog = false;

function screen () {
  if (S.view === 'run' && run()?._stub) return '<section class="screen"><p class="hint">Opening…</p></section>';
  if (S.view === 'run' && run()) return runScreen(run());
  if (S.view === 'new') return kindScreen();
  if (S.view === 'pigeons') return pigeons.render();
  if (S.view === 'bench') return bench.render();
  return runsScreen();
}

/* Rust holds a window close while a job is running (lib.rs) and asks the page
   to ask. Nothing has been stopped yet: Keep leaves everything alone; Close
   is the only way out, so it names what is thrown away. */
function busyWords () {
  const teaching = Object.values(S.runs).find(r => r.teach?.state === 'running');
  if (teaching) return `“${teaching.name || 'A new job'}” is still teaching, ${Math.round(100 * teaching.teach.iter / teaching.teach.of)}% done`;
  if (S.ui.record?.step === 'live') return 'A recording is still going';
  return 'A job is still running';
}
function closingDialog () {
  return dialogHtml('Close Flight School?', `<p>${esc(busyWords())}. Closing stops it, and it cannot be resumed from where it stopped.</p>`,
    `<button type="button" class="btn primary" data-do="close-keep">Keep going</button>
     <button type="button" class="btn ghost" data-do="close-quit">Close and lose it</button>`, { ask: true });
}
/* What the ⋯ menu for a run leads to (the menu itself is menu.js): renaming,
   and a Trash move that says what goes with it. */
function runMenuDialog (d) {
  const r = S.runs[d.id];
  if (!r) return '';
  const name = r.name || 'A new job';
  if (d.kind === 'run-rename') {
    return dialogHtml('Rename this run', `<form id="rename-form" data-submit="run-rename-save" data-id="${esc(r.id)}">
        <input id="rename-name" class="text-input" type="text" maxlength="60" value="${esc(r.name)}" autocomplete="off" spellcheck="false" aria-label="The run’s name">
      </form>`,
      `<button type="button" class="btn" data-do="dialog-close">Cancel</button>
       <button type="submit" form="rename-form" class="btn primary">Rename</button>`, { ask: true });
  }
  if (d.kind === 'run-trash') {
    const busy = r.teach?.state === 'running';
    return dialogHtml(`Move “${name}” to the Trash?`, `<p>${busy ? 'It is still teaching. Stop the run first, then move it.' : `Its ${n(r._stub ? (r._summary?.rows ?? 0) : tableRows(r).length)} rows, its pictures and anything it was taught go with it. You can put it back from the Trash.`}</p>`,
      `<button type="button" class="btn ghost" data-do="dialog-close">Keep it</button>
       <button type="button" class="btn danger" data-do="run-trash-confirm" data-id="${esc(r.id)}" ${busy ? 'disabled' : ''}>Move to the Trash</button>`, { ask: true });
  }
  return '';
}
function dialogFor () {
  const d = S.ui.dialog;
  if (d?.kind === 'closing') return closingDialog();
  if (d?.kind === 'settings') return settingsDialog(d);
  if (d?.kind === 'run-rename' || d?.kind === 'run-trash') return runMenuDialog(d);
  if (d?.kind === 'pigeon-remove') return pigeons.dialog(d);
  return S.view === 'run' && run() ? runDialog(run()) : '';
}
/* A save that failed: one line that stays, can be copied, and offers a retry. */
function saveBar () {
  const e = S.ui.saveError;
  if (!e) return '';
  return `<div class="save-bar" role="alert">${ICON.warn}<span><b>Not saved.</b> ${esc(e.text)}</span>
    <button type="button" class="btn sm" data-do="save-retry">Try again</button></div>`;
}

export function render () {
  const active = document.activeElement;
  const keep = active && active.id && active !== document.body
    ? { id: active.id, sel: typeof active.selectionStart === 'number' ? [active.selectionStart, active.selectionEnd] : null }
    : null;
  /* What opened a dialog, as a selector, so focus can go back to it: an id,
     or the button's own data-do (and data-id) when it has no id. */
  const opener = active && active !== document.body
    ? (active.id ? `#${CSS.escape(active.id)}` : active.dataset?.do ? `[data-do="${active.dataset.do}"]${active.dataset.id ? `[data-id="${CSS.escape(active.dataset.id)}"]` : ''}` : null)
    : null;
  const main = document.getElementById('main');
  const scroll = main ? main.scrollTop : 0;

  root.innerHTML = `<div class="app">
    ${rail()}
    <main class="main" id="main">${saveBar()}${screen()}</main>
    ${dialogFor()}
    ${toastHtml()}
  </div>`;

  const m2 = document.getElementById('main');
  /* A dialog is modal: the page behind it is inert, and focus goes back to
     whatever opened it when it closes. */
  const dlg = Boolean(S.ui.dialog);
  for (const el of root.querySelectorAll('.rail, .main')) el.inert = dlg;
  if (dlg && !hadDialog) S.ui.dialogOpener = opener;
  if (!dlg && hadDialog && S.ui.dialogOpener && !S.ui.focusNext) {
    const back = root.querySelector(S.ui.dialogOpener);
    S.ui.dialogOpener = null;
    if (back) { back.focus({ preventScroll: true }); moveFocus = false; }
  }
  if (moveFocus) {
    moveFocus = false;
    if (m2) m2.scrollTop = 0;
    const h = document.getElementById('stage-title') || root.querySelector('main h1');
    if (h && !S.ui.focusNext) { if (!h.hasAttribute('tabindex')) h.setAttribute('tabindex', '-1'); h.focus({ preventScroll: true }); }
  } else {
    if (m2) m2.scrollTop = scroll;
    if (S.ui.dialog && !hadDialog) document.getElementById('dlg-title')?.focus();
    else if (keep) {
      const el = document.getElementById(keep.id);
      if (el) {
        el.focus({ preventScroll: true });
        if (keep.sel && el.setSelectionRange) { try { el.setSelectionRange(keep.sel[0], keep.sel[1]); } catch { /* not text */ } }
      }
    }
  }
  if (S.view === 'pigeons') pigeons.applyFilter();
  /* A refusal can ask for focus on the field that fixes it. */
  if (S.ui.focusNext) { document.getElementById(S.ui.focusNext)?.focus({ preventScroll: !moveFocus && !hadDialog }); S.ui.focusNext = null; }
  hadDialog = Boolean(S.ui.dialog);
}

/* ── routing ─────────────────────────────────────────────────────────────
   #/runs · #/pigeons · #/bench · #/new · #/run/<id>[/<stage>[/<tab>]]
   A deep link opens exactly what it names — even a stage the rules keep
   shut, which then says so. Only the stage bar enforces the rules.
   #/new is the kinds screen; a run exists only once a kind is picked. */

const hashFor = () => S.view === 'run' && S.runId
  ? `#/run/${encodeURIComponent(S.runId)}/${S.stage}${S.tab ? `/${S.tab}` : ''}`
  : `#/${S.view}`;

function resetRunUi (id) {
  Object.assign(S.ui, { filter: 'all', page: 1, editing: null, adding: false, conflict: 0, target: null, merge: [], rideFilter: 'all', dialog: null, repeats: null, repeatPage: 1, session: null, dropping: false });
  /* Record set up for the run being opened ("record again") is kept. */
  if (S.ui.record?.step !== 'live' && S.ui.record?.forRun !== id) S.ui.record = null;
}

function route () {
  const [view, id, stage, tab] = location.hash.replace(/^#\/?/, '').split('/').filter(Boolean).map(decodeURIComponent);
  S.ui.notice = null;
  if (view === 'new') { S.view = 'new'; S.runId = null; S.ui.dialog = null; moveFocus = true; render(); return; }
  if (view === 'run' && S.runs[id]) {
    if (S.runId !== id) resetRunUi(id);
    S.view = 'run';
    S.runId = id;
    /* A run from the list is a stub until its table is read. */
    if (S.runs[id]._stub) {
      engine.open(id).then(route).catch(e => { S.ui.loadError = String(e); S.view = 'runs'; S.runId = null; render(); });
      moveFocus = true; render(); return;
    }
    const nextStage = STAGES.some(s => s.id === stage) ? stage : currentStage(S.runs[id]);
    if (nextStage !== S.stage) Object.assign(S.ui, { editing: null, adding: false });
    S.stage = nextStage;
    S.tab = tab || null;
    if (!stage) { location.replace(hashFor()); return; }
  } else if (view === 'pigeons' || view === 'bench') {
    S.view = view; S.runId = null;
  } else {
    S.view = 'runs'; S.runId = null;
  }
  moveFocus = true;
  render();
}

/* After an action, the address follows the state, so a link copied at any
   moment reopens that moment. */
function settle () {
  /* Every change a person makes is saved as it is made. */
  engine.persist(run());
  const h = hashFor();
  if (h !== location.hash) location.hash = h; else render();
}

function goStage (stage, tab = null) {
  const r = run();
  const g = gate(r, stage);
  if (!g.open) { S.ui.notice = g.why; render(); document.getElementById('stage-why')?.focus?.(); return; }
  S.stage = stage; S.tab = tab;
  settle();
}

/* ── events ──────────────────────────────────────────────────────────── */

const GENERIC = {
  'new-run' () { location.hash = '#/new'; },
  /* A kind card, on the #/new screen or inside Describe. On #/new it makes the
     run; in a run it sets the kind, or says why the table will not allow it. */
  'kind-pick' (el) {
    const r = S.view === 'run' ? run() : null;
    if (r) {
      const why = setKind(r, el.dataset.kind);
      if (why) { toast(why); render(); return; }
      S.ui.focusNext = 'describe-name';
      settle();
      return;
    }
    const rid = newRun();
    setKind(S.runs[rid], el.dataset.kind);
    engine.persist(S.runs[rid]);
    S.ui.focusNext = 'describe-name';
    location.hash = `#/run/${encodeURIComponent(rid)}/describe`;
  },
  stage (el) { goStage(el.dataset.stage); },
  go (el) { goStage(el.dataset.stage); },
  fix (el) {
    if (el.dataset.filter) Object.assign(S.ui, { filter: el.dataset.filter, page: 1 });
    goStage(el.dataset.stage, el.dataset.tab || null);
  },
  refuse (el) { toast(el.dataset.why); render(); },
  'dialog-close' () { S.ui.dialog = null; render(); },
  'toast-dismiss' () { S.ui.toast = null; render(); },
  'close-keep' () { S.ui.dialog = null; render(); },
  async 'close-quit' () {
    try { await window.__TAURI__.core.invoke('force_quit'); } catch (e) { S.ui.dialog = null; toast(String(e), null, { sticky: true }); render(); }
  },
  'save-retry' () {
    const id = S.ui.saveError?.run;
    const r = (id && S.runs[id]) || run();
    if (r) engine.saveNow(r).then(render); else { S.ui.saveError = null; render(); }
  },
  /* This Mac */
  'setup-env' () { engine.setup('env'); },
  'setup-vision' () { engine.setup('vision'); },
  'env-recheck' () { engine.refreshEnv(); toast('Checking this Mac…'); render(); },
  /* The run menu */
  /* Pigeon's floating menu under the ⋯; Rename and the Trash go on to a dialog. */
  'run-menu' (el, e, at = el) {
    const id = el.dataset.id;
    openMenu(at, [
      { id: 'run-rename', label: 'Rename…', icon: 'rename', title: 'The run’s name. The Pigeon it makes is named on Send.' },
      { id: 'run-duplicate', label: 'Duplicate', icon: 'copy', title: 'A new run with this table and its pictures, not taught yet.' },
      { rule: true },
      { id: 'run-trash', label: 'Move to the Trash…', icon: 'trash', danger: true, title: 'Rows, pictures and training, all of it, into the Trash.' }
    ], (what) => GENERIC[what]({ dataset: { id } }));
  },
  'run-rename' (el) { S.ui.dialog = { kind: 'run-rename', id: el.dataset.id }; S.ui.focusNext = 'rename-name'; render(); document.getElementById('rename-name')?.select?.(); },
  'run-rename-save' (form) {
    const v = form.querySelector('#rename-name')?.value.trim() ?? '';
    S.ui.dialog = null;
    engine.open(form.dataset.id).then(r => {
      if (r) { r.name = v; r.updated = new Date().toISOString(); engine.persist(r); }
      render();
    }).catch(e => { toastError(String(e)); render(); });
  },
  'run-duplicate' (el) {
    const r = S.runs[el.dataset.id];
    if (!r) return;
    S.ui.dialog = null; render();
    engine.duplicate(r.id)
      .then(nid => { toast(`Copied as “${S.runs[nid].name || 'A new job'}”.`); location.hash = `#/run/${encodeURIComponent(nid)}/describe`; })
      .catch(e => { toastError(`Could not duplicate it: ${e}`); render(); });
  },
  'run-trash' (el) { S.ui.dialog = { kind: 'run-trash', id: el.dataset.id }; render(); },
  'run-trash-confirm' (el) {
    const id = el.dataset.id;
    const r = S.runs[id];
    if (!r || r.teach?.state === 'running') return;
    const name = r.name || 'A new job';
    engine.trash(id)
      .then(() => { S.ui.dialog = null; toast(`“${name}” is in the Trash.`); if (S.runId === id) location.hash = '#/runs'; else render(); })
      .catch(e => { S.ui.dialog = null; toastError(`Could not move it to the Trash: ${e}`); render(); });
  },
  undo () {
    const u = S.ui.toast?.undo;
    S.ui.toast = null;
    if (u) { u(); const r = run(); if (r) { r.version++; engine.persist(r); } toast('Undone.'); }
    render();
  },
  reload () { S.ui.loadError = null; render(); engine.load().catch(e => { S.ui.loadError = String(e); }).finally(route); }
};

root.addEventListener('click', (e) => {
  const el = e.target.closest('[data-do]');
  if (!el) return;
  const what = el.dataset.do;
  if (el.getAttribute('aria-disabled') === 'true' && what === 'go') { goStage(el.dataset.stage); return; }
  if (GENERIC[what]) { if (el.tagName === 'A') e.preventDefault(); GENERIC[what](el, e); return; }
  if (settingsActions[what]) { settingsActions[what](el, e); render(); return; }
  if (pigeons.actions[what]) { e.preventDefault(); pigeons.actions[what](el, e); return; }
  const r = run();
  if (ACTIONS[what] && r) {
    e.preventDefault();
    const res = ACTIONS[what](r, el, e);
    if (res !== false) settle();
  }
});

/* Right-click on a run in the list: the same menu as its ⋯, at the pointer. */
root.addEventListener('contextmenu', (e) => {
  const more = e.target.closest('.run-item')?.querySelector('[data-do="run-menu"]');
  if (!more) return;
  e.preventDefault();
  GENERIC['run-menu'](more, e, { x: e.clientX, y: e.clientY });
});

/* A drag on something marked data-drag (the crop box): the module follows
   the pointer itself and redraws when it lets go. */
root.addEventListener('pointerdown', (e) => {
  const el = e.target.closest('[data-drag]');
  const r = run();
  if (!el || !r || !DRAGS[el.dataset.drag] || e.button !== 0) return;
  e.preventDefault();
  DRAGS[el.dataset.drag](r, el, e);
});

/* A picture asked for in the moment before its table is saved is refused;
   it is asked again a few times, a little later, before it is given up on. */
root.addEventListener('error', (e) => {
  const img = e.target;
  if (!(img instanceof HTMLImageElement) || !img.src.startsWith('pic:')) return;
  const n = Number(img.dataset.retry || 0);
  if (n >= 4) return;
  img.dataset.retry = String(n + 1);
  const base = img.src.replace(/([?&])r=\d+/, '$1').replace(/[?&]$/, '');
  setTimeout(() => { if (img.isConnected) img.src = `${base}${base.includes('?') ? '&' : '?'}r=${n + 1}`; }, 500 * (n + 1));
}, true);

root.addEventListener('input', (e) => {
  const el = e.target;
  if (el.dataset?.input && pigeons.inputs[el.dataset.input]) { pigeons.inputs[el.dataset.input](el, e); return; }
  const h = el.dataset?.input && INPUTS[el.dataset.input];
  const r = run();
  if (!h || !r) return;
  const res = h(r, el, e);
  engine.persist(r);
  if (res === 'soft') render();
});

root.addEventListener('change', (e) => {
  const el = e.target;
  if (el.dataset?.change && pigeons.changes[el.dataset.change]) { pigeons.changes[el.dataset.change](el, e); return; }
  const r = run();
  if (!r) return;
  if (el.dataset?.change && CHANGES[el.dataset.change]) { CHANGES[el.dataset.change](r, el, e); settle(); return; }
  /* A field that updates live redraws the rest of the screen when you leave it
     — but not synchronously. 'change' fires while focus is moving to whatever
     you clicked next; redrawing then would destroy that element under the
     click. Waiting a tick lets focus land, and render() carries it across. */
  if (el.dataset?.input && BLURS[el.dataset.input]) setTimeout(render, 0);
});

root.addEventListener('submit', (e) => {
  const form = e.target;
  if (form.dataset?.submit && GENERIC[form.dataset.submit]) { e.preventDefault(); GENERIC[form.dataset.submit](form, e); return; }
  const h = form.dataset?.submit && SUBMITS[form.dataset.submit];
  const r = run();
  if (!h || !r) return;
  e.preventDefault();
  h(r, form, e);
  settle();
});

document.addEventListener('keydown', (e) => {
  /* Enter in the name field carries on to Collect, as the button would. */
  if (e.key === 'Enter' && e.target.id === 'describe-name' && !e.isComposing) { e.preventDefault(); goStage('collect'); return; }
  /* ⌘, opens Settings — the menu item's key, for a page with no menu bar. */
  if (e.key === ',' && (e.metaKey || e.ctrlKey)) { e.preventDefault(); settingsActions.settings({ dataset: {} }); render(); return; }
  /* Settings' tabs and choices move with the arrow keys, as their roles promise. */
  if ((e.key === 'ArrowLeft' || e.key === 'ArrowRight') && e.target.closest?.('.settings-card [role="tablist"], .settings-card [role="radiogroup"]')) {
    const group = [...e.target.parentElement.querySelectorAll('button')];
    const next = group[(group.indexOf(e.target) + (e.key === 'ArrowRight' ? 1 : group.length - 1)) % group.length];
    if (next) { e.preventDefault(); next.click(); document.getElementById(next.id)?.focus(); }
    return;
  }
  if (e.key === 'Escape') {
    if (S.ui.dialog) { S.ui.dialog = null; render(); return; }
    if (S.ui.editing || S.ui.adding) { S.ui.editing = null; S.ui.adding = false; render(); return; }
  }
  /* The viewer's answer menu keeps focus after a choice; its keys still work. */
  const typing = e.target.closest?.('input, textarea, select:not(#viewer-answer), [contenteditable]');
  if (!typing && viewerKey(run(), e)) {
    e.preventDefault();
    const r = run();
    if (r) engine.persist(r);
    render();
    return;
  }
  if (!typing && S.view === 'run' && S.stage === 'review' && S.tab === 'conflicts' && (e.key === 'j' || e.key === 'k')) {
    S.ui.conflict = Math.max(0, S.ui.conflict + (e.key === 'j' ? 1 : -1));
    render();
  }
});

/* Background work redraws what depends on it: the runs list, or the open
   stage of the run that changed. Anywhere else, only the header's progress
   chip would change, and that is not worth tearing down an open menu for. */
engine.onChange((r) => {
  if (S.ui.record?.step === 'live' && S.view === 'run' && S.tab === 'record') return render();
  if (S.view === 'runs') return render();
  if (S.view !== 'run' || !r || S.runId !== r.id) return;
  if (S.stage === 'teach' || S.tab === 'record' || S.stage === 'ride' || S.stage === 'send') return render();
  const chip = document.querySelector('.chip.busy');
  if (chip && r.teach?.state === 'running') chip.lastChild.textContent = `Teaching ${Math.round(100 * r.teach.iter / r.teach.of)}%`;
  else if (r.teach?.state !== 'running') render();
});

window.addEventListener('proto:render', render);

/* A .pigeon or .nest double-clicked in Finder is added by Rust and shown on
   the Pigeons screen. */
async function handOpens () {
  let results = [];
  try { results = await window.__TAURI__.core.invoke('pigeon_take_opens') || []; } catch { return; }
  if (!results.length) return;
  const all = { kind: 'add', added: 0, already: 0, refused: 0, results: [] };
  for (const x of results) { all.added += x.added || 0; all.already += x.already || 0; all.refused += x.refused || 0; all.results.push(...(x.results || [])); }
  S.ui.pigeonBand = all; S.ui.pigeonPick = null; S.pigeons = null;
  if (S.view === 'pigeons') render(); else location.hash = '#/pigeons';
}
window.__TAURI__.event.listen('pigeon-added', handOpens);
/* A file dragged from the Finder onto the window: a Reader's Collect stage
   takes it as material; anywhere else says where to drop it instead. */
function dropTarget () {
  const r = run();
  return S.view === 'run' && r && r.kind === 'reader' && S.stage === 'collect' && !S.ui.dialog ? r : null;
}
window.__TAURI__.event.listen('tauri://drag-enter', () => { if (dropTarget() && !S.ui.dropping) { S.ui.dropping = true; render(); } });
window.__TAURI__.event.listen('tauri://drag-leave', () => { if (S.ui.dropping) { S.ui.dropping = false; render(); } });
window.__TAURI__.event.listen('tauri://drag-drop', (e) => {
  const paths = (e.payload && e.payload.paths) || [];
  S.ui.dropping = false;
  const r = dropTarget();
  if (!r) { if (paths.length) toast('To make a document the material, drop it on a Reader’s Collect stage.'); render(); return; }
  MODULES.collect.addMaterial(r, paths);
  render();
});
window.__TAURI__.event.listen('close-requested', () => { S.ui.dialog = { kind: 'closing' }; render(); });
/* The menu bar (lib.rs): ⌘N and ⌘1–3. */
window.__TAURI__.event.listen('menu', (e) => {
  const id = String(e.payload || '');
  if (id === 'new-run') location.hash = '#/new';
  else if (id === 'settings') { settingsActions.settings({ dataset: {} }); render(); }
  else if (id.startsWith('view-')) location.hash = `#/${id.slice(5)}`;
});
handOpens();
window.addEventListener('hashchange', route);
/* Back from System Settings: ask macOS again what recording may do. */
window.addEventListener('focus', () => { if (S.ui.dialog?.kind === 'settings' && S.ui.dialog.tab === 'privacy') { S.ui.settingsPerm = undefined; render(); } });
reset();
engine.load()
  .catch(e => { S.ui.loadError = String(e); })
  .finally(route);

export { stale };
