/* pigeons.js — the lineup: every .pigeon this Mac can hand to Pigeon, as a
 * card each: add a file (or a .nest), export the ticked ones as a .nest,
 * show the file, take one out (to the Trash).
 * A .pigeon double-clicked in the Finder lands here too (app.js handOpens).
 *
 * Nothing here is a run: the list comes from `pigeon_list` and is refreshed
 * after every change, so the screen never drifts from the folder. */

import { S } from '../state.js';
import { needs } from '../engine.js';
import { esc, ICON, n, when, avatar } from './html.js';
import { toast, toastError, dialogHtml } from './cells.js';
import { patch } from './patch.js';

const tauri = window.__TAURI__;
const invoke = (cmd, args) => tauri.core.invoke(cmd, args);
const redraw = () => window.dispatchEvent(new Event('proto:render'));

export const mb = (b) => b >= 1e9 ? `${(b / 1e9).toFixed(1)} GB` : b >= 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${Math.max(1, Math.round((b || 0) / 1e3))} KB`;
const KIND = { classifier: 'sorts', router: 'routes', freeform: 'writes', grounded: 'answers from a document', persona: 'a voice', skillpack: 'a skill pack' };
export const kindWords = (k) => KIND[k] || (k ? String(k) : 'a job');
const baseWords = (b) => b ? String(b).replace(/\.gguf$/i, '').replace(/-Instruct.*$/i, '') : 'its base';

/* The lineup is asked for once per visit and again after every change. */
let loading = false;
export async function refresh () {
  if (loading) return;
  loading = true;
  try { S.pigeons = await invoke('pigeon_list') || []; S.ui.pigeonError = null; }
  catch (e) { S.pigeons = S.pigeons || []; S.ui.pigeonError = String(e); }
  loading = false;
  S.ui.pigeonsAt = Date.now();
  redraw();
}

/* What an add or an export did, in plain words across the top of the lineup. */
function band () {
  const b = S.ui.pigeonBand;
  if (!b) return '';
  if (b.kind === 'nest') return `<p class="ready">${ICON.check}Exported ${n(b.count)} ${b.count === 1 ? 'Pigeon' : 'Pigeons'} to <b>${esc(b.file)}</b> (${mb(b.bytes)}). Send that one file — AirDrop, a flash drive, anything — and <i>Add a Pigeon…</i> on the other Mac takes the whole flock.</p>`;
  const rs = b.results || [];
  const refused = rs.filter(r => r.status === 'refused');
  const parts = [];
  if (b.added) parts.push(`<b>Added ${n(b.added)}</b>`);
  if (b.already) parts.push(`${n(b.already)} already here`);
  if (b.refused) parts.push(`${n(b.refused)} refused`);
  return `<div class="${b.added ? 'ready' : b.refused ? 'warn-line' : 'hint'} band">${b.added ? ICON.check : b.refused ? ICON.warn : ''}<span>${parts.join(' · ') || 'Nothing to add.'}
    ${rs.filter(r => r.status !== 'refused').map(r => `<br>${esc(r.name)}${r.from_nest ? ` <span class="hint">from ${esc(r.from_nest)}</span>` : ''}${r.status === 'already' ? ' — the same training is already in the lineup' : ''}${(r.warnings || []).length ? ` — ${esc(r.warnings.join(' · '))}` : ''}`).join('')}
    ${refused.map(r => `<br><b>${esc(r.name)}</b>${r.from_nest ? ` (in ${esc(r.from_nest)})` : ''}: ${esc((r.errors || []).join(' · '))}${r.base_needed ? ' — download that model in Pigeon, then add the file again.' : ''}`).join('')}
  </span></div>`;
}

function card (p, ticked) {
  const score = p.passed != null
    ? `<span class="pg-score${p.graded_by_model ? ' judged' : ''}" title="${p.passed} of ${p.total} passed with the training on${p.control_passed != null ? `; ${p.control_passed} of ${p.total} with it removed` : ''}.${p.graded_by_model ? ' Graded by a model — an opinion.' : ''} ${p.source === 'run' ? 'Measured here' : 'Measured on the Mac that made it'}${p.engine ? `, asked through ${p.engine}` : ''}.">${p.passed}/${p.total}${p.graded_by_model ? ' ~' : ''}</span>`
    : '<span class="pg-score none" title="This file carries no test results.">no score</span>';
  return `<div class="pg-card" data-hay="${esc([p.name, kindWords(p.kind), p.base || '', p.vision ? 'pictures' : ''].join(' ').toLowerCase())}">
    <label class="pg-tick" title="Ticked Pigeons go into the .nest when you export."><input type="checkbox" data-change="pigeon-tick" data-id="${esc(p.id)}" ${ticked ? 'checked' : ''}><span class="sr">In the flock</span></label>
    <div class="pg-top">${avatar(p.name, { kind: p.kind, src: p.icon })}
      <span class="pg-name"><b>${esc(p.name)}</b><span class="hint">${esc(kindWords(p.kind))}${p.vision ? ' · from pictures' : ''}</span></span>${score}</div>
    <p class="hint pg-meta">needs ${esc(baseWords(p.base))}${p.base_match === 'same_base_other_quant' ? ' <span title="Same model, a different size — it loads, with a warning">(close match)</span>' : ''} · ${mb(p.bytes)}${p.engine ? ` · <span title="Its score was asked through ${esc(p.engine)}, the engine Pigeon serves it with">scored in pigeon-engine</span>` : ''} · ${p.source === 'run' ? 'made here' : 'added'} ${p.added_ms ? esc(when(new Date(p.added_ms).toISOString())) : ''}</p>
    <div class="pg-act">
      <button type="button" class="btn sm" data-do="pigeon-reveal" data-id="${esc(p.id)}">Show the file</button>
      ${p.run_id && S.runs[p.run_id] ? `<a class="btn sm" href="#/run/${esc(p.run_id)}/ride">The check ride</a>` : ''}
      <button type="button" class="btn sm ghost" data-do="pigeon-remove" data-id="${esc(p.id)}" data-name="${esc(p.name)}">Remove…</button>
    </div>
  </div>`;
}

export function render () {
  if (S.pigeons == null && !loading) refresh();
  const list = S.pigeons || [];
  const pick = S.ui.pigeonPick;   // null: all of them
  const ticked = (id) => pick == null || pick.includes(id);
  const nTicked = list.filter(p => ticked(p.id)).length;
  const need = needs(null);
  const tools = !need.length;
  const why = need[0]?.text || '';
  return `<section class="screen">
    <header class="screen-head">
      <div><h1>Pigeons</h1><p class="lede">Everything this Mac can hand to Pigeon. Each one is a single <span class="mono">.pigeon</span> file: the training, the exact way to ask it, the answers it may give, and its scorecard.</p></div>
      <div class="pg-bar">
        <button type="button" class="btn" data-do="pigeon-nest" ${list.length && nTicked && tools && !S.ui.pigeonBusy ? '' : `aria-disabled="true"`} title="${tools ? 'Every ticked Pigeon, in one .nest file' : esc(why)}">${ICON.send}Export the flock…${list.length && nTicked < list.length ? ` (${nTicked} of ${list.length})` : ''}</button>
        <button type="button" class="btn primary" data-do="pigeon-add" ${tools && !S.ui.pigeonBusy ? '' : 'aria-disabled="true"'} title="${tools ? 'A .pigeon or a .nest someone sent you' : esc(why)}">${ICON.plus}${S.ui.pigeonBusy === 'add' ? 'Checking…' : 'Add a Pigeon…'}</button>
      </div>
    </header>
    ${S.ui.pigeonError ? `<p class="warn-line" role="alert">${ICON.warn}${esc(S.ui.pigeonError)} <button type="button" class="btn sm ghost" data-do="pigeon-refresh">Try again</button></p>` : ''}
    ${band()}
    ${S.pigeons == null ? '<p class="hint">Reading the lineup…</p>'
      : !list.length ? `<div class="panel empty-home">${patch('look', { size: 72, play: true })}<p><b>Nothing in the lineup yet.</b> A run that reaches Send drops its Pigeon here, and <i>Add a Pigeon…</i> copies in a file someone sent you. Double-clicking a <span class="mono">.pigeon</span> in the Finder does the same.</p>
          <p><button type="button" class="btn primary" data-do="new-run">${ICON.plus}Start a run</button></p></div>`
      : `${list.length > 6 ? `<label class="search">${ICON.search}<input type="search" id="pigeon-filter" placeholder="Filter by name, kind or model" value="${esc(S.ui.pigeonFilter || '')}" data-input="pigeon-filter" autocomplete="off" spellcheck="false" aria-label="Filter the lineup by name, kind or model"></label>` : ''}
         <div class="pg-grid">${list.map(p => card(p, ticked(p.id))).join('')}</div>
         <p class="hint" id="pigeon-nomatch" hidden>Nothing in the lineup matches that. The ticks are untouched.</p>
         <p class="hint">A double-clicked <span class="mono">.pigeon</span> or <span class="mono">.nest</span> lands here too. <b>~</b> marks a score graded by a model rather than exact-matched.</p>`}
  </section>`;
}

export function dialog (d) {
  if (d.kind !== 'pigeon-remove') return '';
  return dialogHtml(`Take “${d.name}” out of the lineup?`, '<p>The file goes to the Trash, not away. A run that made it keeps its own copy.</p>',
    `<button type="button" class="btn ghost" data-do="dialog-close">Keep it</button>
     <button type="button" class="btn danger" data-do="pigeon-remove-confirm" data-id="${esc(d.id)}">Move to the Trash</button>`, { ask: true });
}

/* The filter hides cards in place: a redraw per keystroke would lose the
   caret, and what is ticked for the flock is deliberately untouched. */
export function applyFilter () {
  const q = (S.ui.pigeonFilter || '').trim().toLowerCase();
  let shown = 0;
  for (const el of document.querySelectorAll('.pg-card[data-hay]')) { const hit = !q || el.dataset.hay.includes(q); el.hidden = !hit; if (hit) shown++; }
  const none = document.getElementById('pigeon-nomatch');
  if (none) none.hidden = !(q && shown === 0);
}

export const actions = {
  'pigeon-refresh' () { S.ui.pigeonError = null; refresh(); },
  async 'pigeon-add' (el) {
    if (el.getAttribute('aria-disabled') === 'true') { toast(el.title); return; }
    S.ui.pigeonBusy = 'add'; redraw();
    try {
      const r = await invoke('pigeon_add_pick');
      if (r) { S.ui.pigeonBand = { kind: 'add', ...r }; S.ui.pigeonPick = null; await refresh(); }
    } catch (e) { toastError(String(e)); }
    S.ui.pigeonBusy = null; redraw();
  },
  async 'pigeon-nest' (el) {
    if (el.getAttribute('aria-disabled') === 'true') { toast(el.title); return; }
    const list = S.pigeons || [];
    const ids = list.filter(p => S.ui.pigeonPick == null || S.ui.pigeonPick.includes(p.id)).map(p => p.id);
    S.ui.pigeonBusy = 'nest'; redraw();
    try {
      const r = await invoke('pigeon_export_nest', { ids, name: null });
      if (r) S.ui.pigeonBand = { kind: 'nest', file: String(r.path).split('/').pop(), bytes: r.bytes, count: ids.length };
    } catch (e) { toastError(String(e)); }
    S.ui.pigeonBusy = null; redraw();
  },
  'pigeon-reveal' (el) { invoke('pigeon_reveal', { id: el.dataset.id }).catch(e => { toastError(String(e)); }); },
  'pigeon-remove' (el) { S.ui.dialog = { kind: 'pigeon-remove', id: el.dataset.id, name: el.dataset.name }; redraw(); },
  async 'pigeon-remove-confirm' (el) {
    S.ui.dialog = null; redraw();
    try { await invoke('pigeon_remove', { id: el.dataset.id }); toast('Moved to the Trash.'); await refresh(); }
    catch (e) { toastError(String(e)); redraw(); }
  }
};

export const changes = {
  'pigeon-tick' (el) {
    const all = (S.pigeons || []).map(p => p.id);
    const cur = new Set(S.ui.pigeonPick == null ? all : S.ui.pigeonPick);
    if (el.checked) cur.add(el.dataset.id); else cur.delete(el.dataset.id);
    S.ui.pigeonPick = cur.size === all.length ? null : [...cur];
    redraw();
  }
};

export const inputs = {
  'pigeon-filter' (el) { S.ui.pigeonFilter = el.value; applyFilter(); }
};
