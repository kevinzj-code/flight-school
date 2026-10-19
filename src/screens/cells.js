/* cells.js — CARRIES FORWARD. The pieces every stage draws a row with: the
 * input cell, the answer chip, the honesty tags, the toast and the dialog. */

import { S, labelOf, run, answerList, updateRow, leaveOut, putBack } from '../state.js';
import { esc, ICON } from './html.js';
import { engine } from '../engine.js';

/* Where a number came from. Every figure that is not a plain count carries
   one, so nobody mistakes the prototype's inventions for measurements. */
const TAGS = {
  measured: ['measured', 'Measured on a 16 GB M1 Pro. Times on another Mac scale with its speed.'],
  estimate: ['estimate', 'Worked out between two measurements, not measured itself.']
};
export const tag = (kind) => kind && TAGS[kind] ? `<span class="tag ${kind}" title="${esc(TAGS[kind][1])}">${TAGS[kind][0]}</span>` : '';

/* A row's input: its words, or its picture — with the spot, when it was a
   click. The picture is fetched once and drawn when it arrives. A picture is
   also a button: it opens in the picture viewer, beside every other picture
   on the screen. */
export function inputCell (row, { big = false, r = run(), view = true, full = false, size = '' } = {}) {
  const i = row.input;
  /* A Reader's row: the question, and the passage it was asked against
     folded under it — what trains is both, in one input. */
  if (i.kind === 'text' && row.question != null) {
    const passage = i.text.endsWith(row.question) ? i.text.slice(0, i.text.length - row.question.length).trim() : i.text;
    return `<div class="cell-reader"><p class="cell-text${big ? ' big' : ''}">${esc(row.question)}</p>
      <details class="passage"><summary><span class="badge sec">§${esc(String(row.shown?.[0] ?? '?'))}</span><span class="passage-title">${esc(row.sectionTitle || 'the passage')}</span>${row.refusal ? '<span class="badge refusal" title="The passage does not answer this. The right answer says so, and cites nothing.">refusal</span>' : ''}</summary>
      <p class="passage-text">${esc(passage)}</p></details></div>`;
  }
  if (i.kind === 'text') return `<p class="cell-text${big ? ' big' : ''}">${esc(i.text)}</p>`;
  const url = r && i.path ? engine.image(r, i.path, { full }) : null;
  const w = r?.window?.w, h = r?.window?.h;
  const dot = i.click && w && h
    ? `<i class="cell-dot" style="left:${(100 * i.click[0] / w).toFixed(2)}%;top:${(100 * i.click[1] / h).toFixed(2)}%"></i>` : '';
  const cls = `cell-art${big ? ' big' : ''}${size ? ` ${size}` : ''}${w && w > h * 1.3 ? ' wide' : ''}`;
  /* A table cut in Collect: the part that trains is clear, the rest shaded. */
  const c = r?.picCrop;
  const shade = c ? `<i class="cell-crop" style="left:${c[0] * 100}%;top:${c[1] * 100}%;width:${c[2] * 100}%;height:${c[3] * 100}%" aria-hidden="true"></i>` : '';
  const alt = i.click ? 'The window, just before the click' : 'The picture';
  const inner = `${url
    ? `<img src="${url}" alt="${alt}">` : `<span class="cell-wait" data-alt="${alt}" aria-label="Loading the picture"></span>`}${shade}${dot}`;
  return view && row.id
    ? `<button type="button" class="${cls} art-open" data-do="view" data-row="${row.id}" aria-label="Open the picture larger">${inner}</button>`
    : `<span class="${cls}">${inner}</span>`;
}

export function chip (r, a, { tone = '' } = {}) {
  if (a == null) return '<span class="fn empty">no answer</span>';
  return `<span class="fn${tone ? ` ${tone}` : ''}">${esc(labelOf(r, a))}</span>`;
}

/* The toast: one line, bottom left, with Undo when the change can be undone.
   It is a live region, so a screen reader hears it. */
let toastTimer;
/* A second undoable change inside the six seconds joins the first: the toast
   counts them and Undo undoes them all, newest first. */
export function toast (text, undo = null, { sticky = false } = {}) {
  const prev = S.ui.toast;
  const undos = undo && prev?.undos?.length && Date.now() - prev.at < 6000 ? [...prev.undos, undo] : (undo ? [undo] : []);
  const shown = undos.length > 1 ? `${text} (${undos.length} changes)` : text;
  S.ui.toast = { text: shown, undo: undos.length ? () => { for (const u of [...undos].reverse()) u(); } : null, undos, sticky, at: Date.now() };
  clearTimeout(toastTimer);
  if (!sticky) toastTimer = setTimeout(() => { S.ui.toast = null; window.dispatchEvent(new Event('proto:render')); }, 6000);
}
/* Something went wrong: the words stay until dismissed, and can be selected
   and copied — an error that vanishes in six seconds is an error nobody can
   report. */
export const toastError = (text) => toast(text, null, { sticky: true });
export function toastHtml () {
  const t = S.ui.toast;
  return `<div class="toast-slot" role="${t?.sticky ? 'alert' : 'status'}" aria-live="${t?.sticky ? 'assertive' : 'polite'}">${t
    ? `<div class="toast${t.sticky ? ' sticky' : ''}"><span>${esc(t.text)}</span>${t.undo ? '<button type="button" class="toast-undo" data-do="undo">Undo</button>' : ''}${t.sticky ? '<button type="button" class="toast-undo" data-do="toast-dismiss">Dismiss</button>' : ''}</div>`
    : ''}</div>`;
}

/* A dialog says it is one, traps nothing it does not need to, and Escape
   closes it (app.js). */
/* `ask`: a question with its answers — Pigeon's small sheet, narrow and with
   no ×, because Cancel is one of the answers. Escape and the scrim still close it. */
export function dialogHtml (title, body, actions, { wide = false, ask = false } = {}) {
  return `<div class="scrim" data-do="dialog-close"></div>
  <div class="dialog${wide ? ' wide' : ''}${ask ? ' ask' : ''}" role="${ask ? 'alertdialog' : 'dialog'}" aria-modal="true" aria-labelledby="dlg-title">
    <div class="dialog-head">
      <h2 id="dlg-title" tabindex="-1">${esc(title)}</h2>
      ${ask ? '' : `<button type="button" class="icon-btn" data-do="dialog-close" aria-label="Close">${ICON.close}</button>`}
    </div>
    <div class="dialog-body">${body}</div>
    <div class="dialog-actions">${actions}</div>
  </div>`;
}

/* ── the picture viewer ─────────────────────────────────────────────────
   Any picture on the screen opens here at its own size, with the pictures
   that were on the screen beside it one arrow key away. */

export function openViewer (el) {
  const ids = [...new Set([...document.querySelectorAll('[data-do="view"][data-row]')].map(b => b.dataset.row))];
  S.ui.dialog = { kind: 'viewer', ids, at: Math.max(0, ids.indexOf(el.dataset.row)) };
}

export function stepViewer (by) {
  const d = S.ui.dialog;
  if (d?.kind !== 'viewer') return false;
  d.at = Math.min(d.ids.length - 1, Math.max(0, d.at + by));
  return true;
}

/* In the viewer the keys do the work, so a play's drafts or a whole answer's
   pictures can be checked without the mouse: 1–9 give the answer (and
   approve a draft), A approves a draft as it is, D discards a draft, and
   each of those moves on to the next picture. */
export function viewerHtml (r, d) {
  d.ids = d.ids.filter(id => r.rows.some(x => x.id === id));
  if (!d.ids.length) return '';
  d.at = Math.min(d.at, d.ids.length - 1);
  const row = r.rows.find(x => x.id === d.ids[d.at]);
  const answers = r.output === 'prose' ? [] : answerList(r);
  const badges = [
    row.holdout ? '<span class="badge exam">exam</span>' : '',
    row.draft ? '<span class="badge draft">draft</span>' : '',
    row.left ? `<span class="badge left">left out${row.left === 'repeat' ? ' — repeat' : row.left === 'conflict' ? ' — conflict' : ''}</span>` : '',
    row.session ? `<span class="session">${esc(row.session)}</span>` : ''
  ].join('');
  const pick = answers.length ? `<label class="field-label small" for="viewer-answer">Answer</label>
    <select id="viewer-answer" class="select viewer-select" data-change="view-answer" data-row="${row.id}">
      ${answers.map((a, i) => `<option value="${esc(a)}" ${a === row.answer ? 'selected' : ''}>${i < 9 ? `${i + 1} · ` : ''}${esc(labelOf(r, a))}</option>`).join('')}
    </select>` : chip(r, row.answer);
  const act = row.draft
    ? `<button type="button" class="btn sm ghost" data-do="view-discard" data-row="${row.id}">Discard <kbd>D</kbd></button>
       <button type="button" class="btn sm primary" data-do="view-approve" data-row="${row.id}">Approve <kbd>A</kbd></button>`
    : row.left
      ? `<button type="button" class="btn sm ghost" data-do="view-back" data-row="${row.id}">Put back</button>`
      : `<button type="button" class="btn sm ghost" data-do="view-leave" data-row="${row.id}">Leave out</button>`;
  return dialogHtml(`Picture ${d.at + 1} of ${d.ids.length}`, `
    <div class="viewer-art">${inputCell(row, { r, view: false, full: true, size: 'full' })}</div>
    <div class="viewer-meta">${pick}${badges}</div>`,
    `<p class="hint viewer-keys">← → move${answers.length ? ` · 1–${Math.min(9, answers.length)} set the answer${row.draft ? ' and approve' : ''}` : ''}${row.draft ? ' · A approve · D discard' : ''}</p>${act}
    <button type="button" class="btn sm" data-do="view-step" data-by="-1" ${d.at === 0 ? 'disabled' : ''}>Previous</button>
    <button type="button" class="btn sm" data-do="view-step" data-by="1" ${d.at >= d.ids.length - 1 ? 'disabled' : ''}>Next</button>`, { wide: true });
}

/* What the viewer's keys and buttons do to the row on screen. Each returns a
   toast line, and moves on when the row is settled. */
export function viewerAct (r, what, value) {
  const d = S.ui.dialog;
  if (d?.kind !== 'viewer') return;
  const id = d.ids[d.at];
  const row = r.rows.find(x => x.id === id);
  if (!row) return;
  const next = () => { if (d.at < d.ids.length - 1) d.at++; };
  if (what === 'answer') {
    const was = { answer: row.answer, draft: row.draft };
    updateRow(r, id, { answer: value, draft: false });
    toast(`${was.draft ? 'Approved as' : 'Now'} ${labelOf(r, value)}.`, () => updateRow(r, id, was));
    next();
  } else if (what === 'approve' && row.draft) {
    updateRow(r, id, { draft: false });
    toast('Approved.', () => updateRow(r, id, { draft: true }));
    next();
  } else if (what === 'discard' && row.draft) {
    const at = r.rows.indexOf(row);
    r.rows = r.rows.filter(x => x !== row);
    r.version++;
    toast('Draft discarded.', () => { r.rows.splice(at, 0, row); d.ids.splice(d.at, 0, id); });
    d.ids = d.ids.filter(x => x !== id);
  } else if (what === 'leave') {
    leaveOut(r, id);
    toast('Row left out. It is still under “Left out”.', () => putBack(r, id));
  } else if (what === 'back') {
    putBack(r, id);
    toast('Row put back.');
  }
  if (!d.ids.length) S.ui.dialog = null;
}

/* The viewer's keys; true when one was used. */
export function viewerKey (r, e) {
  const d = S.ui.dialog;
  if (d?.kind !== 'viewer' || !r || e.metaKey || e.ctrlKey || e.altKey) return false;
  const row = r.rows.find(x => x.id === d.ids[d.at]);
  if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') { stepViewer(e.key === 'ArrowLeft' ? -1 : 1); return true; }
  if (/^[1-9]$/.test(e.key) && r.output !== 'prose') {
    const a = answerList(r)[Number(e.key) - 1];
    if (a == null) return false;
    viewerAct(r, 'answer', a); return true;
  }
  if (row?.draft && (e.key === 'a' || e.key === 'A')) { viewerAct(r, 'approve'); return true; }
  if (row?.draft && (e.key === 'd' || e.key === 'D')) { viewerAct(r, 'discard'); return true; }
  return false;
}
