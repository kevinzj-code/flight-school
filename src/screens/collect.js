/* collect.js — CARRIES FORWARD. Stage 2: the table.
 *
 * Flow-review changes folded in:
 *   - Every way in is on one bar, and the one that suits the job leads:
 *     Record for pictures, Pair files for prose from documents.
 *   - The exam is visible here, marked on its rows. Flight School already
 *     shows you the exam before anything trains; the first plan lost that.
 *   - Leaving a row out is undoable, and a left-out row is still one filter
 *     away — nothing vanishes.
 *   - Under the floor, the table offers a prompt instead of pretending a
 *     training run could work. */

import {
  S, floorFor, tableRows, exam, drafts, mismatched, needsThink, answerList, labelOf,
  addRow, addRows, updateRow, leaveOut, putBack, trainingSet, setPicCrop, approveRows, badCites, staleRows, bareRows, annotateReaderRow, recorded
} from '../state.js';
import { minutesFor, engine } from '../engine.js';
import { openRecordAgain } from './teach.js';
import { esc, ICON, BUSY, n } from './html.js';
import { inputCell, chip, tag, toast, toastError, dialogHtml, openViewer, stepViewer, viewerAct } from './cells.js';

/* Work that finishes later redraws when it lands, and saves what it changed. */
const later = (r) => { engine.persist(r); window.dispatchEvent(new Event('proto:render')); };

const PAGE = 40;

function filters (r) {
  const all = tableRows(r);
  const f = [['all', `All rows`, all.length]];
  if (r.output !== 'prose') {
    const counts = new Map();
    for (const x of all) counts.set(x.answer, (counts.get(x.answer) || 0) + 1);
    for (const a of answerList(r)) if (counts.get(a)) f.push([`a:${a}`, labelOf(r, a), counts.get(a)]);
  }
  const ex = exam(r).length; if (ex) f.push(['exam', 'The exam', ex]);
  const left = r.rows.filter(x => x.left).length; if (left) f.push(['left', 'Left out', left]);
  const d = drafts(r).length; if (d) f.push(['drafts', 'Drafts to check', d]);
  const t = needsThink(r).length; if (t) f.push(['needs-think', 'Needs reasoning', t]);
  const m = mismatched(r).length; if (m) f.push(['mixed', 'Wrong kind', m]);
  const bare = bareRows(r).length; if (bare) f.push(['bare', 'Citation only', bare]);
  const bc = badCites(r).length; if (bc) f.push(['bad-cite', 'Cites the unseen', bc]);
  const st = staleRows(r).length; if (st) f.push(['stale', 'Older document', st]);
  return f;
}

/* Every session in the table — a recording, a folder, a play — newest last. */
const sessionsOf = (r) => [...new Set(r.rows.map(x => x.session).filter(Boolean))];

function filtered (r) {
  let list = byFilter(r);
  if (r.kind === 'reader' && S.ui.filter === 'drafts') list = [...list].sort((a, b) => (a.section ?? 1e9) - (b.section ?? 1e9) || Number(a.refusal) - Number(b.refusal));
  return S.ui.session ? list.filter(x => x.session === S.ui.session) : list;
}

/* A Reader's drafts, grouped by the section they are about, each group with
   one button: a hundred drafts is a dozen clicks, not a hundred. */
function draftGroups (r, rows) {
  if (r.kind !== 'reader' || S.ui.filter !== 'drafts') return rows.map(x => rowHtml(r, x)).join('');
  const out = [];
  let last;
  for (const x of rows) {
    const k = x.section ?? null;
    if (k !== last) {
      const all = drafts(r).filter(y => (y.section ?? null) === k);
      const title = k == null ? 'No section' : `§${k} ${r._chunks?.find(c => c.ordinal === k)?.title || ''}`;
      out.push(`<div class="t-group"><span><b>${esc(title)}</b> · ${n(all.length)} ${all.length === 1 ? 'draft' : 'drafts'}${all.some(y => y.refusal) ? ` · ${n(all.filter(y => y.refusal).length)} ${all.filter(y => y.refusal).length === 1 ? 'refusal' : 'refusals'}` : ''}</span>
        <button type="button" class="btn sm" data-do="approve-section" data-section="${k == null ? '' : k}">Approve all ${n(all.length)}</button></div>`);
      last = k;
    }
    out.push(rowHtml(r, x));
  }
  return out.join('');
}

function byFilter (r) {
  const f = S.ui.filter;
  if (f === 'left') return r.rows.filter(x => x.left);
  if (f === 'drafts') return drafts(r);
  if (f === 'exam') return exam(r);
  if (f === 'needs-think') return needsThink(r);
  if (f === 'mixed') return mismatched(r);
  if (f === 'bad-cite') return badCites(r);
  if (f === 'bare') return bareRows(r);
  if (f === 'stale') return staleRows(r);
  if (f.startsWith('a:')) return tableRows(r).filter(x => x.answer === f.slice(2));
  return [...drafts(r), ...tableRows(r)];
}

function answerEditor (r, row, id) {
  if (r.output === 'prose') {
    return `<textarea id="${id}" class="text-input" rows="4" data-input="edit-answer" data-row="${row.id}">${esc(row.answer)}</textarea>`;
  }
  return `<select id="${id}" class="select" data-change="edit-answer" data-row="${row.id}">
    ${answerList(r).map(a => `<option value="${esc(a)}" ${a === row.answer ? 'selected' : ''}>${esc(labelOf(r, a))}</option>`).join('')}
  </select>`;
}

function rowHtml (r, row) {
  const editing = S.ui.editing === row.id;
  const wrongKind = row.input.kind !== r.input;
  const badges = [
    row.holdout ? '<span class="badge exam" title="Set aside for the check ride. It never trains.">exam</span>' : '',
    row.draft ? '<span class="badge draft">draft</span>' : '',
    row.fresh ? '<span class="badge fresh">new</span>' : '',
    row.left ? `<span class="badge left">left out${row.left === 'conflict' ? ' — conflict' : row.left === 'split' ? ' — narrower job' : row.left === 'repeat' ? ' — repeat' : ''}</span>` : '',
    wrongKind ? `<span class="badge bad">${r.input === 'image' ? 'text' : 'picture'} in a table of ${r.input === 'image' ? 'pictures' : 'text'}</span>` : '',
    row.session ? `<span class="session">${esc(row.session)}</span>` : ''
  ].join('');

  let out;
  if (editing) {
    out = `${r.reasoning ? `<label class="field-label small" for="think-${row.id}">Why this answer?</label>
      <textarea id="think-${row.id}" class="text-input think-input" rows="2" data-input="edit-think" data-row="${row.id}" placeholder="What in the input decides it?">${esc(row.think || '')}</textarea>` : ''}
      <label class="field-label small" for="ans-${row.id}">Answer</label>${answerEditor(r, row, `ans-${row.id}`)}`;
  } else if (r.output === 'prose') {
    out = `<p class="cell-text prose">${esc(row.answer)}</p>`;
  } else {
    out = `${r.reasoning ? (row.think ? `<p class="think">&lt;think&gt; ${esc(row.think)} &lt;/think&gt;</p>` : (row.holdout ? '' : '<p class="think missing">No reasoning yet</p>')) : ''}${chip(r, row.answer)}`;
  }

  const acts = row.draft
    ? `<button type="button" class="btn sm" data-do="approve" data-row="${row.id}">Approve</button><button type="button" class="btn sm ghost" data-do="edit" data-row="${row.id}">Fix</button><button type="button" class="btn sm ghost" data-do="discard" data-row="${row.id}">Discard</button>`
    : editing
      ? `<button type="button" class="btn sm primary" data-do="edit-done">Done</button>`
      : row.left
        ? `<button type="button" class="btn sm ghost" data-do="put-back" data-row="${row.id}">Put back</button>`
        : `<button type="button" class="btn sm ghost" data-do="edit" data-row="${row.id}" aria-label="Edit this row">Edit</button><button type="button" class="btn sm ghost" data-do="leave-out" data-row="${row.id}" aria-label="Leave this row out">Leave out</button>`;

  return `<div class="t-row${row.left ? ' is-left' : ''}${row.draft ? ' is-draft' : ''}${wrongKind ? ' is-bad' : ''}${editing ? ' is-editing' : ''}">
    <div class="t-in">${inputCell(row)}</div>
    <div class="t-out">${out}<div class="t-badges">${badges}</div></div>
    <div class="t-act">${acts}</div>
  </div>`;
}

/* A picture in the grid: the picture, its answer under it, and a mark for
   the exam, a draft or a row left out. Opening it opens the viewer on the
   pictures in the grid. */
function gridCell (r, row) {
  const mark = row.holdout ? '<span class="g-mark exam" title="On the exam">exam</span>'
    : row.draft ? '<span class="g-mark draft" title="A draft, waiting to be approved">draft</span>'
    : row.left ? '<span class="g-mark left" title="Left out">out</span>' : '';
  return `<div class="g-cell${row.left ? ' is-left' : ''}">${inputCell(row)}${mark}<span class="g-ans">${r.output === 'prose' ? '' : esc(labelOf(r, row.answer))}</span></div>`;
}

/* One session picked: its rows can leave the table together, or come back
   together — a fumbled recording or the wrong folder is one click, not four
   hundred. */
function sessionBar (r) {
  if (!S.ui.session) return '';
  const rows = r.rows.filter(x => x.session === S.ui.session);
  if (!rows.length) return '';
  const inTable = rows.filter(x => !x.left && !x.draft).length;
  const out = rows.filter(x => x.left).length;
  return `<div class="session-bar" role="status"><span><b>${esc(S.ui.session)}</b> · ${n(rows.length)} rows${out ? `, ${n(out)} left out` : ''}</span>
    ${inTable ? `<button type="button" class="btn sm" data-do="session-leave">Leave this session out</button>` : ''}
    ${out ? `<button type="button" class="btn sm ghost" data-do="session-back">Put it all back</button>` : ''}</div>`;
}

function addForm (r) {
  if (!S.ui.adding) return '';
  const answers = answerList(r);
  const text = r.input === 'text';
  return `<form class="add-row" data-submit="add-row">
    <div class="add-in">
      ${text
        ? `<label class="field-label small" for="add-input">What goes in</label><textarea id="add-input" class="text-input" rows="3" required placeholder="Paste the input…"></textarea>`
        : `<p class="field-label small">What goes in</p><p class="hint">Pictures come in by recording or from a folder — Add a row is for text.</p>`}
    </div>
    <div class="add-out">
      <label class="field-label small" for="add-answer">What should come out</label>
      ${r.output === 'prose'
        ? '<textarea id="add-answer" class="text-input" rows="3" required placeholder="The answer you would write…"></textarea>'
        : answers.length
          ? `<select id="add-answer" class="select">${answers.map(a => `<option value="${esc(a)}">${esc(labelOf(r, a))}</option>`).join('')}</select>`
          : '<input id="add-answer" class="text-input mono-input" required placeholder="e.g. TAG(result)">'}
    </div>
    <div class="add-act"><button type="submit" class="btn primary sm">Add the row</button><button type="button" class="btn sm ghost" data-do="add-cancel">Cancel</button></div>
  </form>`;
}

export function render (r) {
  const rows = tableRows(r);
  const list = filtered(r);
  const grid = r.input === 'image' && S.ui.layout === 'grid';
  const shown = list.slice(0, (grid ? PAGE * 4 : PAGE) * S.ui.page);
  const floor = floorFor(r);
  const under = rows.length < floor;
  const ts = trainingSet(r);
  const image = r.input === 'image';
  const fs = filters(r);
  /* The way in that suits the kind leads; the others follow. A text kind has
     no Record, because a recording makes picture rows. Something else keeps
     the whole bar, as it always has. */
  const lead = { player: 'record', looker: 'import', sorter: 'import', writer: 'pair' }[r.kind] || (r.kind ? null : image ? 'record' : r.output === 'prose' && r.grounded ? 'pair' : null);
  const primary = (id) => (lead === id ? 'primary' : '');
  const textKind = ['sorter', 'writer', 'reader'].includes(r.kind);
  const job = S.ui.draftJob;
  const ways = [
    ...(r.kind === 'reader' && r.material?.sections && !job ? [`<button type="button" class="btn primary" data-do="draft-material-open">${ICON.spark}${r.answerStyle === 'partner' ? 'Read it and draft conversations' : 'Read it and draft questions'}</button>`] : []),
    ...(textKind ? [] : [`<button type="button" class="btn ${primary('record')}" data-do="record-open">${ICON.rec}Record</button>`]),
    `<button type="button" class="btn ${primary('pair')}" data-do="pair-open" ${S.ui.importing ? 'aria-busy="true" disabled' : ''}>${ICON.pair}Pair files</button>`,
    `<button type="button" class="btn ${primary('import')}" data-do="import-open" ${S.ui.importing ? 'aria-busy="true" disabled' : ''}>${ICON.folder}Import a folder</button>`,
    ...(image && rows.length ? [recorded(r)
      ? `<button type="button" class="btn" data-do="pic-crop-open" title="Record a new table that keeps only part of the window">${ICON.image}Record again with only part of the window…</button>`
      : `<button type="button" class="btn${r.picCrop ? ' on' : ''}" data-do="pic-crop-open" title="Cut every picture to the part that matters">${ICON.image}${r.picCrop ? 'Cropped' : 'Crop pictures'}…</button>`] : []),
    ...(image ? [`<button type="button" class="btn" data-do="play-pick" ${S.ui.importing ? 'aria-busy="true" disabled' : ''} title="The moments a Pigeon played, from Pigeon's play log — to approve or correct">${ICON.rec}Import a play</button>`] : []),
    `<button type="button" class="btn" data-do="add-open">${ICON.plus}Add a row</button>`,
    image
      ? `<button type="button" class="btn" aria-disabled="true" data-do="refuse" data-why="A model can only draft text rows. For pictures, record or import a folder.">${ICON.spark}Let a model draft</button>`
      : r.kind === 'reader' ? '' : `<button type="button" class="btn" data-do="draft" ${S.ui.drafting ? 'aria-busy="true" disabled' : ''}>${ICON.spark}${S.ui.drafting ? 'Drafting…' : 'Let a model draft'}</button>`
  ].join('');
  const progress = job ? `<div class="draft-progress" role="status">${BUSY}<span><b>${job.phase === 'questions' ? `Reading section ${n(Math.min(job.done + 1, job.total))} of ${n(job.total)}` : `Writing refusals, ${n(job.done)} of ${n(job.total)}`}</b> · ${n(job.added)} ${job.added === 1 ? 'draft' : 'drafts'} so far${job.failed ? ` · ${n(job.failed)} ${job.failed === 1 ? 'reply' : 'replies'} unusable` : ''}${job.rejected ? ` · ${n(job.rejected)} ${job.rejected === 1 ? 'draft' : 'drafts'} refused` : ''}${job.stop ? ' · stopping after this one' : ''}</span>
    <span class="bar-track"><span class="bar" style="width:${job.total ? Math.round(100 * job.done / job.total) : 0}%"></span></span>
    <button type="button" class="btn sm ghost" data-do="draft-stop" ${job.stop ? 'disabled' : ''}>Stop</button></div>` : '';
  const dr = drafts(r).length;
  const approveAll = r.kind === 'reader' && dr && S.ui.filter === 'drafts' && !job
    ? `<div class="session-bar" role="status"><span><b>${n(dr)} ${dr === 1 ? 'draft' : 'drafts'}</b> waiting. Read a few from each section; approve the section when they hold.</span><button type="button" class="btn sm" data-do="approve-all">Approve all ${n(dr)}</button></div>` : '';

  return `<div class="collect">
    ${materialPanel(r)}
    <div class="ways" role="toolbar" aria-label="Ways to add rows">${ways}</div>
    ${progress}
    ${addForm(r)}

    ${under ? `<div class="floor">
        <div><b>${n(rows.length)} ${rows.length === 1 ? 'row' : 'rows'} is not a training set yet.</b>
        <p>${r.kind === 'reader'
          ? `A Reader needs at least ${floor} approved rows — they are drafted from the document, so the cost is checking, not writing.`
          : `A training run needs at least ${floor}; a narrow job does well from about 150. Until then, a prompt with your examples in it will do more than an adapter trained on too little.`}</p></div>
        ${r.kind === 'reader' ? '' : '<button type="button" class="btn" data-do="prompt-open">Use a prompt instead</button>'}
      </div>` : ''}

    ${rows.length || drafts(r).length || r.rows.some(x => x.left) ? `
    <div class="filter-line">
      <div class="filterbar" role="tablist" aria-label="Show">
        ${fs.map(([id, label, count]) => `<button type="button" role="tab" id="flt-${esc(id)}" class="flt" aria-selected="${S.ui.filter === id}" data-do="filter" data-filter="${esc(id)}">${esc(label)} <span>${n(count)}</span></button>`).join('')}
      </div>
      ${sessionsOf(r).length > 1 ? `<label class="session-pick" for="session-filter"><span class="field-label small">Session</span>
        <select id="session-filter" class="select" data-change="session-filter">
          <option value="">All sessions</option>
          ${sessionsOf(r).map(x => `<option value="${esc(x)}" ${S.ui.session === x ? 'selected' : ''}>${esc(x)} (${n(r.rows.filter(y => y.session === x).length)})</option>`).join('')}
        </select></label>` : ''}
      ${image ? `<div class="pic-size seg" role="group" aria-label="Layout">
        ${[['list', 'List'], ['grid', 'Grid']].map(([k, label]) => `<button type="button" class="seg-btn" data-do="layout" data-layout="${k}" aria-pressed="${(S.ui.layout || 'list') === k}">${label}</button>`).join('')}
      </div>
      <div class="pic-size" role="group" aria-label="Picture size">
        <span class="field-label small">Pictures</span>
        <span class="seg">${[['s', 'Small'], ['m', 'Medium'], ['l', 'Large']].map(([k, label]) => `<button type="button" class="seg-btn" data-do="pic-size" data-size="${k}" aria-pressed="${S.ui.picSize === k}">${label}</button>`).join('')}</span>
      </div>` : ''}
    </div>
    ${sessionBar(r)}${approveAll}
    ${grid
      ? `<div class="pic-grid pics-${S.ui.picSize}" aria-label="The table, as pictures">${shown.map(x => gridCell(r, x)).join('') || '<p class="t-empty">Nothing here.</p>'}</div>`
      : `<div class="table pics-${image ? S.ui.picSize : 's'}" role="table" aria-label="The table">
      <div class="t-head" role="row"><span>What goes in</span><span>What comes out</span><span></span></div>
      ${draftGroups(r, shown) || '<p class="t-empty">Nothing here.</p>'}
    </div>`}
    ${list.length > shown.length ? `<button type="button" class="btn more" data-do="more">Show ${Math.min(grid ? PAGE * 4 : PAGE, list.length - shown.length)} more of ${n(list.length - shown.length)}</button>` : ''}
    ` : `<div class="empty-table">
      <h3>The table is empty</h3>
      <p>${emptyWords(r)}</p>
    </div>`}

    <div class="tiles">
      <div class="tile"><b>${n(rows.length)}</b><span>rows in the table</span></div>
      ${r.output !== 'prose' ? `<div class="tile"><b>${answerList(r).length}</b><span>answers it may give</span></div>` : ''}
      <div class="tile"><b>${n(exam(r).length)}</b><span>set aside as the exam</span></div>
      <div class="tile"><b>${n(ts.rows.length)}</b><span>will train, after Review’s defaults</span></div>
      ${r.input ? `<div class="tile"><b>~${minutesFor(r)} min</b><span>to teach ${tag('measured')}</span></div>` : ''}
    </div>

    <div class="stage-foot">
      <p class="hint">${under ? `Review opens at ${floor} rows.` : 'Every change saves as you make it.'}</p>
      <button type="button" class="btn primary" data-do="go" data-stage="review">Review the table${ICON.arrow}</button>
    </div>
  </div>`;
}

/* ── a Reader's document ──────────────────────────────────────────────
   Above the table, because the rows come from it. Files in, sections out
   (material.rs); the drop zone takes a drop from the Finder (app.js) or
   opens the file dialog. */
function materialPanel (r) {
  if (r.kind !== 'reader') return '';
  const m = r.material;
  const busy = S.ui.importing;
  const files = m?.files || [];
  return `<section class="material" aria-label="The document">
    <div class="material-head"><h3>The document</h3>
      <p class="hint">What the Reader answers from. It travels inside the .pigeon, section by section, and the best-matching section is pasted in with every question.</p></div>
    ${files.length ? `<ul class="material-files">${files.map(f => `<li>
        <span class="material-file">${ICON.text}<b>${esc(f.file)}</b></span>
        <span class="hint">${n(f.sections)} ${f.sections === 1 ? 'section' : 'sections'} · ~${n(f.est_tokens)} tokens${f.pages ? ` · ${n(f.pages)} ${f.pages === 1 ? 'page' : 'pages'}` : ''}${f.ocr_pages ? ` · ${n(f.ocr_pages)} read by text recognition` : ''}</span>
        <button type="button" class="icon-btn" data-do="material-remove" data-file="${esc(f.file)}" aria-label="Take ${esc(f.file)} out" title="Take this file out of the material" ${busy ? 'disabled' : ''}>${ICON.close}</button>
      </li>`).join('')}</ul>
      <p class="material-total"><b>${n(m.sections)} sections · ~${n(m.estTokens)} tokens.</b> At ask time the section that best matches the question is pasted in above it — the shape every row trains in.</p>` : ''}
    <button type="button" class="dropzone${S.ui.dropping ? ' dropping' : ''}" data-do="material-pick" ${busy ? 'aria-busy="true" disabled' : ''}>
      ${busy ? BUSY : ICON.folder}<b>${busy ? 'Reading…' : files.length ? 'Drop another file here, or choose one…' : 'Drop the document here, or choose it…'}</b>
      <span class="hint">.md, .txt, .html or .pdf, or a folder of them. A scanned PDF is read by macOS’s own text recognition.</span>
    </button>
  </section>`;
}

/* Files into the material: from the dialog (paths null) or a drop. One
   toast says what came in and names anything that could not be read. */
export function addMaterial (r, paths = null) {
  if (S.ui.importing) return false;
  S.ui.importing = true; later(r);
  engine.material.add(r, paths).then(got => {
    if (!got) return;
    const bad = got.problems || [];
    const line = `${n(got.sections)} sections from ${got.files.length} ${got.files.length === 1 ? 'file' : 'files'}.`;
    if (bad.length) toastError(`${line} Could not read: ${bad.join(' ')}`); else toast(line);
  }).catch(e => toastError(String(e))).finally(() => { S.ui.importing = false; later(r); });
  return false;
}

/* What an empty table says, by kind: the one way in that suits it. */
function emptyWords (r) {
  switch (r.kind) {
    case 'player': return 'Press <b>Record</b> and use the app or game you want it to learn. Every key you press and every click is kept, with the screen from just before it.';
    case 'looker': return 'Press <b>Import a folder</b> and choose a folder with one sub-folder per label. The folder names become the labels, and every picture inside becomes a row.';
    case 'sorter': return 'Press <b>Import a folder</b> with one sub-folder per label, add rows by hand, or let a model draft some from your labels for you to check.';
    case 'writer': return 'Press <b>Pair files</b> and choose two folders matched by file name: what went in, and what you wrote from it. Or add pairs by hand, or let a model draft some.';
    case 'reader': return r.material?.sections
      ? `The document is in, with ${n(r.material.sections)} sections. Press <b>Read it and draft questions</b>: a model reads each section and drafts questions with their answers, and refusals for the questions a section does not answer, for you to check.`
      : 'Drop the document in above. Every row is drafted from it: a question, the section it is asked against, and the answer to check.';
    default: return r.input === 'image'
      ? 'Press <b>Record</b> and use the app you want it to learn. Every key you press and every click is kept, with the screen from just before it.'
      : 'Add rows by hand, pair a folder of inputs with a folder of the answers you wrote, or let a model draft some for you to check.';
  }
}

/* ── importing a folder ────────────────────────────────────────────────
   Each answer a folder brings, and what it becomes in this table: an answer
   the table already has (the same words — a recorded KEY(Up) included), one
   that differs from an existing answer only in capitals, spacing or
   punctuation (offered to be joined to it, and joined unless you say no), or
   a new one. Two new answers that differ the same way are offered as one. */
const loose = (s) => String(s).toLowerCase().replace(/[^a-z0-9]+/g, '');

export function importPlan (r, rows) {
  const counts = new Map();
  for (const x of rows) counts.set(x.answer, (counts.get(x.answer) || 0) + 1);
  const labels = [...counts.keys()].sort((a, b) => counts.get(b) - counts.get(a));
  const plan = [];
  for (const label of labels) {
    const same = answerList(r).find(a => labelOf(r, a) === label);
    if (same != null) { plan.push({ label, count: counts.get(label), to: same, known: true }); continue; }
    const near = answerList(r).find(a => loose(labelOf(r, a)) === loose(label))
      ?? plan.find(p => !p.known && !p.near && loose(p.label) === loose(label))?.label;
    plan.push({ label, count: counts.get(label), to: label, known: false, near: near ?? null, join: near != null });
  }
  return plan;
}

/* ── dialogs ──────────────────────────────────────────────────────────── */

const pct = (v) => `${(v * 100).toFixed(2)}%`;

export function dialog (r, d) {
  if (d.kind === 'pic-crop') {
    const sample = tableRows(r).find(x => x.input.path);
    const url = sample ? engine.image(r, sample.input.path, { full: true }) : null;
    const c = d.crop;
    const picker = url ? `<div class="crop-pick" data-drag="pic-crop" role="img" aria-label="${c ? `A picture from the table, with ${Math.round(c[2] * 100)}% by ${Math.round(c[3] * 100)}% kept` : 'A picture from the table, all of it kept'}">
        <img src="${url}" alt="" draggable="false">
        ${c ? `<div class="crop-box" style="left:${pct(c[0])};top:${pct(c[1])};width:${pct(c[2])};height:${pct(c[3])}"></div>` : ''}
      </div>` : '<p class="hint">Loading a picture…</p>';
    const keeping = `<p class="hint" data-live="pic-crop-size">${c ? `Keeping ${Math.round(c[2] * 100)}% of the width and ${Math.round(c[3] * 100)}% of the height.` : 'Keeping all of it.'}</p>`;
    /* Recorded pictures train as recorded: a part of them is a new recording. */
    if (recorded(r)) {
      return dialogHtml('Record again with only part of the window', `
      <p>These pictures were recorded at ${r.recordSide || 768} px. Cutting them now would train your Pigeon on pictures it will never see when it plays. Record a new table with only this part of the window instead.</p>
      ${picker}${keeping}
      <p class="hint">The new table starts empty, with this table's answers and question, and opens on Record with this part chosen. This table stays as it is.</p>`,
      `<button type="button" class="btn ghost" data-do="dialog-close">Cancel</button>
       <button type="button" class="btn primary" data-do="record-part" ${c ? '' : 'disabled'}>Start a new table</button>`);
    }
    return dialogHtml('Crop every picture', `
      <p>Drag over the part of the picture that matters. Everything outside it — a sidebar, a toolbar, a score — is cut away from every row and the exam when it trains${r.sessionsIn.length ? ', and Pigeon cuts the same part away when it plays' : ''}. The pictures in the table stay whole; the part cut away is shaded.</p>
      ${url ? `<div class="crop-pick" data-drag="pic-crop" role="img" aria-label="${c ? `A picture from the table, with ${Math.round(c[2] * 100)}% by ${Math.round(c[3] * 100)}% kept` : 'A picture from the table, all of it kept'}">
        <img src="${url}" alt="" draggable="false">
        ${c ? `<div class="crop-box" style="left:${pct(c[0])};top:${pct(c[1])};width:${pct(c[2])};height:${pct(c[3])}"></div>` : ''}
      </div>` : '<p class="hint">Loading a picture…</p>'}
      <p class="hint" data-live="pic-crop-size">${c ? `Keeping ${Math.round(c[2] * 100)}% of the width and ${Math.round(c[3] * 100)}% of the height.` : 'Keeping all of it.'}</p>
      ${r.teach ? '<p class="hint">It has been taught already: cropping makes that Pigeon out of date, and the next round trains on the cut pictures.</p>' : ''}`,
      `<button type="button" class="btn ghost" data-do="dialog-close">Cancel</button>
       ${r.picCrop ? '<button type="button" class="btn" data-do="pic-crop-clear">Keep whole pictures</button>' : ''}
       <button type="button" class="btn primary" data-do="pic-crop-apply" ${c ? '' : 'disabled'}>Crop every picture</button>`);
  }
  if (d.kind === 'import-preview') {
    const { got, plan } = d;
    const wrong = r.input ? got.rows.filter(x => x.input.kind !== r.input).length : 0;
    const line = (p, i) => `<li class="imp-row"><span class="fn">${esc(p.label)}</span><span class="imp-n">${n(p.count)}</span>
      <span class="imp-to">${p.known ? '<span class="hint">already an answer</span>'
        : p.near ? `<label class="check" for="join-${i}"><input type="checkbox" id="join-${i}" ${p.join ? 'checked' : ''} data-change="import-join" data-label="${esc(p.label)}"> Join to <span class="fn">${esc(labelOf(r, p.near))}</span></label>`
        : '<span class="badge fresh">new answer</span>'}</span></li>`;
    const fresh = plan.filter(p => !p.known && !(p.near && p.join)).length;
    return dialogHtml(`Add ${n(got.rows.length)} rows from ${got.folder || 'the folder'}?`, `
      <p>What each answer in the folder becomes in this table.${fresh ? ` <b>${fresh} ${fresh === 1 ? 'is new' : 'are new'}</b> — check ${fresh === 1 ? 'it is' : 'they are'} meant to be, not a typo.` : ''}</p>
      <ul class="imp-list">${plan.map(line).join('')}</ul>
      ${plan.some(p => p.near) ? '<p class="hint">An answer that differs from another only in capitals, spacing or punctuation is joined to it unless you untick it — otherwise the table would teach two answers that mean one.</p>' : ''}
      ${wrong ? `<p class="warn-line">${ICON.warn}${n(wrong)} of these are ${r.input === 'image' ? 'text' : 'pictures'}, in a table of ${r.input === 'image' ? 'pictures' : 'text'}. They will be marked, and must be left out before it can teach.</p>` : ''}
      ${got.problems.length ? `<details><summary>${n(got.problems.length)} ${got.problems.length === 1 ? 'file' : 'files'} could not be read</summary><ul class="plain">${got.problems.slice(0, 8).map(p => `<li>${esc(p)}</li>`).join('')}</ul></details>` : ''}`,
      `<button type="button" class="btn ghost" data-do="dialog-close">Cancel</button><button type="button" class="btn primary" data-do="import-add">Add ${n(got.rows.length)} rows</button>`);
  }
  if (d.kind === 'import') {
    return dialogHtml('Import a folder', `
      <p>Choose a folder laid out one of two ways — the same two Flight School reads today:</p>
      <ul class="plain">
        <li><b>One folder per answer.</b> <span class="mono">BREED(boxer)/</span> holds the boxer photos, and so on.</li>
        <li><b>A <span class="mono">labels.json</span></b> naming each file’s answer.</li>
      </ul>
      <p class="hint">Pictures stay where they are; the table names them. ${r.input === 'text' ? 'For text, each file’s words are one input.' : ''}</p>`,
      '<button type="button" class="btn ghost" data-do="dialog-close">Cancel</button><button type="button" class="btn primary" data-do="import-pick">Choose a folder…</button>');
  }
  if (d.kind === 'pair') {
    return dialogHtml('Pair files', `
      <p>Two folders, matched by file name: what went in on one side, what you wrote from it on the other.</p>
      <div class="pair-demo"><div><span class="mono">statements/</span><span class="mono">northwind-q2.txt</span><span class="mono">halden-q2.txt</span></div>${ICON.arrow}<div><span class="mono">articles/</span><span class="mono">northwind-q2.md</span><span class="mono">halden-q2.md</span></div></div>
      <p>This is the fastest way in for anyone with an archive: the rows are work you already did. A file with no partner is listed, not guessed at.</p>
      <p class="hint">You pick the two folders one after the other: first what went in, then what you wrote.</p>`,
      '<button type="button" class="btn ghost" data-do="dialog-close">Cancel</button><button type="button" class="btn primary" data-do="pair-pick">Choose the folders…</button>');
  }
  if (d.kind === 'draft-material') {
    const total = r.material?.sections || 0;
    const partner = r.answerStyle === 'partner';
    const per = S.ui.draftPer || 3;
    /* A long book drafts from a sample; a thought partner defaults to 120
       sections (Kevin, 2026-10-16), a fact Reader to all of them. */
    const sampleChoices = [...new Set([60, 120, 240].filter(k => k < total).concat(total))];
    const from = Math.min(total, S.ui.draftSections || (partner ? 120 : total));
    const answering = from * per;
    const refusals = Math.round(answering * (partner ? 0.175 : 0.25));
    const calls = from + Math.ceil(refusals / 3);
    /* Measured 2026-09-30: 6.3 s a call on Qwen3.5-4B, the drafter a 16 GB Mac picks.
       A thought partner's replies are longer, so its calls are guessed at twice that. */
    const mins = Math.max(1, Math.round(calls * (partner ? 14 : 7) / 60));
    const what = partner
      ? `<p>A model reads ${from === total ? `each of the ${n(total)} sections` : `${n(from)} of the ${n(total)} sections, spread through the document,`} and writes messages a student would send about it — a concept they don't get, a discussion prompt, a draft to push on, a request to just write it, a mistake to correct — each with the reply a thought partner gives, citing its section. Then, for about one row in six, a <b>not covered</b> reply: a question from one section asked against its neighbour, which says so and helps from general knowledge without citing.</p>`
      : `<p>A model reads ${from === total ? `each of the ${n(total)} sections` : `${n(from)} of the ${n(total)} sections, spread through the document,`} and writes questions it answers, every answer citing its section. Then, for about one row in five, it writes a <b>refusal</b>: a question from one section asked against the section next to it, declined without a citation — the near miss a Reader most often gets wrong.</p>`;
    return dialogHtml(partner ? 'Read it and draft conversations' : 'Read it and draft questions', `
      ${what}
      <div class="field-pair">
        <div><label class="field-label small" for="draft-sections">Sections to read</label>
        <select id="draft-sections" class="select" data-change="draft-sections">${sampleChoices.map(k => `<option value="${k}" ${from === k ? 'selected' : ''}>${k === total ? `All ${n(total)}` : n(k)}</option>`).join('')}</select></div>
        <div><label class="field-label small" for="draft-per">${partner ? 'Messages per section' : 'Questions per section'}</label>
        <select id="draft-per" class="select" data-change="draft-per">${[2, 3, 5].map(k => `<option value="${k}" ${per === k ? 'selected' : ''}>${k}</option>`).join('')}</select></div>
      </div>
      <p class="hint">About ${n(answering + refusals)} drafts, about ${n(mins)} min ${tag(partner ? 'estimate' : 'measured')}. Each is checked before it lands: ${partner ? 'at most 130 words, exactly one question and it comes last, citing only its own section' : 'it must answer, from its own section, about the subject'}. They wait as drafts and train only once you approve them, section by section. You can stop at any point and keep what is done.</p>`,
      '<button type="button" class="btn ghost" data-do="dialog-close">Cancel</button><button type="button" class="btn primary" data-do="draft-material">Start reading</button>');
  }
  if (d.kind === 'prompt') {
    const rows = tableRows(r).slice(0, 8);
    const list = answerList(r).map(a => labelOf(r, a));
    const prompt = `${r.output === 'prose' ? 'Write the answer in the same way as these examples.' : `Answer with exactly one of: ${list.join(', ')}.`}\n\n${rows.map(x => `Input: ${x.input.kind === 'text' ? x.input.text : '[picture]'}\nAnswer: ${labelOf(r, x.answer)}`).join('\n\n')}\n\nInput: `;
    return dialogHtml('Use a prompt instead', `
      <p>With ${n(tableRows(r).length)} examples, a prompt that shows them to a model will do better than an adapter trained on them. Here is one, built from your rows. Come back to training at ${floorFor(r)}.</p>
      <label class="field-label small" for="prompt-text">The prompt</label>
      <textarea id="prompt-text" class="text-input mono-input" rows="12" readonly>${esc(prompt)}</textarea>`,
      '<button type="button" class="btn ghost" data-do="dialog-close">Close</button><button type="button" class="btn primary" data-do="copy-prompt">Copy the prompt</button>');
  }
  return '';
}

/* ── actions ──────────────────────────────────────────────────────────── */

export const actions = {
  filter (r, el) { S.ui.filter = el.dataset.filter; S.ui.page = 1; },
  /* The picture viewer opens from any screen's pictures (cells.js). */
  view (r, el) { openViewer(el); },
  'view-step' (r, el) { stepViewer(Number(el.dataset.by)); },
  'view-leave' (r) { viewerAct(r, 'leave'); },
  'view-back' (r) { viewerAct(r, 'back'); },
  'view-approve' (r) { viewerAct(r, 'approve'); },
  'view-discard' (r) { viewerAct(r, 'discard'); },
  'pic-crop-open' (r) { S.ui.dialog = { kind: 'pic-crop', crop: r.picCrop ? [...r.picCrop] : null }; },
  'pic-crop-apply' (r) {
    const c = S.ui.dialog?.crop;
    if (!c) return;
    const was = r.picCrop;
    setPicCrop(r, c);
    S.ui.dialog = null;
    toast(`Every picture will be cut to that part when it trains${r.sessionsIn.length ? ', and Pigeon will see the same part' : ''}.`, () => setPicCrop(r, was));
  },
  'record-part' (r) {
    const c = S.ui.dialog?.crop;
    if (!c) return;
    S.ui.dialog = null;
    return openRecordAgain(r, { part: c }, 'Choose the window, then record.');
  },
  'pic-crop-clear' (r) {
    const was = r.picCrop;
    setPicCrop(r, null);
    S.ui.dialog = null;
    toast('Pictures train whole again.', () => setPicCrop(r, was));
  },
  layout (r, el) {
    S.ui.layout = el.dataset.layout; S.ui.page = 1;
    try { localStorage.setItem('fs.layout', S.ui.layout); } catch { /* this session only */ }
  },
  'session-leave' (r) {
    const ids = r.rows.filter(x => x.session === S.ui.session && !x.left && !x.draft).map(x => x.id);
    const exam = r.rows.filter(x => ids.includes(x.id) && x.holdout).length;
    for (const id of ids) r.rows.find(x => x.id === id).left = 'dropped';
    r.version++;
    toast(`${ids.length} rows left out.${exam ? ` ${exam} of them were on the exam, which is now smaller.` : ''} They are under “Left out”.`,
      () => { for (const id of ids) r.rows.find(x => x.id === id).left = null; });
  },
  'session-back' (r) {
    const ids = r.rows.filter(x => x.session === S.ui.session && x.left).map(x => x.id);
    for (const id of ids) r.rows.find(x => x.id === id).left = null;
    r.version++;
    toast(`${ids.length} rows put back.`, () => { for (const id of ids) r.rows.find(x => x.id === id).left = 'dropped'; });
  },
  'pic-size' (r, el) {
    S.ui.picSize = el.dataset.size;
    try { localStorage.setItem('fs.picSize', S.ui.picSize); } catch { /* remembered for this session only */ }
  },
  more () { S.ui.page++; },
  edit (r, el) { S.ui.editing = el.dataset.row; },
  'edit-done' () { S.ui.editing = null; toast('Saved.'); },
  'leave-out' (r, el) {
    const id = el.dataset.row;
    leaveOut(r, id);
    toast('Row left out. It is still under “Left out”.', () => putBack(r, id));
  },
  'put-back' (r, el) { putBack(r, el.dataset.row); toast('Row put back.'); },
  approve (r, el) { approveRows(r, [el.dataset.row]); toast('Approved — it is in the table.'); },
  'approve-section' (r, el) {
    const k = el.dataset.section === '' ? null : Number(el.dataset.section);
    const ids = drafts(r).filter(x => (x.section ?? null) === k).map(x => x.id);
    approveRows(r, ids);
    toast(`${ids.length} ${ids.length === 1 ? 'draft' : 'drafts'} approved.`, () => { for (const id of ids) { const x = r.rows.find(y => y.id === id); if (x) x.draft = true; } });
  },
  'approve-all' (r) {
    const ids = drafts(r).map(x => x.id);
    approveRows(r, ids);
    S.ui.filter = 'all';
    toast(`${ids.length} drafts approved. They are in the table.`, () => { for (const id of ids) { const x = r.rows.find(y => y.id === id); if (x) x.draft = true; } });
  },
  'draft-material-open' () { S.ui.dialog = { kind: 'draft-material' }; },
  'draft-stop' () { if (S.ui.draftJob) S.ui.draftJob.stop = true; },
  'draft-material' (r) {
    if (S.ui.draftJob) return false;
    S.ui.dialog = null;
    later(r);
    const total = r.material?.sections || 0;
    const sections = Math.min(total, S.ui.draftSections || (r.answerStyle === 'partner' ? 120 : total));
    engine.draftFromMaterial(r, { perSection: S.ui.draftPer || 3, sections }).then(got => {
      S.ui.filter = 'drafts';
      toast(got.stopped ? `Stopped with ${got.added} drafts to check.` : `${got.added} drafts to check, by section. They do not train until you approve them.${got.rejected ? ` ${got.rejected} ${got.rejected === 1 ? 'draft was' : 'drafts were'} refused: no answer, not from the passage, or asked about the text.` : ''}${got.failed ? ` ${got.failed} replies could not be read.` : ''}`);
    }).catch(e => toastError(`The model could not draft: ${e}`)).finally(() => later(r));
    return false;
  },
  discard (r, el) {
    const id = el.dataset.row;
    const row = r.rows.find(x => x.id === id);
    r.rows = r.rows.filter(x => x.id !== id);
    toast('Draft discarded.', () => { r.rows.push(row); });
  },
  'add-open' () { S.ui.adding = true; },
  'add-cancel' () { S.ui.adding = false; },
  'import-open' () { S.ui.dialog = { kind: 'import' }; },
  'pair-open' () { S.ui.dialog = { kind: 'pair' }; },
  'material-pick' (r) { return addMaterial(r, null); },
  'material-remove' (r, el) {
    if (S.ui.importing) return false;
    const file = el.dataset.file;
    S.ui.importing = true; later(r);
    engine.material.remove(r, file).then(() => toast(`${file} is out of the material.`))
      .catch(e => toastError(String(e))).finally(() => { S.ui.importing = false; later(r); });
    return false;
  },
  'prompt-open' () { S.ui.dialog = { kind: 'prompt' }; },
  'record-open' () { S.ui.record = { step: 'setup', window: null, keys: true, clicks: true }; S.tab = 'record'; },
  /* A folder is looked at before it is added: what each answer becomes, and
     what could not be read. Nothing is in the table until "Add". */
  'import-pick' (r) {
    S.ui.dialog = null;
    S.ui.importing = true; later(r);
    engine.importFolder().then(got => {
      if (!got) return;
      if (!got.rows.length) { toast(got.problems[0] || 'Nothing in that folder could be read.'); return later(r); }
      S.ui.dialog = { kind: 'import-preview', got, plan: importPlan(r, got.rows) };
      later(r);
    }).catch(e => { toastError(String(e)); later(r); }).finally(() => { S.ui.importing = false; later(r); });
    return false;
  },
  'import-add' (r) {
    const d = S.ui.dialog;
    if (d?.kind !== 'import-preview') return;
    const to = (label) => { const p = d.plan.find(x => x.label === label); return p.join && p.near ? p.near : p.to; };
    const add = d.got.rows.map(x => ({ input: x.input, answer: to(x.answer) }));
    if (r.output !== 'prose') for (const a of new Set(add.map(x => x.answer))) if (!answerList(r).includes(a) && !r.answers.includes(a)) r.answers.push(a);
    const added = addRows(r, add, d.got.folder ? `Folder ${d.got.folder}` : 'Imported folder');
    S.ui.dialog = null;
    engine.findConflicts(r);
    const joined = d.plan.filter(p => p.join && p.near).length;
    toast(`Added ${added.length} rows from ${d.got.folder || 'the folder'}.${joined ? ` ${joined} ${joined === 1 ? 'answer that differed only in capitals or spacing was' : 'answers that differed only in capitals or spacing were'} joined to ${joined === 1 ? 'its match' : 'their matches'}.` : ''}`,
      () => { r.rows = r.rows.filter(x => !added.includes(x)); });
  },
  /* Decided 2026-09-21 (Pigeon's docs/PLAY-LOOP.md, P5): a play comes back
     as drafts. Approve the moments it got right, fix the ones it got wrong,
     and teach again — the exam stays the same, so the next round is fair. */
  'play-pick' (r) {
    S.ui.importing = true; later(r);
    engine.importPlay(r).then(got => {
      if (!got) return;
      if (!got.rows.length) { toast('That play log has no turns in it.'); return later(r); }
      const add = got.rows.map(x => {
        const known = answerList(r).find(a => labelOf(r, a) === x.answer);
        return { input: x.input, answer: known ?? x.answer, draft: true };
      });
      for (const x of add) if (r.output !== 'prose' && !answerList(r).includes(x.answer)) r.answers.push(x.answer);
      const when = got.started_ms ? new Date(got.started_ms).toLocaleString('en-GB', { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' }) : '';
      addRows(r, add, `Play ${when}`.trim());
      S.ui.filter = 'drafts';
      toast(`${add.length} moments from ${got.pigeon || 'the play'} to check. Approve the right ones, fix the wrong ones — they do not train until you do.`);
      later(r);
    }).catch(e => { toastError(String(e)); later(r); }).finally(() => { S.ui.importing = false; later(r); });
    return false;
  },
  'pair-pick' (r) {
    S.ui.dialog = null;
    S.ui.importing = true; later(r);
    engine.pairFiles().then(got => {
      if (!got) return;
      if (!got.rows.length) { toast(`No files matched by name.${got.alone.length ? ` ${got.alone.length} had no partner.` : ''}`); return later(r); }
      /* A Reader's pairs in the drafter's shape carry their section and
         whether they refuse, like drafted rows do. */
      const rows = r.kind === 'reader' ? got.rows.map(x => ({ ...annotateReaderRow(x), materialVersion: r.material?.version ?? null })) : got.rows;
      const added = addRows(r, rows, 'Paired files');
      engine.findConflicts(r);
      toast(`Paired ${added.length} files. ${got.alone.length ? `${got.alone.length} had no partner: ${got.alone.slice(0, 3).join(', ')}${got.alone.length > 3 ? '…' : ''}` : 'Every file had a partner.'}`,
        () => { r.rows = r.rows.filter(x => !added.includes(x)); });
      later(r);
    }).catch(e => { toastError(String(e)); later(r); }).finally(() => { S.ui.importing = false; later(r); });
    return false;
  },
  draft (r) {
    if (S.ui.drafting) return false;
    S.ui.drafting = true;
    toast('A model is drafting rows — about a minute. It is loaded for this and put away after.');
    engine.draft(r, 8).then(rows => {
      const add = rows.map(x => ({ input: { kind: 'text', text: x.input }, answer: r.output === 'prose' ? x.output : (answerList(r).find(a => labelOf(r, a) === x.output) ?? x.output), draft: true }));
      if (r.output !== 'prose') for (const x of add) if (!answerList(r).includes(x.answer)) r.answers.push(x.answer);
      addRows(r, add, 'Drafted by a model');
      S.ui.filter = 'drafts';
      toast(`${add.length} drafts to check. They do not train until you approve them.`);
    }).catch(e => toastError(`The model could not draft: ${e}`))
      .finally(() => { S.ui.drafting = false; later(r); });
    return false;
  },
  'copy-prompt' () {
    const el = document.getElementById('prompt-text');
    const done = () => toast('Prompt copied.');
    if (navigator.clipboard) navigator.clipboard.writeText(el.value).then(done, () => { el.select(); toast('Select-all is done — press ⌘C to copy.'); });
    else { el.select(); toast('Select-all is done — press ⌘C to copy.'); }
  }
};

export const inputs = {
  'edit-answer' (r, el) { const x = r.rows.find(y => y.id === el.dataset.row); if (x) { x.answer = el.value; r.version++; } return 'none'; },
  'edit-think' (r, el) { const x = r.rows.find(y => y.id === el.dataset.row); if (x) { x.think = el.value; r.version++; } return 'none'; }
};

export const changes = {
  'draft-per' (r, el) { S.ui.draftPer = Number(el.value); },
  'draft-sections' (r, el) { S.ui.draftSections = Number(el.value); },
  'edit-answer' (r, el) { updateRow(r, el.dataset.row, { answer: el.value }); },
  'view-answer' (r, el) { viewerAct(r, 'answer', el.value); },
  'session-filter' (r, el) { S.ui.session = el.value || null; S.ui.page = 1; },
  'import-join' (r, el) {
    const p = S.ui.dialog?.plan?.find(x => x.label === el.dataset.label);
    if (p) p.join = el.checked;
  }
};

export const submits = {
  'add-row' (r, form) {
    const answer = form.querySelector('#add-answer').value.trim();
    if (!answer) return;
    if (r.input !== 'text') { toast('Pictures come in by recording or from a folder.'); return; }
    const input = { kind: 'text', text: form.querySelector('#add-input').value.trim() };
    if (!input.text) return;
    if (r.output === 'function' && !answerList(r).includes(answer)) r.answers.push(answer);
    addRow(r, { input, answer, fresh: true });
    S.ui.adding = false;
    toast('Row added.');
  }
};

/* Dragging out the part of every picture to keep (the Crop dialog). */
export const drags = {
  'pic-crop' (r, el, e) {
    const d = S.ui.dialog;
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
      const live = document.querySelector('[data-live="pic-crop-size"]');
      if (live) live.textContent = `Keeping ${Math.round(rect[2] * 100)}% of the width and ${Math.round(rect[3] * 100)}% of the height.`;
    };
    const up = () => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
      if (rect && rect[2] > 0.03 && rect[3] > 0.03 && d?.kind === 'pic-crop') d.crop = rect.map(v => +v.toFixed(4));
      window.dispatchEvent(new Event('proto:render'));
    };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
  }
};
