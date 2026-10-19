/* review.js — CARRIES FORWARD. Stage 3: everything that must be looked at
 * before a training run, and nothing that is busywork.
 *
 * Flow-review changes folded in:
 *   - Review opens on a SUMMARY, not a tab of problems: what will train, what
 *     is set aside and why, what blocks. Each line links to where it is fixed.
 *   - Only things with no safe default block. Conflicts, NONE over its cap and
 *     stray clicks each state their default and let you change it.
 *   - The exam has its own tab, with Flight School's rule and a way to swap a
 *     question — the first plan had dropped it from view.
 *   - Thousands of conflicts (every row disagreeing) get a verdict, not a
 *     queue: the input cannot decide this job, so split it. */

import {
  S, RULES, tableRows, exam, trainingSet, blocks, defaults, openConflictList, answerList, labelOf,
  resolveConflict, reopenConflict, renameAction, mergeActions, decideStray, setNoneMode, reshuffleExam,
  labelsTxt, actionById, actionLabel, keyActions, clickActions, noneShare, updateRow
, startNarrower, nameProblem, badCites, staleRows, refusalRows, refusalShare, sectionCoverage,
  repeatKeeper, keepOneOf, putBackRows, keepAllOf, lookAgain, answerGroup, REPEAT} from '../state.js';
import { engine } from '../engine.js';
import { esc, ICON, n } from './html.js';
import { inputCell, chip, toast } from './cells.js';
import { targetsOverlay, crop } from './art.js';

function tabs (r) {
  const t = [['ready', 'Summary', null]];
  const oc = openConflictList(r).length;
  if (r.conflicts.length) t.push(['conflicts', 'Rows that disagree', oc || null]);
  const rep = S.ui.repeats;
  const openRep = rep?.state === 'done' && rep.run === r.id ? openRepeats(r).length : 0;
  t.push(['repeats', 'Repeats', openRep || null]);
  if (r.actions.length) t.push(['actions', 'Your actions', clickActions(r).filter(x => !x.name).length + r.strays.filter(s => !s.decision).length || null]);
  if (r.output !== 'prose') t.push(['balance', 'Balance', noneShare(r) > RULES.noneCap ? '!' : null]);
  if (r.kind === 'reader') t.push(['grounding', 'Grounding', badCites(r).length + staleRows(r).length || null]);
  t.push(['exam', 'The exam', null]);
  return t;
}

/* ── summary ──────────────────────────────────────────────────────────── */

function ready (r) {
  const ts = trainingSet(r);
  const b = blocks(r);
  const d = defaults(r);
  /* Repeats are worked out here too, in the background, so an exam picture
     that is also in the training rows is said on the first screen. */
  if (repeatsStale(r) && S.ui.repeats?.state !== 'working') engine.findRepeats(r);
  const rep = S.ui.repeats?.state === 'done' && S.ui.repeats.run === r.id ? openRepeats(r) : [];
  const leak = rep.filter(g => g.exam && g.rows.some(x => !x.holdout));
  const leakRows = leak.reduce((n2, g) => n2 + g.rows.filter(x => !x.holdout).length, 0);
  const copies = rep.reduce((n2, g) => n2 + g.rows.length - 1, 0);
  const repLines = [
    leak.length ? `<li><span><b>${n(leak.length)} exam ${leak.length === 1 ? 'picture is' : 'pictures are'} also in the training rows</b> (${n(leakRows)} ${leakRows === 1 ? 'copy' : 'copies'}), so the check ride would partly test memory and score better than the Pigeon is.</span><button type="button" class="btn sm" data-do="rep-exam">Fix</button></li>` : '',
    copies ? `<li><span>${n(copies)} ${copies === 1 ? 'row repeats' : 'rows repeat'} another with the same answer. They all train unless you keep one of each.</span><button type="button" class="btn sm ghost" data-do="tab" data-tab="repeats">Look</button></li>` : ''
  ].join('');
  const fixBtn = (i) => `<button type="button" class="btn sm ghost" data-do="fix" data-stage="${i.fix.stage}" data-tab="${i.fix.tab || ''}" data-filter="${i.fix.filter || ''}">${i.level === 'block' ? 'Fix' : 'Change'}</button>`;
  const strayOut = r.strays.filter(s => !s.decision || s.decision === 'leave').length;
  return `<div class="ready-view">
    <div class="sum-grid">
      <div class="sum big"><b>${n(ts.rows.length)}</b><span>rows will train</span></div>
      <div class="sum"><b>${n(exam(r).length)}</b><span>set aside as the exam — never trained on</span></div>
      ${ts.conflictRowsLeftOut ? `<div class="sum"><b>${n(ts.conflictRowsLeftOut)}</b><span>left out: rows that disagree</span></div>` : ''}
      ${ts.noneDropped ? `<div class="sum"><b>${n(ts.noneDropped)}</b><span>NONE rows sampled away</span></div>` : ''}
      ${strayOut ? `<div class="sum"><b>${strayOut}</b><span>stray clicks left out</span></div>` : ''}
    </div>

    ${b.length ? `<section class="issue-list block">
      <h3>${ICON.lock}Before it can be taught</h3>
      <ul>${b.map(i => `<li><span>${esc(i.text)}</span>${fixBtn(i)}</li>`).join('')}</ul>
    </section>` : ''}

    ${d.length || repLines ? `<section class="issue-list default">
      <h3>${ICON.check}Settled for you — change any of these</h3>
      <ul>${d.map(i => `<li><span>${esc(i.text)}</span>${fixBtn(i)}</li>`).join('')}${repLines}</ul>
    </section>` : ''}

    ${!b.length && !d.length && !repLines ? `<p class="ready">${ICON.check}Nothing to settle. The table is ready to teach.</p>` : ''}

    <div class="stage-foot">
      <p class="hint">${b.length ? 'Teach opens when nothing above blocks it.' : 'Everything above has a default. Teaching now uses them.'}</p>
      <button type="button" class="btn primary" data-do="go" data-stage="teach" ${b.length ? 'aria-disabled="true"' : ''}>Teach it${ICON.arrow}</button>
    </div>
  </div>`;
}

/* ── conflicts ─────────────────────────────────────────────────────────── */

function conflicts (r) {
  if (S.ui.conflictError) {
    return `<p class="warn-line" role="alert">${ICON.warn}The pictures could not be compared: ${esc(S.ui.conflictError)}. <button type="button" class="btn sm ghost" data-do="conf-retry">Try again</button></p>`;
  }
  const open = openConflictList(r);
  const settled = r.conflicts.filter(c => c.resolution);
  /* When most of the table is in conflict, a queue is the wrong tool. */
  const everything = r.conflicts.length > 50 && r.conflicts.length * 2 >= r.rows.length * 0.8;
  const aside = r.rows.filter(x => x.left === 'split').length;
  const narrow = aside ? `<div class="narrow"><p><b>${n(aside)} rows are set aside for a narrower job.</b> A narrower job is its own run: the rows go with it, and this table carries on without them.</p>
      <button type="button" class="btn" data-do="narrow">Start a run with them${ICON.arrow}</button></div>` : '';
  if (!open.length) {
    return `<div class="conf-done"><p class="ready">${ICON.check}Every pair is settled.</p>${narrow}
      ${settled.length ? `<ul class="settled">${settled.map((c, i) => `<li>Pair ${i + 1} — ${esc({ a: 'kept the first', b: 'kept the second', drop: 'both left out', split: 'set aside for a narrower job' }[c.resolution])} <button type="button" class="btn sm ghost" data-do="conf-reopen" data-id="${c.id}">Reopen</button></li>`).join('')}</ul>` : ''}</div>`;
  }
  if (everything) {
    return `<div class="verdict">
      <h3>Every row here disagrees with another one.</h3>
      <p>${n(open.length)} pairs of identical inputs carry different answers. That is not a labelling mistake: <b>the input alone cannot decide this job</b>. The right answer depends on something that happened before it — and that is not in the input.</p>
      <p>Two ways out, and neither is settling pairs one by one:</p>
      <ul class="plain">
        <li><b>Narrow the job</b> to inputs that do decide the answer — a dialog asking a question, a menu waiting for a choice.</li>
        <li><b>Show what came before</b>: put the last few pictures in one, so it can see what led here. The table keeps its shape.</li>
      </ul>
      <div class="stage-foot"><button type="button" class="btn" data-do="conf-all" data-how="drop">Leave all ${n(open.length * 2)} rows out</button><button type="button" class="btn primary" data-do="conf-all" data-how="split">Set them aside for a narrower job</button></div>
    </div>`;
  }
  const i = Math.min(S.ui.conflict, open.length - 1);
  const c = open[i];
  const a = r.rows.find(x => x.id === c.a);
  const b = r.rows.find(x => x.id === c.b);
  const side = (row, which) => `<div class="conf-side">${inputCell(row, { big: true })}<div class="conf-meta">${chip(r, row.answer)}<span class="session">${esc(row.session || '')}</span></div>
    <button type="button" class="btn sm" data-do="conf" data-id="${c.id}" data-how="${which}">Keep this one</button></div>`;
  return `<div class="conf">
    <div class="conf-top">
      <p><b>${i + 1} of ${n(open.length)}</b> · these two inputs are <b>${c.similarity}% identical</b> and carry different answers.</p>
      <div class="pager"><button type="button" class="icon-btn" data-do="conf-prev" aria-label="Previous pair (K)" ${i === 0 ? 'disabled' : ''}>${ICON.back}</button><button type="button" class="icon-btn" data-do="conf-next" aria-label="Next pair (J)" ${i >= open.length - 1 ? 'disabled' : ''}>${ICON.arrow}</button></div>
    </div>
    <div class="conf-pair">${side(a, 'a')}<span class="vs">vs</span>${side(b, 'b')}</div>
    <p class="hint">The input does not say which is right — what happened before it does. If neither answer is wrong, leave both out, or split off a job the input can decide.</p>
    <div class="stage-foot">
      <p class="hint">Unsettled pairs are left out when it trains. J and K move between pairs.</p>
      <button type="button" class="btn ghost" data-do="conf" data-id="${c.id}" data-how="drop">Leave both out</button>
      <button type="button" class="btn" data-do="conf" data-id="${c.id}" data-how="split">Narrower job…</button>
    </div>
  </div>`;
}

/* ── repeats ────────────────────────────────────────────────────────────
   The same input with the same answer, grouped. Worked out when the tab is
   opened (and again whenever the table or the setting changes), never
   saved: the pictures decide it, not a stored list. Nothing here is a
   default — every row trains until you choose. */

const REP_PAGE = 20;
const openRepeats = (r) => (S.ui.repeats?.groups || []).filter(g => !(r.repeatsKept || []).includes(g.key));

function repeatsStale (r) {
  const rep = S.ui.repeats;
  return !rep || rep.run !== r.id || rep.version !== r.version || rep.mode !== S.ui.repeatMode || rep.tol !== S.ui.repeatTol;
}

function repeatGroupHtml (r, g, kept) {
  const keep = repeatKeeper(g);
  const others = g.rows.filter(x => x !== keep);
  const strip = others.slice(0, 8);
  const extra = others.length - strip.length;
  const leaves = g.rows.filter(x => x !== keep && !x.holdout).length;
  return `<div class="rep-group${kept ? ' kept' : ''}">
    <div class="rep-main">${inputCell(keep, { big: true })}</div>
    <div class="rep-info">
      <p class="rep-count"><b>×${n(g.rows.length)}</b> ${chip(r, g.answer)}${g.exam ? '<span class="badge exam" title="One of these is on the exam. Keeping one keeps the exam row, so none of its copies train.">exam</span>' : ''}</p>
      ${strip.length ? `<div class="rep-strip">${strip.map(x => inputCell(x, { size: 'mini' })).join('')}${extra > 0 ? `<span class="rep-more">+${n(extra)}</span>` : ''}</div>` : ''}
      ${kept
        ? `<div class="rep-acts"><span class="hint">All ${n(g.rows.length)} kept.</span><button type="button" class="btn sm ghost" data-do="rep-again" data-key="${g.key}">Look again</button></div>`
        : `<div class="rep-acts">
        <button type="button" class="btn sm" data-do="rep-one" data-key="${g.key}" ${leaves ? '' : 'disabled'}>Keep one${leaves ? `, leave ${n(leaves)} out` : ''}</button>
        <button type="button" class="btn sm ghost" data-do="rep-all" data-key="${g.key}">Keep all</button>
        ${r.output !== 'prose' ? `<label class="field-label small" for="rep-ans-${g.key}">Answer for all</label>
        <select id="rep-ans-${g.key}" class="select sm" data-change="rep-answer" data-key="${g.key}">
          ${answerList(r).map(a => `<option value="${esc(a)}" ${a === g.answer ? 'selected' : ''}>${esc(labelOf(r, a))}</option>`).join('')}
        </select>` : ''}
      </div>`}
    </div>
  </div>`;
}

function repeats (r) {
  if (repeatsStale(r) && S.ui.repeats?.state !== 'working') engine.findRepeats(r);
  const rep = S.ui.repeats;
  const image = r.input === 'image';
  const setting = image ? `<fieldset class="rep-mode">
      <legend class="field-label">Count two pictures as the same when they are</legend>
      <label class="radio-row" for="rep-exact"><input type="radio" id="rep-exact" name="rep-mode" value="exact" ${S.ui.repeatMode === 'exact' ? 'checked' : ''} data-change="rep-mode">
        <span><b>Identical</b> — every part within ${REPEAT.exactTol} grey levels, which is only compression noise</span></label>
      <label class="radio-row" for="rep-close"><input type="radio" id="rep-close" name="rep-mode" value="close" ${S.ui.repeatMode === 'close' ? 'checked' : ''} data-change="rep-mode">
        <span><b>Nearly identical</b> — no part differs by more than you allow</span></label>
      ${S.ui.repeatMode === 'close' ? `<div class="rep-tol">
        <label for="rep-tol" class="field-label small">Most any part may differ</label>
        <input type="range" id="rep-tol" min="4" max="60" step="1" value="${S.ui.repeatTol}" data-change="rep-tol" data-input="rep-tol">
        <span class="mono" id="rep-tol-v">${S.ui.repeatTol} of 255</span>
      </div>
      <p class="hint">Each picture is compared at 32 × 32. A sprite that moved or a number that changed shows up as one part that differs a lot; a small setting keeps those apart.</p>` : ''}
    </fieldset>` : '';
  const intro = `<p>Rows with the same ${image ? 'picture' : 'words'} <em>and</em> the same answer. A recording is full of them — the same key pressed on the same screen — and every copy after the first teaches nothing new while it weighs the table toward that moment. ${r.conflicts.length ? 'The same input with a <em>different</em> answer is under “Rows that disagree”.' : ''}</p>`;
  if (!rep || rep.run !== r.id || (rep.state === 'working' && !rep.groups.length)) {
    return `<div class="repeats">${intro}${setting}<p class="hint" role="status">${image ? 'Comparing the pictures…' : 'Comparing the rows…'}</p></div>`;
  }
  if (rep.state === 'failed') {
    return `<div class="repeats">${intro}${setting}<p class="warn-line" role="alert">${ICON.warn}The pictures could not be compared: ${esc(rep.error)}. <button type="button" class="btn sm ghost" data-do="rep-retry">Try again</button></p></div>`;
  }
  const open = openRepeats(r);
  const kept = (rep.groups || []).filter(g => (r.repeatsKept || []).includes(g.key));
  const copies = open.reduce((s, g) => s + g.rows.filter(x => x !== repeatKeeper(g) && !x.holdout).length, 0);
  const shown = open.slice(0, REP_PAGE * S.ui.repeatPage);
  const leftNow = r.rows.filter(x => x.left === 'repeat').length;
  return `<div class="repeats">
    ${intro}${setting}
    ${rep.unread ? `<p class="hint">${n(rep.unread)} ${rep.unread === 1 ? 'picture' : 'pictures'} could not be read, so ${rep.unread === 1 ? 'it is' : 'they are'} not compared.</p>` : ''}
    ${open.length ? `<div class="rep-sum" role="status">
        <p><b>${n(open.length)} ${open.length === 1 ? 'group' : 'groups'}</b> of repeats. Keeping one of each leaves <b>${n(copies)} ${copies === 1 ? 'row' : 'rows'}</b> out.${rep.state === 'working' ? ' <span class="hint">Checking again…</span>' : ''}</p>
        <button type="button" class="btn" data-do="rep-every" ${copies ? '' : 'disabled'}>Keep one of each</button>
      </div>
      <div class="rep-list">${shown.map(g => repeatGroupHtml(r, g, false)).join('')}</div>
      ${open.length > shown.length ? `<button type="button" class="btn more" data-do="rep-more">Show ${Math.min(REP_PAGE, open.length - shown.length)} more of ${n(open.length - shown.length)}</button>` : ''}`
      : `<p class="ready" role="status">${ICON.check}No repeats${kept.length ? ' left to look at' : ''}${image ? ` at this setting` : ''}.${leftNow ? ` ${n(leftNow)} ${leftNow === 1 ? 'copy is' : 'copies are'} left out — they are under “Left out” in Collect.` : ''}</p>`}
    ${kept.length ? `<details class="rep-kept"><summary>${n(kept.length)} ${kept.length === 1 ? 'group' : 'groups'} you chose to keep whole</summary>
      <div class="rep-list">${kept.map(g => repeatGroupHtml(r, g, true)).join('')}</div></details>` : ''}
  </div>`;
}

/* ── your actions ──────────────────────────────────────────────────────
   Everything you did while recording, as answers. Keys are named after the
   key until you rename them, so they never block. Clicks have no name until
   you give one. Either kind can be merged when two mean the same thing. */

function mergeBar (r) {
  const n2 = S.ui.merge.length;
  const picked = S.ui.merge.map(id => actionById(r, id)).filter(Boolean);
  const same = picked.every(a => a.kind === picked[0]?.kind);
  if (n2 >= 2 && !same) return '<p class="warn-line">A key and a click cannot be merged — tick two of the same kind.</p>';
  return n2 >= 2 ? `<button type="button" class="btn" data-do="merge">${ICON.merge}Merge ${n2} into one answer</button>` : '';
}

function actionCard (r, a, sel) {
  const ticked = S.ui.merge.includes(a.id);
  const isKey = a.kind === 'key';
  const label = isKey ? 'Name' : (a.name ? 'Name' : 'Name this one');
  const pic = isKey
    ? `<span class="keycap" aria-hidden="true">${esc(a.raw)}</span>`
    : `<button type="button" class="crop-btn" data-do="target-select" data-id="${a.id}" aria-label="Show ${esc(a.name || 'this click')} on the window">${crop(r, a)}</button>`;
  return `<div class="tgt${sel === a.id ? ' on' : ''}${!isKey && !a.name ? ' unnamed' : ''}">
    <input type="checkbox" id="merge-${a.id}" class="merge-box" ${ticked ? 'checked' : ''} data-change="merge-tick" data-id="${a.id}" aria-label="Pick ${esc(actionLabel(a))} to merge">
    ${pic}
    <div class="tgt-name">
      <label for="name-${a.id}" class="field-label small">${label}${isKey ? ` <span class="hint">— pressed ${esc(a.raw)}</span>` : ''}</label>
      <input id="name-${a.id}" class="text-input" type="text" value="${esc(a.name || '')}" placeholder="${isKey ? esc(a.raw) : 'e.g. Archive'}" data-input="action-name" data-id="${a.id}" autocomplete="off" aria-describedby="why-${a.id}">
      <span class="hint bad" id="why-${a.id}" data-live="why-${a.id}" aria-live="polite"></span>
    </div>
    <div class="tgt-out"><span class="fn${!isKey && !a.name ? ' empty' : ''}" data-live="chip-${a.id}">${esc(actionLabel(a))}</span><span class="count">used ${n(a.uses)}×</span></div>
  </div>`;
}

function actionsView (r) {
  const sel = S.ui.target;
  const keys = keyActions(r);
  const clicks = clickActions(r);
  const strays = r.strays.map((s, i) => ({ ...s, i })).filter(s => !s.decision);
  return `<div class="actions-view">
    <p>Everything you did while recording is an answer the Pigeon can give. Give each one a name that says what it <em>does</em> — the name is what it learns, and what you will read in its answers.</p>
    ${keys.length ? `<section class="act-group">
      <h3>Keys you pressed</h3>
      <p class="hint">Named after the key until you rename them — <span class="mono">KEY(Return)</span> works as it is. Rename a key when what it does matters more than where it is: <span class="mono">KEY(Z)</span> → <span class="mono">KEY(Confirm)</span>.</p>
      <div class="key-list">${keys.map(a => actionCard(r, a, sel)).join('')}</div>
    </section>` : ''}
    ${clicks.length ? `<section class="act-group">
      <h3>Places you clicked</h3>
      <div class="targets">
        <div class="tg-window">
          <div class="tg-bar"><span class="field-label small">Where you clicked</span>
            <label class="check" for="grid-on"><input type="checkbox" id="grid-on" ${S.ui.grid ? 'checked' : ''} data-change="grid"> Grid</label></div>
          ${targetsOverlay(r, clicks, r.strays, { grid: S.ui.grid, selected: sel })}
          <p class="hint">The grid is a guide for your eye. The Pigeon learns the name — <span class="mono">CLICK(Send)</span>, never a grid square — and finds each place on screen by its picture.</p>
        </div>
        <div class="tg-list">
          ${clicks.map(a => actionCard(r, a, sel)).join('')}
          ${strays.length ? `<div class="strays"><b>${strays.length} ${strays.length === 1 ? 'click landed' : 'clicks landed'} away from anything else</b><p>Probably clicking to focus the window. Left out unless you say otherwise.</p>
            <div class="stage-foot tight"><button type="button" class="btn sm" data-do="strays" data-how="leave">Leave them out</button><button type="button" class="btn sm" data-do="strays" data-how="target">Make an action</button></div></div>` : ''}
        </div>
      </div>
    </section>` : ''}
    ${mergeBar(r) || '<p class="hint">Two that mean the same thing — W and the up arrow, two boxes that are one button? Tick both to merge them.</p>'}
    <details class="labels-preview"><summary>labels.txt — generated from these</summary><pre>${esc(labelsTxt(r))}</pre></details>
  </div>`;
}

/* ── balance ───────────────────────────────────────────────────────────── */

function balance (r) {
  const rows = tableRows(r);
  const counts = new Map();
  for (const x of rows) counts.set(x.answer, (counts.get(x.answer) || 0) + 1);
  const max = Math.max(1, ...counts.values());
  const share = noneShare(r);
  const ts = trainingSet(r);
  const trainedNone = ts.rows.filter(x => x.answer === 'NONE').length;
  const bars = answerList(r).filter(a => counts.get(a)).map(a => {
    const c = counts.get(a);
    const none = a === 'NONE';
    return `<div class="bar-row${none ? ' none' : ''}"><span class="bar-label">${esc(labelOf(r, a))}</span>
      <span class="bar-track"><span class="bar" style="width:${(c / max) * 100}%"></span>${none && r.noneMode === 'sample' && ts.noneDropped ? `<span class="bar after" style="width:${(trainedNone / max) * 100}%"></span>` : ''}</span>
      <span class="bar-n">${n(c)}${none && r.noneMode === 'sample' && ts.noneDropped ? ` → ${n(trainedNone)}` : ''}</span></div>`;
  }).join('');
  const small = [...counts].filter(([a, c]) => a !== 'NONE' && c < 10);
  return `<div class="balance">
    <p>How many rows teach each answer. <b>NONE</b> — do nothing — is most of a recording, because most of the time you are reading or waiting.</p>
    <div class="bars">${bars}</div>
    ${small.length ? `<p class="warn-line">${ICON.warn}${small.map(([a, c]) => `${esc(labelOf(r, a))} has only ${c}`).join('; ')}. It may never learn ${small.length === 1 ? 'it' : 'them'}. Record the moments you press ${small.length === 1 ? 'it' : 'them'}, or leave ${small.length === 1 ? 'it' : 'them'} out of the list.</p>` : ''}
    <fieldset class="choices">
      <legend class="field-label">NONE is ${Math.round(share * 100)}% of the rows. The cap is ${Math.round(RULES.noneCap * 100)}%.</legend>
      <label class="radio-row" for="none-sample"><input type="radio" id="none-sample" name="none" value="sample" ${r.noneMode === 'sample' ? 'checked' : ''} data-change="none-mode">
        <span><b>Sample NONE down to ${Math.round(RULES.noneCap * 100)}% of what trains</b> — recommended<br><span class="hint">Nothing is deleted; the extra rows just sit this run out.</span></span></label>
      <label class="radio-row" for="none-all"><input type="radio" id="none-all" name="none" value="all" ${r.noneMode === 'all' ? 'checked' : ''} data-change="none-mode">
        <span><b>Train on every NONE row</b><br><span class="hint">A Pigeon that always answers NONE would then score well and do nothing — the check ride counts misses separately so you would see it.</span></span></label>
    </fieldset>
  </div>`;
}

/* ── grounding ─────────────────────────────────────────────────────────
   What a Reader's table teaches about reading: how much of the document its
   rows cover, how often it is shown a passage that does not answer the
   question, and the two deterministic checks — citations of the unseen, and
   rows drafted against a document that has since changed. */

function grounding (r) {
  const rows = tableRows(r);
  const ref = refusalRows(r).length;
  const share = refusalShare(r);
  const cov = sectionCoverage(r);
  const bc = badCites(r).length;
  const st = staleRows(r).length;
  const fix = (filter, label) => `<button type="button" class="btn sm ghost" data-do="fix" data-stage="collect" data-tab="" data-filter="${filter}">${label}</button>`;
  return `<div class="grounding">
    <p>A Reader is graded on two things: whether it answers <em>from the passage</em>, and whether it says <b>not in here</b> when the passage does not cover the question. Both are measured on the check ride against the same model untrained. Nothing here is a judge’s opinion.</p>
    <div class="sum-grid">
      <div class="sum big"><b>${n(rows.length - ref)}</b><span>rows answer from their section</span></div>
      <div class="sum"><b>${n(ref)}</b><span>refusals — ${Math.round(share * 100)}% of the table, about 20% is the aim</span></div>
      <div class="sum"><b>${n(cov.covered)} of ${n(cov.total)}</b><span>sections have at least one question</span></div>
    </div>
    <ul class="plain checks">
      <li>${bc ? `${ICON.warn}<span><b>${n(bc)} ${bc === 1 ? 'row cites' : 'rows cite'} a section ${bc === 1 ? 'it was' : 'they were'} not shown.</b> Measured on the first grounded adapter: handed nothing relevant, it invented an answer and cited a source anyway in 11 of 14 cases. A citation that cannot be checked does not train.</span>${fix('bad-cite', 'Look')}` : `${ICON.check}<span>Every citation points at the section the row was shown.</span>`}</li>
      <li>${st ? `${ICON.warn}<span><b>${n(st)} ${st === 1 ? 'row was' : 'rows were'} drafted against an earlier version of the document.</b> Its section numbers no longer match. Discard them and draft again.</span>${fix('stale', 'Look')}` : `${ICON.check}<span>Every row matches the document as it is now.</span>`}</li>
      <li>${share < 0.05 ? `${ICON.warn}<span><b>${share ? `Only ${Math.round(share * 100)}%` : 'None'} of the rows are refusals.</b> A Reader that never sees a passage that does not answer the question learns that every passage does.</span>` : `${ICON.check}<span>The refusals are near misses: a question from one section, asked against the section beside it, declined without a citation.</span>`}</li>
    </ul>
    <p class="hint">Measured so far: refusal is trainable — 1 in 10 to 10 in 10 — but no adapter yet beats the untrained model at it by more than noise. The check ride will show both numbers, side by side.</p>
  </div>`;
}

/* ── the exam ──────────────────────────────────────────────────────────── */

function examView (r) {
  const ex = exam(r);
  return `<div class="exam">
    <p>These ${n(ex.length)} rows are the check ride. They are set aside <b>before anything trains</b> and never trained on — the exam is never in the textbook. ${r.kind === 'reader' ? 'A Reader’s exam is drawn by <b>section</b>: whole sections go, with every question about them, so no exam question’s own passage was ever trained on.' : 'Flight School’s rule: at least 5, at most 20, a fair share of every answer.'}</p>
    <div class="exam-grid">${ex.map(x => `<div class="exam-cell">${inputCell(x)}${r.output === 'prose' ? `<p class="cell-text small">${esc(x.answer.slice(0, 120))}…</p>` : chip(r, x.answer)}
      <button type="button" class="btn sm ghost" data-do="exam-swap" data-row="${x.id}">Swap</button></div>`).join('')}</div>
    <div class="stage-foot">
      <p class="hint">Each round of teaching is graded on this same exam, so round 2 against round 1 is a fair comparison.</p>
      <button type="button" class="btn" data-do="exam-redraw">Draw a different exam</button>
    </div>
  </div>`;
}

export function render (r) {
  r.reviewSeen = true;
  /* Conflicts are worked out when rows arrive; a table changed since (or
     judged by an older rule) is looked at again, in the background. */
  if (r.input && r.conflictsAt !== r.version) engine.findConflicts(r);
  const t = tabs(r);
  const tab = t.some(x => x[0] === S.tab) ? S.tab : 'ready';
  const body = { ready, conflicts, repeats, actions: actionsView, balance, grounding, exam: examView }[tab](r);
  return `<div class="review">
    <div class="tabs" role="tablist" aria-label="Review">
      ${t.map(([id, label, badge]) => `<button type="button" role="tab" id="tab-${id}" class="tab" aria-selected="${tab === id}" data-do="tab" data-tab="${id}">${esc(label)}${badge ? `<span class="tab-badge">${badge}</span>` : ''}</button>`).join('')}
    </div>
    <div class="tab-body" role="tabpanel" aria-labelledby="tab-${tab}">${body}</div>
  </div>`;
}

/* ── actions ──────────────────────────────────────────────────────────── */

export const actions = {
  tab (r, el) { S.tab = el.dataset.tab === 'ready' ? null : el.dataset.tab; },
  conf (r, el) {
    const how = el.dataset.how;
    resolveConflict(r, el.dataset.id, how);
    const words = { a: 'Kept the first.', b: 'Kept the second.', drop: 'Both left out.', split: 'Set aside for a narrower job.' }[how];
    toast(words, () => reopenConflict(r, el.dataset.id));
  },
  'conf-all' (r, el) {
    const ids = openConflictList(r).map(c => c.id);
    for (const id of ids) resolveConflict(r, id, el.dataset.how);
    toast(el.dataset.how === 'split' ? `${ids.length * 2} rows set aside for a narrower job.` : `${ids.length * 2} rows left out.`, () => { for (const id of ids) reopenConflict(r, id); });
  },
  'conf-reopen' (r, el) { reopenConflict(r, el.dataset.id); },
  'rep-one' (r, el) {
    const g = openRepeats(r).find(x => x.key === el.dataset.key);
    if (!g) return;
    const out = keepOneOf(r, g);
    toast(`Kept one; ${out.length} ${out.length === 1 ? 'copy' : 'copies'} left out.`, () => putBackRows(r, out));
  },
  /* Only the groups that hold an exam row: each keeps its exam row, and its
     copies stop training. */
  'rep-exam' (r) {
    const out = openRepeats(r).filter(g => g.exam).flatMap(g => keepOneOf(r, g));
    toast(`${out.length} ${out.length === 1 ? 'copy' : 'copies'} of exam pictures left out of training.`, () => putBackRows(r, out));
  },
  'rep-every' (r) {
    const out = openRepeats(r).flatMap(g => keepOneOf(r, g));
    toast(`Kept one of each; ${out.length} ${out.length === 1 ? 'copy' : 'copies'} left out.`, () => putBackRows(r, out));
  },
  'rep-all' (r, el) { keepAllOf(r, el.dataset.key); toast('Kept all of them. The group is under “chose to keep whole”.', () => lookAgain(r, el.dataset.key)); },
  'rep-again' (r, el) { lookAgain(r, el.dataset.key); },
  'rep-more' () { S.ui.repeatPage++; },
  'conf-retry' (r) { S.ui.conflictError = null; r.conflictsAt = null; engine.findConflicts(r); return false; },
  'rep-retry' () { S.ui.repeats = null; },
  /* Decided 2026-09-21: "Narrower job" starts a new run holding the rows set
     aside for it — with the actions they answer with — and opens it. */
  narrow (r) {
    const rows = r.rows.filter(x => x.left === 'split');
    if (!rows.length) return false;
    const id = startNarrower(r, rows);
    location.hash = `#/run/${encodeURIComponent(id)}/describe`;
    toast(`${rows.length} rows moved into a new run. Name it, then review it.`);
    return false;
  },
  'conf-next' () { S.ui.conflict++; },
  'conf-prev' () { S.ui.conflict = Math.max(0, S.ui.conflict - 1); },
  'target-select' (r, el) { S.ui.target = S.ui.target === el.dataset.id ? null : el.dataset.id; },
  merge (r) {
    const ids = [...S.ui.merge];
    mergeActions(r, ids);
    S.ui.merge = [];
    toast(`Merged into ${actionLabel(actionById(r, ids[0]))}.`);
  },
  strays (r, el) {
    r.strays.forEach((s, i) => { if (!s.decision) decideStray(r, i, el.dataset.how); });
    toast(el.dataset.how === 'target' ? 'Made an action. Give it a name.' : 'Stray clicks left out.');
  },
  'exam-swap' (r, el) {
    const x = r.rows.find(y => y.id === el.dataset.row);
    const alt = tableRows(r).find(y => !y.holdout && y.answer === x.answer && y.id !== x.id);
    if (!alt) { toast(`No other ${labelOf(r, x.answer)} row to swap in.`); return; }
    updateRow(r, x.id, { holdout: false });
    updateRow(r, alt.id, { holdout: true });
    toast('Swapped for another row with the same answer.');
  },
  'exam-redraw' (r) { reshuffleExam(r); toast('A new exam is drawn. Earlier rounds were graded on the old one.'); }
};

export const inputs = {
  /* The number follows the slider as it moves; the groups are worked out
     again when it is let go. */
  'rep-tol' (r, el) { const v = document.getElementById('rep-tol-v'); if (v) v.textContent = `${el.value} of 255`; return 'none'; },
  'action-name' (r, el) {
    const problem = nameProblem(el.value);
    el.setAttribute('aria-invalid', problem ? 'true' : 'false');
    const why = document.querySelector(`[data-live="why-${el.dataset.id}"]`);
    if (why) why.textContent = problem || '';
    if (problem) return 'none';
    renameAction(r, el.dataset.id, el.value);
    const a = actionById(r, el.dataset.id);
    const live = document.querySelector(`[data-live="chip-${el.dataset.id}"]`);
    if (live) { live.textContent = actionLabel(a); live.classList.toggle('empty', a.kind === 'click' && !a.name); }
    return 'none';
  }
};

export const changes = {
  'merge-tick' (r, el) {
    const id = el.dataset.id;
    S.ui.merge = el.checked ? [...new Set([...S.ui.merge, id])] : S.ui.merge.filter(x => x !== id);
  },
  grid (r, el) { S.ui.grid = el.checked; },
  'rep-mode' (r, el) { S.ui.repeatMode = el.value; S.ui.repeatPage = 1; },
  'rep-tol' (r, el) { S.ui.repeatTol = Number(el.value); S.ui.repeatPage = 1; },
  'rep-answer' (r, el) {
    const g = openRepeats(r).find(x => x.key === el.dataset.key);
    if (!g) return;
    const was = g.answer;
    const ids = g.rows.map(x => x.id);
    answerGroup(r, g, el.value);
    toast(`${ids.length} rows now answer ${labelOf(r, el.value)}.`, () => { for (const id of ids) { const x = r.rows.find(y => y.id === id); if (x) x.answer = was; } });
  },
  'none-mode' (r, el) { setNoneMode(r, el.value); }
};

/* Renaming updates the chip as you type without a redraw; the rest of the
   screen (tab badges, the summary) catches up when you leave the field. */
export const blurs = { 'action-name': true };
