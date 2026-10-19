/* ride.js — CARRIES FORWARD. Stage 5: the check ride.
 *
 * Flow-review changes folded in:
 *   - The headline is never one number. A function job gets three — right
 *     answers on real actions, false presses, misses — because a miss stalls
 *     and a false press buys the wrong item.
 *   - It is graded beside the SAME model untrained, and beside the last round,
 *     so a score always has something to be compared with.
 *   - It closes the loop. The mistakes it makes most point back to the rows
 *     that would fix them — teach, notice, teach again — instead of ending on
 *     a score.
 *   - Reasoning is shown under the answer, collapsed, never as a receipt. */

import { S, exam, answerList, labelOf } from '../state.js';
import { engine, summarise } from '../engine.js';
import { esc, ICON, n } from './html.js';
import { inputCell, chip, toast } from './cells.js';

const OUTCOME = {
  right: ['Right', 'good'], miss: ['Miss', 'warn'], 'false-press': ['False press', 'bad'], wrong: ['Wrong answer', 'bad'],
  'off-list': ['Off the list', 'bad'], 'no-answer': ['No answer', 'bad'], 'not-found': ['Not on screen', 'warn']
};
const pct = (x) => `${Math.round(x * 100)}%`;

function notYet (r) {
  const ex = exam(r).length;
  return `<div class="ride-start">
    <p>The exam is ${n(ex)} rows it has never seen. Each one is asked twice: once of the taught Pigeon, once of the same model with the training switched off. It runs by itself when teaching finishes${r.teach?.state === 'done' ? ' — this loads its results' : ''}.</p>
    ${r.output === 'function' ? '<p>Only the answer is graded — the last <span class="mono">&lt;do&gt;</span> in the reply, matched exactly against the list. Anything else it says is not scored.</p>' : '<p>Prose is graded by a judge model. That is an opinion, and it is shown as one.</p>'}
    ${S.ui.rideError ? `<p class="warn-line" role="alert">${ICON.warn}${esc(S.ui.rideError)}</p>` : ''}
    <div class="stage-foot"><button type="button" class="btn primary" data-do="ride" ${S.ui.rideBusy ? 'aria-busy="true" disabled' : ''}>${S.ui.rideBusy ? 'Loading the ride…' : `Give it the check ride${ICON.arrow}`}</button></div>
  </div>`;
}

function confusions (r, rows) {
  const m = new Map();
  for (const x of rows) {
    if (x.outcome === 'right') continue;
    const k = `${x.expected}→${x.got}`;
    m.set(k, (m.get(k) || 0) + 1);
  }
  return [...m].sort((a, b) => b[1] - a[1]).slice(0, 3).map(([k, c]) => { const [e, g] = k.split('→'); return { e, g, c }; });
}

function functionView (r) {
  const ride = r.ride;
  const s = summarise(ride);
  const prev = ride.previous;
  const f = S.ui.rideFilter;
  const rows = f === 'all' ? ride.rows : ride.rows.filter(x => x.outcome === f);
  const delta = (now, then, lowerIsBetter = false) => {
    if (then == null) return '';
    const d = Math.round((now - then) * 100);
    if (!d) return '<span class="delta">same as last round</span>';
    const good = lowerIsBetter ? d < 0 : d > 0;
    return `<span class="delta ${good ? 'good' : 'bad'}">${d > 0 ? '+' : ''}${d} pts on round ${ride.round - 1}</span>`;
  };
  const conf = confusions(r, ride.rows);
  return `<div class="ride-done">
    <p class="round">Round ${ride.round} · ${n(s.n)} exam rows, asked on this Mac</p>
    ${answerList(r).includes('NONE') ? `<div class="head3">
      <div class="h3cell"><b>${pct(s.accuracy)}</b><span>right on the ${n(s.actions)} rows that needed an action</span>${delta(s.accuracy, prev?.accuracy)}</div>
      <div class="h3cell"><b>${n(s.falsePresses)}</b><span>false presses — acted when it should have waited (of ${n(s.nones)})</span></div>
      <div class="h3cell"><b>${n(s.misses)}</b><span>misses — waited when it should have acted</span></div>
    </div>` : `<div class="head3">
      <div class="h3cell"><b>${pct(s.accuracy)}</b><span>right, of the ${n(s.n)} exam rows</span>${delta(s.accuracy, prev?.accuracy)}</div>
      <div class="h3cell"><b>${n(s.counts.wrong || 0)}</b><span>another answer from the list</span></div>
      <div class="h3cell"><b>${n((s.counts['off-list'] || 0) + (s.counts['no-answer'] || 0))}</b><span>not an answer on the list at all</span></div>
    </div>`}
    <p class="base-line">The same model, untrained: <b>${n(s.baseRight)} of ${n(s.n)}</b> right.${(s.n - s.baseRight) > s.n / 2 ? ' Most of its answers were not on the list at all.' : ''}</p>

    ${conf.length ? `<section class="loop">
      <h3>What it gets wrong most</h3>
      <ul>${conf.map(c => `<li><span>${chip(r, c.e)} ${ICON.arrow} ${c.g === 'null' ? '<span class="fn empty">no answer</span>' : chip(r, c.g)} <b>×${c.c}</b></span>
        <button type="button" class="btn sm" data-do="more-like" data-answer="${esc(c.e)}">Add rows like these</button></li>`).join('')}</ul>
      <p class="hint">More rows where the answer is ${esc(labelOf(r, conf[0].e))}, then teach again. Round ${ride.round + 1} is graded on this same exam.</p>
    </section>` : ''}

    <div class="filter-line">
    <div class="filterbar" role="tablist" aria-label="Show">
      <button type="button" role="tab" id="rf-all" class="flt" aria-selected="${f === 'all'}" data-do="ride-filter" data-filter="all">All <span>${ride.rows.length}</span></button>
      ${Object.entries(s.counts).sort((a, b) => b[1] - a[1]).map(([k, c]) => `<button type="button" role="tab" id="rf-${k}" class="flt" aria-selected="${f === k}" data-do="ride-filter" data-filter="${k}">${OUTCOME[k][0]} <span>${c}</span></button>`).join('')}
    </div>
      <label class="check push" for="show-base"><input type="checkbox" id="show-base" ${S.ui.showBase ? 'checked' : ''} data-change="show-base"> Show the untrained model</label>
    </div>
    <div class="ride-rows">
      ${rows.map(x => `<div class="rr ${OUTCOME[x.outcome][1]}">
        <div class="rr-in">${inputCell(x)}</div>
        <div class="rr-body">
          <div class="rr-line"><span class="rr-k">Expected</span>${chip(r, x.expected)}</div>
          <div class="rr-line"><span class="rr-k">It said</span>${chip(r, x.got, { tone: OUTCOME[x.outcome][1] })}<span class="pill ${OUTCOME[x.outcome][1]}">${OUTCOME[x.outcome][0]}</span></div>
          ${x.think ? `<details class="think-fold"><summary>What it said it was thinking</summary><p>${esc(x.think)}</p><p class="hint">Not graded, and not a reason — the model writes this alongside the answer.</p></details>` : ''}
          ${S.ui.showBase && x.base ? `<div class="rr-line base"><span class="rr-k">Untrained</span>${x.base.got ? `<span class="fn muted">${esc(x.base.raw ? x.base.got : labelOf(r, x.base.got))}</span>` : '<span class="fn empty">no answer</span>'}<span class="pill">${OUTCOME[x.base.outcome][0]}</span></div>` : ''}
        </div>
      </div>`).join('')}
    </div>
  </div>`;
}

/* A Reader's three numbers, each beside the untrained model's, none of
   them a judge's opinion. The measured caution goes with them: refusal is
   trainable, and not yet better than no adapter by more than noise. */
function groundingView (g) {
  const cell = (title, a, an, b, bn, note) => `<div class="h3cell"><b>${n(a)} of ${n(an)}</b><span>${title}${bn ? ` · <span class="untrained">untrained ${n(b)} of ${n(bn)}</span>` : ''}<br><span class="hint">${note}</span></span></div>`;
  return `<section class="grounded-ride">
    <h3>Did it read the passage?</h3>
    <div class="head3">
      ${g.source ? cell('came from the same sentence of the passage as what you wrote', g.source.sentence, g.source.n, g.source.baseSentence, g.source.baseN, `Counted without a judge, so right answers in other words count. ${n(g.source.same)} of them also cite the section, which is what Pigeon checks; the untrained model never learned to cite.`) : ''}
      ${cell('said “not in here” when the passage did not answer', g.refusals.refused, g.refusals.n, g.refusals.baseRefused, g.refusals.baseN, 'Counted as: cited nothing. A refusal row has nothing to cite.')}
      ${cell('answered a covered question without a citation — a false refusal', g.covered.falseRefusals, g.covered.n, g.covered.baseFalseRefusals, g.covered.baseN, `Of the covered rows, ${n(g.covered.unsupported)} cited a section they were not shown${g.covered.baseN ? ` (untrained: ${n(g.covered.baseUnsupported)})` : ''}.`)}
      ${g.swaps ? cell('followed the passage when a number in it was changed', g.swaps.followed, g.swaps.n, g.swaps.baseFollowed, g.swaps.baseN, `${n(g.swaps.memorised)} answered from memory instead${g.swaps.baseN ? ` (untrained: ${n(g.swaps.baseMemorised)})` : ''}. Exam answers that quote a number from their passage, asked again with that number changed — ${n(g.swaps.asked)} asked, counted where the answer states the number when nothing is changed.`) : '<div class="h3cell"><b>—</b><span>the fact swap needs an exam answer that quotes a number from its passage<br><span class="hint">None did this round.</span></span></div>'}
    </div>
    <p class="hint">Measured before this: refusal is trainable, and no adapter yet beats the untrained model at it by more than noise on twenty cases. Read the two columns, not one.</p>
  </section>`;
}

/* Whether the answer leans on the same sentence as what you wrote; on a miss
   the badge's hover shows both sentences. */
function sourceLine (c) {
  const s = c && c.source;
  if (!s) return '';
  if (s.same) return '<span class="badge sec" title="Leans on the same sentence of the passage as what you wrote, and cites its section.">same source</span>';
  if (s.sentence) return '<span class="badge exam" title="Leans on the same sentence as what you wrote, but does not cite its section.">same sentence, no citation</span>';
  const said = (t) => (t ? `“${t}”` : 'no sentence');
  return `<span class="badge bad" title="${esc(`What you wrote leans on: ${said(s.expected)}\nThis answer leans on: ${said(s.got)}`)}">different sentence</span>`;
}

function citeLine (x, who) {
  const c = who ? x.base : x;
  if (!c) return '';
  return citeBadge(x, c) + (x.refusal === false || x.refusal === true ? ` ${sourceLine(c)}` : '');
}

function citeBadge (x, c) {
  const cited = c.cited || [];
  const bad = c.unsupported || [];
  if (!cited.length) return `<span class="badge ${x.refusal ? 'exam' : 'refusal'}" title="${x.refusal ? 'Right: nothing to cite.' : 'A false refusal: the passage answered it.'}">no citation</span>`;
  return bad.length ? `<span class="badge bad" title="Cited a section it was not shown.">cited §${bad.join(', §')} — not shown</span>` : `<span class="badge sec">cited §${cited.join(', §')}</span>`;
}

function proseView (r) {
  const ride = r.ride;
  const s = summarise(ride);
  const g = ride.grounded;
  return `<div class="ride-done">
    <p class="round">Round ${ride.round} · ${n(s.n)} exam rows, asked on this Mac</p>
    ${g ? groundingView(g) : ''}
    <div class="head3">
      <div class="h3cell"><b>${n(s.good)} of ${n(s.n)}</b><span>judged as good as what you wrote — by a model, so an opinion, not a measurement</span></div>
      <div class="h3cell"><b>${n(s.baseGood)} of ${n(s.n)}</b><span>the same model, untrained</span></div>
    </div>
    <div class="ride-rows prose">${ride.rows.map(x => `<div class="rr ${x.passed ? 'good' : 'bad'}">
      <div class="rr-body">
        ${x.question != null ? inputCell(x) : x.input.kind === 'text' ? `<p class="cell-text small">${esc(x.input.text)}</p>` : inputCell(x)}
        <div class="two"><div><span class="rr-k">What you wrote</span><p class="cell-text">${esc(x.expected)}</p></div>
        <div><span class="rr-k">What it wrote · ${x.passed ? 'judged good' : 'judged not good'} ${citeLine(x)}</span><p class="cell-text">${esc(x.got || '')}</p></div></div>
        ${S.ui.showBase && x.base ? `<div><span class="rr-k">Untrained · ${x.base.passed ? 'judged good' : 'judged not good'} ${citeLine(x, true)}</span><p class="cell-text muted">${esc(x.base.got || '')}</p></div>` : ''}
      </div></div>`).join('')}</div>
    <label class="check" for="show-base"><input type="checkbox" id="show-base" ${S.ui.showBase ? 'checked' : ''} data-change="show-base"> Show the untrained model</label>
  </div>`;
}

export function render (r) {
  if (!r.ride || r.ride.version !== r.teach?.version) return notYet(r);
  const body = r.ride.kind === 'prose' ? proseView(r) : functionView(r);
  return `${body}<div class="stage-foot">
    <button type="button" class="btn" data-do="go" data-stage="collect">Back to the table</button>
    <button type="button" class="btn primary" data-do="go" data-stage="send">Send it to Pigeon${ICON.arrow}</button>
  </div>`;
}

export const actions = {
  ride (r) { S.ui.rideFilter = 'all'; engine.checkRide(r); return false; },
  'ride-filter' (r, el) { S.ui.rideFilter = el.dataset.filter; },
  'more-like' (r, el) {
    S.stage = 'collect'; S.tab = null; S.ui.filter = `a:${el.dataset.answer}`; S.ui.page = 1;
    toast(`Showing your ${labelOf(r, el.dataset.answer)} rows. Record or add more like them, then teach again.`);
  }
};

export const changes = {
  'show-base' (r, el) { S.ui.showBase = el.checked; }
};
