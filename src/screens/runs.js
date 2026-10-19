/* runs.js — CARRIES FORWARD. The home screen: every run, grouped by the
 * Pigeon it makes, each with the ONE thing to do next.
 *
 * Flow-review change: the first version showed which stage a run had reached.
 * That answers "where is it", when the question a person opens the app with
 * is "what do I do now". Each row now says that, and goes straight there. */

import { S, STAGES, allRuns, progress, nextStep, tableRows, INPUT_WORDS, outputWords, kindOf } from '../state.js';
import { summarise } from '../engine.js';
import { esc, ICON, when, n } from './html.js';
import { patch } from './patch.js';
import { KIND_ICON } from './kind.js';

function dots (r) {
  const m = r._summary;
  const at = m ? STAGES.findIndex(s => s.id === (m.stage || m.next?.stage)) : -1;
  const p = (s, i) => !m ? progress(r, s.id) : m.done ? 'done' : i < at ? 'done' : i === at ? 'current' : 'ahead';
  return `<span class="dots" aria-hidden="true">${STAGES.map((s, i) => `<span class="dot ${p(s, i)}"></span>`).join('')}</span>`;
}

function score (r) {
  const m = r._summary;
  if (m) {
    if (!m.score) return '<span class="score none">—</span>';
    return m.score.kind === 'prose'
      ? `<span class="score" title="Judged by a model — an opinion">${m.score.good}/${m.score.n} judged good</span>`
      : `<span class="score">${Math.round(m.score.accuracy * 100)}% right</span>`;
  }
  if (!r.ride || r.ride.state !== 'done') return '<span class="score none">—</span>';
  const s = summarise(r.ride);
  return s.kind === 'prose'
    ? `<span class="score" title="Judged by a model — an opinion">${s.good}/${s.n} judged good</span>`
    : `<span class="score">${Math.round(s.accuracy * 100)}% right</span>`;
}

function row (r) {
  const next = r._summary?.next || nextStep(r);
  const rows = r._summary ? r._summary.rows : tableRows(r).length;
  const k = kindOf(r);
  const named = k && k.id !== 'custom';
  return `<div class="run-item"><a class="run-row" href="#/run/${esc(r.id)}/${esc(next.stage)}">
    <span class="run-kind" title="${esc(named ? k.name : INPUT_WORDS[r.input] || '')}">${named ? KIND_ICON[k.id] : r.input === 'image' ? ICON.image : ICON.text}</span>
    <span class="run-main">
      <span class="run-name">${esc(r.name || 'Untitled run')}</span>
      <span class="run-sub">${named ? `${esc(k.name)} · ` : ''}${esc(INPUT_WORDS[r.input] || 'Not described yet')}${r.input ? ` · ${esc(outputWords(r))}` : ''} · ${n(rows)} rows</span>
    </span>
    <span class="run-next ${next.tone}">${dots(r)}<span>${esc(next.text)}</span></span>
    <span class="run-score">${score(r)}</span>
    <span class="run-when">${esc(when(r.updated))}</span>
  </a><button type="button" class="icon-btn run-more" data-do="run-menu" data-id="${esc(r.id)}" aria-label="More for ${esc(r.name || 'this run')}" title="Rename, duplicate, or move to the Trash">${ICON.more}</button></div>`;
}

export function runsScreen () {
  const runs = allRuns();
  if (S.ui.loadError) {
    return `<section class="screen"><h1>The runs could not be read</h1><p class="warn-line">${esc(S.ui.loadError)}</p>
      <p><button type="button" class="btn primary" data-do="reload">Try again</button></p></section>`;
  }
  if (!runs.length) {
    return `<section class="screen empty-home">
      ${patch('idle', { size: 96, play: true })}
      <h1>No runs yet</h1>
      <p>Every Pigeon starts as a table: what goes in on the left, what should come out on the right. You bring the rows — a recording, a folder, an archive of your own work.</p>
      <button type="button" class="btn primary" data-do="new-run">${ICON.plus}Start your first run</button>
    </section>`;
  }
  const groups = new Map();
  for (const r of runs) {
    const k = r.sent ? r.pigeon : (r.pigeon ? `${r.pigeon} · not sent yet` : 'Not sent yet');
    if (!groups.has(k)) groups.set(k, []);
    groups.get(k).push(r);
  }
  return `<section class="screen">
    <header class="screen-head">
      <div><h1>Runs</h1><p class="lede">Each run is one table becoming one Pigeon. Each row says what to do next.</p></div>
      <button type="button" class="btn primary" data-do="new-run">${ICON.plus}New run</button>
    </header>
    ${[...groups].map(([g, rs]) => `<div class="run-group"><h2 class="group-name">${esc(g)}</h2>
      <div class="run-list">${rs.map(row).join('')}</div></div>`).join('')}
  </section>`;
}
