/* bench.js — every run and every Pigeon on this Mac, in one table with the
 * two columns that matter: the score with its training, and the score with
 * the training removed. The second is what makes the first believable — a
 * score without it cannot tell "it learned" from "the instructions got
 * shorter".
 *
 * Rows come from the lineup (each .pigeon carries its own scorecard) and
 * from the runs, which carry theirs in table.json — or, before a run is
 * opened, in its summary. */

import { S, allRuns } from '../state.js';
import { esc, n, when } from './html.js';
import { refresh as refreshPigeons } from './pigeons.js';

/* What a run's last check ride measured, whether the run is open or a stub. */
function rideOf (r) {
  if (r._summary) return r._summary.bench || null;
  const x = r.ride;
  if (!x || x.state !== 'done') return null;
  return { passed: x.passed, total: x.total, control: x.controlPassed ?? null, judged: (x.gradedByModel || 0) > 0 };
}

function rows () {
  const lineup = S.pigeons || [];
  const inLineup = new Set(lineup.map(p => p.run_id).filter(Boolean));
  const out = lineup.map(p => ({
    id: p.run_id && S.runs[p.run_id] ? p.run_id : null, name: p.name, at: p.added_ms ? new Date(p.added_ms).toISOString() : null,
    passed: p.passed, total: p.total, control: p.control_passed ?? null, judged: (p.graded_by_model || 0) > 0,
    elsewhere: p.source !== 'run', packaged: true
  }));
  for (const r of allRuns()) {
    if (inLineup.has(r.id)) continue;
    const ride = rideOf(r);
    out.push({ id: r.id, name: r.name || 'Untitled run', at: r.updated, ...(ride || {}), taught: Boolean(ride || r.teach), packaged: false });
  }
  return out;
}

export function render () {
  if (S.pigeons == null) refreshPigeons();
  const list = rows();
  const dashes = list.filter(r => r.passed != null && r.control == null).length;
  const cell = (r) => {
    return `<td class="num ${r.passed != null ? 'good' : 'dim'}">${r.passed != null ? `${n(r.passed)}/${n(r.total)}${r.judged ? ' ~' : ''}` : '—'}</td>
      <td class="num ${r.control == null ? 'dim' : ''}">${r.control == null ? '—' : `${n(r.control)}/${n(r.total)}`}</td>`;
  };
  return `<section class="screen">
    <header class="screen-head"><div><h1>Bench</h1><p class="lede">Every run on this Mac, with two columns: the score with its training, and the score with the training removed. The second is what makes the first believable.</p></div></header>
    ${list.length ? `<div class="table bench"><table>
      <thead><tr><th>Run</th><th class="num">With training</th><th class="num">Training removed</th><th class="num">Last</th></tr></thead>
      <tbody>${list.map(r => `<tr>
        <td>${r.id ? `<a href="#/run/${esc(r.id)}/ride">${esc(r.name)}</a>` : `<b>${esc(r.name)}</b>`}
          ${r.elsewhere ? ' <span class="hint" title="Added from a file: its numbers were measured on the Mac that made it.">measured elsewhere</span>' : ''}
          ${r.packaged || r.taught ? '' : ' <span class="hint">not taught yet</span>'}</td>
        ${cell(r)}
        <td class="num hint">${r.at ? esc(when(r.at)) : '—'}</td>
      </tr>`).join('')}</tbody></table></div>`
      : '<div class="panel empty-home"><p><b>Nothing has been taught yet.</b> A run that reaches its check ride appears here with both numbers.</p></div>'}
    <p class="hint"><b>The dashes are the point.</b> ${dashes ? `The training-removed check was never run for ${n(dashes)} of these — they predate it.` : 'Every scored run here has its training-removed number.'} It runs on every new check ride. <b>~</b> marks a score graded by a model rather than exact-matched: weaker evidence, and labelled as such.</p>
  </section>`;
}
