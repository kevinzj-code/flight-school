/* run.js — CARRIES FORWARD. One run: its header, the stage bar, and the open
 * stage. The stage bodies live in their own modules. */

import { S, STAGES, gate, progress, stale, INPUT_WORDS, outputWords, kindOf } from '../state.js';
import { esc, ICON } from './html.js';
import { viewerHtml } from './cells.js';
import { KIND_ICON } from './kind.js';
import * as describe from './describe.js';
import * as collect from './collect.js';
import * as record from './record.js';
import * as review from './review.js';
import * as teach from './teach.js';
import * as ride from './ride.js';
import * as send from './send.js';
import * as prompted from './prompted.js';

export const MODULES = { describe, collect, record, review, teach, ride, send, prompted };

function header (r) {
  const inIcon = r.input === 'image' ? ICON.image : r.input === 'text' ? ICON.text : '';
  const outIcon = r.output === 'prose' ? ICON.prose : ICON.fn;
  const busy = r.teach?.state === 'running' ? `<span class="chip busy">${ICON.spark}Teaching ${Math.round(100 * r.teach.iter / r.teach.of)}%</span>` : '';
  const k = kindOf(r);
  return `<header class="run-head" data-tauri-drag-region>
    <div class="run-title">
      <a class="crumb" href="#/runs">${ICON.back}Runs</a>
      <h1 data-live="run-name">${r.name ? esc(r.name) : '<span class="untitled">Untitled run</span>'}</h1>
    </div>
    <div class="run-chips">
      ${busy}
      ${k && k.id !== 'custom' ? `<span class="chip kind" title="${esc(k.line)}">${KIND_ICON[k.id]}${esc(k.name)}</span>` : ''}
      ${r.input ? `<span class="chip">${inIcon}${esc(INPUT_WORDS[r.input])}</span>` : ''}
      <span class="chip">${outIcon}${esc(outputWords(r))}</span>
      ${r.base ? `<span class="chip base" title="The model it is trained on">${esc(r.base)}</span>` : ''}
      <button type="button" class="icon-btn run-more" data-do="run-menu" data-id="${esc(r.id)}" aria-label="More for this run" title="Rename, duplicate, or move to the Trash">${ICON.more}</button>
    </div>
  </header>`;
}

function stageBar (r) {
  const items = STAGES.map((s, i) => {
    const g = gate(r, s.id);
    const p = progress(r, s.id);
    const here = S.stage === s.id;
    const warn = s.id === 'teach' && stale(r);
    const cls = ['stage-step', p, here ? 'here' : '', g.open ? '' : 'locked', warn ? 'stale' : ''].filter(Boolean).join(' ');
    const mark = !g.open ? ICON.lock : warn ? ICON.warn : p === 'done' ? ICON.check : `<span class="num">${i + 1}</span>`;
    return `<li><button type="button" id="stage-${s.id}" class="${cls}" data-do="stage" data-stage="${s.id}"
      ${here ? 'aria-current="step"' : ''} ${g.open ? '' : `aria-disabled="true" aria-describedby="stage-why"`}>
      <span class="mark">${mark}</span><span class="label">${esc(s.name)}</span></button></li>`;
  }).join('');
  return `<nav class="stage-bar" aria-label="Stages"><ol>${items}</ol></nav>
    <p class="notice" id="stage-why" role="status" ${S.ui.notice ? '' : 'hidden'}>${ICON.lock}<span>${esc(S.ui.notice || '')}</span></p>`;
}

export function runScreen (r) {
  /* A Prompted Pigeon trains nothing, so it has no stages: one page. */
  if (r.kind === 'prompted') {
    return `<section class="screen run-screen">
      ${header(r)}
      <section class="stage-body" aria-labelledby="stage-title">
        <div class="stage-intro"><h2 id="stage-title" tabindex="-1">Package a model</h2>
        <p class="does">Write what it is told, ask it a few test questions, and pack it for other people.</p></div>
        ${prompted.render(r)}
      </section>
    </section>`;
  }
  const id = S.stage === 'collect' && S.tab === 'record' ? 'record' : S.stage;
  const s = STAGES.find(x => x.id === S.stage);
  const g = gate(r, S.stage);
  const mod = MODULES[id];
  return `<section class="screen run-screen">
    ${header(r)}
    ${stageBar(r)}
    <section class="stage-body" aria-labelledby="stage-title">
      ${g.open ? '' : `<p class="by-link">${ICON.lock}<span>Opened by link. For this run the stage is shut: ${esc(g.why)}</span></p>`}
      <div class="stage-intro">
        <h2 id="stage-title" tabindex="-1">${esc(s.name)}</h2>
        <p class="does">${esc(s.does)}</p>
      </div>
      ${mod.render(r)}
    </section>
  </section>`;
}

export function runDialog (r) {
  const d = S.ui.dialog;
  if (!d) return '';
  if (d.kind === 'viewer') return viewerHtml(r, d);
  if (d.kind.startsWith('teach-')) return teach.dialog(r, d);
  return collect.dialog(r, d);
}
