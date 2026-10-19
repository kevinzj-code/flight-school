/* kind.js — the first question a new run asks: what kind of Pigeon?
 *
 * Seven cards. Five kinds each fix some of Describe's choices and hide them —
 * a Reader is text in, prose out, grounded, and there is nothing to choose —
 * Something else is Describe whole, and Prompted trains nothing at all: it
 * packages the base model with a system prompt (spec D83, prompted.js). Each card says which model it trains
 * on and about how long, from measured numbers, so the base model is decided
 * by the card rather than by a dial three screens later.
 *
 * Drawn two ways: as the #/new screen, before a run exists, and inside
 * Describe for a run that has no kind yet (an old link, or Change). Both use
 * the one `kind-pick` action in app.js. the kinds design note is the design. */

import { KINDS } from '../state.js';
import { MEASURED, ITERS } from '../engine.js';
import { esc, ICON } from './html.js';
import { tag } from './cells.js';

export const KIND_ICON = { reader: ICON.prose, player: ICON.rec, sorter: ICON.fn, looker: ICON.image, writer: ICON.pen, custom: ICON.wrench, prompted: ICON.send };

const minutes = (itPerSec) => Math.max(1, Math.round(ITERS.standard / itPerSec / 60));

/* The cost line under a card: which model, and the measured minutes. */
function cost (k) {
  if (k.model === 'vision') {
    const { small } = MEASURED.qwen3vl2b;
    return `Trains on Qwen3-VL-2B · about ${minutes(small.itPerSec)} min for small pictures ${tag('measured')}; full-size ones not measured yet`;
  }
  if (k.model === 'text') {
    const m = k.id === 'reader' ? MEASURED.text.long : MEASURED.text.short;
    return `Trains on Qwen3-VL-2B · about ${minutes(m.itPerSec)} min ${k.id === 'reader' ? 'for rows that carry a passage' : 'for short rows'} ${tag('measured')}`;
  }
  if (k.model === 'none') return `Asks Qwen3-VL-2B as it is · no training · a minute or two to ask its test questions`;
  return 'The model follows what you choose';
}

const card = (k, current) => `<button type="button" role="listitem" id="kind-${k.id}" class="choice kind-card${current === k.id ? ' on' : ''}" data-do="kind-pick" data-kind="${k.id}">
      <span class="choice-title">${KIND_ICON[k.id]}${esc(k.name)}</span>
      <span class="kind-line">${esc(k.line)}</span>
      <span class="choice-body">${esc(k.body)}</span>
      <span class="choice-meta">${cost(k)}</span>
    </button>`;

/* The trained kinds in the grid; Prompted on its own row beneath, because it
   is a different thing: a model packaged with a prompt, for other people. */
export function kindCards (current = null) {
  const trained = KINDS.filter(k => k.model !== 'none');
  const as_is = KINDS.filter(k => k.model === 'none');
  return `<div class="kind-grid" role="list">${trained.map(k => card(k, current)).join('')}</div>
    <p class="kind-or">Or package a model as it is</p>
    <div class="kind-grid kind-grid-one" role="list">${as_is.map(k => card(k, current)).join('')}</div>`;
}

/* The #/new screen. */
export function kindScreen () {
  return `<section class="screen kinds">
    <header class="screen-head">
      <div><a class="crumb" href="#/runs">${ICON.back}Runs</a><h1 id="stage-title" tabindex="-1">What kind of Pigeon?</h1>
      <p class="lede">Each kind fills the table a different way. Every kind is taught, given a check ride against the same model untrained, and packed as one .pigeon file — except Prompted, which packages the model as it is.</p></div>
    </header>
    ${kindCards()}
  </section>`;
}
