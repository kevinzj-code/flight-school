/* describe.js — CARRIES FORWARD. Stage 1: name it, and choose the two columns.
 *
 * Flow-review changes folded in:
 *   - The answers are NOT demanded up front. A recorded job learns them from
 *     what you press and click; a list is optional.
 *   - What goes in is fixed once there are rows. Switching a table of pictures
 *     to text would silently void every row, so it is refused with the reason
 *     and the alternative (start a new run), rather than allowed.
 *   - Every choice shows its cost where you make it, from measured numbers.
 *   - The kind comes first (kind.js). A kind fixes some of these choices and
 *     they are not drawn; Something else draws every one. A run with no kind
 *     yet — an old link, or Change — asks here. */

import { tableRows, setInput, setOutput, setAnswerSet, INPUT_WORDS, kindOf, fixedFields, clearKind } from '../state.js';
import { MEASURED, ITERS } from '../engine.js';
import { esc, ICON, n } from './html.js';
import { tag, toast } from './cells.js';
import { kindCards, KIND_ICON } from './kind.js';

function card (group, value, checked, title, body, meta, disabled = false) {
  const id = `${group}-${value}`;
  return `<label class="choice${checked ? ' on' : ''}${disabled ? ' off' : ''}" for="${id}">
    <input type="radio" id="${id}" name="${group}" value="${value}" ${checked ? 'checked' : ''} ${disabled ? 'disabled' : ''} data-change="${group}">
    <span class="choice-title">${title}</span>
    <span class="choice-body">${body}</span>
    ${meta ? `<span class="choice-meta">${meta}</span>` : ''}
  </label>`;
}

/* The kind, as one line at the top, with the way back to the question. */
function kindLine (r) {
  const k = kindOf(r);
  return `<div class="kind-chosen">
    <span class="kind-chosen-icon">${KIND_ICON[k.id]}</span>
    <span><b>${esc(k.name)}</b> — ${esc(k.line)}</span>
    <button type="button" class="btn sm ghost" data-do="kind-change" title="Back to the kinds. What you chose here stays until the next kind decides.">Change</button>
  </div>`;
}

export function render (r) {
  if (!r.kind) {
    return `<div class="describe">
      <div class="field"><p class="field-label">What kind of Pigeon?</p>
      <p class="hint">Each kind fills the table a different way, and decides what goes in and what comes out.</p></div>
      ${kindCards()}
    </div>`;
  }
  const rows = tableRows(r).length;
  const locked = rows > 0;
  const fixed = fixedFields(r);
  const out = r.output === 'prose' ? 'prose' : r.reasoning ? 'reasoned' : 'function';
  const smallMin = Math.max(1, Math.round(ITERS.standard / MEASURED.qwen3vl2b.small.itPerSec / 60));
  const labelsOnly = fixed.has('answerSet') && r.answerSet === 'custom';

  return `<div class="describe">
    ${kindLine(r)}

    <div class="field">
      <label for="describe-name" class="field-label">What is it called?</label>
      <input id="describe-name" class="text-input big" type="text" value="${esc(r.name)}" placeholder="${esc(placeholder(r))}" data-input="name" autocomplete="off" spellcheck="false">
      <p class="hint">The name of the run. The Pigeon it makes gets its own name when you send it.</p>
    </div>

    ${r.kind === 'reader' ? readerStyle(r, rows) : ''}

    ${fixed.has('input') ? '' : `
    <fieldset class="choices">
      <legend class="field-label">What goes in — the left column</legend>
      ${locked ? `<p class="hint lock-note">${ICON.lock}This table already has ${n(rows)} ${r.input === 'image' ? 'pictures' : 'rows of text'}. To teach from ${r.input === 'image' ? 'text' : 'pictures'} instead, start a new run — switching here would leave every row out.</p>` : ''}
      <div class="choice-row">
        ${card('input', 'text', r.input === 'text', `${ICON.text}Text`, 'A sentence, a message, a whole document.',
          `Trains on Qwen3-VL-2B · about ${Math.max(1, Math.round(ITERS.standard / MEASURED.text.short.itPerSec / 60))} min for short rows ${tag('measured')}`, locked && r.input !== 'text')}
        ${card('input', 'image', r.input === 'image', `${ICON.image}A picture`, 'A window, a screenshot, a photo.',
          `Trains on Qwen3-VL-2B · about ${smallMin} min for small pictures (256×256) ${tag('measured')}; full-size ones not measured yet`, locked && r.input !== 'image')}
      </div>
    </fieldset>`}

    ${fixed.has('output') ? '' : `
    <fieldset class="choices">
      <legend class="field-label">What comes out — the right column</legend>
      <div class="choice-row three">
        ${card('output', 'prose', out === 'prose', `${ICON.prose}Prose`, 'Free text: an article, a summary, a reply.',
          'Graded by a model’s opinion')}
        ${card('output', 'function', out === 'function', `${ICON.fn}A function`, 'One answer from a list you can read: <span class="mono">KEY(Enter)</span>, <span class="mono">CLICK(Send)</span>, <span class="mono">TAG(urgent)</span>.',
          'Graded by exact match · the table can fill itself')}
        ${card('output', 'reasoned', out === 'reasoned', `${ICON.fn}Reasoning, then a function`, 'It explains, then answers. Only the answer is graded. Slower to answer, since every reply is longer.',
          'Optional. Every row needs a line saying why, written in Collect — a recording knows the key, not why. <b>A function</b> needs none.')}
      </div>
      ${out === 'reasoned' && rows ? `<p class="warn-line">${ICON.warn}${n(rows)} rows have no reasoning yet. They will each need it before this can be taught.</p>` : ''}
    </fieldset>`}

    ${r.output === 'prose' ? (fixed.has('grounded') ? '' : `
    <fieldset class="choices">
      <legend class="field-label">Does each answer need facts from a document?</legend>
      <label class="switch-row" for="grounded">
        <input type="checkbox" id="grounded" ${r.grounded ? 'checked' : ''} data-change="grounded">
        <span><b>Yes — the facts come from something I supply each time</b><br>
        <span class="hint">A statement, a manual, a transcript. Only the writing is trained; the facts are handed over with every question, so it cannot invent a figure.</span></span>
      </label>
    </fieldset>`) : labelsOnly ? `
    <fieldset class="choices">
      <legend class="field-label">${r.kind === 'sorter' ? 'Your labels, one per line' : 'Your labels, one per line'}</legend>
      <p class="hint">${r.kind === 'looker' ? 'A folder with one sub-folder per label fills these in too — the folder names become the labels.' : 'Every answer is exactly one of these. “None of these” is allowed unless you say otherwise in Review.'}</p>
      <textarea id="answers-custom" class="text-input mono-input" rows="4" data-input="answers-custom" spellcheck="false" placeholder="${r.kind === 'looker' ? 'penny\nnickel\ndime' : 'TAG(urgent)\nTAG(billing)\nTAG(other)'}">${esc(r.answers.join('\n'))}</textarea>
    </fieldset>` : fixed.has('answerSet') ? '' : `
    <fieldset class="choices">
      <legend class="field-label">Where do the answers come from?</legend>
      <div class="choice-row">
        ${card('answers', 'record', r.answerSet === 'record', 'From what I do', 'Record yourself using any app. Every key you press and every place you click becomes an answer — <span class="mono">KEY(Enter)</span>, <span class="mono">CLICK(Send)</span> — and you name them afterwards.', 'Nothing to set up first')}
        ${card('answers', 'custom', r.answerSet === 'custom', 'My own list', 'Write the answers, one per line — <span class="mono">TAG(result)</span>, <span class="mono">CATEGORY(travel)</span>.', 'For sorting, tagging and routing')}
      </div>
      ${r.answerSet === 'custom' ? `<label class="field-label small" for="answers-custom">Your answers, one per line</label>
        <textarea id="answers-custom" class="text-input mono-input" rows="4" data-input="answers-custom" spellcheck="false">${esc(r.answers.join('\n'))}</textarea>` : ''}
    </fieldset>`}

    <div class="stage-foot">
      <p class="hint">${r.name && r.input ? `${esc(INPUT_WORDS[r.input])}, ${esc(out === 'prose' ? 'prose out' : out === 'reasoned' ? 'reasoning then a function out' : 'a function out')}.` : r.input ? 'Name it to carry on.' : 'Name it and choose what goes in to carry on.'}</p>
      <button type="button" class="btn primary" data-do="go" data-stage="collect">${esc(nextLabel(r))}${ICON.arrow}</button>
    </div>
  </div>`;
}

/* A Reader answers from its document in one of two ways. The choice decides
   how its rows are drafted and checked (engine.draftFromMaterial,
   state.js draftProblems); rows already drafted keep the style they were
   drafted in. */
function readerStyle (r, rows) {
  const style = r.answerStyle || 'facts';
  return `<fieldset class="choices">
      <legend class="field-label">How it answers</legend>
      <div class="choice-row">
        ${card('style', 'facts', style === 'facts', 'Facts', 'Answers the question from the passage in one to three sentences, citing it, and says so when the passage does not answer.', 'For a handbook, a policy, a manual')}
        ${card('style', 'partner', style === 'partner', 'Thought partner', 'Helps someone think it through: explains, offers angles, pushes on a draft, corrects a mistake. Cites the passage, and never writes the work for them. Ends with one question.', 'For course material, a textbook, readings')}
      </div>
      ${rows ? `<p class="hint">The ${n(rows)} rows already in the table keep the style they were drafted in. Drafting again uses this one.</p>` : ''}
      <label class="field-label small" for="describe-does">What it is for <span class="hint">(optional, one line; the drafter reads it)</span></label>
      <input id="describe-does" class="text-input" type="text" value="${esc(r.does || '')}" data-input="does" autocomplete="off" placeholder="${style === 'partner' ? 'e.g. a thought partner for graduate students in a data science course' : 'e.g. answering staff questions about the leave policy'}">
    </fieldset>`;
}

/* A name that suits the kind, so the field is not blank for someone new. */
function placeholder (r) {
  return { reader: 'e.g. handbook-reader', player: 'e.g. tetris-player', sorter: 'e.g. inbox-sorter', looker: 'e.g. coin-looker', writer: 'e.g. headline-writer' }[r.kind] || 'e.g. inbox-helper';
}

function nextLabel (r) {
  return { reader: 'Bring the document', player: 'Record yourself', sorter: 'Collect the rows', looker: 'Import the pictures', writer: 'Show it pairs' }[r.kind] || 'Collect the rows';
}

export const inputs = {
  /* Renaming changes nothing that trains, so it does not bump `version` — a
     taught Pigeon does not go stale over its name. The header follows as you
     type; the rest of the screen catches up when you leave the field. */
  name (r, el) {
    r.name = el.value; r.updated = new Date().toISOString();
    const h = document.querySelector('[data-live="run-name"]');
    if (h) h.innerHTML = el.value ? esc(el.value) : '<span class="untitled">Untitled run</span>';
    return 'none';
  },
  does (r, el) { r.does = el.value; r.updated = new Date().toISOString(); return 'none'; },
  'answers-custom' (r, el) { r.answers = el.value.split('\n').map(s => s.trim()).filter(Boolean); r.version++; return 'none'; }
};

export const blurs = { name: true, does: true };

export const actions = {
  'kind-change' (r) { clearKind(r); }
};

export const changes = {
  input (r, el) { setInput(r, el.value); },
  output (r, el) {
    const before = r.output;
    setOutput(r, el.value);
    const rows = tableRows(r).length;
    if (rows && before !== r.output) {
      toast(r.output === 'prose'
        ? `Prose now: the ${n(rows)} rows keep their answers as free text, and the exam is graded by a model's opinion.`
        : `A function now: each of the ${n(rows)} rows must answer with one item from the list. Check the answers still fit.`);
    }
  },
  answers (r, el) { setAnswerSet(r, el.value); },
  grounded (r, el) { r.grounded = el.checked; r.version++; },
  style (r, el) { r.answerStyle = el.value === 'partner' ? 'partner' : 'facts'; r.updated = new Date().toISOString(); }
};
