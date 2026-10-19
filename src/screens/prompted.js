/* prompted.js — a Prompted Pigeon (spec D83): the base model, as it is, with
 * a system prompt, packaged for other people. Nothing trains, so there are
 * no stages: one page says what it is, asks its test questions, and packs.
 *
 * The test questions are the maker's test. They are asked exactly as a
 * Runner asks (python/pigeon/basepack.py): the system prompt, any fixed
 * example turns, then the question. A label or JSON answer with an expected
 * answer is scored; prose is recorded as a sample with no score (§12.2). */

import { S, promptedReady, limitOf, PROMPT_LIMITS } from '../state.js';
import { engine } from '../engine.js';
import { esc, ICON, n, BUSY } from './html.js';
import { toast, toastError } from './cells.js';

const p = (r) => r.prompt;

/* What base-table.json says, from the page's settings. Throws, naming the
   field, when typed JSON does not read. */
export function tableFor (r) {
  const q = p(r);
  const parse = (text, what) => {
    if (!text.trim()) return null;
    try { return JSON.parse(text); } catch { throw new Error(`${what} is not JSON — check its brackets and quotes`); }
  };
  const shots = parse(q.shots, 'The example turns');
  if (shots && !(Array.isArray(shots) && shots.every(t => typeof t?.user === 'string' && typeof t?.pigeon === 'string'))) {
    throw new Error('The example turns are a list of {"user": …, "pigeon": …}, both text');
  }
  const schema = q.output === 'json' ? parse(q.schema, 'The answer\'s shape') : null;
  const hostValues = q.output === 'json' ? q.hostValues.split(',').map(s => s.trim()).filter(Boolean) : [];
  const text = q.output === 'text';
  const kind = q.output === 'label' ? 'classifier' : q.output === 'json' ? 'extractor' : q.system.trim() ? 'persona' : 'freeform';
  return {
    id: (r.name || '').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '') || r.id,
    kind,
    prompt_template: q.looks ? `{image}\n${q.template}` : q.template,
    temperature: text ? Number(q.temperature) || 0 : 0,
    ...(q.system.trim() ? { system_prompt: q.system } : {}),
    ...(shots?.length ? { shots } : {}),
    ...(q.output === 'label' ? { labels: q.labels } : {}),
    ...(text && q.memory > 0 ? { history: { shape: 'as_sent', max_turns: Number(q.memory) } } : {}),
    output_contract: {
      type: q.output, max_tokens: limitOf(q),
      ...(q.output === 'label' || (q.output === 'json' && schema) ? { constrained: true } : {}),
      ...(schema ? { output_schema: schema } : {}),
      ...(hostValues.length ? { host_values: hostValues } : {})
    },
    card: { does: (r.does || '').trim() },
    rows_source: `${n(rowsFor(r).length)} test questions written in Flight School, run ${r.id}`
  };
}

export function rowsFor (r) {
  const q = p(r);
  return q.rows.filter(x => (x.input || '').trim()).map(x => {
    const exp = (x.expected || '').trim();
    let expected;
    if (exp && q.output === 'json') { try { expected = JSON.parse(exp); } catch { throw new Error(`The expected answer to “${x.input.slice(0, 40)}” is not JSON`); } }
    else if (exp && q.output === 'label') expected = exp;
    return { input: x.input, ...(expected !== undefined ? { expected } : {}), ...(x.image ? { image: x.image } : {}) };
  });
}

function rowHtml (r, x, i) {
  const q = p(r);
  const showExpected = q.output !== 'text';
  return `<li class="prompted-row">
    <label class="field-label small" for="pq-${i}">Question ${i + 1}</label>
    <textarea id="pq-${i}" class="text-input" rows="2" data-input="prompted-q" data-i="${i}" spellcheck="true">${esc(x.input || '')}</textarea>
    ${showExpected ? `<label class="field-label small" for="pe-${i}">Expected answer <span class="hint">(optional; scored when given)</span></label>
    <input id="pe-${i}" class="text-input mono-input" type="text" value="${esc(x.expected || '')}" data-input="prompted-e" data-i="${i}" spellcheck="false">` : ''}
    <div class="prompted-row-foot">
      ${q.looks ? `<button type="button" class="btn sm" data-do="prompted-picture" data-i="${i}">${ICON.image}${x.image ? esc(x.image.split('/').pop()) : 'Choose a picture…'}</button>` : ''}
      <button type="button" class="btn sm ghost" data-do="prompted-drop" data-i="${i}" aria-label="Remove question ${i + 1}">${ICON.trash}Remove</button>
    </div>
  </li>`;
}

function result (r) {
  const got = p(r).packed;
  if (!got) return '';
  const c = got.check_ride || {};
  const score = c.total
    ? `<b>${n(c.passed)} of ${n(c.total)}</b> right (higher is better)${c.samples ? `, and ${n(c.samples)} recorded as samples` : ''}`
    : `<b>${n(c.samples || 0)}</b> answers recorded as samples, with no score`;
  return `<div class="prompted-result" role="status">
    <p>${ICON.check}Packed ${esc(got.path.split('/').pop())} · ${(got.bytes / 1024).toFixed(0)} KB, no trained weights.</p>
    <p>Its test: ${score}${c.off_contract ? ` · ${n(c.off_contract)} answered off its contract` : ''}.</p>
    ${got.openedWith ? `<p>Opened in the Pigeon at <span class="mono">${esc(got.openedWith)}</span>.</p>` : ''}
  </div>`;
}

export function render (r) {
  const q = p(r);
  const why = promptedReady(r);
  const busy = S.ui.packing;
  return `<div class="describe prompted">
    <div class="kind-chosen">
      <span class="kind-chosen-icon">${ICON.send}</span>
      <span><b>Prompted</b> — Qwen3-VL-2B as it is, with your system prompt. Nothing is trained.</span>
    </div>

    <div class="field">
      <label for="describe-name" class="field-label">What is it called?</label>
      <input id="describe-name" class="text-input big" type="text" value="${esc(r.name)}" placeholder="e.g. plain-english" data-input="name" autocomplete="off" spellcheck="false">
    </div>
    <div class="field">
      <label for="pigeon-does" class="field-label">What does it do? <span class="hint">One line, shown under its name in Pigeon</span></label>
      <input id="pigeon-does" class="text-input" type="text" maxlength="60" value="${esc(r.does || '')}" data-input="prompted-does" placeholder="e.g. Explains a letter in plain English.">
    </div>

    <div class="field">
      <label for="prompted-system" class="field-label">System prompt</label>
      <p class="hint">Sent before every question, exactly as written. Leave it empty to send none.</p>
      <textarea id="prompted-system" class="text-input" rows="6" data-input="prompted-system">${esc(q.system)}</textarea>
    </div>
    <div class="field">
      <label for="prompted-template" class="field-label">Each question, as it is asked</label>
      <p class="hint"><span class="mono">{input}</span> is where the person's words go. Most need nothing else.</p>
      <input id="prompted-template" class="text-input mono-input" type="text" value="${esc(q.template)}" data-input="prompted-template" spellcheck="false">
      <label class="switch-row" for="prompted-looks">
        <input type="checkbox" id="prompted-looks" ${q.looks ? 'checked' : ''} data-change="prompted-looks">
        <span><b>It looks at a picture</b><br><span class="hint">The picture goes first, then the words. Each test question needs one.</span></span>
      </label>
    </div>

    <fieldset class="choices">
      <legend class="field-label">What comes back</legend>
      <div class="choice-row three">
        ${['text', 'label', 'json'].map(o => `<label class="choice${q.output === o ? ' on' : ''}" for="prompted-out-${o}">
          <input type="radio" id="prompted-out-${o}" name="prompted-out" value="${o}" ${q.output === o ? 'checked' : ''} data-change="prompted-output">
          <span class="choice-title">${o === 'text' ? `${ICON.prose}Prose` : o === 'label' ? `${ICON.fn}One of my labels` : `${ICON.fn}JSON`}</span>
          <span class="choice-body">${o === 'text' ? 'Free text. Recorded as samples, not scored.' : o === 'label' ? 'Exactly one line from your list, held to it as it is written.' : 'An object in the shape you give, scored on the fields an expected answer names.'}</span>
        </label>`).join('')}
      </div>
      ${q.output === 'text' ? `<div class="inline-fields">
        <label class="field-label small" for="prompted-temp">Temperature <input id="prompted-temp" class="text-input num" type="number" min="0" max="2" step="0.1" value="${esc(String(q.temperature))}" data-input="prompted-temp"></label>
        <label class="field-label small" for="prompted-memory">Earlier turns it is sent <input id="prompted-memory" class="text-input num" type="number" min="0" max="50" step="1" value="${esc(String(q.memory))}" data-input="prompted-memory"></label>
      </div>` : ''}
      ${q.output === 'label' ? `<label class="field-label small" for="prompted-labels">Labels, one per line</label>
        <textarea id="prompted-labels" class="text-input mono-input" rows="4" data-input="prompted-labels" spellcheck="false">${esc(q.labels.join('\n'))}</textarea>` : ''}
      ${q.output === 'json' ? `<label class="field-label small" for="prompted-schema">The answer's shape, as a JSON Schema</label>
        <textarea id="prompted-schema" class="text-input mono-input" rows="6" data-input="prompted-schema" spellcheck="false">${esc(q.schema)}</textarea>
        <label class="field-label small" for="prompted-host">Fields whose choices the app supplies with each ask <span class="hint">(optional, comma-separated, e.g. <span class="mono">steps[].skill</span>)</span></label>
        <input id="prompted-host" class="text-input mono-input" type="text" value="${esc(q.hostValues)}" data-input="prompted-host" spellcheck="false">` : ''}
      <label class="field-label small" for="prompted-max">Longest answer, in tokens <input id="prompted-max" class="text-input num" type="number" min="1" max="4096" value="${esc(String(limitOf(q)))}" data-input="prompted-max"></label>
    </fieldset>

    <details class="field"${q.shots.trim() ? ' open' : ''}>
      <summary class="field-label">Fixed example turns <span class="hint">(optional)</span></summary>
      <p class="hint">Sent before every question, after the system prompt, word for word: <span class="mono">[{"user": "…", "pigeon": "…"}]</span>.</p>
      <textarea id="prompted-shots" class="text-input mono-input" rows="5" data-input="prompted-shots" spellcheck="false">${esc(q.shots)}</textarea>
    </details>

    <fieldset class="choices">
      <legend class="field-label">Test questions</legend>
      <p class="hint">Its maker's test. They are asked when it is packed, and every answer goes into the file's report with them.</p>
      <ol class="prompted-rows">${q.rows.map((x, i) => rowHtml(r, x, i)).join('')}</ol>
      <button type="button" class="btn" data-do="prompted-add">${ICON.plus}Add a question</button>
    </fieldset>

    ${q.error ? `<p class="warn-line" role="alert">${ICON.warn}${esc(q.error)}</p>` : ''}
    ${result(r)}

    <div class="stage-foot">
      <p class="hint">${why ? esc(why) + ', then it can be packed.' : `Asks ${n(rowsFor(r).length)} test ${rowsFor(r).length === 1 ? 'question' : 'questions'}, then packs.`}</p>
      <button type="button" class="btn" data-do="prompted-save" ${why || busy ? 'disabled' : ''}>Save file…</button>
      <button type="button" class="btn primary" data-do="prompted-try" ${why || busy ? 'disabled' : ''} ${busy ? 'aria-busy="true"' : ''}>${busy ? `${BUSY}Asking its test questions…` : `${ICON.send}Try it in Pigeon`}</button>
    </div>
  </div>`;
}

function pack (r, toPigeon) {
  if (S.ui.packing || promptedReady(r)) return false;
  let table, rows;
  try { table = tableFor(r); rows = rowsFor(r); } catch (e) { p(r).error = e.message; return; }
  p(r).error = null;
  S.ui.packing = true;
  engine.basePack(r, table, rows, toPigeon).then(out => {
    if (!out) return;
    if (!toPigeon) toast(`Saved — ${out.path}`);
  }).catch(e => { p(r).error = String(e); toastError(String(e)); })
    .finally(() => { S.ui.packing = false; engine.redraw(); });
}

const edit = (r, key, value) => { p(r)[key] = value; p(r).packed = null; r.updated = new Date().toISOString(); };

export const inputs = {
  'prompted-does' (r, el) { r.does = el.value; r.pigeon = r.name; return 'none'; },
  'prompted-system' (r, el) { edit(r, 'system', el.value); return 'none'; },
  'prompted-template' (r, el) { edit(r, 'template', el.value); return 'none'; },
  'prompted-temp' (r, el) { edit(r, 'temperature', Math.min(2, Math.max(0, Number(el.value) || 0))); return 'none'; },
  'prompted-memory' (r, el) { edit(r, 'memory', Math.max(0, Math.round(Number(el.value) || 0))); return 'none'; },
  'prompted-labels' (r, el) { edit(r, 'labels', el.value.split('\n').map(s => s.trim()).filter(Boolean)); return 'none'; },
  'prompted-schema' (r, el) { edit(r, 'schema', el.value); return 'none'; },
  'prompted-host' (r, el) { edit(r, 'hostValues', el.value); return 'none'; },
  'prompted-max' (r, el) {
    edit(r, 'limits', { ...PROMPT_LIMITS, ...p(r).limits, [p(r).output]: Math.max(1, Math.round(Number(el.value) || 1)) });
    return 'none';
  },
  'prompted-shots' (r, el) { edit(r, 'shots', el.value); return 'none'; },
  'prompted-q' (r, el) { p(r).rows[Number(el.dataset.i)].input = el.value; p(r).packed = null; return 'none'; },
  'prompted-e' (r, el) { p(r).rows[Number(el.dataset.i)].expected = el.value; p(r).packed = null; return 'none'; }
};

export const blurs = { 'prompted-does': true, 'prompted-template': true, 'prompted-q': true, 'prompted-labels': true };

export const changes = {
  'prompted-looks' (r, el) { edit(r, 'looks', el.checked); },
  'prompted-output' (r, el) {
    edit(r, 'output', el.value);
    if (el.value !== 'text') p(r).memory = 0;
  }
};

export const actions = {
  'prompted-add' (r) { p(r).rows.push({ input: '', expected: '' }); S.ui.focusNext = `pq-${p(r).rows.length - 1}`; },
  'prompted-drop' (r, el) { p(r).rows.splice(Number(el.dataset.i), 1); p(r).packed = null; },
  'prompted-picture' (r, el) {
    engine.basePicture(r).then(path => {
      if (!path) return;
      p(r).rows[Number(el.dataset.i)].image = path; p(r).packed = null;
      engine.persist(r); engine.redraw();
    }).catch(e => toastError(String(e)));
    return false;
  },
  'prompted-save' (r) { return pack(r, false); },
  'prompted-try' (r) { return pack(r, true); }
};
