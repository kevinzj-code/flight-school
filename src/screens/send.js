/* send.js — CARRIES FORWARD. Stage 6: pack it and hand it to Pigeon.
 *
 * Flow-review changes folded in:
 *   - Sending again REPLACES the Pigeon as a new revision; it says so, and
 *     says the chats with it keep working.
 *   - What Pigeon will show — the card, the one line of what it does, the
 *     examples — is written and seen here, before it goes, including the
 *     format's rule that "what it does" fits in 60 characters.
 *   - A picture Pigeon needs its base in Pigeon too. That is said before the
 *     hand-off, not discovered after it. */

import { S, exam, stale } from '../state.js';
import { engine } from '../engine.js';
import { esc, ICON, avatar, BUSY } from './html.js';
import { tag, toast, toastError, chip } from './cells.js';
import { refresh as refreshLineup } from './pigeons.js';
import { pref } from './settings.js';

const mb = (x) => x >= 1000 ? `${(x / 1000).toFixed(1)} GB` : x >= 1 ? `${x.toFixed(1)} MB` : '< 1 MB';

/* ── the icon ──────────────────────────────────────────────────────────
   Pigeon shows a Pigeon by its icon, so the icon is made here, where the
   name and the line are: a pigeon drawn from named parts (python/pigeonlook.py)
   — the name's own random look to start, the Icon Maker's when the lineup has
   one, or any mix the person clicks through — or a PNG they choose instead.
   The preview is drawn from the same parts table the packer draws the PNG
   from, so the two cannot disagree. The choice is kept on the run. */
const kits = new Map();      // run id → { recipe, icon } | 'loading'
let skill = null;            // the parts table, the same for every run
const asking = new Set();    // run ids the Icon Maker is drawing for

function lookSrc (recipe) {
  if (!skill || !recipe) return null;
  const parts = skill.head + skill.order.map(k => recipe[k] ? skill.parts[k][recipe[k]] || '' : '').join('');
  return 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(skill.wrapper.replace('{parts}', () => parts))
    .replace(/'/g, '%27').replace(/\(/g, '%28').replace(/\)/g, '%29');
}
const choices = (k) => [...Object.keys(skill.parts[k]), ...(skill.optional.includes(k) ? [null] : [])];
/* An answer that arrives later lands on the run as it is then: the object
   may have been swapped for the whole table since the ask went out. */
const current = (r) => S.runs[r.id] || r;
const iconMaker = () => (S.pigeons || []).find(p => p.skill === 'pigeon-look');

/** What the card shows for this run: its PNG, its look, or (still loading) nothing. */
export function iconSrc (r) {
  const kit = kits.get(r.id);
  if (r.iconChosen && kit && kit !== 'loading' && kit.icon) return kit.icon;
  return lookSrc(r.look?.recipe);
}

function loadKit (r) {
  if (kits.has(r.id)) return;
  kits.set(r.id, 'loading');
  if (S.pigeons == null) refreshLineup();
  engine.lookKit(r).then(k => {
    skill = k.skill;
    kits.set(r.id, { recipe: k.recipe, icon: k.icon });
    const cur = current(r);
    if (!cur.look) { cur.look = { recipe: k.recipe, source: 'random' }; engine.persist(cur); }
    redraw();
  }).catch(e => { kits.set(r.id, { error: String(e) }); redraw(); });
}

/* The Icon Maker is asked once per run, as soon as there is one to ask, and
   again only when asked for. Its answer lands only if nobody has touched the
   icon since: a late answer never overwrites a choice. */
function ask (r, { auto = false } = {}) {
  if (asking.has(r.id) || !iconMaker()) return;
  asking.add(r.id);
  r.look = { ...r.look, asked: true, reason: null };
  engine.persist(r);
  const before = JSON.stringify(r.look.recipe);
  engine.askIconMaker(r).then(res => {
    const cur = current(r);
    if (cur.iconChosen || JSON.stringify(cur.look?.recipe) !== before) return;
    if (res?.source === 'icon-maker' && res.recipe) cur.look = { recipe: res.recipe, source: 'icon-maker', by: res.pigeon, asked: true };
    else cur.look = { ...cur.look, reason: res?.reason || 'the Icon Maker did not answer' };
    engine.persist(cur);
  }).catch(e => { if (!auto) toastError(String(e)); })
    .finally(() => { asking.delete(r.id); redraw(); });
}

function iconField (r) {
  loadKit(r);
  const kit = kits.get(r.id);
  const name = r.pigeon || r.name;
  /* Asked after this paint, not during it. */
  if (pref('iconAsk') === 'auto' && r.look?.source === 'random' && !r.look.asked && !r.iconChosen && skill && iconMaker()) queueMicrotask(() => ask(r, { auto: true }));
  if (!skill || kit === 'loading' || kit?.error) {
    return `<div class="field"><span class="field-label">Its icon</span>
      <div class="look-row">${avatar(name, { size: 88 })}<div class="look-side">
        <p class="hint">${kit?.error ? `The icon tools did not answer — ${esc(kit.error)}. It is packed with a pigeon drawn from its name.` : '<span class="look-note">${BUSY}Getting the parts…</span>'}</p>
      </div></div></div>`;
  }
  const chosen = r.iconChosen && kit.icon;
  const busy = asking.has(r.id);
  const recipe = r.look?.recipe || kit.recipe;
  const note = chosen ? 'Your PNG, as it will ship.'
    : busy ? `<span class="look-note">${BUSY}The Icon Maker is drawing one — change anything and yours stays.</span>`
    : r.look?.source === 'icon-maker' ? `Drawn by ${esc(r.look.by || 'the Icon Maker')} from the name and what it does.`
    : r.look?.source === 'you' ? 'Your mix.'
    : `A random look, picked from the name${r.look?.reason ? ` (${esc(r.look.reason)})` : ''}.`;
  return `<div class="field"><span class="field-label">Its icon</span>
    <div class="look-row">${avatar(name, { size: 88, src: iconSrc(r) })}<div class="look-side">
      ${chosen ? '' : `<div class="look-chips" role="group" aria-label="The parts of the icon">${skill.order.map(k => `<button type="button" class="chip look-chip" data-do="look-slot" data-slot="${esc(k)}" title="Click for the next ${esc(k)}">${esc(k)}<b>${esc(recipe[k] || 'none')}</b></button>`).join('')}</div>`}
      <p class="hint">${note}${chosen ? '' : ' Every part goes with every other; it is drawn at 512 × 512 when you send.'}</p>
      <div class="look-acts">
        ${chosen ? '<button type="button" class="btn sm" data-do="icon-drawn">Use a drawn one</button>' : '<button type="button" class="btn sm" data-do="look-roll">Re-roll</button>'}
        ${!chosen && iconMaker() ? `<button type="button" class="btn sm" data-do="look-ask" ${busy ? 'disabled aria-busy="true"' : ''}>Ask ${esc(iconMaker().name)}</button>` : ''}
        <button type="button" class="btn sm ghost" data-do="icon-pick">${chosen ? 'Choose another PNG…' : 'Choose a PNG…'}</button>
      </div>
    </div></div></div>`;
}

function card (r) {
  const does = r.does ?? '';
  const examples = r.input === 'text' ? exam(r).slice(0, 3) : [];
  return `<div class="pcard">
    <div class="pcard-head">${avatar(r.pigeon || r.name, { src: iconSrc(r) })}
      <div><b>${esc(r.pigeon || r.name)}</b><span>${esc(does || 'What it does goes here')}</span></div></div>
    ${examples.length ? `<div class="pcard-ex">${examples.map(x => `<p><span class="mono">${esc(x.input.text.slice(0, 60))}${x.input.text.length > 60 ? '…' : ''}</span> ${ICON.arrow} ${r.output === 'prose' ? esc(x.answer.slice(0, 40)) + '…' : chip(r, x.answer)}</p>`).join('')}</div>` : '<p class="hint">Picture rows are not copied onto the card — the format leaves them out.</p>'}
  </div>`;
}

function sent (r) {
  return `<div class="sent">
    <p class="ready">${ICON.check}Sent to Pigeon as revision ${r.sent.revision}.</p>
    <div class="pigeon-mock" aria-label="What Pigeon shows">
      <div class="pm-bar"><span>Pigeon</span></div>
      <div class="pm-row">${avatar(r.pigeon || r.name, { src: iconSrc(r) })}<div><b>${esc(r.pigeon || r.name)}</b><span>${esc(r.does || '')}</span></div><span class="pm-new">new</span></div>
      <p class="pm-note">${r.input === 'image' ? `It takes a picture. The first time, Pigeon asks once to download ${esc(r._parts?.base?.name || 'its base')} — shared by every picture Pigeon on it.` : 'It is ready to message.'}</p>
    </div>
    <p class="hint">${r.sent.openedWith ? `Opened in the Pigeon at <span class="mono">${esc(r.sent.openedWith)}</span>` : 'Pigeon has opened it'}: it shows you what it is and asks before adding it. ${r.sent.path ? `The file is also kept at <span class="mono">${esc(r.sent.path)}</span>.` : ''}</p>
    <div class="stage-foot"><a class="btn" href="#/runs">Back to runs</a></div>
  </div>`;
}

export function render (r) {
  if (r.sent && r.sent.version === r.version && S.tab === 'sent') return sent(r);
  const { rows: parts, total, base, projector } = engine.pack(r);
  const doesLen = (r.does || '').length;
  return `<div class="send">
    ${stale(r) ? `<p class="warn-line">${ICON.warn}The table changed since this was taught. This sends the Pigeon as it was taught, not the table as it is now.</p>` : ''}
    <div class="send-grid">
      <div class="send-form">
        <div class="field"><label for="pigeon-name" class="field-label">The Pigeon’s name</label>
          <input id="pigeon-name" class="text-input" type="text" value="${esc(r.pigeon || r.name)}" data-input="pigeon-name" autocomplete="off"></div>
        <div class="field"><label for="pigeon-does" class="field-label">What it does, in a line</label>
          <input id="pigeon-does" class="text-input" type="text" value="${esc(r.does || '')}" maxlength="60" placeholder="e.g. Files receipts by category" data-input="pigeon-does" autocomplete="off">
          <p class="hint"><span data-live="does-count">${doesLen}</span> of 60 characters. Pigeon shows this under its name.</p></div>
        ${iconField(r)}
        <h3>What goes in the file</h3>
        <ul class="parts">${parts.map(p => `<li><span><b>${esc(p.what)}</b><span class="hint">${esc(p.detail)}</span></span><span class="size">${mb(p.mb)} ${tag(p.tag)}</span></li>`).join('')}
          <li class="total"><span><b>${esc((r.pigeon || r.name).replace(/\s+/g, '-').toLowerCase())}.pigeon</b></span><span class="size">about ${mb(total)}</span></li></ul>
        ${projectorNote(projector, base)}
        ${r.sent ? `<p class="hint">Pigeon has revision ${r.sent.revision}. Sending replaces it with revision ${r.sent.revision + 1}; chats with it keep working.</p>` : ''}
      </div>
      <div class="send-card"><p class="field-label">How Pigeon will show it</p>${card(r)}</div>
    </div>
    <div class="stage-foot">
      <button type="button" class="btn" data-do="save-file" ${S.ui.packing ? 'disabled' : ''}>Save file…</button>
      <button type="button" class="btn primary" data-do="try-pigeon" ${S.ui.packing ? 'aria-busy="true" disabled' : ''}>${ICON.send}${S.ui.packing ? 'Packing…' : 'Try it in Pigeon'}</button>
    </div>
  </div>`;
}

/* Under the box: where the projector comes from. Left out of the file, it
   comes with the base; carried, it is most of the file. */
export function projectorNote (projector, base) {
  const named = base ? `${esc(base.name)}, ${mb(base.bytes / 1e6)}` : 'the one it was taught on';
  if (projector === 'stock') return `<p class="hint">The projector is not in the file. It comes with the base, ${esc(base.name)}, which Pigeon downloads once (${mb(base.bytes / 1e6)}).</p>`;
  if (projector === 'carried' || projector === 'unknown') return `<p class="hint">The projector is most of the file — it travels with this Pigeon, so it runs off a USB stick with nothing else installed. Pigeon also needs the base, ${named}, downloaded once.</p>`;
  return '';
}

const redraw = () => window.dispatchEvent(new Event('proto:render'));

function pack (r, toPigeon) {
  if (!(r.does || '').trim()) { toast('Give it a line saying what it does — Pigeon shows it under the name.'); S.ui.focusNext = 'pigeon-does'; return; }
  if (S.ui.packing) return false;
  S.ui.packing = true;
  engine.send(r, toPigeon).then(out => {
    if (!out) return;
    if (toPigeon) { S.tab = 'sent'; location.hash = `#/run/${encodeURIComponent(r.id)}/send/sent`; }
    else toast(`Saved — ${out.path}`);
  }).catch(e => toastError(String(e)))
    .finally(() => { S.ui.packing = false; redraw(); });
  return false;
}

export const actions = {
  'save-file' (r) { return pack(r, false); },
  'try-pigeon' (r) { return pack(r, true); },
  /* The next of one part — clicking round every choice comes back to the first. */
  'look-slot' (r, el) {
    if (!skill || !r.look) return false;
    const k = el.dataset.slot, names = choices(k);
    r.look = { ...r.look, recipe: { ...r.look.recipe, [k]: names[(names.indexOf(r.look.recipe[k] ?? null) + 1) % names.length] }, source: 'you', reason: null };
  },
  'look-roll' (r) {
    if (!skill) return false;
    const pick = (a) => a[Math.floor(Math.random() * a.length)];
    r.look = { ...r.look, recipe: Object.fromEntries(skill.order.map(k => [k, pick(choices(k))])), source: 'you', reason: null };
  },
  'look-ask' (r) { ask(r); },
  'icon-pick' (r) {
    engine.pickIcon(r).then(k => {
      if (!k) return;
      kits.set(r.id, { recipe: k.recipe, icon: k.icon });
      const cur = current(r);
      cur.iconChosen = Boolean(k.icon);
      engine.persist(cur); redraw();
    }).catch(e => toastError(String(e)));
    return false;
  },
  'icon-drawn' (r) { r.iconChosen = false; }
};

export const inputs = {
  'pigeon-name' (r, el) { r.pigeon = el.value; return 'none'; },
  'pigeon-does' (r, el) {
    r.does = el.value;
    const c = document.querySelector('[data-live="does-count"]');
    if (c) c.textContent = el.value.length;
    return 'none';
  }
};

/* The card preview catches up when you leave either field. */
export const blurs = { 'pigeon-name': true, 'pigeon-does': true };
