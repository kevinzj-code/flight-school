/* rows.js — becomes test data. Builds each example job's rows, deterministically,
 * so two people opening the same link see the same table.
 *
 * A row is one line of the two-column table:
 *   { id, input: { kind:'image', art, seed, click? } | { kind:'text', text },
 *     answer, think?, session?, holdout?, left?: null|'conflict'|'split'|'dropped' }
 * `answer` is the label string, or '@<targetId>' for a click, so renaming a
 * target renames every row that points at it. */

import { rng } from '../../src/state.js';

let counter = 0;
export const rowId = () => `r${String(++counter).padStart(5, '0')}`;

/* Flight School's own exam rule (instructor/importset.py): at least 5, at most
   20, a fifth of the rows in between — drawn round-robin across answers so the
   exam covers what the table covers. */
export function holdoutCount (n) { return Math.min(20, Math.max(5, Math.floor(n / 5))); }

export function markHoldout (rows, seed = 1) {
  for (const r of rows) r.holdout = false;
  const pool = rows.filter(r => !r.left && !r.draft);
  const want = holdoutCount(pool.length);
  if (pool.length < 10) return;
  const byAnswer = new Map();
  const shuffle = rng(seed);
  for (const r of [...pool].sort(() => shuffle() - .5)) {
    if (!byAnswer.has(r.answer)) byAnswer.set(r.answer, []);
    byAnswer.get(r.answer).push(r);
  }
  const lanes = [...byAnswer.values()];
  let picked = 0;
  for (let i = 0; picked < want; i++) {
    const lane = lanes[i % lanes.length];
    const r = lane.shift();
    if (r) { r.holdout = true; picked++; }
    if (lanes.every(l => !l.length)) break;
  }
}

/* ── recorded key presses: the game example ─────────────────────────────
   The art is example data — a game screen stands in for whatever window was
   recorded. Nothing below the art knows it is a game: the answers are the
   keys that were pressed, as key actions, named or not. */

const SCENES = {
  NONE: ['walk', 'talk'], W: ['cave', 'forest', 'town'], S: ['forest', 'town', 'walk'], A: ['town', 'walk'],
  D: ['walk', 'forest', 'cave'], Z: ['talk', 'battle'], X: ['menu', 'battle'], Return: ['town', 'forest'], Backspace: ['menu']
};
export const sceneFor = (raw) => SCENES[raw] || ['walk'];

/* byKey: { W: 212, …, NONE: 398 }; idOf: raw key → its action id. */
export function keyRows (byKey, idOf, { sessions = 9, seed = 7 } = {}) {
  const r = rng(seed);
  const rows = [];
  for (const [raw, count] of Object.entries(byKey)) {
    const scenes = sceneFor(raw);
    for (let i = 0; i < count; i++) {
      rows.push({
        id: rowId(),
        input: { kind: 'image', art: scenes[i % scenes.length], seed: Math.floor(r() * 1e6) },
        answer: raw === 'NONE' ? 'NONE' : `@${idOf(raw)}`,
        session: `Session ${1 + Math.floor(r() * sessions)}`
      });
    }
  }
  return rows.sort(() => r() - .5);
}

/* Two rows with the SAME frame and different answers — the thing recording
   normal play produces, and Review exists to catch. */
/* pairs: [[answerA, answerB], …] — answers as stored ('@k1'). */
export function conflictPairs (rows, pairs) {
  const out = [];
  for (const [k, [a1, a2]] of pairs.entries()) {
    const seed = 900000 + k;
    const art = k % 2 ? 'town' : 'forest';
    const a = { id: rowId(), input: { kind: 'image', art, seed }, answer: a1, session: 'Session 4' };
    const b = { id: rowId(), input: { kind: 'image', art, seed }, answer: a2, session: 'Session 9' };
    rows.push(a, b);
    out.push({ id: `c${k + 1}`, a: a.id, b: b.id, resolution: null, similarity: 98 - (k % 3) });
  }
  return out;
}

/* ── recorded clicks: the mail example ──────────────────────────────────── */

/* Where clicks landed in the example mail window, gathered into click actions.
   `uses` is how many recorded clicks each one came from. */
export const MAIL_CLICKS = [
  { id: 't1', kind: 'click', name: 'Compose', rect: [16, 54, 138, 34], uses: 88 },
  { id: 't2', kind: 'click', name: 'Top message', rect: [184, 106, 224, 34], uses: 61 },
  { id: 't3', kind: 'click', name: 'Send', rect: [684, 440, 66, 30], uses: 50 },
  { id: 't4', kind: 'click', name: null, rect: [690, 52, 72, 26], uses: 12 }
];
export const MAIL_STRAYS = [{ x: 560, y: 260, decision: null }, { x: 598, y: 312, decision: null }, { x: 512, y: 336, decision: null }];

export function clickRows (clicks, noneCount, seed = 3) {
  const r = rng(seed);
  const rows = [];
  for (const t of clicks) {
    const [x, y, w, h] = t.rect;
    for (let i = 0; i < t.uses; i++) {
      rows.push({
        id: rowId(),
        input: { kind: 'image', art: 'mail', click: [Math.round(x + 8 + r() * (w - 16)), Math.round(y + 6 + r() * (h - 12))] },
        answer: `@${t.id}`,
        session: `Session ${1 + Math.floor(r() * 3)}`
      });
    }
  }
  for (let i = 0; i < noneCount; i++) {
    rows.push({ id: rowId(), input: { kind: 'image', art: 'mail', click: null }, answer: 'NONE', session: `Session ${1 + Math.floor(r() * 3)}` });
  }
  return rows.sort(() => r() - .5);
}

/* ── the quarterly write-up: statements paired with articles ───────────── */

const COMPANIES = ['Northwind Tools', 'Halden Freight', 'Orchard Labs', 'Pike & Wren', 'Lumen Grid', 'Tarn Mills', 'Brightwater Foods', 'Castor Aero'];

export function quarterlyRows (count, seed = 11) {
  const r = rng(seed);
  const rows = [];
  for (let i = 0; i < count; i++) {
    const co = COMPANIES[i % COMPANIES.length];
    const q = `Q${1 + (i % 4)} ${2022 + Math.floor(i / 32)}`;
    const rev = (40 + r() * 400).toFixed(1);
    const growth = (r() * 18 - 3).toFixed(1);
    const recv = (Number(growth) + 4 + r() * 12).toFixed(1);
    const margin = (6 + r() * 20).toFixed(1);
    const cash = (Number(rev) * (0.05 + r() * .15)).toFixed(1);
    rows.push({
      id: rowId(),
      input: { kind: 'text', text: `${co} — ${q} statement. Revenue $${rev}m (${growth >= 0 ? '+' : ''}${growth}% on the year). Operating margin ${margin}%. Receivables up ${recv}%. Cash from operations $${cash}m.` },
      answer: Number(recv) > Number(growth) + 6
        ? `${co}'s quarter turned on one line of the cash-flow statement: receivables grew ${recv}% while revenue grew ${growth}%. When customers take longer to pay than sales take to grow, the margin of ${margin}% flatters the business. Cash from operations, at $${cash}m, is the number to watch.`
        : `${co} had a quiet ${q}. Revenue moved ${growth}% on the year to $${rev}m and receivables kept pace, so the ${margin}% operating margin looks earned rather than borrowed from next quarter. $${cash}m of operating cash backs it up.`,
      session: null
    });
  }
  return rows;
}

/* ── chat titles: Flight School's own finished lesson ──────────────────── */

const CHATS = [
  ['can you help me figure out why my sourdough is so dense every time', 'Dense sourdough fixes'],
  ['what’s the difference between a roth and traditional ira again', 'Roth vs traditional IRA'],
  ['I need a polite way to tell my landlord the heater is broken again', 'Heater repair request'],
  ['how do I get the kids to sleep when it’s still light out', 'Summer bedtime tricks'],
  ['write a short toast for my brother’s wedding, he loves sailing', 'Sailing wedding toast'],
  ['my tomato leaves are turning yellow from the bottom up', 'Yellowing tomato leaves'],
  ['explain what a p-value actually means like I’m 12', 'P-values, simply'],
  ['plan a 3 day trip to lisbon on a budget', 'Budget Lisbon weekend']
];
export function titleRows (count) {
  return Array.from({ length: count }, (_, i) => ({ id: rowId(), input: { kind: 'text', text: CHATS[i % CHATS.length][0] }, answer: CHATS[i % CHATS.length][1], session: null }));
}

/* ── the edge cases ─────────────────────────────────────────────────────── */

const NOTES = [
  ['Ran the assay twice with fresh buffer; same drift at 40 minutes.', 'TAG(result)'],
  ['Why does the control lane smear only on Tuesdays?', 'TAG(question)'],
  ['Switched to the 0.2 µm filter before loading.', 'TAG(method)']
];
export function noteRows (count) {
  return Array.from({ length: count }, (_, i) => ({ id: rowId(), input: { kind: 'text', text: NOTES[i % 3][0] }, answer: NOTES[i % 3][1], session: null }));
}

export function receiptRows (images, texts) {
  const cats = ['CATEGORY(food)', 'CATEGORY(travel)', 'CATEGORY(office)'];
  const rows = Array.from({ length: images }, (_, i) => ({ id: rowId(), input: { kind: 'image', art: 'receipt', seed: i + 1 }, answer: cats[i % 3], session: null }));
  for (let i = 0; i < texts; i++) rows.splice(i * 9 + 4, 0, { id: rowId(), input: { kind: 'text', text: `Taxi to airport, $${38 + i * 3}.00` }, answer: 'CATEGORY(travel)', session: null, mismatch: true });
  return rows;
}

export function photoRows (count, answers) {
  return Array.from({ length: count }, (_, i) => ({ id: rowId(), input: { kind: 'image', art: 'photo', seed: i + 1 }, answer: answers[i % answers.length], session: null }));
}

export function walkerRows (pairs, ids) {
  const rows = [];
  const conflicts = [];
  for (let k = 0; k < pairs; k++) {
    const seed = 500000 + k;
    const a = { id: rowId(), input: { kind: 'image', art: 'forest', seed }, answer: `@${ids[k % 4]}`, session: 'Session 1' };
    const b = { id: rowId(), input: { kind: 'image', art: 'forest', seed }, answer: `@${ids[(k + 2) % 4]}`, session: 'Session 2' };
    rows.push(a, b);
    conflicts.push({ id: `c${k + 1}`, a: a.id, b: b.id, resolution: null, similarity: 99 });
  }
  return { rows, conflicts };
}
