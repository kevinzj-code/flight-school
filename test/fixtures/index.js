/* fixtures — becomes the redesign's test data.
 *
 * Three full jobs that together touch every screen, one finished run taken from
 * Flight School's measured history, and the edge cases. Rows and answer splits
 * are EXAMPLE data. Anything about speed, memory or quality is either MEASURED
 * (and says so, from the state record or the 2026-09-20 vision runs) or
 * SIMULATED by engine.fake.js (and says so). Nothing is presented as a
 * measurement that was not one.
 */

/* The one measured figure these examples lean on (the chat-titles lesson). */
const MEASURED = { titles: { rows: 276, minutes: 1.9, peakGb: 1.2, compliant: '97% compliant' }, pilotOom: { atIter: 320, peakGb: 10.808 } };
import {
  keyRows, conflictPairs, clickRows, MAIL_CLICKS, MAIL_STRAYS, quarterlyRows, titleRows,
  noteRows, receiptRows, photoRows, walkerRows, markHoldout
} from './rows.js';

/* Key actions as a recording finds them: the key that was pressed (`raw`),
   and the name the person gave it, if any. Keys keep their own name until
   renamed — so these two runs show both: most keys named for what they do in
   the game, Return and Backspace left as they were pressed. */
const keyActions = (pairs) => pairs.map(([raw, name], i) => ({ id: `k${i + 1}`, kind: 'key', raw, name, uses: 0 }));
const idOfRaw = (actions) => (raw) => actions.find(a => a.raw === raw).id;

const base = (over) => ({
  kind: 'custom', edge: false, reasoning: false, grounded: false, answers: [], answerSet: 'record',
  rows: [], actions: [], strays: [], conflicts: [], noneMode: 'sample',
  holdoutSeed: 1, version: 1, reviewSeen: false,
  teach: null, ride: null, sent: null, flags: {}, picture: null,
  ...over
});

function gameboy () {
  const actions = keyActions([['W', 'Up'], ['A', 'Left'], ['S', 'Down'], ['D', 'Right'], ['Z', 'A'], ['X', 'B'], ['Return', null], ['Backspace', null]]);
  const id = idOfRaw(actions);
  const byKey = { NONE: 398, W: 212, S: 176, D: 131, A: 118, Z: 164, X: 49, Return: 27, Backspace: 9 };
  const rows = keyRows(byKey, id);
  for (const a of actions) a.uses = byKey[a.raw];
  const conflicts = conflictPairs(rows, [[`@${id('W')}`, `@${id('S')}`], [`@${id('D')}`, `@${id('A')}`]]);
  markHoldout(rows);
  return base({
    id: 'gameboy', name: 'gb-rpg-mover', pigeon: 'Game Boy RPG',
    note: 'Recorded key presses. Two conflicts, and NONE over its cap.',
    input: 'image', output: 'function', answerSet: 'record', answers: ['NONE'],
    base: 'Qwen3-VL-2B', picture: 'small', rows, actions, conflicts,
    updated: '2026-09-21T10:40:00Z'
  });
}

function mail () {
  const actions = structuredClone(MAIL_CLICKS);
  const rows = clickRows(actions, 26);
  markHoldout(rows);
  return base({
    id: 'mail', name: 'inbox-clicker', pigeon: 'Inbox helper',
    note: 'Recorded clicks. One click still to name, three stray clicks.',
    input: 'image', output: 'function', answerSet: 'record', answers: ['NONE'],
    base: 'Qwen3-VL-2B', picture: 'full', rows, actions, strays: structuredClone(MAIL_STRAYS),
    updated: '2026-09-21T09:05:00Z'
  });
}

function quarterly () {
  const rows = quarterlyRows(142);
  markHoldout(rows);
  return base({
    id: 'quarterly', name: 'quarterly-write-up', pigeon: 'Quarterly write-up',
    note: 'Statements paired with the articles written from them.',
    input: 'text', output: 'prose', grounded: true,
    base: 'Qwen2.5-1.5B', rows, reviewSeen: true,
    updated: '2026-09-20T16:20:00Z'
  });
}

function titles () {
  const rows = titleRows(MEASURED.titles.rows);
  markHoldout(rows);
  return base({
    id: 'titles', name: 'chat-titles', pigeon: 'Chat titler',
    note: 'A finished run, from Flight School’s measured history.',
    input: 'text', output: 'prose', base: 'Qwen2.5-1.5B', rows, reviewSeen: true,
    teach: { state: 'done', minutes: MEASURED.titles.minutes, peakGb: MEASURED.titles.peakGb, measured: true, version: 1, iter: 600, of: 600, loss: [] },
    ride: { state: 'done', kind: 'prose', measured: true, version: 1, round: 1, passed: 19, total: 20, controlPassed: 11, gradedByModel: 20,
      rows: rows.filter(x => x.holdout).map((x, i) => ({ id: x.id, input: x.input, expected: x.answer, got: x.answer, passed: i !== 3, how: 'judged', base: { got: x.answer, passed: i % 2 === 0 } })) },
    sent: { revision: 1, at: '2026-09-18T12:00:00Z' },
    updated: '2026-09-18T12:00:00Z'
  });
}

/* ── edge cases: in the Prototype panel, one screen deep ─────────────────── */

function tiny () {
  return base({
    id: 'tiny', edge: true, name: 'lab-notes-tagger', pigeon: 'Lab notes',
    note: 'Twelve rows — under the floor. Collect offers a prompt instead.',
    input: 'text', output: 'function', answerSet: 'custom', answers: ['TAG(method)', 'TAG(result)', 'TAG(question)'],
    base: 'Qwen2.5-1.5B', rows: noteRows(12), updated: '2026-09-21T08:00:00Z'
  });
}
function mixed () {
  const rows = receiptRows(58, 6);
  return base({
    id: 'mixed', edge: true, name: 'receipt-reader', pigeon: 'Receipts',
    note: 'Six text rows in a table of pictures — refused at the row.',
    input: 'image', output: 'function', answerSet: 'custom', answers: ['CATEGORY(food)', 'CATEGORY(travel)', 'CATEGORY(office)'],
    base: 'Qwen3-VL-2B', picture: 'full', rows, updated: '2026-09-21T08:00:00Z'
  });
}
function oom () {
  const answers = ['BREED(boxer)', 'BREED(beagle)', 'BREED(husky)', 'BREED(poodle)', 'BREED(collie)', 'BREED(terrier)'];
  const rows = photoRows(150, answers);
  markHoldout(rows);
  return base({
    id: 'oom', edge: true, name: 'dog-breeds-pilot', pigeon: 'Dog breeds',
    note: 'The real pilot: out of memory at iteration 320.',
    input: 'image', output: 'function', answerSet: 'custom', answers,
    base: 'Qwen3.5-4B', picture: 'full', rows, reviewSeen: true,
    teach: { state: 'failed', reason: 'out-of-memory', iter: MEASURED.pilotOom.atIter, of: 600, peakGb: MEASURED.pilotOom.peakGb, measured: true, loss: [], checkpoint: 300, version: 1 },
    updated: '2026-08-30T14:00:00Z'
  });
}
function walker () {
  const actions = keyActions([['W', 'Up'], ['A', 'Left'], ['S', 'Down'], ['D', 'Right']]);
  const { rows, conflicts } = walkerRows(240, actions.map(a => a.id));
  for (const a of actions) a.uses = 120;
  markHoldout(rows);
  return base({
    id: 'walker', edge: true, name: 'overworld-walker', pigeon: 'Overworld',
    note: 'Every row conflicts — the picture cannot say which key comes next.',
    input: 'image', output: 'function', answerSet: 'record', answers: [],
    base: 'Qwen3-VL-2B', picture: 'small', rows, actions, conflicts, updated: '2026-09-21T08:00:00Z'
  });
}
function denied () {
  return base({
    id: 'denied', edge: true, name: 'desk-helper', pigeon: 'Desk helper',
    note: 'Recording refused: screen recording is off for Flight School.',
    input: 'image', output: 'function', answerSet: 'record', answers: [],
    base: 'Qwen3-VL-2B', flags: { recordDenied: true },
    updated: '2026-09-21T08:00:00Z'
  });
}

export function buildFixtures () {
  return [gameboy(), mail(), quarterly(), titles(), tiny(), mixed(), oom(), walker(), denied()];
}

/* The list the Prototype panel shows, without building every row twice. */
export const FIXTURE_INDEX = [
  ['gameboy', 'gb-rpg-mover', 'Recorded key presses. Two conflicts, and NONE over its cap.', false],
  ['mail', 'inbox-clicker', 'Recorded clicks. One click still to name, three stray clicks.', false],
  ['quarterly', 'quarterly-write-up', 'Statements paired with the articles written from them.', false],
  ['titles', 'chat-titles', 'A finished run, from Flight School’s measured history.', false],
  ['tiny', 'lab-notes-tagger', 'Twelve rows — under the floor.', true],
  ['mixed', 'receipt-reader', 'Six text rows in a table of pictures.', true],
  ['oom', 'dog-breeds-pilot', 'The real pilot: out of memory at iteration 320.', true],
  ['walker', 'overworld-walker', 'Every row conflicts.', true],
  ['denied', 'desk-helper', 'Screen recording is off for Flight School.', true]
].map(([id, name, note, edge]) => ({ id, name, note, edge }));
