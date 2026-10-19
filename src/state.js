/* state.js — CARRIES FORWARD into the redesign.
 *
 * The one state object every screen is drawn from, the rules that decide what
 * a run still needs, and every change a person can make to a run. It mirrors
 * `S` in src/index.html on purpose, so the real app can keep its shape.
 *
 * Nothing in here fetches, trains or records — that is engine.js, behind the
 * seam. So the rules below can be read, and tested, without a model.
 *
 * The rule that shapes the whole flow (from the 2026-09-21 flow review):
 *   ONLY SOMETHING WITH NO SAFE DEFAULT BLOCKS. A conflict, NONE over its cap,
 *   a stray click — each has an obvious default (leave it out, sample it down),
 *   and Review states that default and lets you change it. None of them makes
 *   you do busywork before you may train. What blocks is what cannot be
 *   guessed: too few rows, a click target with no name, an answer with no
 *   reasoning when reasoning was asked for, a table that mixes text and
 *   pictures.
 */

/* A seeded shuffle, so the exam drawn for a table is the same every time the
   same seed draws it. */
export function rng (seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* Flight School's own exam rule (instructor/importset.py): at least 5, at most
   20, a fifth of the rows in between — drawn round-robin across answers so the
   exam covers what the table covers. Drawn again whenever rows arrive, until
   the first time it is taught: from then on every round is graded on the same
   exam, and new rows only ever train. */
export function holdoutCount (n) { return Math.min(20, Math.max(5, Math.floor(n / 5))); }

export function markHoldout (r, seed = 1) {
  const rows = r.rows;
  for (const x of rows) x.holdout = false;
  const pool = rows.filter(x => !x.left && !x.draft);
  const want = holdoutCount(pool.length);
  if (pool.length < 10) return;
  const shuffle = rng(seed);
  /* A Reader's exam is drawn by SECTION, never by row: a held-out question
     whose own section was trained on tests memory, not reading (the baseline notes,
     Step 0b). Whole sections go, with every row asked about them — their
     refusal grafts included. Rows with no section (paired by hand) are
     sections of one. */
  if (r.kind === 'reader') {
    const key = (x) => (x.section != null ? `s${x.section}` : `r${x.id}`);
    const sections = [...new Set(pool.map(key))].sort(() => shuffle() - .5);
    let picked = 0;
    for (const k of sections) {
      if (picked >= want) break;
      for (const x of pool) if (key(x) === k) { x.holdout = true; picked++; }
    }
    return;
  }
  const byAnswer = new Map();
  for (const x of [...pool].sort(() => shuffle() - .5)) {
    if (!byAnswer.has(x.answer)) byAnswer.set(x.answer, []);
    byAnswer.get(x.answer).push(x);
  }
  const lanes = [...byAnswer.values()];
  let picked = 0;
  for (let i = 0; picked < want; i++) {
    const lane = lanes[i % lanes.length];
    const x = lane.shift();
    if (x) { x.holdout = true; picked++; }
    if (lanes.every(l => !l.length)) break;
  }
}

/* A run as it starts: the shape every screen draws from, and what table.json
   holds (src-tauri/src/table.rs). */
export function blankRun (id) {
  return {
    id, name: '', pigeon: null, note: '', edge: false, does: '',
    kind: null,
    answerStyle: 'facts', // a Reader's: 'facts' answers from the passage; 'partner' is a thought partner (docs/THOUGHT-PARTNER-READER.md)
    material: null,   // a Reader's document, as material.rs summarises it: { files, sections, estTokens, whole, version }
    input: null, output: 'function', reasoning: false, grounded: false,
    answerSet: 'record', answers: [], base: null, picture: null, question: null, preset: null,
    recordSide: null, maxSide: null, crop: null, picCrop: null,
    rows: [], actions: [], strays: [], conflicts: [], repeatsKept: [], noneMode: 'sample', alias: {}, sessionsIn: [],
    holdoutSeed: 1, version: 1, reviewSeen: false, nextRow: 1,
    teach: null, ride: null, sent: null, flags: {},
    updated: new Date().toISOString()
  };
}

/* Row ids are per run and never reused, so a row keeps its id for life. */
export const nextRowId = (r) => `r${String(r.nextRow++).padStart(5, '0')}`;

export const STAGES = [
  { id: 'describe', name: 'Describe', does: 'Name the job, and choose what goes in and what comes out.' },
  { id: 'collect', name: 'Collect', does: 'Fill the table — record, paste, import a folder, pair files, or let a model draft.' },
  { id: 'review', name: 'Review', does: 'Check what the table will teach before a run rather than after one. Every suggestion here has a default; only real gaps stop you.' },
  { id: 'teach', name: 'Teach', does: 'Train it, with progress drawn from measured numbers.' },
  { id: 'ride', name: 'Check ride', does: 'Grade it on the exam it has never seen, beside the same model untrained, with each kind of mistake counted on its own.' },
  { id: 'send', name: 'Send', does: 'Pack the .pigeon, see what it weighs, and try it in Pigeon.' }
];
export const stageIndex = (id) => STAGES.findIndex(s => s.id === id);

export const RULES = {
  floor: 30,        // under this it is not a training set; a prompt is offered instead
  readerFloor: 150, // a Reader's rows are drafted from its document, so it can afford more — and needs them (the kinds design note)
  noneCap: 0.20,    // NONE may be at most this share of what trains
  wholeMaterial: 3000 // up to this many estimated tokens, a Reader's whole document is pasted into every ask (material.rs, manifest.py)
};

/* ── kinds ─────────────────────────────────────────────────────────────── *
 * The first question a new run asks: what kind of Pigeon? Each kind fixes
 * some of Describe's choices and hides them; `custom` fixes nothing and shows
 * Describe whole. The manifest keeps the spec's own vocabulary (`grounded`,
 * `classifier`, `freeform`); these names are Flight School's, for the cards
 * and the rail. the kinds design note is the design. */
export const KINDS = [
  { id: 'reader', name: 'Reader', line: 'Answers questions from a document you give it.',
    body: 'The facts come from the document each time it is asked; only the way it answers is trained. It reads your document — it does not memorise it.',
    fix: { input: 'text', output: 'prose', grounded: true }, model: 'text' },
  { id: 'player', name: 'Player', line: 'Watches you use an app or a game, then presses the keys itself.',
    body: 'Record yourself for a few minutes. Every key and click becomes an answer, with the screen from just before it.',
    fix: { input: 'image', output: 'function', answerSet: 'record' }, model: 'vision' },
  { id: 'sorter', name: 'Sorter', line: 'Puts each message, note or request into one of your labels.',
    body: 'Write the labels, then bring examples: a folder per label, pasted rows, or drafts from a model for you to check.',
    fix: { input: 'text', output: 'function', answerSet: 'custom' }, model: 'text' },
  { id: 'looker', name: 'Looker', line: 'Puts each picture into one of your labels.',
    body: 'Write the labels, then import a folder with one sub-folder per label. Coins by type, screenshots by state.',
    fix: { input: 'image', output: 'function', answerSet: 'custom' }, model: 'vision' },
  { id: 'writer', name: 'Writer', line: 'Turns what you give it into what you would have written.',
    body: 'A title from a note, a headline from an article, a reply in your shape. Show it pairs: what went in, what you wrote.',
    fix: { input: 'text', output: 'prose', grounded: false }, model: 'text' },
  { id: 'custom', name: 'Something else', line: 'Choose every part yourself: what goes in, what comes out.',
    body: 'Every dial, for a job the five kinds do not name — or when you are not sure yet.',
    fix: {}, model: null },
  /* A base Pigeon (spec D83): nothing trains. Its own screen (prompted.js),
     not the six stages, because it has no Teach and no exam to hold out. */
  { id: 'prompted', name: 'Prompted', line: 'Packages a model with your system prompt, for other people.',
    body: 'Nothing is trained. Write the system prompt and a few test questions; Flight School asks them, records the answers, and packs a .pigeon.',
    fix: {}, model: 'none' }
];
export const kindOf = (r) => KINDS.find(k => k.id === r?.kind) || null;
/* Which of Describe's fields the kind decided, so Describe can hide them. */
export const fixedFields = (r) => new Set(Object.keys(kindOf(r)?.fix || {}));
/* A run saved before kinds existed: whatever it was, it was built with every
   dial showing, which is what `custom` means. A run never described stays
   kindless and asks. */
export const legacyKind = (t) => (t.kind !== undefined ? t.kind : t.input ? 'custom' : null);
/* The floor a run's table must reach before Review opens. */
export const floorFor = (r) => (r?.kind === 'reader' ? RULES.readerFloor : RULES.floor);

/* ── a Reader's rows ───────────────────────────────────────────────────── *
 * A Reader's row is a passage and a question, and the answer written from
 * that passage:
 *   input.text     `[§n] Title\nbody\n\nquestion` — the exact shape the Pigeon is
 *                  asked in (chunker.rs::material_block), so it trains on it
 *   question       the question alone, for the table to show
 *   section        the section the question is ABOUT (its own)
 *   shown          the ordinals in the input — one, today
 *   sectionTitle   the shown section's title, to notice a document that changed
 *   refusal        the passage does not answer it; the right answer declines,
 *                  says what the passage is about, and cites nothing
 * `materialBlock` is a mirror of chunker.rs::material_block, and the citation
 * readers of chunker.rs::cited_ordinals — one shape, on both sides. */
export const materialBlock = (c) => `[§${c.ordinal}] ${c.title}\n${c.text}`;
const CITE = /\[§(\d+)\]/g;
export function citedOrdinals (answer) {
  const out = [];
  for (const m of String(answer ?? '').matchAll(CITE)) { const k = Number(m[1]); if (!out.includes(k)) out.push(k); }
  return out;
}
export const unsupportedCitations = (answer, shown) => citedOrdinals(answer).filter(k => !shown.includes(k));
const stripCitation = (answer, k) => answer.replace(new RegExp(`\\s*\\[§${k}\\]`, 'g'), '');
export const stripCitations = (answer) => citedOrdinals(answer).reduce(stripCitation, String(answer)).replace(/\s+([.,;:!?])/g, '$1').trim();

/* One row from a section, a question and a drafted answer. An answering row
   ends up citing exactly the section it was shown — a citation of anything
   else is removed, a missing one is added — and a refusal cites nothing, so
   the table can never teach a citation that was not checkable. */
export function readerRow (chunk, question, answer, { refusal = false, section = chunk.ordinal, materialVersion = null } = {}) {
  const shown = [chunk.ordinal];
  let a = String(answer ?? '').trim();
  if (refusal) {
    a = stripCitations(a);
  } else {
    for (const k of unsupportedCitations(a, shown)) a = stripCitation(a, k);
    a = a.replace(/\s+([.,;:!?])/g, '$1').trim();
    if (!citedOrdinals(a).length) a = `${a} [§${chunk.ordinal}]`;
  }
  const q = String(question ?? '').trim();
  return { input: { kind: 'text', text: `${materialBlock(chunk)}\n\n${q}` }, answer: a, question: q, section, shown, sectionTitle: chunk.title, refusal, materialVersion, draft: true };
}

/* Which questions to ask against the WRONG passage, and which passage: the
   neighbouring section, because the near miss — right chapter, wrong
   paragraph — is the case a Reader gets wrong, and a random section is too
   easy to decline (the baseline notes, Step 0b). `drafted` is every answering row's
   { section, question }; the plan is spread across sections, and takes
   enough that refusals are `share` of the whole. */
export function graftPlan (chunks, drafted, share = 0.2) {
  const want = Math.round(drafted.length * share / (1 - share));
  if (!want || chunks.length < 2) return [];
  const bySection = new Map();
  for (const d of drafted) { if (!bySection.has(d.section)) bySection.set(d.section, []); bySection.get(d.section).push(d.question); }
  const sections = [...bySection.keys()].sort((a, b) => a - b);
  const depth = Math.max(...[...bySection.values()].map(q => q.length));
  const candidates = [];
  for (let d = 0; d < depth; d++) for (const k of sections) if (bySection.get(k)[d] != null) candidates.push({ section: k, question: bySection.get(k)[d] });
  const take = Math.min(want, candidates.length);
  const plan = [];
  for (let i = 0; i < take; i++) {
    const c = candidates[Math.floor(i * candidates.length / take)];
    const at = chunks.findIndex(x => x.ordinal === c.section);
    const after = chunks[at + 1], before = chunks[at - 1];
    const shown = after && before ? (i % 2 ? before : after) : (after || before);
    if (shown) plan.push({ section: c.section, question: c.question, shown });
  }
  return plan;
}

/* A Reader row brought in as a pair of files in the shape the drafter writes
   (`[§n] Title`, the section, a blank line, the question) gets the fields a
   drafted row has, so Review's checks and the section-wise exam apply to it.
   A refusal is an answer that cites nothing. The files do not say which
   section a refusal's question is really about, so the shown section stands
   in for it. Anything else comes back as it was. */
export function annotateReaderRow (row) {
  if (row.input?.kind !== 'text') return row;
  const m = /^\[§(\d+)\] ([^\n]*)\n([\s\S]*?)\n\n([^\n]+)$/.exec(String(row.input.text).trim());
  if (!m) return row;
  const n = Number(m[1]);
  return { ...row, question: m[4].trim(), shown: [n], section: n, sectionTitle: m[2].trim(), refusal: citedOrdinals(row.answer).length === 0 };
}

/* Why a drafted row cannot go in, or null. Measured on the first Constitution
   run (2026-09-30): the drafter answered some sections with a bare citation,
   invented a ratification story for Article VII, and asked "what does the
   passage say about" — each of which trained the Pigeon to do the same. A
   draft must answer, must answer FROM its passage, and must ask about the
   subject rather than about the text. */
const CONTENT_STOP = new Set(['about', 'after', 'again', 'because', 'before', 'between', 'could', 'during', 'every', 'other', 'shall', 'should', 'their', 'there', 'these', 'those', 'through', 'under', 'until', 'where', 'which', 'while', 'whose', 'would', 'within', 'without', 'being', 'having']);
const contentWords = (text) => new Set(String(text ?? '').toLowerCase().match(/[a-z]{5,}/g)?.filter(w => !CONTENT_STOP.has(w)) ?? []);
/* A question put by someone looking at the text rather than asking about
   the subject: "this article", "the passage", "mentioned in this preamble",
   "the subject of this section". Seen in the first drafts, 2026-09-30. */
const ABOUT_THE_TEXT = /\b(the|this|that|above|following|given)\s+(passage|section|text|excerpt|document|paragraph|preamble|article|amendment|clause)\b|\bpassage\b|\b(subject|topic|purpose|main (idea|point)) of (this|the)\b|\bmentioned (in|here)\b/i;
export function draftProblems (chunk, question, answer, { refusal = false, style = 'facts' } = {}) {
  const q = String(question ?? '').trim();
  /* The citation, and any stray "§2" the model wrote in its place, do not
     count as an answer; a real one has a word in it. */
  const a = stripCitations(answer).replace(/§\s*\d+/g, '').trim();
  if (!/[a-z]{2,}/i.test(a)) return 'no answer beyond a citation';
  if (style === 'partner') return partnerProblems(chunk, q, a, answer, refusal);
  if (q.length < 8 || !/\?$/.test(q)) return 'not a question';
  if (ABOUT_THE_TEXT.test(q)) return 'asks about the passage instead of the subject';
  if (refusal) {
    if (citedOrdinals(answer).length) return 'a refusal that cites';
    /* A refusal that never says no is an answer from the wrong passage. */
    return /\b(not|no|nothing|neither|never)\b|n['’]t\b/i.test(a) ? null : 'a refusal that does not refuse';
  }
  const have = contentWords(`${chunk.title} ${chunk.text}`);
  const want = [...contentWords(a)];
  if (!want.length) return null;
  const shared = want.filter(w => have.has(w)).length;
  return shared * 3 < want.length ? 'not from the passage' : null;
}

/* A thought partner's row (answerStyle 'partner'). The student's message may
   be a request or a pasted draft, so it need not end in "?"; it still must not
   talk about the passage, which the student has not seen. The reply is the
   shape the base could not keep to with a prompt alone (measured 2026-10-16,
   datasets/thought-partner-pds): at most PARTNER.words words, exactly one
   question and it comes last. An answering reply cites its own section or
   shares a fifth of its words with it; a "not covered" reply says so and
   cites nothing, though it may help from general knowledge. */
export const PARTNER = { words: 130 };
const wordCount = (s) => (String(s).match(/\S+/g) || []).length;
function partnerProblems (chunk, q, a, answer, refusal) {
  if (q.length < 8) return 'not a message';
  if (ABOUT_THE_TEXT.test(q)) return 'asks about the passage instead of the subject';
  if (wordCount(a) > PARTNER.words) return 'too long';
  if ((a.match(/\?/g) || []).length !== 1 || !/\?["”’)]?$/.test(a)) return 'not exactly one question, at the end';
  const cites = citedOrdinals(answer);
  if (refusal) {
    if (cites.length) return 'a refusal that cites';
    return /\b(not|no|nothing|neither|never)\b|n['’]t\b/i.test(a) ? null : 'a refusal that does not refuse';
  }
  if (cites.some(n => n !== chunk.ordinal)) return 'cites a section it was not shown';
  if (cites.includes(chunk.ordinal)) return null;
  const have = contentWords(`${chunk.title} ${chunk.text}`);
  const want = [...contentWords(a)];
  return !want.length || want.filter(w => have.has(w)).length * 5 >= want.length ? null : 'not from the passage';
}

/* The kinds of student message a thought partner's rows are drafted as, in
   the proportions of docs/THOUGHT-PARTNER-READER.md (the sixth kind, "not
   covered", comes from the refusal graft). Dealt round-robin across sections,
   so every kind meets every part of the book. */
export const PARTNER_KINDS = {
  concept: 'says they do not understand an idea from this topic, or asks why it works',
  brainstorm: 'pastes a discussion prompt an instructor might set on this topic and asks where to start',
  feedback: 'pastes two to four sentences of their own draft on this topic, with one weak or overstated claim in it, and asks what you think',
  write: 'asks you to just write their discussion post or reply on this topic for them',
  misconception: 'states something about this topic that is wrong, as if it were true, and asks you to confirm it'
};
const PARTNER_CYCLE = ['concept', 'brainstorm', 'feedback', 'concept', 'misconception', 'brainstorm', 'concept', 'write', 'feedback',
  'concept', 'brainstorm', 'misconception', 'concept', 'feedback', 'write', 'brainstorm', 'concept'];
export const partnerKindsFor = (sectionIndex, per) =>
  Array.from({ length: per }, (_, i) => PARTNER_CYCLE[(sectionIndex * per + i) % PARTNER_CYCLE.length]);

/* `n` sections spread evenly through the document, first and last included,
   in order: drafting from a sample rather than all of a long book. */
export function sampleSections (chunks, n) {
  if (!n || n >= chunks.length) return chunks;
  if (n === 1) return [chunks[0]];
  return Array.from({ length: n }, (_, i) => chunks[Math.round(i * (chunks.length - 1) / (n - 1))]);
}

/* Rows in the table whose answer is a citation and nothing else. */
export const bareRows = (r) => (r.kind !== 'reader' ? [] : tableRows(r).filter(x => x.shown && !stripCitations(x.answer)));

/* Rows in the table whose answer cites a section they were not shown — a
   refusal that cites anything included. Deterministic, no judge. */
export const badCites = (r) => (r.kind !== 'reader' ? [] : tableRows(r).filter(x => x.shown && (x.refusal ? citedOrdinals(x.answer).length : unsupportedCitations(x.answer, x.shown).length)));
/* Rows drafted against a section the document no longer has at that number
   — a file replaced or taken out since. Only known once the sections are
   loaded (`r._chunks`). */
export const staleRows = (r) => (r.kind !== 'reader' || !r._chunks ? [] : tableRows(r).filter(x => x.shown && x.sectionTitle != null && r._chunks.find(c => c.ordinal === x.shown[0])?.title !== x.sectionTitle));
export const refusalRows = (r) => tableRows(r).filter(x => x.refusal);
export function refusalShare (r) { const rows = tableRows(r); return rows.length ? refusalRows(r).length / rows.length : 0; }
export function sectionCoverage (r) {
  const total = r.material?.sections || 0;
  const seen = new Set(tableRows(r).map(x => x.section).filter(k => k != null));
  return { covered: [...seen].filter(k => k >= 1 && k <= total).length, total };
}

/* ── the state ─────────────────────────────────────────────────────────── */

export const S = {
  view: 'runs',
  runId: null,
  stage: null,
  tab: null,
  runs: {},
  ui: {
    notice: null,
    toast: null,          // { text, undo? }
    dialog: null,         // { kind, ... }
    filter: 'all',        // Collect: 'all' | an answer | 'left' | 'exam' | 'drafts' | 'needs-think'
    page: 1,
    editing: null,        // a row id being edited
    adding: false,        // the inline add-a-row form
    conflict: 0,          // index into open conflicts
    target: null,         // selected target id
    merge: [],            // target ids ticked for merging
    grid: true,
    rideFilter: 'all',
    showBase: false,
    repeats: null,        // Review: { state, groups, mode, tol, version } — worked out, never saved
    repeatMode: 'exact',  // 'exact' | 'close'
    repeatTol: 12,        // close: the most any part of two pictures may differ, in grey levels (0–255)
    repeatPage: 1,
    layout: (() => { try { return localStorage.getItem('fs.layout') || 'list'; } catch { return 'list'; } })(),   // Collect: 'list' | 'grid'
    session: null,        // Collect: one session's rows only
    picSize: (() => { try { return localStorage.getItem('fs.picSize') || 's'; } catch { return 's'; } })(),   // Collect: 's' | 'm' | 'l' — how big the table draws pictures
    record: null          // { step, window, keys, clicks, ... } while recording
  }
};

export function reset () {
  S.ui.notice = null; S.ui.toast = null; S.ui.dialog = null; S.ui.record = null;
}

export const run = () => (S.runId ? S.runs[S.runId] : null);

export function allRuns () {
  return Object.values(S.runs).filter(r => !r.edge)
    .sort((a, b) => String(b.updated).localeCompare(String(a.updated)));
}

/* ── answers ───────────────────────────────────────────────────────────── */

/* ── actions ───────────────────────────────────────────────────────────── *
 * What a person DOES while recording becomes an answer — universally, with no
 * idea what app they were using. Two kinds:
 *   key    { id, kind:'key',   raw:'Z', name:'A'|null, uses }   → KEY(A), or KEY(Z) unnamed
 *   click  { id, kind:'click', rect:[x,y,w,h], name|null, uses } → CLICK(Send)
 * A key is already an answer — it is called what was pressed until you rename
 * it. A click has no name until you give it one, and a click with no name is
 * not an answer, so it blocks. Rows store '@<action id>', so a rename renames
 * every row that points at it. */
export const actionById = (r, id) => r.actions.find(t => t.id === id);
export const keyActions = (r) => r.actions.filter(a => a.kind === 'key');
export const clickActions = (r) => r.actions.filter(a => a.kind === 'click');

export function actionLabel (a) {
  if (!a) return 'CLICK(…)';
  return a.kind === 'key' ? `KEY(${a.name || a.raw})` : `CLICK(${a.name || '…'})`;
}

export function labelOf (r, a) {
  if (typeof a === 'string' && a.startsWith('@')) return actionLabel(actionById(r, a.slice(1)));
  return a;
}

/* The closed list this Pigeon may answer with — `labels.txt`, generated,
   never typed by hand. */
export function answerList (r) {
  if (r.output === 'prose') return [];
  const out = [...r.answers.filter(a => a !== 'NONE')];
  for (const t of r.actions) out.push(`@${t.id}`);
  const seen = new Set(r.rows.map(x => x.answer));
  for (const a of seen) if (!out.includes(a) && a !== 'NONE' && !String(a).startsWith('@')) out.push(a);
  if (r.answers.includes('NONE') || seen.has('NONE')) out.push('NONE');
  return out;
}

export const labelsTxt = (r) => answerList(r).map(a => labelOf(r, a)).join('\n');

/* ── the table, as it will train ───────────────────────────────────────── */

const openConflictList = (r) => r.conflicts.filter(c => !c.resolution);
export const openConflicts = (r) => openConflictList(r).length;
export { openConflictList };

/* Rows a person can see as the table — everything not deliberately left out
   and not a draft still waiting for approval. */
export const tableRows = (r) => r.rows.filter(x => !x.left && !x.draft);
export const exam = (r) => tableRows(r).filter(x => x.holdout);
export const drafts = (r) => r.rows.filter(x => x.draft);
export const mismatched = (r) => tableRows(r).filter(x => x.input.kind !== r.input);
export const needsThink = (r) => (r.reasoning && r.output === 'function') ? tableRows(r).filter(x => !x.holdout && !x.think) : [];

/* What would actually train, and why each other row will not. Every number on
   the Ready checklist comes from here, so the checklist cannot disagree with
   the run. */
export function trainingSet (r) {
  const all = tableRows(r).filter(x => !x.holdout);
  const inConflict = new Set();
  for (const c of openConflictList(r)) { inConflict.add(c.a); inConflict.add(c.b); }
  const clean = all.filter(x => !inConflict.has(x.id));
  let use = clean;
  let noneDropped = 0;
  if (r.output !== 'prose' && r.noneMode === 'sample') {
    const none = clean.filter(x => x.answer === 'NONE');
    const rest = clean.length - none.length;
    const allowed = Math.floor(rest * RULES.noneCap / (1 - RULES.noneCap));
    if (none.length > allowed) {
      noneDropped = none.length - allowed;
      const keep = new Set(none.slice(0, allowed).map(x => x.id));
      use = clean.filter(x => x.answer !== 'NONE' || keep.has(x.id));
    }
  }
  return { rows: use, conflictRowsLeftOut: all.length - clean.length, noneDropped };
}

export function noneShare (r, rows = tableRows(r)) {
  if (!rows.length) return 0;
  return rows.filter(x => x.answer === 'NONE').length / rows.length;
}

/* ── what a run still needs ────────────────────────────────────────────── *
 * Each issue: { id, level: 'block' | 'default', text, fix: { stage, tab, filter } }
 * 'default' issues say what will happen if you do nothing. */
export function issues (r) {
  const out = [];
  const n = tableRows(r).length;
  if (!r.kind) out.push({ id: 'kind', level: 'block', text: 'Choose what kind of Pigeon this is.', fix: { stage: 'describe' } });
  if (!r.name.trim()) out.push({ id: 'name', level: 'block', text: 'It needs a name.', fix: { stage: 'describe' } });
  if (!r.input) out.push({ id: 'input', level: 'block', text: 'Choose what goes in: text or a picture.', fix: { stage: 'describe' } });
  if (r.kind === 'reader' && !(r.material?.sections > 0)) out.push({ id: 'material', level: 'block', text: 'Drop the document in first — a Reader answers from it.', fix: { stage: 'collect' } });
  const floor = floorFor(r);
  if (n < floor) out.push({ id: 'floor', level: 'block', text: `It has ${n} ${n === 1 ? 'row' : 'rows'}. ${r.kind === 'reader' ? 'A Reader' : 'A training run'} needs at least ${floor}.`, fix: { stage: 'collect' } });
  const mm = mismatched(r).length;
  if (mm) out.push({ id: 'mixed', level: 'block', text: `${mm} ${mm === 1 ? 'row is' : 'rows are'} ${r.input === 'image' ? 'text in a table of pictures' : 'a picture in a table of text'}. One table takes one kind of input.`, fix: { stage: 'collect', filter: 'mixed' } });
  const unnamed = clickActions(r).filter(t => !t.name).length;
  if (unnamed) out.push({ id: 'actions', level: 'block', text: `${unnamed} ${unnamed === 1 ? 'click has' : 'clicks have'} no name yet — a click with no name is not an answer.`, fix: { stage: 'review', tab: 'actions' } });
  if (cutAfterRecording(r)) out.push({ id: 'recut', level: 'block', text: `These pictures were ${r.picCrop ? 'cut' : 'shrunk'} after they were recorded, so your Pigeon would train on pictures it never sees when it plays. Train them as recorded, or record a new table.`, fix: { stage: 'teach' } });
  const nt = needsThink(r).length;
  if (nt) out.push({ id: 'think', level: 'block', text: `${nt.toLocaleString('en-GB')} rows have no reasoning yet. Reasoning is on, so every practice row needs it.`, fix: { stage: 'collect', filter: 'needs-think' } });
  if (r.kind === 'reader') {
    const bare = bareRows(r).length;
    if (bare) out.push({ id: 'bare', level: 'block', text: `${bare} ${bare === 1 ? 'row has' : 'rows have'} no answer beyond a citation. A Pigeon taught on them learns that a citation alone is an answer.`, fix: { stage: 'collect', filter: 'bare' } });
    const bc = badCites(r).length;
    if (bc) out.push({ id: 'cites', level: 'block', text: `${bc} ${bc === 1 ? 'row cites' : 'rows cite'} a section ${bc === 1 ? 'it was' : 'they were'} not shown. A citation nobody can check is decoration, so it does not train.`, fix: { stage: 'collect', filter: 'bad-cite' } });
    const st = staleRows(r).length;
    if (st) out.push({ id: 'stale', level: 'block', text: `${st} ${st === 1 ? 'row was' : 'rows were'} drafted against an earlier version of the document. Discard ${st === 1 ? 'it' : 'them'} and draft again.`, fix: { stage: 'collect', filter: 'stale' } });
    const share = refusalShare(r);
    if (n && share < 0.05) out.push({ id: 'refusals', level: 'default', text: `${share ? `Only ${Math.round(share * 100)}%` : 'None'} of the rows are refusals. A Reader that never sees “not in here” learns to answer everything, whatever it is shown.`, fix: { stage: 'collect' } });
  }
  const dr = drafts(r).length;
  if (dr) out.push({ id: 'drafts', level: 'default', text: `${dr} model ${dr === 1 ? 'draft is' : 'drafts are'} waiting for you. Unapproved drafts do not train.`, fix: { stage: 'collect', filter: 'drafts' } });

  const oc = openConflicts(r);
  if (oc) out.push({ id: 'conflicts', level: 'default', text: `${oc.toLocaleString('en-GB')} ${oc === 1 ? 'pair of rows disagrees' : 'pairs of rows disagree'}. Unless you settle them, both rows of each pair are left out.`, fix: { stage: 'review', tab: 'conflicts' } });
  const strays = r.strays.filter(s => !s.decision).length;
  if (strays) out.push({ id: 'strays', level: 'default', text: `${strays} ${strays === 1 ? 'click landed' : 'clicks landed'} away from anything else. Unless you make an action of them, they are left out.`, fix: { stage: 'review', tab: 'actions' } });
  if (r.output !== 'prose' && n) {
    const share = noneShare(r);
    if (share > RULES.noneCap) {
      out.push(r.noneMode === 'sample'
        ? { id: 'none', level: 'default', text: `NONE is ${Math.round(share * 100)}% of the rows. It will be sampled down to ${Math.round(RULES.noneCap * 100)}% of what trains.`, fix: { stage: 'review', tab: 'balance' } }
        : { id: 'none', level: 'default', text: `NONE is ${Math.round(share * 100)}% of the rows, and you chose to train on all of them. A Pigeon that always answers NONE will score well and do nothing.`, fix: { stage: 'review', tab: 'balance' } });
    }
  }
  return out;
}

export const blocks = (r) => issues(r).filter(i => i.level === 'block');
export const defaults = (r) => issues(r).filter(i => i.level === 'default');

/* A trained Pigeon goes stale the moment its table changes. */
export const stale = (r) => r.teach?.state === 'done' && r.teach.version !== r.version;

/* ── stage rules ───────────────────────────────────────────────────────── */

export function gate (r, stageId) {
  if (!r) return { open: false, why: 'Open a run first.' };
  const b = blocks(r);
  const before = (ids) => b.find(i => ids.includes(i.id));
  switch (stageId) {
    case 'describe': return { open: true };
    case 'collect': {
      const x = before(['kind', 'name', 'input']);
      return x ? { open: false, why: `Collect opens once Describe is done. ${x.text}` } : { open: true };
    }
    case 'review': {
      const x = before(['kind', 'name', 'input', 'material', 'floor']);
      return x ? { open: false, why: `Review needs a table to look at. ${x.text}` } : { open: true };
    }
    case 'teach': {
      if (r.teach?.state === 'running' || r.teach?.state === 'done' || r.teach?.state === 'failed') return { open: true };
      /* A table cut after recording is settled on Teach itself, so that one
         block opens Teach (to show the choice) and only keeps Start shut. */
      const rest = b.filter(i => i.id !== 'recut');
      return rest.length ? { open: false, why: `Teach opens once nothing blocks it. ${rest[0].text}` } : { open: true };
    }
    case 'ride':
      return r.teach?.state === 'done' ? { open: true } : { open: false, why: 'The check ride needs a finished training run.' };
    case 'send':
      return r.ride?.state === 'done' ? { open: true } : { open: false, why: 'Send opens once it has been through a check ride.' };
    default: return { open: false, why: 'No such stage.' };
  }
}

/* Where a run actually is — worked out from the run, never stored, so it
   cannot drift from the truth. */
export function currentStage (r) {
  if (r.kind === 'prompted') return 'describe';
  const b = blocks(r).map(i => i.id);
  if (b.includes('kind') || b.includes('name') || b.includes('input')) return 'describe';
  if (b.includes('material') || b.includes('floor') || b.includes('mixed') || b.includes('think') || b.includes('cites') || b.includes('stale') || b.includes('bare')) return 'collect';
  if (b.some(id => id !== 'recut')) return 'review';
  if (b.length) return 'teach';
  if (!r.reviewSeen && defaults(r).length && !r.teach) return 'review';
  if (!r.teach || r.teach.state !== 'done' || stale(r)) return 'teach';
  if (!r.ride || r.ride.state !== 'done' || r.ride.version !== r.teach.version) return 'ride';
  return 'send';
}

export function progress (r, stageId) {
  const at = stageIndex(currentStage(r));
  const i = stageIndex(stageId);
  if (r.sent && !stale(r) && currentStage(r) === 'send') return 'done';
  return i < at ? 'done' : i === at ? 'current' : 'ahead';
}

/* The one line the runs list shows for a run: what to do next. */
export function nextStep (r) {
  if (r.kind === 'prompted') {
    if (r.sent) return { text: `Sent to Pigeon as revision ${r.sent.revision}`, stage: 'describe', tone: 'done' };
    if (r.prompt?.packed) return { text: 'Ridden and packed — send it', stage: 'describe', tone: 'good' };
    return { text: promptedReady(r) || 'Ready to ride and pack', stage: 'describe', tone: promptedReady(r) ? '' : 'good' };
  }
  const b = blocks(r);
  if (r.teach?.state === 'running') return { text: `Teaching — ${Math.round(100 * r.teach.iter / r.teach.of)}%`, stage: 'teach', tone: 'busy' };
  if (r.teach?.state === 'failed') return { text: r.teach.outOfMemory ? 'It ran out of memory while teaching — see why' : 'Teaching stopped before it finished — see why', stage: 'teach', tone: 'bad' };
  if (b.length) return { text: b[0].text, stage: b[0].fix.stage, tone: 'warn' };
  if (stale(r)) return { text: 'The table changed since it was taught — teach it again', stage: 'teach', tone: 'warn' };
  const at = currentStage(r);
  if (at === 'review') return { text: `${defaults(r).length} ${defaults(r).length === 1 ? 'suggestion' : 'suggestions'} to look over, then teach`, stage: 'review', tone: '' };
  if (at === 'teach') return { text: 'Ready to teach', stage: 'teach', tone: 'good' };
  if (at === 'ride') return { text: 'Taught — give it the check ride', stage: 'ride', tone: 'good' };
  if (r.sent) return { text: `Sent to Pigeon as revision ${r.sent.revision}`, stage: 'send', tone: 'done' };
  return { text: 'Ready to send to Pigeon', stage: 'send', tone: 'good' };
}

/* ── changes ───────────────────────────────────────────────────────────── *
 * Every change that alters what would train bumps `version`, which is how a
 * trained Pigeon knows it is stale. */

const touch = (r) => { r.version++; r.updated = new Date().toISOString(); };

/* The same id shape Rust gives a run folder (qb-<milliseconds>), made here
   so a new run exists the moment it is started; its first save creates the
   folder. */
export function newRun () {
  let id = `qb-${Date.now()}`;
  while (S.runs[id]) id = `qb-${Number(id.slice(3)) + 1}`;
  S.runs[id] = blankRun(id);
  return id;
}

/* The kind decides the columns. It refuses, saying why, when the table already
   holds rows of the other input — the same rule Describe applies to switching
   text and pictures by hand. Returns null when it took, or the reason. */
export function setKind (r, id) {
  const k = KINDS.find(x => x.id === id);
  if (!k) return 'No such kind.';
  /* By what the run holds, not its kind: Change clears the kind first. */
  if (id === 'prompted' && tableRows(r).length) {
    return 'This run already has rows to train on. Start a new run for a Prompted Pigeon — nothing in it trains.';
  }
  if (id !== 'prompted' && r.prompt?.rows?.some(x => (x.input || '').trim())) {
    return 'This run is a Prompted Pigeon with its own test questions. Start a new run to train one.';
  }
  if (id === 'prompted' && !r.prompt) r.prompt = blankPrompt();
  const rows = tableRows(r).length;
  if (rows && k.fix.input && r.input && k.fix.input !== r.input) {
    return `This table already has ${rows} ${r.input === 'image' ? 'pictures' : 'rows of text'}, and a ${k.name} takes ${k.fix.input === 'image' ? 'pictures' : 'text'}. Start a new run for it — switching here would leave every row out.`;
  }
  r.kind = id;
  if (k.fix.input) setInput(r, k.fix.input);
  if (k.fix.output) setOutput(r, k.fix.output);
  if ('grounded' in k.fix) r.grounded = k.fix.grounded;
  if (k.fix.answerSet) setAnswerSet(r, k.fix.answerSet);
  touch(r);
  return null;
}

/* A Prompted Pigeon's settings: everything base-table.json says
   (python/pigeon/basepack.py), as the page edits them. `shots`, `schema`
   and `hostValues` are kept as typed and read when it is packed. */
export function blankPrompt () {
  return { system: '', template: '{input}', looks: false, temperature: 0.7, memory: 0, output: 'text',
    labels: [], limits: { ...PROMPT_LIMITS }, shots: '', schema: '', hostValues: '', rows: [], packed: null, error: null };
}

/* The longest answer, one per kind of answer, so switching between them
   never rewrites another's: the page shows and the pack writes limitOf(p).
   A page saved before there was one each kept a single `maxTokens`. */
export const PROMPT_LIMITS = { text: 400, label: 8, json: 260 };
export const limitOf = (p) => p.limits?.[p.output] ?? (p.output === 'text' ? p.maxTokens : null) ?? PROMPT_LIMITS[p.output] ?? 400;

/* Why it cannot be packed yet, in the words the page shows, or null. */
export function promptedReady (r) {
  const p = r.prompt || {};
  if (!(r.name || '').trim()) return 'Name it';
  if (!(r.does || '').trim()) return 'Say in one line what it does';
  if (!p.template.includes('{input}')) return 'The question template needs {input}, where the question goes';
  if (p.output === 'label' && p.labels.length < 2) return 'Write at least two labels';
  if (!p.rows.some(x => (x.input || '').trim())) return 'Add a test question';
  if (p.looks && p.rows.some(x => (x.input || '').trim() && !x.image)) return 'Every test question needs its picture';
  return null;
}

/* Back to the question. What was chosen stays until the next kind decides. */
export function clearKind (r) { r.kind = null; touch(r); }

/* A Reader's material, as material.rs just summarised it (null when the last
   file was taken out). The version counts every change, because a row's
   [§n] labels are only right against the index they were drafted from, and
   a taught Pigeon carries the material — so this bumps `version` too. */
export function setMaterial (r, got) {
  r.material = got ? { files: got.files, sections: got.sections, estTokens: got.est_tokens, whole: got.whole, version: (r.material?.version || 0) + 1 } : null;
  touch(r);
}

export function setInput (r, kind) {
  r.input = kind;
  r.base = 'Qwen3-VL-2B';
  /* What every picture row is asked. Recorded rows ask what to do next; a
     folder of photos asks which one it is. One question for every row, so
     Pigeon can ask it the same way when a picture arrives alone. */
  if (kind === 'image' && !r.question) r.question = r.answerSet === 'record' ? 'What do you do next?' : 'Which one is this?';
  touch(r);
}

export function setOutput (r, mode) {
  if (mode === 'reasoned') { r.output = 'function'; r.reasoning = true; } else { r.output = mode; r.reasoning = false; }
  if (r.output === 'prose') r.grounded = r.grounded || false;
  touch(r);
}

/* Where the answers come from: what you do while recording, or a list you
   write. Nothing in between knows what app is being recorded. */
export function setAnswerSet (r, set, custom) {
  r.answerSet = set;
  if (set === 'custom') r.answers = (custom ?? r.answers.filter(a => a !== 'NONE')).filter(Boolean);
  else r.answers = ['NONE'];
  touch(r);
}

export function addRow (r, row) {
  r.rows.push({ id: nextRowId(r), session: null, ...row });
  touch(r);
}

/* Many rows at once — a folder, a pair of folders, a model's drafts. Until
   the first teach the exam is drawn again over the whole table. */
export function addRows (r, rows, session) {
  const added = rows.map(x => ({ id: nextRowId(r), session, fresh: true, ...x }));
  r.rows.push(...added);
  if (!r.teach && !added.every(x => x.draft)) markHoldout(r, r.holdoutSeed);
  touch(r);
  return added;
}

/* Drafts into the table. The exam is drawn again once they are in, because
   a table that came entirely as drafts (a Reader's does) has no exam until
   something approved draws one. */
export function approveRows (r, ids) {
  const set = new Set(ids);
  let k = 0;
  for (const x of r.rows) if (set.has(x.id) && x.draft) { x.draft = false; k++; }
  if (k && !r.teach) markHoldout(r, r.holdoutSeed);
  touch(r);
  return k;
}

export function updateRow (r, id, patch) {
  const x = r.rows.find(y => y.id === id);
  if (!x) return;
  Object.assign(x, patch);
  touch(r);
}

/* Deleting a row is undoable: it is marked, not removed. */
export function leaveOut (r, id, why = 'dropped') { updateRow(r, id, { left: why }); }
export function putBack (r, id) { updateRow(r, id, { left: null }); }

export function resolveConflict (r, cid, how) {
  const c = r.conflicts.find(x => x.id === cid);
  if (!c) return;
  const prev = c.resolution;
  c.resolution = how;
  const rowA = r.rows.find(x => x.id === c.a);
  const rowB = r.rows.find(x => x.id === c.b);
  if (rowA) rowA.left = how === 'b' || how === 'drop' ? 'conflict' : how === 'split' ? 'split' : null;
  if (rowB) rowB.left = how === 'a' || how === 'drop' ? 'conflict' : how === 'split' ? 'split' : null;
  touch(r);
  return prev;
}

export function reopenConflict (r, cid) {
  const c = r.conflicts.find(x => x.id === cid);
  if (!c) return;
  c.resolution = null;
  for (const id of [c.a, c.b]) { const x = r.rows.find(y => y.id === id); if (x && (x.left === 'conflict' || x.left === 'split')) x.left = null; }
  touch(r);
}

/* ── repeats ──────────────────────────────────────────────────────────── *
 * Rows that show the same input and carry the same answer. A recording is
 * full of them — the same key on the same screen, again and again — and every
 * copy after the first teaches nothing new while it weighs the table toward
 * that one moment. (The same input with a DIFFERENT answer is a conflict.)
 *
 * Pictures are judged on the pixels: each picture at 32 × 32 grey levels
 * (fs-recorder `hash`), compared cell by cell. "Exact" allows only JPEG
 * noise; "close" allows any part to differ by up to `tol` grey levels. The
 * 64-bit fingerprint only picks which pairs are worth comparing. Text is the
 * same words, ignoring case and spacing. */

export const REPEAT = { exactTol: 3, candidateBits: 12, closeCandidateBits: 20 };

/* The largest difference between two pictures in any one of their 1,024
   cells, or Infinity as soon as it passes `limit`. */
export function pixelGap (a, b, limit = 255) {
  let most = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] > b[i] ? a[i] - b[i] : b[i] - a[i];
    if (d > most) { most = d; if (most > limit) return Infinity; }
  }
  return most;
}

const popcount = (x) => { x -= (x >>> 1) & 0x55555555; x = (x & 0x33333333) + ((x >>> 2) & 0x33333333); return (((x + (x >>> 4)) & 0x0F0F0F0F) * 0x01010101) >>> 24; };

/* prints: path → { hi, lo, sig } (hi/lo: the fingerprint's two halves;
   sig: 1,024 grey levels). Returns groups, largest first:
   { key, rows, answer, exam } — `key` is the group's first row id. */
/* Two rows that show the same input and carry different answers: the same
   words, or the same pixels — the 64-bit fingerprint only picks which pairs
   to compare (on a game board it cannot see a small sprite move at all), and
   the 32 × 32 signature decides. `[aId, bId, similarity]`, at most maxPairs. */
export function conflictPairs (r, prints = {}, { maxPairs = 2000 } = {}) {
  const rows = r.rows.filter(x => !x.left && !x.draft && !x.holdout && x.answer != null);
  const pairs = [];
  if (r.input === 'text') {
    const seen = new Map();
    for (const x of rows) {
      if (x.input.kind !== 'text') continue;
      const k = String(x.input.text || '').trim().toLowerCase().replace(/\s+/g, ' ');
      if (!seen.has(k)) seen.set(k, []);
      seen.get(k).push(x);
    }
    for (const list of seen.values()) for (let i = 0; i < list.length; i++) for (let j = i + 1; j < list.length; j++) {
      if (list[i].answer !== list[j].answer) pairs.push([list[i].id, list[j].id, 100]);
    }
    return pairs;
  }
  if (r.input !== 'image') return pairs;
  const list = rows.filter(x => x.input.path && prints[x.input.path]);
  for (let i = 0; i < list.length && pairs.length < maxPairs; i++) {
    const a = prints[list[i].input.path];
    for (let j = i + 1; j < list.length && pairs.length < maxPairs; j++) {
      if (list[i].answer === list[j].answer) continue;
      const b = prints[list[j].input.path];
      if (popcount(a.hi ^ b.hi) + popcount(a.lo ^ b.lo) > REPEAT.candidateBits) continue;
      const gap = pixelGap(a.sig, b.sig, REPEAT.exactTol);
      if (gap <= REPEAT.exactTol) pairs.push([list[i].id, list[j].id, Math.round(100 * (1 - gap / 255))]);
    }
  }
  return pairs;
}

export function repeatGroups (r, prints = {}, { mode = 'exact', tol = 12 } = {}) {
  const rows = r.rows.filter(x => !x.left && !x.draft && x.answer != null);
  const parent = new Map(rows.map(x => [x.id, x.id]));
  const find = (id) => { let x = id; while (parent.get(x) !== x) { parent.set(x, parent.get(parent.get(x))); x = parent.get(x); } return x; };
  const join = (a, b) => { const x = find(a), y = find(b); if (x !== y) parent.set(x < y ? y : x, x < y ? x : y); };
  const byAnswer = new Map();
  for (const x of rows) { if (!byAnswer.has(x.answer)) byAnswer.set(x.answer, []); byAnswer.get(x.answer).push(x); }
  const limit = mode === 'close' ? tol : REPEAT.exactTol;
  const near = mode === 'close' ? REPEAT.closeCandidateBits : REPEAT.candidateBits;
  for (const list of byAnswer.values()) {
    if (r.input === 'text') {
      const seen = new Map();
      for (const x of list) {
        if (x.input.kind !== 'text') continue;
        const k = String(x.input.text || '').trim().toLowerCase().replace(/\s+/g, ' ');
        if (seen.has(k)) join(seen.get(k), x.id); else seen.set(k, x.id);
      }
      continue;
    }
    const pics = list.filter(x => x.input.path && prints[x.input.path]);
    for (let i = 0; i < pics.length; i++) {
      const a = prints[pics[i].input.path];
      for (let j = i + 1; j < pics.length; j++) {
        const b = prints[pics[j].input.path];
        if (popcount(a.hi ^ b.hi) + popcount(a.lo ^ b.lo) > near) continue;
        if (find(pics[i].id) === find(pics[j].id)) continue;
        if (pixelGap(a.sig, b.sig, limit) <= limit) join(pics[i].id, pics[j].id);
      }
    }
  }
  const groups = new Map();
  for (const x of rows) { const k = find(x.id); if (!groups.has(k)) groups.set(k, []); groups.get(k).push(x); }
  return [...groups.values()].filter(g => g.length > 1)
    .map(g => ({ key: g[0].id, rows: g, answer: g[0].answer, exam: g.some(x => x.holdout) }))
    .sort((a, b) => b.rows.length - a.rows.length || a.key.localeCompare(b.key));
}

/* The row a group keeps: an exam row if it holds one — so none of the exam's
   copies go on training, which would let the exam be learned — otherwise its
   first. */
export const repeatKeeper = (g) => g.rows.find(x => x.holdout) || g.rows[0];

/* Leaves out every copy but the one kept (exam rows are never left out; they
   are the exam). Returns the ids it left out, for Undo. */
export function keepOneOf (r, g) {
  const keep = repeatKeeper(g);
  const out = g.rows.filter(x => x !== keep && !x.holdout).map(x => x.id);
  for (const id of out) { const x = r.rows.find(y => y.id === id); if (x) x.left = 'repeat'; }
  if (out.length) touch(r);
  return out;
}

export function putBackRows (r, ids) {
  for (const id of ids) { const x = r.rows.find(y => y.id === id); if (x && x.left === 'repeat') x.left = null; }
  touch(r);
}

/* "Keep all" is remembered by the group's first row, so the group stops
   being offered; "Look again" forgets it. Neither changes what trains. */
export function keepAllOf (r, key) { r.repeatsKept = [...new Set([...(r.repeatsKept || []), key])]; }
export function lookAgain (r, key) { r.repeatsKept = (r.repeatsKept || []).filter(k => k !== key); }

export function answerGroup (r, g, answer) {
  for (const x of g.rows) { const y = r.rows.find(z => z.id === x.id); if (y) y.answer = answer; }
  touch(r);
}

/* A name is an answer, and the check ride reads the answer list back out of
   "exactly one of: a, b, c" — so a comma or an "or" would split one name in
   two, and it drops a label of 40 characters or more (instructor/fly.py). */
export function nameProblem (name) {
  const t = String(name || '').trim();
  if (!t) return null;
  if (t.length > 30) return 'At most 30 characters.';
  if (/[(),]/.test(t)) return 'No brackets or commas — the name goes inside KEY(…) or CLICK(…).';
  if (/\bor\b/i.test(t)) return 'Not the word “or” — the answer list is read back split on it.';
  if (t.toUpperCase() === 'NONE') return 'NONE already means “do nothing”.';
  return null;
}

/* A narrower job: its own run, holding the rows set aside for it and the
   actions they answer with. The rows leave this table. */
export function startNarrower (r, rows) {
  const id = newRun();
  const n = S.runs[id];
  const used = new Set(rows.map(x => x.answer).filter(a => String(a).startsWith('@')).map(a => a.slice(1)));
  Object.assign(n, {
    name: '', kind: r.kind, input: r.input, output: r.output, reasoning: r.reasoning, grounded: r.grounded,
    answerSet: r.answerSet, answers: [...r.answers], base: r.base, picture: r.picture, question: r.question,
    window: r.window, noneMode: r.noneMode, note: `Split from ${r.name || 'another run'}.`,
    actions: r.actions.filter(a => used.has(a.id)).map(a => ({ ...a, uses: rows.filter(x => x.answer === `@${a.id}`).length }))
  });
  /* Recorded frames live in the first run's folder; the new run names them
     by where they are. */
  const from = (p) => p && !p.startsWith('/') ? `${S.runsRoot || ''}/${r.id}/${p}` : p;
  n.rows = rows.map(x => ({ ...x, id: nextRowId(n), left: null, holdout: false, fresh: true, input: { ...x.input, ...(x.input.path ? { path: from(x.input.path) } : {}) } }));
  markHoldout(n, n.holdoutSeed);
  r.rows = r.rows.filter(x => x.left !== 'split');
  r.conflicts = r.conflicts.filter(c => c.resolution !== 'split');
  touch(r);
  return id;
}

export function renameAction (r, id, name) {
  const t = actionById(r, id);
  if (!t) return;
  t.name = name.trim() || null;
  touch(r);
}

/* Two actions that mean the same thing become one: two boxes that are really
   one button, or two keys that do the same job (W and the up arrow). */
export function mergeActions (r, ids) {
  const [keepId, ...rest] = ids;
  const keep = actionById(r, keepId);
  if (!keep || !rest.length) return;
  for (const id of rest) {
    const t = actionById(r, id);
    if (!t || t.kind !== keep.kind) continue;
    if (keep.kind === 'click') {
      const [x1, y1, w1, h1] = keep.rect;
      const [x2, y2, w2, h2] = t.rect;
      const x = Math.min(x1, x2), y = Math.min(y1, y2);
      keep.rect = [x, y, Math.max(x1 + w1, x2 + w2) - x, Math.max(y1 + h1, y2 + h2) - y];
    }
    keep.uses += t.uses;
    keep.name = keep.name || t.name;
    for (const row of r.rows) if (row.answer === `@${id}`) row.answer = `@${keepId}`;
    r.actions = r.actions.filter(z => z.id !== id);
    /* A later recording's clicks on the merged place follow it here. */
    (r.alias ||= {})[id] = keepId;
  }
  touch(r);
}

export function decideStray (r, i, decision) {
  const s = r.strays[i];
  if (!s) return;
  s.decision = decision;
  if (decision === 'target') {
    let id = `s${i + 1}`;
    while (actionById(r, id)) id = `${id}x`;
    r.actions.push({ id, kind: 'click', name: null, rect: [s.x - 12, s.y - 12, 24, 24], uses: 1 });
    r.rows.push({ id: nextRowId(r), input: { kind: 'image', path: s.path, click: [s.x, s.y] }, answer: `@${id}`, session: s.session || null });
  }
  touch(r);
}

export function setNoneMode (r, mode) { r.noneMode = mode; touch(r); }

export function reshuffleExam (r) {
  r.holdoutSeed++;
  markHoldout(r, r.holdoutSeed);
  touch(r);
}

/* ── picture size ─────────────────────────────────────────────────────── *
 * The longer side of every picture, in pixels. `recordSide` is what the
 * recorder captured at (every session of a table the same, like frames per
 * row); `maxSide` is what it trains at, when Teach shrinks it further. The
 * trained size goes into the Pigeon (actions.json `frames.max_side`), and
 * Pigeon's eye shrinks what it sees to match — so a Pigeon always sees what
 * it was taught on. Nothing is ever made larger. */
export const SIDES = [768, 512, 384, 256, 160];

/* The size a recorded table was captured at. Null for imported pictures. */
export function sourceSide (r) {
  if (r.input !== 'image' || !(r.sessionsIn || []).length) return null;
  return r.recordSide || 768;
}

/* The size it trains at, and the size its Pigeon sees at. */
export function pictureSide (r) {
  const src = sourceSide(r);
  return Math.min(r.maxSide || src || 768, src || Infinity);
}

/* The size its pictures really come out at: a window smaller than the size
   is kept at its own size, by the recorder and by Pigeon alike — and a table
   cropped after recording (picCrop) keeps only that part of each frame. */
export function seenSide (r, side = pictureSide(r)) {
  const [, , cw, ch] = r.picCrop || [0, 0, 1, 1];
  const win = r.window?.w && r.window?.h ? Math.round(Math.max(r.window.w * cw, r.window.h * ch)) : Infinity;
  return Math.min(side, win);
}

/* A crop inside a crop, as one crop of the whole: the part of the window a
   window to keep when part of a recorded picture is chosen: the picture is
   already `outer` of the window, so `inner` of it is this of the window. Only
   used to start a new recording; a recorded table itself is never cut.
   Both are fractions [x, y, w, h] from the top left; null is all of it. */
export function partOfWindow (outer, inner) {
  if (!inner) return outer || null;
  const [ox, oy, ow, oh] = outer || [0, 0, 1, 1];
  const [ix, iy, iw, ih] = inner;
  return [ox + ix * ow, oy + iy * oh, iw * ow, ih * oh].map(v => +v.toFixed(4));
}

/* A table recorded in Record: its pictures are the recorder's own, made in
   one pass, the way Pigeon makes the pictures it plays from (spec §13.2). So
   they train as recorded: cut or shrunk afterwards, they would be pictures
   Pigeon never sees. A different part or size is a new recording. */
export const recorded = (r) => r.input === 'image' && (r.sessionsIn || []).length > 0;

/* A recorded table cut or shrunk after recording, before that was stopped. */
export function cutAfterRecording (r) {
  return recorded(r) && (Boolean(r.picCrop) || Boolean(r.maxSide && r.maxSide < sourceSide(r)));
}

export function trainAsRecorded (r) {
  r.picCrop = null;
  r.maxSide = null;
  touch(r);
}

/* A new table to record again, with only part of the window (`part`, of a
   recorded picture) or at a smaller size (`side`). It takes the job, not the
   rows: its own sessions make its own pictures. The old table is untouched.
   Returns the new run's id and what Record should start with. */
export function recordAgain (r, { part = null, side = null } = {}) {
  const id = newRun();
  const n = S.runs[id];
  Object.assign(n, {
    name: `${r.name}${part ? ' (part)' : side ? ` (${side} px)` : ''}`,
    kind: r.kind, input: r.input, output: r.output, reasoning: r.reasoning, grounded: r.grounded,
    answerSet: r.answerSet, answers: r.answers.filter(a => !String(a).startsWith('@')),
    base: r.base, picture: r.picture, question: r.question, stack: r.stack || 1, noneMode: r.noneMode,
    window: r.window, note: `Recorded again from ${r.name || 'another run'}${part ? ', with only part of the window' : side ? `, at ${side} px` : ''}.`
  });
  return { id, record: { crop: part ? partOfWindow(r.crop, part) : (r.crop || null), side: side || sourceSide(r) || 768, stack: r.stack || 1 } };
}

/* Cut every picture in the table to part of it, from Collect. It trains cut;
   Pigeon is told the part, and sees it at the size it trained at. */
export function setPicCrop (r, crop) {
  r.picCrop = crop;
  touch(r);
}

export function setPictureSide (r, side) {
  r.maxSide = side;
  touch(r);
}

/* ── words ─────────────────────────────────────────────────────────────── */

export const INPUT_WORDS = { text: 'Text in', image: 'Picture in' };
export function outputWords (r) {
  if (r.kind === 'prompted') return { text: 'Prose out', label: 'A label out', json: 'JSON out' }[r.prompt?.output] || 'Prose out';
  if (r.output === 'prose') return r.grounded ? 'Prose out, from facts you supply' : 'Prose out';
  return r.reasoning ? 'Reasoning, then a function' : 'A function out';
}
