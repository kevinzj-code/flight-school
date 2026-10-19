/* The new screens' rules, checked without a window (node src/test/run.mjs).
 *
 * state.js holds every rule a run is drawn from — what trains, what the exam
 * is, what blocks — and engine.js turns what the pipeline wrote into what the
 * Check ride shows. Both are checked here against real files where they exist
 * (a recording made on this Mac, a check ride run by the pipeline), and
 * against the prototype's example runs for the rules. */
import { readFileSync, existsSync } from 'node:fs';
import { homedir } from 'node:os';

let invokeReply = async () => null;
globalThis.window = {
  __TAURI__: { core: { invoke: (cmd, a) => invokeReply(cmd, a) }, event: { listen: async () => () => {} } },
  addEventListener () {}, dispatchEvent () {}
};

const S_ = await import('../src/state.js');
const { S, trainingSet, exam, gate, currentStage, nextStep, markHoldout, holdoutCount, nameProblem, startNarrower, mergeActions, blankRun, tableRows, answerList, labelOf } = S_;
const { engine, summarise, addRecording, groundingOf, rideMatcher } = await import('../src/engine.js');
const await_import = { groundingOf };
const { buildFixtures } = await import('./fixtures/index.js');

let failed = 0, passed = 0;
const check = (name, fn) => {
  try { const r = fn(); if (r === true || r === undefined) { passed++; } else { failed++; console.log(`  ✗ ${name}\n    ${r}`); } }
  catch (e) { failed++; console.log(`  ✗ ${name}\n    ${e.stack}`); }
};
const checkAsync = async (name, fn) => {
  try { const r = await fn(); if (r === true || r === undefined) passed++; else { failed++; console.log(`  ✗ ${name}\n    ${r}`); } }
  catch (e) { failed++; console.log(`  ✗ ${name}\n    ${e.stack}`); }
};

/* ── a ride's rows find their exam rows ──────────────────────────────── */
check('a ride row finds its picture exactly, then through a resized copy', () => {
  const find = rideMatcher([['q', '/t/imports/set/board.jpg', 'jpg'], ['q', '/t/imports/set/board.png', 'png'],
                            ['q', '/t/recording/s1/f000001.jpg', 'frame']]);
  const got = [find('q', '/t/imports/set/board.jpg'), find('q', '/t/imports/set/board.png'),
               find('q', '/t/.sized/64/imports/set/board.jpg.png'), find('q', '/t/.sized/64/imports/set/board.png.png'),
               find('q', '/t/.sized/64/s1/f000001.jpg.png'), find('q', '/t/.sized/64/recording/s1/f000001.jpg.png'),
               find('other', '/t/imports/set/board.jpg')];
  const want = ['jpg', 'png', 'jpg', 'png', null, 'frame', null];
  return JSON.stringify(got) === JSON.stringify(want) || `got ${JSON.stringify(got)}`;
});

/* ── the rules, on every example run ─────────────────────────────────── */
for (const r of buildFixtures()) {
  check(`${r.id}: every rule answers without throwing`, () => {
    currentStage(r); nextStep(r); trainingSet(r);
    for (const s of ['describe', 'collect', 'review', 'teach', 'ride', 'send']) gate(r, s);
  });
}

check('the exam is a fifth of the table, 5 to 20 rows, drawn across every answer', () => {
  const r = blankRun('t');
  for (let i = 0; i < 60; i++) r.rows.push({ id: `x${i}`, input: { kind: 'text', text: `t${i}` }, answer: ['A', 'B', 'C'][i % 3] });
  markHoldout(r, 1);
  const ex = r.rows.filter(x => x.holdout);
  if (ex.length !== holdoutCount(60) || ex.length !== 12) return `drew ${ex.length}`;
  const answers = new Set(ex.map(x => x.answer));
  return answers.size === 3 || `only ${[...answers]} in the exam`;
});

check('exam rows never train', () => {
  const r = blankRun('t');
  Object.assign(r, { name: 'n', input: 'text' });
  for (let i = 0; i < 40; i++) r.rows.push({ id: `x${i}`, input: { kind: 'text', text: `t${i}` }, answer: i % 2 ? 'A' : 'B' });
  markHoldout(r, 1);
  const ts = trainingSet(r).rows.map(x => x.id);
  return exam(r).every(x => !ts.includes(x.id)) || 'an exam row is in the training set';
});

check('a name that the check ride would split or drop is refused, with why', () => {
  const bad = ['Open, chat', 'Save or quit', 'a(b)', 'x'.repeat(31), 'none'];
  const good = ['Open chat', 'Send', 'Top message', 'x'.repeat(30), 'Ordinary'];
  const b = bad.filter(n => !nameProblem(n));
  const g = good.filter(n => nameProblem(n));
  return (!b.length && !g.length) || `let through: ${b.join(' | ')}; refused: ${g.join(' | ')}`;
});

/* ── a real recording, into a table ──────────────────────────────────── */
const runs = `${homedir()}/Library/Application Support/mode.airoplane.flightschool/quickbuild/runs`;
const recPath = `${runs}/qb-1790016211625/recording.json`;
if (existsSync(recPath)) {
  const book = JSON.parse(readFileSync(recPath, 'utf8'));
  check('a recording becomes a table: keys named after themselves, one-off clicks set aside as strays', () => {
    const r = blankRun('rec');
    Object.assign(r, { name: 'rec', input: 'image', output: 'function', answers: ['NONE'] });
    addRecording(r, book);
    const strays = book.actions.filter(a => a.kind === 'click' && a.uses < 2).length;
    if (r.strays.length !== strays) return `${r.strays.length} strays, want ${strays}`;
    if (r.rows.length !== book.rows.length - strays) return `${r.rows.length} rows`;
    if (!r.actions.some(a => a.kind === 'key' && labelOf(r, `@${a.id}`) === `KEY(${a.raw})`)) return 'a key is not named after itself';
    if (addRecording(r, book) !== 0) return 'the same session was added twice';
    return true;
  });
  check('a later session follows a merge: its clicks land on the place they were merged into', () => {
    const r = blankRun('rec2');
    Object.assign(r, { name: 'rec', input: 'image', output: 'function', answers: ['NONE'] });
    const first = { ...book, sessions: book.sessions.slice(0, 1), rows: book.rows.filter(x => x.session === book.sessions[0].name) };
    addRecording(r, first);
    const keys = r.actions.filter(a => a.kind === 'key');
    if (keys.length < 2) return 'too few keys in session 1 to merge';
    mergeActions(r, [keys[0].id, keys[1].id]);
    addRecording(r, book);
    return r.rows.every(x => x.answer !== `@${keys[1].id}`) || 'a row still points at the merged-away key';
  });
}

check('a narrower job is a new run holding the rows set aside for it', () => {
  S.runs = {};
  const r = blankRun('big'); S.runs.big = r;
  Object.assign(r, { name: 'big', input: 'text', output: 'function', answers: ['A', 'B'] });
  for (let i = 0; i < 40; i++) r.rows.push({ id: `x${i}`, input: { kind: 'text', text: `t${i}` }, answer: i % 2 ? 'A' : 'B', left: i < 6 ? 'split' : null });
  const id = startNarrower(r, r.rows.filter(x => x.left === 'split'));
  const n = S.runs[id];
  if (!n || n.rows.length !== 6) return `new run has ${n?.rows.length}`;
  return (r.rows.length === 34 && n.rows.every(x => !x.left)) || 'rows did not move';
});

/* ── a real check ride, read back ────────────────────────────────────── */
const ridePath = process.env.FS_RIDE_RUN;
if (ridePath && existsSync(`${ridePath}/export/check-ride.json`)) {
  await checkAsync('a check ride the pipeline wrote reads back as outcomes, with the untrained model beside it', async () => {
    const ride = JSON.parse(readFileSync(`${ridePath}/export/check-ride.json`, 'utf8'));
    const examRows = readFileSync(`${ridePath}/exam.jsonl`, 'utf8').trim().split('\n').map(l => JSON.parse(l));
    const r = blankRun('ride'); S.runs.ride = r;
    const labels = readFileSync(`${ridePath}/labels.txt`, 'utf8').trim().split('\n');
    Object.assign(r, { name: 'ride', input: examRows[0].image ? 'image' : 'text', output: 'function', answers: labels, question: examRows[0].image ? examRows[0].question : null,
      teach: { state: 'done', version: 1 } });
    r.rows = examRows.map((x, i) => ({ id: `e${i}`, input: x.image ? { kind: 'image', path: x.image } : { kind: 'text', text: x.question }, answer: x.answer, holdout: true }));
    invokeReply = async (cmd) => cmd === 'table_ride' ? ride : null;
    await engine.checkRide(r);
    const s = summarise(r.ride);
    const matched = r.ride.rows.filter(x => !x.id.startsWith('e') || r.rows.some(y => y.id === x.id)).length;
    if (matched !== ride.rows.length) return `${matched} of ${ride.rows.length} exam rows found`;
    const right = r.ride.rows.filter(x => x.outcome === 'right').length;
    if (right !== ride.passed) return `${right} right here, ${ride.passed} passed in the pipeline`;
    if (s.baseRight !== ride.control_passed) return `untrained ${s.baseRight}, pipeline ${ride.control_passed}`;
    return true;
  });
}

/* A reasoning job: the table holds the reason and the action apart; the
   pipeline was asked the pair and graded the action. Read back, only the
   action counts. */
const reasonedPath = process.env.FS_REASONED_RUN;
if (reasonedPath && existsSync(`${reasonedPath}/export/check-ride.json`)) {
  await checkAsync('a reasoning job reads back graded on its action alone, with its reasoning kept apart', async () => {
    const ride = JSON.parse(readFileSync(`${reasonedPath}/export/check-ride.json`, 'utf8'));
    const doOf = (s) => { const m = [...String(s).matchAll(/<do>\s*([\s\S]*?)\s*<\/do>/gi)]; return m.length ? m[m.length - 1][1].trim() : null; };
    const examRows = readFileSync(`${reasonedPath}/exam.jsonl`, 'utf8').trim().split('\n').map(l => JSON.parse(l));
    const r = blankRun('reasoned'); S.runs.reasoned = r;
    const labels = [...new Set(examRows.map(x => doOf(x.answer)))];
    Object.assign(r, { name: 'reasoned', input: 'text', output: 'function', reasoning: true, answers: labels, teach: { state: 'done', version: 1 } });
    r.rows = examRows.map((x, i) => ({ id: `e${i}`, input: { kind: 'text', text: x.question }, answer: doOf(x.answer), think: 'x', holdout: true }));
    invokeReply = async (cmd) => cmd === 'table_ride' ? ride : null;
    await engine.checkRide(r);
    const right = r.ride.rows.filter(x => x.outcome === 'right').length;
    if (right !== ride.passed) return `${right} right here, ${ride.passed} in the pipeline`;
    return r.ride.rows.every(x => x.think !== null || x.outcome === 'no-answer') || 'a reply lost its reasoning';
  });
}

/* ── repeats ─────────────────────────────────────────────────────────── */
{
  const { repeatGroups, keepOneOf, putBackRows, pixelGap } = S_;
  const flat = (v) => new Uint8Array(1024).fill(v);
  const withSpot = (v, at, to) => { const a = flat(v); a[at] = to; return a; };
  const P = (sig, hi = 0, lo = 0) => ({ hi, lo, sig });
  const pictures = () => {
    const r = blankRun('rep');
    Object.assign(r, { name: 'rep', input: 'image' });
    const row = (id, path, answer, extra = {}) => ({ id, input: { kind: 'image', path }, answer, ...extra });
    r.rows = [
      row('r1', 'a.jpg', 'UP'), row('r2', 'a2.jpg', 'UP'), row('r3', 'a3.jpg', 'UP'),   // one screen, three presses
      row('r4', 'b.jpg', 'UP'),                                                          // a sprite moved: one part differs a lot
      row('r5', 'a4.jpg', 'DOWN'),                                                       // same screen, other answer: a conflict, not a repeat
      row('r6', 'a5.jpg', 'UP', { holdout: true }),                                      // a copy on the exam
      row('r7', 'a6.jpg', 'UP', { left: 'dropped' })                                     // already left out
    ];
    const prints = {
      'a.jpg': P(flat(100)), 'a2.jpg': P(withSpot(100, 5, 102)), 'a3.jpg': P(flat(101)),
      'b.jpg': P(withSpot(100, 500, 180)), 'a4.jpg': P(flat(100)), 'a5.jpg': P(flat(100)), 'a6.jpg': P(flat(100))
    };
    return { r, prints };
  };

  check('the pixel gap is the largest difference in any one part, and stops early past its limit', () => {
    if (pixelGap(flat(10), withSpot(10, 7, 40)) !== 30) return 'gap is not the largest single difference';
    return pixelGap(flat(0), flat(200), 5) === Infinity || 'did not stop past the limit';
  });

  check('identical pictures with the same answer group; a moved sprite and another answer do not', () => {
    const { r, prints } = pictures();
    const g = repeatGroups(r, prints, { mode: 'exact' });
    if (g.length !== 1) return `${g.length} groups`;
    const ids = g[0].rows.map(x => x.id).join(',');
    return ids === 'r1,r2,r3,r6' || `grouped ${ids}`;
  });

  check('"nearly identical" takes in a picture that differs in one part, when the setting allows it', () => {
    const { r, prints } = pictures();
    const tight = repeatGroups(r, prints, { mode: 'close', tol: 20 })[0].rows.length;
    const loose = repeatGroups(r, prints, { mode: 'close', tol: 90 })[0].rows.length;
    return (tight === 4 && loose === 5) || `tight ${tight}, loose ${loose}`;
  });

  check('keeping one keeps the exam row, so none of its copies train; Undo puts them back', () => {
    const { r, prints } = pictures();
    const g = repeatGroups(r, prints)[0];
    const out = keepOneOf(r, g);
    if (out.join(',') !== 'r1,r2,r3') return `left out ${out}`;
    if (r.rows.find(x => x.id === 'r6').left) return 'the exam row was left out';
    if (trainingSet(r).rows.some(x => ['r1', 'r2', 'r3'].includes(x.id))) return 'a copy of an exam picture still trains';
    putBackRows(r, out);
    return r.rows.filter(x => x.left === 'repeat').length === 0 || 'undo left rows out';
  });

  check('text repeats are the same words, ignoring case and spacing', () => {
    const r = blankRun('t');
    Object.assign(r, { name: 't', input: 'text' });
    r.rows = [
      { id: 'r1', input: { kind: 'text', text: 'Hello  there' }, answer: 'A' },
      { id: 'r2', input: { kind: 'text', text: 'hello there' }, answer: 'A' },
      { id: 'r3', input: { kind: 'text', text: 'hello there' }, answer: 'B' }
    ];
    const g = repeatGroups(r);
    return (g.length === 1 && g[0].rows.length === 2) || JSON.stringify(g.map(x => x.rows.map(y => y.id)));
  });
}

/* ── conflicts ───────────────────────────────────────────────────────── */
await checkAsync('two pictures are a conflict only when their pixels match, not just their fingerprints', async () => {
  const sig = (v, at = -1) => { const a = new Uint8Array(1024).fill(v); if (at >= 0) a[at] = 250; return Buffer.from(a).toString('base64'); };
  const same = '00000000000000ff';
  const table = { 'a.jpg': { hash: same, sig: sig(90) }, 'b.jpg': { hash: same, sig: sig(90, 300) }, 'c.jpg': { hash: same, sig: sig(91) } };
  invokeReply = async (cmd, a) => cmd === 'table_signatures' ? Object.fromEntries(a.paths.map(p => [p, table[p]])) : null;
  const r = blankRun('conf');
  Object.assign(r, { name: 'c', input: 'image' });
  r.rows = [
    { id: 'r1', input: { kind: 'image', path: 'a.jpg' }, answer: 'KEY(Up)' },
    { id: 'r2', input: { kind: 'image', path: 'b.jpg' }, answer: 'KEY(Right)' },   // Patch moved: one part differs a lot
    { id: 'r3', input: { kind: 'image', path: 'c.jpg' }, answer: 'KEY(Left)' },    // the same screen
    { id: 'r4', input: { kind: 'image', path: 'b.jpg' }, answer: 'KEY(Right)', left: 'conflict' }  // left out by the old rule
  ];
  await engine.findConflicts(r);
  const pairs = r.conflicts.map(c => `${c.a}-${c.b}`).join(',');
  if (pairs !== 'r1-r3') return `conflicts: ${pairs}`;
  return r.rows.find(x => x.id === 'r4').left === null || 'a row left out by a false conflict was not put back';
});
invokeReply = async () => null;

await checkAsync('importing a folder joins its answers to the ones the table has, and offers near-misses as the same answer', async () => {
  const { importPlan } = await import('../src/screens/collect.js');
  const r = blankRun('imp');
  Object.assign(r, { name: 'i', input: 'image', answers: ['NONE'] });
  r.actions = [{ id: 'k1', kind: 'key', raw: 'Up', name: null, uses: 3 }];
  const rows = [['KEY(Up)', 5], ['KEY(up)', 2], ['Cat', 4], ['cat ', 1], ['Dog', 3]].flatMap(([a, n]) => Array.from({ length: n }, () => ({ answer: a })));
  const plan = Object.fromEntries(importPlan(r, rows).map(p => [p.label, p]));
  if (plan['KEY(Up)'].to !== '@k1') return `KEY(Up) goes to ${plan['KEY(Up)'].to}, not the recorded key`;
  if (plan['KEY(up)'].near !== '@k1' || !plan['KEY(up)'].join) return 'KEY(up) is not offered as the recorded KEY(Up)';
  if (plan['cat '].near !== 'Cat' || !plan['cat '].join) return '"cat " is not offered as Cat';
  return (plan.Dog.near === null && !plan.Dog.known) || 'Dog should be a new answer';
});

check('a picture is a URL the app serves, with its run and path encoded and the viewer asking for the full file', () => {
  const r = blankRun('qb-1');
  const thumb = engine.image(r, 'recording/s1/a b.jpg');
  const full = engine.image(r, '/Users/x/Pictures/c#1.png', { full: true });
  if (thumb !== 'pic://localhost/qb-1/recording/s1/a%20b.jpg') return `thumb: ${thumb}`;
  if (full !== 'pic://localhost/qb-1//Users/x/Pictures/c%231.png?full=1') return `full: ${full}`;
  return engine.image(r, '') === null || 'no path, no picture';
});
invokeReply = async () => null;

/* ── Send's two buttons ─────────────────────────────────────────────── *
 * "Save file…" only writes the file; "Try it in Pigeon" also adds it to the
 * lineup and opens Pigeon (table.rs after_pack). On the page, only the second
 * counts as sent to Pigeon. */
for (const [button, toPigeon] of [['Save file…', false], ['Try it in Pigeon', true]]) {
  await checkAsync(`Send's "${button}" asks Rust for ${toPigeon ? 'the lineup and Pigeon' : 'the file alone'}`, async () => {
    const calls = [];
    invokeReply = async (cmd, a) => { calls.push([cmd, a]); return cmd === 'table_send' ? { path: '/tmp/x.pigeon', bytes: 1 } : null; };
    const r = Object.assign(blankRun(`send-${toPigeon}`), { name: 'Send test', input: 'text', does: 'Tests sending.' });
    await engine.send(r, toPigeon);
    invokeReply = async () => null;
    const sends = calls.filter(([c]) => c === 'table_send');
    if (sends.length !== 1 || sends[0][1].toPigeon !== toPigeon) return `table_send asked with ${JSON.stringify(sends.map(c => c[1].toPigeon))}`;
    if (calls.some(([c]) => c.startsWith('pigeon_add'))) return 'the page added it to the lineup itself';
    if (toPigeon) return (r.sent?.revision === 1) || `sent ${JSON.stringify(r.sent)}`;
    return r.sent == null || `a saved file counted as sent: ${JSON.stringify(r.sent)}`;
  });
}

/* ── picture size ──────────────────────────────────────────────────── */
{
  const { pictureSide, sourceSide, setPictureSide } = S_;
  const { actionsOf, playable, visionCost } = await import('../src/engine.js');
  const rec = () => { const r = blankRun('ps'); Object.assign(r, { name: 'ps', input: 'image', sessionsIn: ['Session 1'], recordSide: 512 }); return r; };

  check('a recorded table trains and plays at the size it was recorded at', () => {
    const r = rec();
    return (pictureSide(r) === 512 && actionsOf(r).frames.max_side === 512) || `trains at ${pictureSide(r)}, plays at ${actionsOf(r).frames.max_side}`;
  });

  check('an imported picture table trains at the size Teach shrinks it to', () => {
    const r = blankRun('imp');
    Object.assign(r, { name: 'imp', input: 'image', answers: ['KEY(Up)'] });
    setPictureSide(r, 256);
    return pictureSide(r) === 256 || `trains at ${pictureSide(r)}`;
  });

  check('a picture is never made larger than it was recorded', () => {
    const r = rec();
    r.maxSide = 768;
    return (pictureSide(r) === 512 && sourceSide(r) === 512) || `trains at ${pictureSide(r)}`;
  });

  check('a table recorded before sizes were chosen is 768, as it always was', () => {
    const r = rec();
    r.recordSide = null;
    return actionsOf(r).frames.max_side === 768 || `plays at ${actionsOf(r).frames.max_side}`;
  });

  check('an imported picture table answering KEY(Up) gets no actions.json: it answers, but is not played (D71)', () => {
    const r = blankRun('keys');
    Object.assign(r, { name: 'k', input: 'image', answers: ['KEY(Up)', 'KEY(Left)', 'Cat'] });
    return (actionsOf(r) === null && playable(r) === null) || JSON.stringify(actionsOf(r));
  });

  check('a recorded table answering KEY(Up) presses the key it names, at the recording\'s size', () => {
    const r = rec();
    Object.assign(r, { answers: ['KEY(Up)', 'KEY(Left)', 'NONE'] });
    const a = playable(r);
    const k = a && a.keys.map(x => `${x.answer}=${x.raw}`).join(' ');
    return (k === 'KEY(Up)=Up KEY(Left)=Left' && a.frames.max_side === 512 && a.frames.gap_ms === 250) || JSON.stringify(a);
  });

  check('a recorded table tells Pigeon exactly the recording\'s part and size, whatever was cut later', () => {
    const r = rec();
    Object.assign(r, { recordSide: 768, crop: [0.5, 0, 0.5, 1], window: { w: 384, h: 480 }, picCrop: [0, 0.5, 1, 0.5], maxSide: 256 });
    const f = actionsOf(r).frames;
    return (JSON.stringify(f.crop) === JSON.stringify([0.5, 0, 0.5, 1]) && f.max_side === 768) || JSON.stringify(f);
  });

  check('a recorded table cut or shrunk after recording blocks Start, and opens Teach to settle it', () => {
    const { cutAfterRecording, trainAsRecorded, gate, issues, currentStage, blocks } = S_;
    const r = rec();
    if (cutAfterRecording(r)) return 'an uncut table is not cut';
    for (const [k, v] of [['picCrop', [0, 0, 0.5, 1]], ['maxSide', 256]]) {
      const x = rec(); x[k] = v;
      if (!cutAfterRecording(x)) return `${k} after recording is not caught`;
      if (!issues(x).some(i => i.id === 'recut' && i.level === 'block')) return `${k}: no block`;
      if (/after they were recorded/.test(gate(x, 'teach').why || '')) return `${k}: Teach should open to show the choice`;
      const ready = Object.assign(rec(), { kind: 'custom', [k]: v, rows: Array.from({ length: 40 }, (_, i) => ({ id: `r${i}`, input: { kind: 'image', path: `recording/s1/f${i}.jpg` }, answer: 'KEY(Up)' })) });
      if (blocks(ready).some(i => i.id !== 'recut')) return `${k}: the ready table has other blocks: ${blocks(ready).map(i => i.id)}`;
      if (currentStage(ready) !== 'teach') return `${k}: a table blocked only by the cut opens on ${currentStage(ready)}, not Teach`;
      trainAsRecorded(x);
      if (cutAfterRecording(x) || x.picCrop || x.maxSide) return `${k}: train as recorded did not clear it`;
    }
    const big = rec(); big.maxSide = 768;   // larger than recorded is the same pictures
    if (cutAfterRecording(big)) return 'a size above the recording is not a shrink';
    const imported = blankRun('imp'); Object.assign(imported, { input: 'image', picCrop: [0, 0, 0.5, 1] });
    return !cutAfterRecording(imported) || 'an imported table may still be cut';
  });

  check('record again makes a new table with the job, not the rows, and leaves the old one alone', () => {
    const { recordAgain } = S_;
    const r = rec();
    Object.assign(r, { name: 'Seed Run', kind: 'player', output: 'function', answers: ['NONE', '@k1', 'KEY(Up)'], stack: 2,
      crop: [0.5, 0, 0.5, 1], question: 'What do you do next?', rows: [{ id: 'r00001' }] });
    const before = JSON.stringify(r);
    const a = recordAgain(r, { part: [0, 0.5, 1, 0.5] });
    const n = S.runs[a.id];
    if (n.name !== 'Seed Run (part)') return `name ${n.name}`;
    if (n.rows.length || n.sessionsIn.length) return 'the new table took rows';
    if (JSON.stringify(n.answers) !== JSON.stringify(['NONE', 'KEY(Up)'])) return `answers ${JSON.stringify(n.answers)}`;
    if (n.kind !== 'player' || n.question !== r.question || n.stack !== 2) return 'the job was not copied';
    if (JSON.stringify(a.record.crop) !== JSON.stringify([0.5, 0.5, 0.5, 0.5]) || a.record.side !== 512) return `record ${JSON.stringify(a.record)}`;
    const b = recordAgain(r, { side: 256 });
    if (S.runs[b.id].name !== 'Seed Run (256 px)' || b.record.side !== 256 || JSON.stringify(b.record.crop) !== JSON.stringify(r.crop)) return `size: ${S.runs[b.id].name} ${JSON.stringify(b.record)}`;
    delete S.runs[a.id]; delete S.runs[b.id];
    return JSON.stringify(r) === before || 'the old table changed';
  });

  check('picture cost is measured at 160 and 768 and an estimate between, getting cheaper as it shrinks', () => {
    const [a, b, c] = [visionCost(160), visionCost(384), visionCost(768)];
    if (!a.measured || !c.measured || b.measured) return 'the ends are not the measurements';
    return (b.tokensPerFrame > a.tokensPerFrame && b.tokensPerFrame < c.tokensPerFrame && b.itPerSec < a.itPerSec && b.itPerSec > c.itPerSec) || JSON.stringify(b);
  });
}

/* ── the look: one family with Pigeon ─────────────────────────────────── */
{
  const { pigeonTokens } = await import('../scripts/tokens.mjs');
  const ours = readFileSync(new URL('../src/tokens.css', import.meta.url), 'utf8');
  const index = readFileSync(new URL('../src/index.html', import.meta.url), 'utf8');
  const app = readFileSync(new URL('../src/app.css', import.meta.url), 'utf8');

  check('tokens.css is Pigeon\'s, unedited', () => {
    if (!/SHARED: Pigeon and Flight School carry this same file/.test(ours)) return 'src/tokens.css is not the shared file';
    const theirs = pigeonTokens();
    if (!existsSync(theirs)) return true;   // Pigeon not checked out beside this repo: nothing to compare
    return readFileSync(theirs, 'utf8') === ours || `src/tokens.css differs from ${theirs} — run npm run tokens`;
  });

  check('the page follows the Mac into dark, and draws in the shared typeface', () => {
    if (!/<html[^>]*data-theme="system"/.test(index)) return 'index.html does not carry data-theme="system", so the dark set never applies';
    if (!/font-family: var\(--sans\)/.test(app) || !/fonts\/hanken-grotesk-variable\.woff2/.test(app)) return 'the body does not draw in Hanken Grotesk';
    return !/:root\s*\{[^}]*--(ink|paper|surface|aqua):/.test(app) || 'app.css defines a colour token of its own';
  });

  check('tabs, filters and segments are Pigeon\'s track, and choose by their ARIA state', () => {
    if (/\.(tab|flt|seg-btn)\.on\b/.test(app)) return 'a segmented control still selects by class';
    if (!/\.tab\[aria-selected="true"\], \.flt\[aria-selected="true"\], \.seg-btn\[aria-pressed="true"\], \.seg-btn\[aria-checked="true"\] \{ background: var\(--surface\)/.test(app)) return 'the chosen segment is not lifted onto the track';
    return true;
  });

  const { settingsDialog, TABS } = await import('../src/screens/settings.js');
  const { rail } = await import('../src/screens/shell.js');
  check('Settings is Pigeon\'s sheet: five tabs on a track, each drawn, the appearance switch in General and not in the rail', () => {
    for (const [k] of TABS) {
      const html = settingsDialog({ kind: 'settings', tab: k });
      if (!/class="settings-card" role="dialog" aria-modal="true"/.test(html)) return `${k}: not a dialog`;
      if ((html.match(/role="tab"/g) || []).length !== 5 || !new RegExp(`id="st-tab-${k}" aria-selected="true"`).test(html)) return `${k}: the tabs are wrong`;
    }
    const general = settingsDialog({ kind: 'settings', tab: 'general' });
    if (!/aria-label="Appearance"[^]*Match Mac[^]*Light[^]*Dark/.test(general)) return 'General has no Match Mac / Light / Dark';
    const r = rail();
    if (/data-do="theme"|theme-pick/.test(r)) return 'the appearance switch is still in the rail';
    return /data-do="settings"/.test(r) || 'the rail has no way into Settings';
  });

  await checkAsync('the glyphs both apps draw are Pigeon\'s, character for character', async () => {
    const { ICON } = await import('../src/screens/html.js');
    const theirs = pigeonTokens().replace(/styles\/tokens\.css$/, 'src/ui/html.js');
    if (!existsSync(theirs)) return true;   // Pigeon not beside this repo: nothing to compare
    const { ICONS } = await import(new URL(`file://${theirs}`).href);
    const SAME = { plus: 'plus', image: 'image', check: 'check', copy: 'copy', send: 'send', back: 'back', more: 'ellipsis', trash: 'trash', stop: 'stop', gear: 'gear', close: 'x', search: 'search', rename: 'rename' };
    const off = Object.entries(SAME).filter(([ours, pig]) => ICON[ours] !== ICONS[pig]).map(([ours]) => ours);
    return !off.length || `drawn differently from Pigeon: ${off.join(', ')}`;
  });

  check('honesty tags are tinted, never outlined, and the toast is Pigeon\'s solid pill', () => {
    const rule = (sel) => { const i = app.indexOf(`${sel} {`); return i < 0 ? '' : app.slice(i, app.indexOf('}', i)); };
    if (/border:/.test(rule('.tag, .badge'))) return 'a tag is drawn as an outline';
    const toast = rule('.toast');
    return (/var\(--solid\)/.test(toast) && /border-radius: 999px/.test(toast)) || 'the toast is not a solid pill';
  });
}

/* ── kinds: the first question, and what each one fixes ───────────────── */
const { KINDS, setKind, clearKind, fixedFields, legacyKind, floorFor, RULES, kindOf } = S_;

check('every kind fixes exactly what its card promises', () => {
  const want = {
    reader: { input: 'text', output: 'prose', grounded: true },
    player: { input: 'image', output: 'function', answerSet: 'record', reasoning: false },
    sorter: { input: 'text', output: 'function', answerSet: 'custom' },
    looker: { input: 'image', output: 'function', answerSet: 'custom' },
    writer: { input: 'text', output: 'prose', grounded: false }
  };
  for (const [id, fields] of Object.entries(want)) {
    const r = blankRun('t');
    const why = setKind(r, id);
    if (why) return `${id}: refused — ${why}`;
    for (const [f, v] of Object.entries(fields)) if (r[f] !== v) return `${id}: ${f} is ${JSON.stringify(r[f])}, wanted ${JSON.stringify(v)}`;
    if (r.base !== 'Qwen3-VL-2B') return `${id}: base is ${r.base}`;
    if (!fixedFields(r).has('input')) return `${id}: input not marked fixed`;
  }
  return true;
});

check('Something else fixes nothing and leaves Describe whole', () => {
  const r = blankRun('t');
  const why = setKind(r, 'custom');
  if (why) return why;
  return (r.kind === 'custom' && r.input === null && fixedFields(r).size === 0) || `input ${r.input}, fixed ${[...fixedFields(r)]}`;
});

check('a run with no kind stops at Describe and asks; a kind and a name open Collect', () => {
  const r = blankRun('t');
  if (currentStage(r) !== 'describe' || gate(r, 'collect').open) return 'a kindless run got past Describe';
  if (!gate(r, 'collect').why.includes('kind')) return `the reason does not name the kind: ${gate(r, 'collect').why}`;
  setKind(r, 'sorter'); r.name = 'inbox';
  if (!gate(r, 'collect').open) return `Collect still shut: ${gate(r, 'collect').why}`;
  clearKind(r);
  return (currentStage(r) === 'describe' && r.input === 'text') || 'Change did not go back to the question, or threw the choice away';
});

check('a kind that takes the other input is refused once the table has rows, with why', () => {
  const r = blankRun('t');
  setKind(r, 'sorter');
  r.rows.push({ id: 'x1', input: { kind: 'text', text: 'a note' }, answer: 'A' });
  const why = setKind(r, 'looker');
  if (!why || !/pictures/.test(why)) return `let it through, or said nothing useful: ${why}`;
  if (r.kind !== 'sorter' || r.input !== 'text') return 'the refused change still took';
  return setKind(r, 'writer') === null || 'a kind with the same input was refused';
});

check('a Reader needs 150 rows; every other kind needs the ordinary floor', () => {
  const r = blankRun('t');
  setKind(r, 'reader'); r.name = 'handbook';
  for (let i = 0; i < 40; i++) r.rows.push({ id: `x${i}`, input: { kind: 'text', text: `p${i}` }, answer: `a${i}` });
  if (floorFor(r) !== RULES.readerFloor || currentStage(r) !== 'collect') return `reader at 40 rows is at ${currentStage(r)}`;
  const w = blankRun('w');
  setKind(w, 'writer'); w.name = 'titles';
  for (let i = 0; i < 40; i++) w.rows.push({ id: `x${i}`, input: { kind: 'text', text: `p${i}` }, answer: `a${i}` });
  return (floorFor(w) === RULES.floor && currentStage(w) !== 'collect') || `writer at 40 rows is at ${currentStage(w)}`;
});

check('a run saved before kinds existed is Something else; one never described still asks', () => {
  if (legacyKind({ input: 'image', output: 'function' }) !== 'custom') return 'a described old run did not become custom';
  if (legacyKind({ input: null }) !== null) return 'an undescribed old run was given a kind';
  if (legacyKind({ kind: 'reader', input: 'text' }) !== 'reader') return 'a saved kind was overwritten';
  return kindOf({ kind: 'nope' }) === null && KINDS.length === 7 && KINDS.at(-1).id === 'prompted';
});

check('a Reader with no document stops at Collect and says so; the document lets it on to the floor', () => {
  const { setMaterial } = S_;
  const r = blankRun('t');
  setKind(r, 'reader'); r.name = 'handbook';
  for (let i = 0; i < 200; i++) r.rows.push({ id: `x${i}`, input: { kind: 'text', text: `p${i}` }, answer: `a${i}` });
  if (currentStage(r) !== 'collect' || !gate(r, 'review').why.includes('document')) return `at ${currentStage(r)}: ${gate(r, 'review').why}`;
  const v = r.version;
  setMaterial(r, { files: [{ file: 'c.md', sections: 3, est_tokens: 900 }], sections: 3, est_tokens: 900, whole: true, chunks: [] });
  if (r.version <= v) return 'the material did not count as a change';
  if (r.material.version !== 1 || !r.material.whole) return `material ${JSON.stringify(r.material)}`;
  if (!gate(r, 'review').open) return `Review still shut: ${gate(r, 'review').why}`;
  setMaterial(r, null);
  return (r.material === null && currentStage(r) === 'collect') || 'taking the last file out did not go back to asking for it';
});

/* ── a Reader's rows ─────────────────────────────────────────────────── */
const { readerRow, graftPlan, citedOrdinals, unsupportedCitations, badCites, staleRows, approveRows, materialBlock, setMaterial } = S_;
const CH = [1, 2, 3, 4, 5, 6].map(k => ({ ordinal: k, file: 'c.md', title: `Section ${k}`, text: `Body of section ${k}.`, est_tokens: 6 }));

check('a drafted answer ends up citing exactly the section it was shown; a refusal cites nothing', () => {
  const a = readerRow(CH[1], 'What is it?', 'It is this.');
  if (a.answer !== 'It is this. [§2]') return `missing citation not added: ${a.answer}`;
  const b = readerRow(CH[1], 'What is it?', 'It is this [§5], see [§2].');
  if (citedOrdinals(b.answer).join() !== '2' || /§5/.test(b.answer)) return `unshown citation kept: ${b.answer}`;
  const c = readerRow(CH[2], 'What is it?', 'Not here [§3]. It is about §3 things [§1].', { refusal: true, section: 2 });
  if (citedOrdinals(c.answer).length || c.section !== 2 || c.shown[0] !== 3) return `refusal: ${JSON.stringify(c)}`;
  if (a.input.text !== `${materialBlock(CH[1])}\n\nWhat is it?` || !a.draft) return 'the input is not passage, blank line, question';
  return unsupportedCitations('x [§4] y [§2]', [2]).join() === '4';
});

check('refusals are grafted onto the NEIGHBOURING section, spread across the document, one in five', () => {
  const drafted = [];
  for (const c of CH) for (let i = 0; i < 3; i++) drafted.push({ section: c.ordinal, question: `q${c.ordinal}.${i}` });
  const plan = graftPlan(CH, drafted, 0.2);
  const want = Math.round(18 * 0.25);
  if (plan.length !== want) return `planned ${plan.length}, wanted ${want}`;
  for (const p of plan) if (Math.abs(p.shown.ordinal - p.section) !== 1) return `${p.question} grafted onto §${p.shown.ordinal}, its own is §${p.section}`;
  const sections = new Set(plan.map(p => p.section));
  if (sections.size < 4) return `only sections ${[...sections]} used`;
  if (new Set(plan.map(p => p.question)).size !== plan.length) return 'a question was used twice';
  return graftPlan(CH.slice(0, 1), drafted.slice(0, 3), 0.2).length === 0 || 'one section cannot have a neighbour';
});

check('a Reader’s exam is whole sections, so no exam question’s section trains', () => {
  const r = blankRun('t');
  setKind(r, 'reader'); r.name = 'c';
  for (const c of CH) for (let i = 0; i < 6; i++) r.rows.push({ id: `x${c.ordinal}.${i}`, ...readerRow(c, `q${c.ordinal}.${i}`, 'a'), draft: false });
  markHoldout(r, 3);
  const ex = exam(r);
  if (!ex.length) return 'no exam drawn';
  const examSections = new Set(ex.map(x => x.section));
  const trained = trainingSet(r).rows;
  const leak = trained.filter(x => examSections.has(x.section));
  if (leak.length) return `${leak.length} training rows share a section with the exam`;
  return ex.every(x => x.section != null) || 'a sectionless row in the exam';
});

check('a citation of the unseen and a row from an older document each block, and approving draws the exam', () => {
  const r = blankRun('t');
  setKind(r, 'reader'); r.name = 'c';
  setMaterial(r, { files: [], sections: 6, est_tokens: 36, whole: true, chunks: CH });
  r._chunks = CH;
  for (const c of CH) for (let i = 0; i < 30; i++) r.rows.push({ id: `x${c.ordinal}.${i}`, ...readerRow(c, `q${c.ordinal}.${i}`, 'a') });
  if (exam(r).length) return 'drafts drew an exam';
  const k = approveRows(r, r.rows.map(x => x.id));
  if (k !== 180 || !exam(r).length) return `approved ${k}, exam ${exam(r).length}`;
  if (!gate(r, 'teach').open) return `teach shut: ${gate(r, 'teach').why}`;
  r.rows[0].answer = 'wrong [§6]';
  if (badCites(r).length !== 1 || !gate(r, 'teach').why.includes('not shown')) return `bad cite not caught: ${gate(r, 'teach').why}`;
  r.rows[0].answer = 'fine [§1]';
  r._chunks = CH.map(c => (c.ordinal === 4 ? { ...c, title: 'Renamed' } : c));
  if (staleRows(r).length !== 30 || !gate(r, 'teach').why.includes('earlier version')) return `stale rows ${staleRows(r).length}: ${gate(r, 'teach').why}`;
  return true;
});

check('a Reader’s ride counts refusals, false refusals, unseen citations and the fact swap, each beside the untrained model', () => {
  const { groundingOf } = await_import;
  const rows = [
    { refusal: true, cited: [], base: { cited: [2] } },
    { refusal: true, cited: [3], base: { cited: [] } },
    { refusal: false, cited: [1], unsupported: [], base: { cited: [], unsupported: [] } },
    { refusal: false, cited: [9], unsupported: [9], base: { cited: [1], unsupported: [] } },
    { refusal: false, cited: [], unsupported: [], base: { cited: [1], unsupported: [] } }
  ];
  const g = groundingOf(rows, { n: 4, adapter: { followed: 3, memorised: 1 }, control: { followed: 1, memorised: 2 }, cases: [] });
  if (g.refusals.refused !== 1 || g.refusals.baseRefused !== 1 || g.refusals.n !== 2) return `refusals ${JSON.stringify(g.refusals)}`;
  if (g.covered.falseRefusals !== 1 || g.covered.unsupported !== 1 || g.covered.baseFalseRefusals !== 1 || g.covered.baseUnsupported !== 0) return `covered ${JSON.stringify(g.covered)}`;
  if (g.swaps.followed !== 3 || g.swaps.baseMemorised !== 2) return `swaps ${JSON.stringify(g.swaps)}`;
  if (g.swaps.n !== 4 || g.swaps.baseN !== 4) return `an old ride without counted n falls back to n: ${JSON.stringify(g.swaps)}`;
  const counted = groundingOf(rows, { n: 12, adapter_n: 9, control_n: 11, adapter: { followed: 8 }, control: { followed: 6 }, cases: [] }).swaps;
  if (counted.asked !== 12 || counted.n !== 9 || counted.baseN !== 11) return `counted swaps ${JSON.stringify(counted)}`;
  return groundingOf(rows, { n: 0 }).swaps === null || 'an empty swap should be null';
});

check('a Reader’s ride leads with the same-source count, beside the untrained model, and shows it only when measured', () => {
  const { groundingOf } = await_import;
  const g = groundingOf([], null, { same: 12, sentence: 14, n: 20, base_same: 0, base_sentence: 15, base_n: 20 });
  if (JSON.stringify(g.source) !== JSON.stringify({ sentence: 14, same: 12, n: 20, baseSentence: 15, baseN: 20 })) return JSON.stringify(g.source);
  if (groundingOf([], null, null).source !== null || groundingOf([], null, { same: 0, n: 0 }).source !== null) return 'an unmeasured ride shows a count';
  const ride = readFileSync(new URL('../src/screens/ride.js', import.meta.url), 'utf8');
  const view = ride.slice(ride.indexOf('function groundingView'), ride.indexOf('function citeLine'));
  if (view.indexOf('g.source ?') < 0 || view.indexOf('g.source ?') > view.indexOf('said “not in here”')) return 'the same-source tile does not lead';
  return (/different sentence/.test(ride) && /same source/.test(ride)) || 'rows carry no source badge';
});

check('a Reader row paired from files gets its section, question and refusal flag; other rows come back as they were', () => {
  const { annotateReaderRow } = S_;
  const text = `${materialBlock(CH[2])}\n\nWhat is it?`;
  const a = annotateReaderRow({ input: { kind: 'text', text }, answer: 'It is this. [§3]' });
  if (a.shown?.[0] !== 3 || a.section !== 3 || a.question !== 'What is it?' || a.sectionTitle !== 'Section 3' || a.refusal) return JSON.stringify(a);
  const b = annotateReaderRow({ input: { kind: 'text', text }, answer: 'Not in here; this is about section 3.' });
  if (!b.refusal) return 'a citation-free answer is a refusal';
  const c = annotateReaderRow({ input: { kind: 'text', text: 'a plain row' }, answer: 'x' });
  return (c.shown === undefined && c.question === undefined) || 'a plain row was annotated';
});

check('a draft must answer, from its passage, about the subject — and a bare-citation row blocks', () => {
  const { draftProblems, bareRows } = S_;
  const c = { ordinal: 25, title: 'Article VII', text: 'The Ratification of the Conventions of nine States, shall be sufficient for the Establishment of this Constitution between the States so ratifying the Same.' };
  if (!draftProblems(c, 'How many States had to ratify?', '[§25]')) return 'a bare citation passed';
  if (!draftProblems(c, 'How many States had to ratify?', ' [§25] ')) return 'whitespace and a citation passed';
  if (!draftProblems(c, 'What does the passage say about ratification?', 'Nine States. [§25]')) return 'a question about the passage passed';
  if (!draftProblems(c, 'Who is this article about?', 'Nine States. [§25]')) return '"this article" passed';
  if (!draftProblems(c, 'How many people are mentioned in this preamble?', 'One (we the People). [§25]')) return '"mentioned in this preamble" passed';
  if (!draftProblems(c, 'How many States had to ratify?', '§25 [§25]')) return 'a stray §25 counted as an answer';
  if (!draftProblems(c, 'What is the frequency of meetings for Congress?', 'The Ratification of the Conventions of nine States shall be sufficient.', { refusal: true })) return 'a refusal that never says no passed';
  if (!draftProblems(c, 'Who signed the document?', 'George Washington signed it along with thirty-nine delegates representing the states that had ratified the Constitution.')) return 'an invented answer passed';
  if (draftProblems(c, 'How many States had to ratify for the Constitution to take effect?', 'Nine. The Ratification of the Conventions of nine States was sufficient for its Establishment. [§25]')) return `a good draft was refused: ${draftProblems(c, 'How many States had to ratify for the Constitution to take effect?', 'Nine. The Ratification of the Conventions of nine States was sufficient for its Establishment. [§25]')}`;
  if (draftProblems(c, 'What is the supremacy clause?', 'This passage does not cover that. It is about the Ratification of the Conventions of nine States.', { refusal: true })) return 'a good refusal was refused';
  if (!draftProblems(c, 'What is the supremacy clause?', 'Not here [§25].', { refusal: true })) return 'a citing refusal passed';
  if (draftProblems(c, 'What is the supremacy clause?', "This passage doesn't cover that; it is about the Ratification of the Conventions.", { refusal: true })) return '"doesn\'t" did not count as saying no';
  const r = blankRun('t');
  setKind(r, 'reader'); r.name = 'c';
  setMaterial(r, { files: [], sections: 1, est_tokens: 30, whole: true, chunks: [c] });
  r.rows.push({ id: 'x1', ...readerRow(c, 'q?', 'Nine States. '), draft: false });
  r.rows.push({ id: 'x2', ...readerRow(c, 'q?', ''), draft: false });
  const block = S_.issues(r).find(i => i.id === 'bare');
  if (bareRows(r).length !== 1 || !block || block.level !== 'block' || block.fix.filter !== 'bare') return `bare ${bareRows(r).length}: ${JSON.stringify(block)}`;
  return true;
});

check('a thought partner draft: one question, last, short, citing only its own section', () => {
  const { draftProblems } = S_;
  const c = { ordinal: 12, title: '3.1 Measures of Center · Mean', text: 'The mean, or average, is the most commonly used measure of the center of a dataset. Sometimes the mean can be skewed by the presence of outliers.' };
  const p = { style: 'partner' };
  const good = 'The mean is the average, and it gets pulled toward outliers [§12]. Think of one very high salary in a small office. Which measure would you trust for that office?';
  const why = draftProblems(c, 'can you just write my post on outliers', good, p);
  if (why) return `a good reply to a request with no "?" was refused: ${why}`;
  if (!draftProblems(c, 'can you write my post', 'Sure. Here it is. Is that good? Want more?', p)) return 'two questions passed';
  if (!draftProblems(c, 'explain the mean please', 'The mean is the average [§12]. Want an example? It helps with outliers.', p)) return 'a question not at the end passed';
  if (!draftProblems(c, 'explain the mean please', `${'word '.repeat(140)}ok?`, p)) return 'a 141-word reply passed';
  if (!draftProblems(c, 'explain the mean please', 'The mean is the average [§13]. Which would you use?', p)) return 'a citation of another section passed';
  if (!draftProblems(c, 'what does the passage say about the mean?', good, p)) return 'a message about the passage passed';
  if (!draftProblems(c, 'explain the mean please', 'Photosynthesis converts sunlight into chemical energy inside chloroplasts. What else?', p)) return 'a reply from nowhere passed';
  const notCovered = "The passage I was given doesn't cover overfitting. From general knowledge: it is a model learning the noise in its training data. Where have you seen it?";
  if (draftProblems(c, 'what is overfitting?', notCovered, { ...p, refusal: true })) return `a good not-covered reply was refused: ${draftProblems(c, 'what is overfitting?', notCovered, { ...p, refusal: true })}`;
  if (!draftProblems(c, 'what is overfitting?', 'It is noise [§12]. Seen it?', { ...p, refusal: true })) return 'a citing not-covered reply passed';
  if (draftProblems(c, 'How is the mean affected by outliers?', 'Outliers can skew the mean of a dataset, as the passage explains. [§12]')) return 'the facts style changed';
  return true;
});

check('a thought partner drafts every kind, in proportion, from a sample spread through the book', () => {
  const { partnerKindsFor, sampleSections, PARTNER_KINDS } = S_;
  const n = {};
  for (let i = 0; i < 120; i++) for (const k of partnerKindsFor(i, 3)) n[k] = (n[k] || 0) + 1;
  if (Object.keys(n).sort().join() !== Object.keys(PARTNER_KINDS).sort().join()) return `kinds ${Object.keys(n)}`;
  const share = (k) => n[k] / 360;
  if (Math.abs(share('concept') - 6 / 17) > 0.02 || Math.abs(share('write') - 2 / 17) > 0.02) return JSON.stringify(n);
  const book = Array.from({ length: 558 }, (_, i) => ({ ordinal: i + 1 }));
  const s = sampleSections(book, 120);
  if (s.length !== 120 || s[0].ordinal !== 1 || s[119].ordinal !== 558 || new Set(s.map(x => x.ordinal)).size !== 120) return `sample ${s.length} ${s[0]?.ordinal} ${s.at(-1)?.ordinal}`;
  if (sampleSections(book, 0) !== book || sampleSections(book, 600) !== book) return 'no sample should mean the whole book';
  return true;
});

check('a new run is a facts Reader until told otherwise', () => (blankRun('t').answerStyle === 'facts') || 'answerStyle');

check('a narrower job keeps its parent’s kind', () => {
  const r = blankRun('t');
  setKind(r, 'player'); r.name = 'game';
  r.actions.push({ id: 'k1', kind: 'key', raw: 'Z', name: 'Jump', uses: 2 });
  for (let i = 0; i < 12; i++) r.rows.push({ id: `x${i}`, input: { kind: 'image', path: `f${i}.png` }, answer: '@k1' });
  const nid = startNarrower(r, r.rows.slice(0, 6));
  return S.runs[nid].kind === 'player' || `got ${S.runs[nid].kind}`;
});

/* ── Send: what goes in the file follows the packer ─────────────────── */
{
  const { fileParts } = await import('../src/engine.js');
  const pic = { input: 'image', output: 'function' };
  const got = { adapter: 29968480, projector: 445053216, projector_carried: false, labels: 44, card: 41025,
                base: { name: 'Qwen3-VL-2B', file: 'Qwen3VL-2B-Instruct-Q4_K_M.gguf', bytes: 1107409952 }, total: 30009549 };
  check('a stock projector is not listed, and the total is the packer’s', () => {
    const p = fileParts(pic, got);
    if (p.projector !== 'stock' || p.rows.some(x => x.what === 'The projector')) return `projector ${p.projector}`;
    return Math.abs(p.total * 1e6 - got.total) < 1 || `total ${p.total} vs ${got.total / 1e6}`;
  });
  check('a carried projector is listed, with its measured size', () => {
    const p = fileParts(pic, { ...got, projector_carried: true, total: got.total + got.projector });
    const row = p.rows.find(x => x.what === 'The projector');
    return (p.projector === 'carried' && row && row.mb === got.projector / 1e6 && Math.abs(p.total * 1e6 - (got.total + got.projector)) < 1) || JSON.stringify(p);
  });
  check('the page names the base the table trained on, never a default', () => {
    const p = fileParts({ ...pic, base: 'Qwen3.5-4B' }, got);
    return (p.base.name === 'Qwen3-VL-2B' && p.rows[0].detail === 'what it learned, on Qwen3-VL-2B') || JSON.stringify(p.rows[0]);
  });
  check('while the packer is asked, nothing is guessed about the projector or the base', () => {
    const p = fileParts(pic, undefined);
    return (p.projector === 'pending' && !p.base && !p.rows.some(x => x.what === 'The projector') && p.rows[0].detail === 'what it learned') || JSON.stringify(p);
  });
  const { projectorNote } = await import('../src/screens/send.js');
  check('under the box: a carried projector travels with this Pigeon; a stock one comes with the base', () => {
    const carried = projectorNote('carried', got.base), stock = projectorNote('stock', got.base);
    if (!carried.includes('it travels with this Pigeon') || carried.includes('every picture Pigeon')) return carried;
    return (stock.includes('The projector is not in the file. It comes with the base, Qwen3-VL-2B, which Pigeon downloads once') && projectorNote('none', null) === '') || stock;
  });
  check('when the packer cannot answer, the projector is shown as carried', () => {
    const p = fileParts(pic, {});
    return (p.projector === 'unknown' && p.rows.some(x => x.what === 'The projector')) || JSON.stringify(p);
  });
}

/* A Prompted Pigeon (spec D83): nothing trains, one page, and the base
   table it packs from says exactly what the page says. */
{
  const { setKind, promptedReady } = S_;
  const await_prompted = await import('../src/screens/prompted.js');
  const { tableFor, rowsFor } = await_prompted;
  const r = blankRun('qb-prompted');
  check('choosing Prompted gives a run its prompt settings and no stages to walk', () => {
    if (setKind(r, 'prompted')) return 'refused';
    return (r.prompt && currentStage(r) === 'describe' && nextStep(r).text === 'Name it') || JSON.stringify(nextStep(r));
  });
  check('a Prompted Pigeon is not packed until it has a name, a line and a question', () => {
    r.name = 'Plain English'; r.does = 'Explains a letter in plain English.';
    if (promptedReady(r) !== 'Add a test question') return promptedReady(r);
    r.prompt.rows.push({ input: 'what does "without prejudice" mean', expected: '' });
    return promptedReady(r) === null || promptedReady(r);
  });
  check('prose with a system prompt is a persona, sent its earlier turns as asked, at its temperature', () => {
    r.prompt.system = 'Explain plainly.'; r.prompt.memory = 6;
    const t = tableFor(r);
    return (t.kind === 'persona' && t.system_prompt === 'Explain plainly.' && t.temperature === 0.7
      && t.history.max_turns === 6 && t.output_contract.type === 'text' && !('constrained' in t.output_contract)
      && t.id === 'plain-english' && t.card.does === r.does) || JSON.stringify(t);
  });
  check('JSON with a shape and host-supplied fields is constrained, and its expected answers are read as JSON', () => {
    Object.assign(r.prompt, { output: 'json', schema: '{"type":"object","properties":{"skill":{"type":"string"}}}', hostValues: 'skill', memory: 0 });
    r.prompt.rows = [{ input: 'add 2 and 3', expected: '{"skill":"sum"}' }, { input: 'a poem', expected: '' }];
    const t = tableFor(r), rows = rowsFor(r);
    return (t.kind === 'extractor' && t.output_contract.constrained === true && t.output_contract.host_values[0] === 'skill'
      && t.temperature === 0 && !t.history && rows[0].expected.skill === 'sum' && !('expected' in rows[1])) || JSON.stringify({ t, rows });
  });
  check('example turns that are not a list of {user, pigeon} are named, not sent', () => {
    r.prompt.shots = '[{"user": "hi"}]';
    try { tableFor(r); return 'accepted'; } catch (e) { return e.message.includes('{"user": …, "pigeon": …}') || e.message; }
  });
  check('the longest answer packed is the one the page shows, whichever answer was chosen on the way', () => {
    const { changes, inputs } = { changes: (await_prompted).changes, inputs: (await_prompted).inputs };
    const q = blankRun('qb-limits'); setKind(q, 'prompted'); q.name = 'Plain'; q.does = 'Plain.';
    q.prompt.rows.push({ input: 'what is a lien', expected: '' });
    changes['prompted-output'](q, { value: 'label' });
    if (tableFor(q).output_contract.max_tokens !== 8) return `label ${tableFor(q).output_contract.max_tokens}`;
    changes['prompted-output'](q, { value: 'text' });
    if (tableFor(q).output_contract.max_tokens !== 400) return `back to prose: ${tableFor(q).output_contract.max_tokens} (the plain-english bug)`;
    inputs['prompted-max'](q, { value: '350' });
    changes['prompted-output'](q, { value: 'json' }); changes['prompted-output'](q, { value: 'text' });
    const old = blankRun('qb-old'); setKind(old, 'prompted'); delete old.prompt.limits; old.prompt.maxTokens = 120;
    return (tableFor(q).output_contract.max_tokens === 350 && tableFor(old).output_contract.max_tokens === 120)
      || `${tableFor(q).output_contract.max_tokens} / ${tableFor(old).output_contract.max_tokens}`;
  });
  check('a Prompted run refuses to become a trained one once it has questions', () => {
    r.prompt.shots = '';
    return (setKind(r, 'writer') || '').includes('Start a new run') || 'switched';
  });
  check('a run with rows to train refuses Prompted, even after Change cleared its kind', () => {
    const w = blankRun('qb-writer');
    setKind(w, 'writer'); w.rows.push({ id: 'r00001', input: { kind: 'text', text: 'a note' }, answer: 'A title' });
    w.kind = null;
    return (setKind(w, 'prompted') || '').includes('rows to train on') || 'became Prompted';
  });
}

console.log(`\n  ${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
