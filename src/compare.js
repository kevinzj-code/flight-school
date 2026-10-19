/* compare.js — a Web Worker. The pairwise picture comparisons (conflicts,
 * repeats) are O(n²) over the table, and on a game recording nearly every
 * cross-answer pair reaches the pixel loop; run on the page they froze it for
 * seconds. Here they run beside it. The rules themselves stay in state.js,
 * so the tests and the no-worker fallback (engine.js) use the same code. */

import { repeatGroups, conflictPairs } from './state.js';

self.onmessage = (e) => {
  const { id, kind, run, prints, opts } = e.data;
  try {
    if (kind === 'repeats') {
      const groups = repeatGroups(run, prints, opts).map(g => ({ key: g.key, ids: g.rows.map(x => x.id), answer: g.answer, exam: g.exam }));
      self.postMessage({ id, groups });
    } else {
      self.postMessage({ id, pairs: conflictPairs(run, prints, opts) });
    }
  } catch (err) {
    self.postMessage({ id, error: String(err) });
  }
};
