/* patch.js — Patch the pigeon, Pigeon's one illustrated character, on the
 * few Flight School screens that are about waiting or finishing.
 *
 * The sheet is Pigeon's own (its assets/patch-sprites.png, lib/patch.js): 12
 * columns of 160 × 144 cells, one animation per row, his feet on the same
 * ground line in every cell. Pigeon steps the frames in JS and gives him a
 * life; here an animation simply loops, in CSS, and a reduced-motion Mac sees
 * its first frame and nothing moves. Copy the sheet over again if Pigeon's
 * changes.
 */

const SHEET = 'assets/patch-sprites.png';
const COLS = 12, ROWS = 17, CELL = [160, 144];

/* Pigeon's TAGS, for the rows used here. */
const TAGS = {
  idle: { row: 0, n: 12, fps: 8 },
  look: { row: 2, n: 12, fps: 8 },
  paint: { row: 6, n: 8, fps: 10 },
  proud: { row: 10, n: 11, fps: 10 },
  loaf: { row: 12, n: 12, fps: 6 }
};

/** One frame of him, or — with `play` — his animation on a loop. */
export function patch (tag = 'idle', { size = 84, frame = 0, play = false, label = '' } = {}) {
  const t = TAGS[tag] || TAGS.idle;
  const h = Math.round(size * CELL[1] / CELL[0]);
  const f = Math.max(0, Math.min(t.n - 1, frame | 0));
  const dur = t.n / t.fps;
  /* The screen is redrawn whole on every teaching tick, and a new element
     starts its animation over. Phased to the clock, each new one picks up
     where the last left off, so he keeps painting rather than stuttering. */
  const phase = play ? `;animation-delay:${(-(performance.now() / 1000) % dur).toFixed(3)}s` : '';
  const style = `width:${size}px;height:${h}px;background-image:url(${SHEET});background-size:${size * COLS}px ${h * ROWS}px;`
    + `background-position:${-f * size}px ${-t.row * h}px;--w:${size}px;--n:${t.n};--dur:${dur.toFixed(3)}s${phase}`;
  return `<span class="patch-sprite${play ? ' plays' : ''}" ${label ? `role="img" aria-label="${label}"` : 'aria-hidden="true"'} style="${style}"></span>`;
}
