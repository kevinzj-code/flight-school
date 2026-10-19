/* art.js — the recorded window, drawn: a frame from the table with the places
 * clicked in it (targetsOverlay), and one place cut out of it (crop). The
 * prototype drew stand-in windows here; these draw the run's own frames, in
 * the frame's own pixels, so a box sits where the clicks landed. */

import { engine } from '../engine.js';
import { esc } from './html.js';

/* The frame to draw a run's places over: the newest recorded frame that has a
   click in it, else any picture row. */
function backdrop (r, pick = null) {
  const rows = r.rows.filter(x => x.input.kind === 'image' && x.input.path);
  const withClick = rows.filter(x => x.input.click && (!pick || x.answer === `@${pick}`));
  const row = withClick[withClick.length - 1] || rows[rows.length - 1];
  return row ? row.input.path : null;
}

const size = (r) => [r.window?.w || 768, r.window?.h || 480];

function frameSvg (r, path, inner, { viewBox = null, cls = 'mail', label = 'The recorded window' } = {}) {
  const [w, h] = size(r);
  const url = path ? engine.image(r, path) : null;
  return `<svg class="art ${cls}" viewBox="${viewBox || `0 0 ${w} ${h}`}" role="img" aria-label="${esc(label)}">
    <rect width="${w}" height="${h}" fill="var(--surface-sunk)"/>
    ${url ? `<image href="${url}" x="0" y="0" width="${w}" height="${h}" preserveAspectRatio="none"/>` : ''}
    ${inner}
  </svg>`;
}

/* One place, cut out of the frame it was clicked in. */
export function crop (r, t) {
  const [x, y, w, h] = t.rect;
  const pad = 10;
  return frameSvg(r, backdrop(r, t.id), '', { viewBox: `${x - pad} ${y - pad} ${w + pad * 2} ${h + pad * 2}`, cls: 'crop', label: t.name ? `Where ${t.name} is` : 'An unnamed click' });
}

/* The click overlay: dashed boxes with counts, the clicks that made them, the
   strays, and a grid as a guide for the eye. */
export function targetsOverlay (r, targets, strays, { grid = true, selected = null } = {}) {
  const [W, H] = size(r);
  const cell = Math.max(W, H) / 20;
  const gridDef = grid
    ? `<defs><pattern id="tg-grid" width="${cell}" height="${cell}" patternUnits="userSpaceOnUse"><path d="M${cell} 0V${cell}M0 ${cell}H${cell}" fill="none" stroke="var(--aqua)" stroke-opacity=".35" stroke-width="1"/></pattern></defs><rect width="${W}" height="${H}" fill="url(#tg-grid)"/>`
    : '';
  const k = W / 790;
  const boxes = targets.map(t => {
    const [x, y, w, h] = t.rect;
    const on = selected === t.id;
    const dots = r.rows.filter(z => z.answer === `@${t.id}` && z.input.click).slice(0, 12)
      .map(z => `<circle cx="${z.input.click[0]}" cy="${z.input.click[1]}" r="${4 * k}" fill="var(--danger)" fill-opacity=".8"/>`).join('');
    return `<g class="tbox${on ? ' on' : ''}" data-do="target-select" data-id="${t.id}" role="button" tabindex="-1" aria-label="${esc(t.name || 'Unnamed click')}, ${t.uses} clicks">
      <rect x="${x - 4}" y="${y - 4}" width="${w + 8}" height="${h + 8}" rx="${6 * k}" fill="${on ? 'var(--aqua-tint)' : 'transparent'}" fill-opacity=".5" stroke="var(--aqua)" stroke-width="${(on ? 3 : 2) * k}" stroke-dasharray="${on ? '0' : `${6 * k} ${4 * k}`}"/>
      ${dots}
      <rect x="${x + w - 30 * k}" y="${y - 22 * k}" width="${36 * k}" height="${18 * k}" rx="${9 * k}" fill="var(--ink)"/>
      <text x="${x + w - 12 * k}" y="${y - 9 * k}" text-anchor="middle" font-family="IBM Plex Mono, monospace" font-size="${11 * k}" font-weight="600" fill="var(--paper)">${t.uses}</text>
    </g>`;
  }).join('');
  const s = strays.map(p => `<circle cx="${p.x}" cy="${p.y}" r="${4 * k}" fill="var(--danger)" fill-opacity="${p.decision === 'leave' ? '.25' : '.75'}"/>`).join('');
  return frameSvg(r, backdrop(r), gridDef + boxes + s);
}
