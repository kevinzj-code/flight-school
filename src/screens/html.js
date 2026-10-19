/* html.js — CARRIES FORWARD. Escaping and the icon set every screen shares.
 * Icons are inline stroke SVG in currentColor, so they take the text's colour
 * and need no request. */

export const esc = (s) => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

/* Pigeon's helper (its src/ui/html.js): 1.9 by default, a glyph may say otherwise. */
const svg = (body, width = '1.9') => `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="${width}" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${body}</svg>`;

/* The glyphs both apps draw — plus, image, check, copy, send, back, more,
   trash, stop, gear, close, search, rename — are Pigeon's, character for character (the test
   suite compares them with Pigeon's src/ui/html.js). The rest are Flight
   School's own, drawn to the same grid and weight. */
export const ICON = {
  plus: svg('<path d="M12 5v14M5 12h14"/>', '2'),
  runs: svg('<path d="M4 6h16M4 12h16M4 18h10"/>'),
  /* Patch, in one colour: Pigeon's own mark (Pigeon's assets/patch-icon.svg),
     its white cut-outs made holes so it takes the text's colour like the rest. */
  pigeons: `<svg viewBox="0 0 416 416" fill="currentColor" aria-hidden="true"><mask id="patch-cut" maskUnits="userSpaceOnUse" x="0" y="0" width="416" height="416"><rect width="416" height="416" fill="#fff"/><g fill="#000"><circle cx="300.3" cy="72" r="44"/><rect x="236.3" y="46" width="147" height="30" rx="15"/><rect x="256.3" y="156" width="18" height="36" rx="8.8"/><rect x="288.3" y="168" width="18" height="36" rx="8.8"/><rect x="144.3" y="208" width="56" height="22" rx="10.8"/></g><path d="M108.3 204c16-32 38.6-42.2 82.6-33l32 7.2" fill="none" stroke="#000" stroke-width="16.8" stroke-linecap="round"/></mask><g mask="url(#patch-cut)"><path d="M372.3 228c8 72-32 124-100 140-80 16-160 0-204-52-32-40-48-92-32-132C.3 172-15.7 112 20.3 84c28-20 52 4 52 32 40-16 92 0 128 20l48 24h124v68Z"/><path d="M244.7 236V76C244.7 24 272.7 0 308.7 0s64 24 64 76v160h-128Z"/><rect x="356.3" y="72" width="59.3" height="27.1" rx="12"/><path d="M373.3 100.9l20.5 1.1c5.1 0 9.3 4.2 9.3 9.3v1.1c0 5.1-4.2 9.3-9.3 9.3h-34.1c-5.1 0-3.1-6.5-3.1-11.6l.3-10.8c0-5.1 9.8.8 16.4 1.7Z"/><path d="M224.3 348l16 48h40M184.3 356v48h44" fill="none" stroke="currentColor" stroke-width="24" stroke-linecap="round" stroke-linejoin="round"/></g><ellipse cx="321.4" cy="72" rx="12" ry="22"/></svg>`,
  bench: svg('<path d="M12 4v16M5 20h14M6 8l-3 6h6zM18 8l-3 6h6zM6 8h12"/>'),
  text: svg('<path d="M5 6h14M12 6v13M9 19h6"/>'),
  image: svg('<rect x="3" y="4" width="18" height="16" rx="2.5"/><circle cx="8.5" cy="9.5" r="1.6"/><path d="M21 15.5l-4.8-4.8a1.5 1.5 0 0 0-2.1 0L6 19"/>'),
  fn: svg('<path d="M9 5c-2 0-3 1-3 3v2c0 1-1 2-2 2 1 0 2 1 2 2v2c0 2 1 3 3 3M15 5c2 0 3 1 3 3v2c0 1 1 2 2 2-1 0-2 1-2 2v2c0 2-1 3-3 3"/>'),
  prose: svg('<path d="M4 7h16M4 12h16M4 17h11"/>'),
  lock: svg('<rect x="5" y="11" width="14" height="9" rx="2"/><path d="M8 11V8a4 4 0 0 1 8 0v3"/>'),
  check: svg('<path d="M5 12.5l4.5 4.5L19 7.5"/>', '2.2'),
  arrow: svg('<path d="M5 12h14M13 6l6 6-6 6"/>'),
  warn: svg('<path d="M12 4l9 16H3z"/><path d="M12 10v4M12 17.5v.5"/>'),
  close: svg('<path d="M6 6l12 12M18 6L6 18"/>', '1.8'),
  wrench: svg('<path d="M14.5 6.5a4 4 0 0 0 5 5L12 19a2.1 2.1 0 0 1-3-3l7.5-7.5a4 4 0 0 1-2-2z"/>'),
  rec: svg('<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="3.5" fill="currentColor"/>'),
  stop: svg('<rect x="6.5" y="6.5" width="11" height="11" rx="2.5"/>'),
  pair: svg('<rect x="3" y="5" width="7" height="14" rx="1.5"/><rect x="14" y="5" width="7" height="14" rx="1.5"/><path d="M10 12h4"/>'),
  folder: svg('<path d="M3.5 7.5a2 2 0 0 1 2-2h4l2 2h7a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2h-13a2 2 0 0 1-2-2z"/>'),
  spark: svg('<path d="M12 3.5l1.9 5.1 5.1 1.9-5.1 1.9L12 17.5l-1.9-5.1L5 10.5l5.1-1.9z"/><path d="M18.5 16.5l.8 2 2 .8-2 .8-.8 2-.8-2-2-.8 2-.8z"/>'),
  window: svg('<rect x="3" y="4.5" width="18" height="15" rx="2"/><path d="M3 8.5h18"/>'),
  send: svg('<path d="M22 2 11 13M22 2l-7 20-4-9-9-4z"/>', '2'),
  merge: svg('<path d="M7 4v6a4 4 0 0 0 4 4h6M17 14l-3-3M17 14l-3 3M7 20v-6"/>'),
  back: svg('<path d="M15 5l-7 7 7 7"/>', '2'),
  /* The app's own mark: the quill from the icon, as a stroke. */
  pen: svg('<path d="M19.5 4.5c-5.5 0-10 3.5-12 9l-3 6 6-3c5.5-2 9-6.5 9-12z"/><path d="M7.5 16.5l8.5-8.5"/>'),
  gear: svg('<circle cx="12" cy="12" r="3"/><path d="M12 3v2.5M12 18.5V21M3 12h2.5M18.5 12H21M5.6 5.6l1.8 1.8M16.6 16.6l1.8 1.8M5.6 18.4l1.8-1.8M16.6 7.4l1.8-1.8"/>'),
  more: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><circle cx="5" cy="12" r="1"/><circle cx="12" cy="12" r="1"/><circle cx="19" cy="12" r="1"/></svg>',
  copy: svg('<rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V6a2 2 0 0 1 2-2h9"/>', '1.8'),
  search: svg('<circle cx="11" cy="11" r="6.5"/><path d="M16 16l4.5 4.5"/>', '2'),
  rename: svg('<path d="M4 7V5h16v2"/><path d="M12 5v14"/><path d="M9 19h6"/>'),
  trash: svg('<path d="M4 7h16M9 7V4.5h6V7M6.5 7l1 13h9l1-13"/>')
};

/* Working on it: Pigeon's three dots, never a spinner — a spinner implies a
   measurable share of something, and nothing here knows how long it will take. */
export const BUSY = '<span class="busy-dots" aria-hidden="true"><i></i><i></i><i></i></span>';

/* A Pigeon's avatar, drawn the way Pigeon draws it (Pigeon's src/ui/pigeons.js
   avatarHtml, styles/cast.css): a squircle; a job — one input, one output, which
   is everything Flight School makes — in lower-case mono initials on the
   writers' ring; a voice in capitals on the warm ground. */
function initials (name) {
  const words = String(name || '?').trim().split(/[\s_-]+/).filter(Boolean);
  if (words.length > 1) return (words[0][0] + words[1][0]).toUpperCase();
  const w = words[0] || '?';
  return (w[0].toUpperCase() + (w[1] || '')).slice(0, 2);
}
/* A Pigeon with an icon wears it, as in Pigeon: the PNG it was packed with, or
   on the Send screen the look being drawn. Only a PNG or an SVG data URL. */
const ICON_SRC = /^data:image\/(png;base64,[A-Za-z0-9+/]+={0,2}|svg\+xml;charset=utf-8,[^"'<>\s]+)$/;
export const avatar = (name, { kind = 'job', size = 40, src = null } = {}) => {
  if (src && ICON_SRC.test(src)) return `<span class="av av-img" style="--s:${size}px" aria-hidden="true"><img alt="" src="${src}"></span>`;
  const voice = !kind || kind === 'persona';
  return `<span class="av ${voice ? 'av-voice' : 'av-job'}" style="--s:${size}px" aria-hidden="true">${esc(voice ? initials(name) : initials(name).toLowerCase())}</span>`;
};

/* Relative time for the runs list. */
export function when (iso) {
  const mins = Math.round((Date.now() - Date.parse(iso)) / 60000);
  if (mins < 60) return `${Math.max(1, mins)} min ago`;
  const h = Math.round(mins / 60);
  if (h < 24) return `${h} h ago`;
  const d = Math.round(h / 24);
  return d === 1 ? 'yesterday' : `${d} days ago`;
}

export const n = (x) => Number(x || 0).toLocaleString('en-GB');
