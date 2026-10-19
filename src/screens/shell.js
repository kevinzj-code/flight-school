/* shell.js — CARRIES FORWARD. The rail, kept exactly as Flight School has it:
 * New run · Runs · Pigeons · Bench. A redesign that moves the furniture as
 * well as the flow makes both harder to judge. */

import { S, allRuns } from '../state.js';
import { needs } from '../engine.js';
import { esc, ICON, BUSY } from './html.js';

/* The quill from the app icon (src-tauri/icons/icon-source.svg), without its
   tile. The ink line it has written takes the page's ink colour, so it stays
   visible on the dark rail too; the marks cut into the vane (.qm-*) are the
   icon's navy in dark mode and white in light (app.css). */
const QUILL = `<svg viewBox="70 20 380 460" aria-hidden="true" focusable="false">
  <path d="M92 446c28 16 54-14 84-2s50 18 80 2 52-16 76-4" fill="none" stroke="currentColor" stroke-width="16" stroke-linecap="round" stroke-linejoin="round"/>
  <g transform="translate(264 220) rotate(40) scale(1.12)">
    <path d="M0-214C74-172 86-48 42 92H-34C-80-18-66-170 0-214Z" fill="#b4bac0"/>
    <path class="qm-stroke" d="M78-92 44-64M-68-20-36 4M68 10 38 34" fill="none" stroke-width="13" stroke-linecap="round"/>
    <path class="qm-stroke" d="M0-176C6-90 8 0 4 96" fill="none" stroke-width="9" stroke-linecap="round"/>
    <rect x="-44" y="-120" width="18" height="36" rx="9" fill="#a58fc0" transform="rotate(-10 -35 -102)"/>
    <rect x="-40" y="-72" width="18" height="36" rx="9" fill="#d8e7f0" transform="rotate(-10 -31 -54)"/>
    <rect x="-9" y="88" width="20" height="34" fill="#d8e7f0"/>
    <rect x="-30" y="116" width="62" height="30" rx="12" fill="#42210b"/>
    <path d="M-26 146H1V238Z" fill="#ffb340"/>
    <path d="M1 146H28L1 238Z" fill="#8f6422"/>
    <circle class="qm-fill" cx="1" cy="176" r="7"/>
    <path class="qm-stroke" d="M1 183V222" stroke-width="4" stroke-linecap="round"/>
  </g>
</svg>`;

export function rail () {
  const item = (view, icon, label, count) => {
    const active = S.view === view || (view === 'runs' && S.view === 'run');
    return `<a id="nav-${view}" class="navitem${active ? ' active' : ''}" href="#/${view}"${active ? ' aria-current="page"' : ''}>
      ${icon}<span>${label}</span>${count != null ? `<span class="count">${count}</span>` : ''}
    </a>`;
  };
  return `<nav class="rail" aria-label="Flight School" data-tauri-drag-region>
    <div class="brand" data-tauri-drag-region><span class="brand-mark">${QUILL}</span><span>Flight School</span></div>
    <button type="button" id="nav-new" class="navitem new" data-do="new-run">${ICON.plus}<span>New run</span></button>
    ${item('runs', ICON.runs, 'Runs', allRuns().length)}
    ${item('pigeons', ICON.pigeons, 'Pigeons')}
    ${item('bench', ICON.bench, 'Bench')}
    <button type="button" id="nav-settings" class="navitem settings" data-do="settings" title="Settings (⌘,)">${ICON.gear}<span>Settings</span></button>
    ${macRow()}
  </nav>`;
}

/* One row at the foot of the rail: whether this Mac can teach, and the one
   button that makes it able to. A fresh Mac learns it here, not at Teach. */
function macRow () {
  const e = S.env;
  if (!e?.checked) return '';
  if (e.busy) return `<div class="mac busy" role="status">${BUSY}<span><b>Setting up this Mac</b><span class="hint" data-live="setup-line">${esc(e.line || '')}</span></span></div>`;
  if (e.error) return `<div class="mac bad" role="alert"><i class="dot"></i><span><b>Setup did not finish</b><span class="hint">${esc(e.error)}</span></span><button type="button" class="btn sm" data-do="setup-env">Try again</button></div>`;
  const need = needs(null);
  if (!need.length) return `<div class="mac ok" title="${e.vision ? 'The training tools and the picture tools are set up.' : 'The training tools are set up. Picture training sets up its own tools the first time.'}"><i class="dot"></i><span><b>This Mac</b><span class="hint">Ready to teach</span></span></div>`;
  const first = need[0];
  const short = { pigeon: 'Needs Pigeon', python: 'Needs Python 3.9+', env: 'Not set up yet', vision: 'Pictures need setup' }[first.id] || 'Needs setup';
  return `<div class="mac warn" title="${esc(first.text)}"><i class="dot"></i><span><b>This Mac</b><span class="hint">${short}</span></span>
    ${first.setup ? `<button type="button" class="btn sm primary" data-do="setup-${first.setup}" title="${esc(first.text)}">Set up</button>` : `<button type="button" class="btn sm ghost" data-do="env-recheck" title="${esc(first.text)}">Check</button>`}</div>`;
}

