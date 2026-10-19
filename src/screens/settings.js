/* settings.js — Settings, as a sheet over the window (⌘,), drawn the way
 * Pigeon draws its own (Pigeon's src/ui/settings.js): a title and a close
 * button, a row of tabs on a track, and rows of a label beside its control.
 *
 * Five tabs, each holding only what Flight School really has:
 *   General   how it looks, how Collect draws pictures, when Send asks the Icon Maker
 *   This Mac  the training tools, the picture tools, the models it borrows from Pigeon
 *   Pigeons   the lineup, and which of them makes icons
 *   Privacy   what leaves this Mac (nothing), and the two permissions recording needs
 *   Advanced  where the runs are kept, and the screens older runs were made with
 *
 * The choices are this Mac's, kept in the page's own storage like Collect's
 * picture size always was.
 */

import { S } from '../state.js';
import { engine, needs, ramGb } from '../engine.js';
import { esc, ICON, avatar, BUSY } from './html.js';
import { kindWords, refresh as refreshLineup } from './pigeons.js';

export const TABS = [['general', 'General'], ['mac', 'This Mac'], ['pigeons', 'Pigeons'], ['privacy', 'Privacy'], ['advanced', 'Advanced']];

/* ── the choices ─────────────────────────────────────────────────────── */

const PREFS = {
  theme: { key: 'fs.theme', values: ['system', 'light', 'dark'], fallback: 'system' },
  layout: { key: 'fs.layout', values: ['list', 'grid'], fallback: 'list' },
  picSize: { key: 'fs.picSize', values: ['s', 'm', 'l'], fallback: 's' },
  iconAsk: { key: 'fs.iconAsk', values: ['auto', 'ask'], fallback: 'auto' }
};
export function pref (name) {
  const p = PREFS[name];
  try { const v = localStorage.getItem(p.key); return p.values.includes(v) ? v : p.fallback; } catch { return p.fallback; }
}
export function setPref (name, value) {
  const p = PREFS[name];
  if (!p || !p.values.includes(value)) return;
  try { localStorage.setItem(p.key, value); } catch { /* this session only */ }
  if (name === 'theme') applyTheme(value);
  /* Collect reads these from S.ui as it draws. */
  if (name === 'layout' || name === 'picSize') S.ui[name] = value;
}
/* tokens.css does the rest from the root's data-theme; the window's own
   appearance — its traffic lights and pickers — is Rust's (appearance.rs). */
export function applyTheme (t = pref('theme')) {
  document.documentElement.dataset.theme = t;
  window.__TAURI__?.core?.invoke('window_theme', { theme: t }).catch(() => {});
}

/* ── drawing ─────────────────────────────────────────────────────────── */

const row = (label, body) => `<div class="st-row"><span class="st-label">${esc(label)}</span><div class="st-val">${body}</div></div>`;
/* A segmented control, named out loud as the label beside it says. */
const seg = (said, name, options) => `<div class="seg st-seg" role="radiogroup" aria-label="${esc(said)}">${options.map(([v, label]) =>
  `<button type="button" class="seg-btn" role="radio" id="st-${name}-${v}" aria-checked="${pref(name) === v}" tabindex="${pref(name) === v ? 0 : -1}" data-do="st-pref" data-pref="${name}" data-value="${v}">${esc(label)}</button>`).join('')}</div>`;
const note = (text) => `<p class="st-note">${text}</p>`;
const state = (ok, words) => `<span class="st-state ${ok ? 'ok' : 'warn'}"><i class="dot"></i>${esc(words)}</span>`;

function general () {
  return row('Appearance', seg('Appearance', 'theme', [['system', 'Match Mac'], ['light', 'Light'], ['dark', 'Dark']])) +
    row('Pictures in Collect', `${seg('Pictures in Collect', 'layout', [['list', 'List'], ['grid', 'Grid']])}
      <div class="st-gap">${seg('Picture size', 'picSize', [['s', 'Small'], ['m', 'Medium'], ['l', 'Large']])}</div>
      ${note('How a picture table draws. Collect’s own buttons change the same thing.')}`) +
    row('Icons on Send', `${seg('Icons on Send', 'iconAsk', [['auto', 'Ask the Icon Maker'], ['ask', 'Only when I ask']])}
      ${note('With an Icon Maker in the lineup, Send can ask it for a look as soon as it opens. It wakes a model for a few seconds; the random look shows meanwhile.')}`);
}

function mac () {
  const e = S.env;
  if (!e?.checked) return note('Checking this Mac…');
  const py = e.py || {};
  const need = needs(null);
  const setupBusy = Boolean(e.busy);
  const training = py.venv_ready
    ? state(true, `Ready · ${py.system_python_version || 'Python'}`)
    : state(false, need.find(n => n.id === 'python' || n.id === 'env')?.text || 'Not set up yet');
  const models = e.models?.pigeon_present
    ? `${state(true, `Kept in Pigeon · ${e.models.files ?? 0} model file${e.models.files === 1 ? '' : 's'}${e.models.gb ? ` · ${e.models.gb.toFixed(1)} GB` : ''}`)}
       ${e.models.store ? note(`<span class="mono">${esc(e.models.store)}</span>`) : ''}`
    : state(false, e.models?.message || 'Pigeon is not on this Mac yet');
  return row('Training tools', `${training}
      ${!py.venv_ready && !need.some(n => n.id === 'python') ? `<div class="st-btns"><button type="button" class="btn sm primary" data-do="setup-env" ${setupBusy ? 'disabled' : ''}>Set up</button></div>` : ''}
      ${note('Python and the trainer, in a folder of their own. Setting up happens once and takes about two minutes.')}`) +
    row('Picture tools', `${e.vision ? state(true, 'Ready') : state(false, 'Not set up yet')}
      ${!e.vision && py.venv_ready ? `<div class="st-btns"><button type="button" class="btn sm" data-do="setup-vision" ${setupBusy ? 'disabled' : ''}>Set up</button></div>` : ''}
      ${note('Only picture runs need these. They set themselves up the first time one teaches.')}`) +
    row('Models', `${models}${note('Flight School borrows its models from Pigeon rather than keeping a second copy.')}`) +
    row('Memory', `<p>${ramGb() ? `${ramGb().toFixed(0)} GB` : '—'}</p>${note('Teach sizes each run to fit it.')}`) +
    (setupBusy ? row('Setting up', `<p class="setup-live">${BUSY}<span data-live="setup-line">${esc(e.line || '')}</span></p>`) : '') +
    `<div class="st-btns"><button type="button" class="btn sm ghost" data-do="env-recheck">Check again</button></div>`;
}

function pigeons () {
  if (S.pigeons == null) { refreshLineup(); return note('Reading the lineup…'); }
  const maker = S.pigeons.find(p => p.skill === 'pigeon-look');
  return row('Icon Maker', maker
      ? `<p><b>${esc(maker.name)}</b> draws icons on Send.</p>`
      : `<p>None in the lineup.</p>${note('Send draws a random look from the name, which you can change part by part.')}`) +
    row('The lineup', S.pigeons.length
      ? `<div class="st-list">${S.pigeons.map(p => `<div class="st-pigeon">${avatar(p.name, { kind: p.kind, src: p.icon, size: 28 })}<span><b>${esc(p.name)}</b><small>${esc(kindWords(p.kind))}${p.vision ? ' · from pictures' : ''}</small></span></div>`).join('')}</div>`
      : note('Nothing yet. A run that reaches Send drops its Pigeon here.')) +
    `<div class="st-btns"><button type="button" class="btn sm" data-do="pigeon-add">${ICON.plus}Add a Pigeon…</button><button type="button" class="btn sm ghost" data-do="st-go" data-view="pigeons">Open Pigeons</button></div>`;
}

function privacy () {
  const perm = S.ui.settingsPerm;
  if (perm === undefined) {
    S.ui.settingsPerm = null;
    engine.recPermission().then(p => { S.ui.settingsPerm = p || { screen: false, input: false }; window.dispatchEvent(new Event('proto:render')); });
  }
  const permRow = (label, ok, which, why) => row(label, `${perm ? state(ok, ok ? 'Allowed' : 'Not allowed') : note('Asking macOS…')}
      ${perm && !ok ? `<div class="st-btns"><button type="button" class="btn sm" data-do="st-privacy" data-which="${which}">Open ${esc(label)} settings</button></div>` : ''}
      ${note(why)}`);
  return row('What leaves this Mac', `<p>Nothing. Collecting, teaching, the check ride and packing all run here, on this Mac’s own models.</p>
      ${note('A run’s rows, pictures and training stay in its own folder until you send the Pigeon somewhere yourself.')}`) +
    permRow('Screen Recording', perm?.screen, 'screen', 'Record sees the one window you pick, only while it records.') +
    permRow('Input Monitoring', perm?.input, 'input', 'Record notes the keys and clicks you make in that window, as the answers.');
}

function advanced () {
  return row('Runs are kept in', `${S.runsRoot ? `<p class="mono st-path">${esc(S.runsRoot)}</p>
      <div class="st-btns"><button type="button" class="btn sm" data-do="st-reveal">Show in Finder</button></div>` : note('—')}
      ${note('One folder per run: its table, its pictures, and what teaching made.')}`);
}

const BODY = { general, mac, pigeons, privacy, advanced };

export function settingsDialog (d) {
  const tab = BODY[d.tab] ? d.tab : 'general';
  return `<div class="scrim" data-do="dialog-close"></div>
  <div class="settings-card" role="dialog" aria-modal="true" aria-labelledby="dlg-title">
    <div class="st-head"><h2 id="dlg-title" tabindex="-1">Settings</h2><button type="button" class="icon-btn" data-do="dialog-close" aria-label="Close settings">${ICON.close}</button></div>
    <div class="tabs st-tabs" role="tablist" aria-label="Settings sections">${TABS.map(([k, label]) =>
      `<button type="button" class="tab" role="tab" id="st-tab-${k}" aria-selected="${tab === k}" data-do="st-tab" data-tab="${k}">${esc(label)}</button>`).join('')}</div>
    <div class="st-body" role="tabpanel" aria-labelledby="st-tab-${tab}">${BODY[tab]()}</div>
  </div>`;
}

/* Clicks inside the sheet, routed from app.js. */
export const actions = {
  'settings' (el) { S.ui.dialog = { kind: 'settings', tab: el.dataset.tab || 'general' }; },
  'st-tab' (el) { S.ui.dialog = { kind: 'settings', tab: el.dataset.tab }; S.ui.focusNext = `st-tab-${el.dataset.tab}`; },
  'st-pref' (el) { setPref(el.dataset.pref, el.dataset.value); },
  'st-privacy' (el) { engine.recOpenSettings(el.dataset.which).catch(() => {}); S.ui.settingsPerm = undefined; },
  'st-go' (el) { S.ui.dialog = null; location.hash = `#/${el.dataset.view}`; },
  'st-reveal' () { window.__TAURI__?.core.invoke('qb_reveal', { path: S.runsRoot }).catch(() => {}); }
};
