/* menu.js — Pigeon's small floating menu (its src/ui/menu.js), for a ⋯ button:
 * it opens under the button rather than as a dialog in the middle of the
 * window. Escape, a click elsewhere or the window losing focus closes it; the
 * arrow keys move through the items, and focus goes back to the ⋯.
 *
 * It lives on <body>, outside the page render() redraws, and is gone before
 * any item acts.
 */

import { esc, ICON } from './html.js';

/** items: [{ id, label, icon, danger, title } | { rule: true }] */
function menuHtml (items) {
  return items.map(it => it.rule
    ? '<div class="rule" role="separator"></div>'
    : `<button type="button" role="menuitem" data-item="${esc(it.id)}"${it.danger ? ' class="danger"' : ''}${it.title ? ` title="${esc(it.title)}"` : ''}>${
      it.icon && ICON[it.icon] ? ICON[it.icon] : ''}<span>${esc(it.label)}</span></button>`).join('');
}

/** Under an element, or at a point `{ x, y }` for a right-click. */
export function openMenu (at, items, onPick) {
  document.querySelectorAll('.ctx').forEach(m => m.remove());
  const menu = document.createElement('div');
  menu.className = 'ctx';
  menu.setAttribute('role', 'menu');
  menu.innerHTML = menuHtml(items);
  document.body.appendChild(menu);

  const fromElement = typeof at.getBoundingClientRect === 'function';
  const r = fromElement ? at.getBoundingClientRect() : { left: at.x, right: at.x, top: at.y, bottom: at.y };
  const below = r.bottom + 4;
  /* Right-aligned to a ⋯, which sits at the right edge of its row; from the pointer, left-aligned to it. */
  const left = fromElement ? r.right - menu.offsetWidth : r.left;
  menu.style.left = `${Math.max(8, Math.min(left, window.innerWidth - menu.offsetWidth - 8))}px`;
  menu.style.top = `${below + menu.offsetHeight > window.innerHeight - 8 ? Math.max(8, r.top - menu.offsetHeight - 4) : below}px`;

  const buttons = [...menu.querySelectorAll('button')];
  const cameFrom = fromElement ? document.activeElement : null;
  const close = () => {
    menu.remove();
    document.removeEventListener('mousedown', away, true);
    document.removeEventListener('keydown', onKey, true);
    window.removeEventListener('blur', close);
    if (cameFrom?.focus && document.contains(cameFrom)) cameFrom.focus();
  };
  function away (e) { if (!menu.contains(e.target)) close(); }
  function onKey (e) {
    if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); close(); return; }
    if (e.key !== 'ArrowDown' && e.key !== 'ArrowUp') return;
    e.preventDefault();
    const i = buttons.indexOf(document.activeElement), n = buttons.length;
    const next = i < 0 ? (e.key === 'ArrowDown' ? 0 : n - 1) : (i + (e.key === 'ArrowDown' ? 1 : n - 1)) % n;
    buttons[next]?.focus();
  }
  menu.addEventListener('click', (e) => {
    const b = e.target.closest('button[data-item]');
    if (!b) return;
    e.stopPropagation();
    close();
    onPick(b.dataset.item);
  });
  setTimeout(() => document.addEventListener('mousedown', away, true), 0);
  document.addEventListener('keydown', onKey, true);
  window.addEventListener('blur', close);
  /* From a ⋯ the keyboard is already there; from a right-click it is not. */
  if (fromElement) buttons[0]?.focus();
  return close;
}
