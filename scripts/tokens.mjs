#!/usr/bin/env node
/* tokens.mjs — copy Pigeon's styles/tokens.css over src/tokens.css.
 *
 * The two apps draw from one tokens file so they read as one family; Pigeon
 * holds the source (see that file's header). Run after changing it there:
 *
 *     npm run tokens              # Pigeon checked out beside this repo
 *     PIGEON_REPO=/path npm run tokens
 */
import { copyFileSync, existsSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const ROOT = resolve(fileURLToPath(new URL('.', import.meta.url)), '..');
export const pigeonTokens = () => join(process.env.PIGEON_REPO || resolve(ROOT, '../pigeon'), 'styles/tokens.css');

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  const from = pigeonTokens();
  if (!existsSync(from)) { console.error(`No Pigeon tokens at ${from} — set PIGEON_REPO to Pigeon's checkout.`); process.exit(1); }
  copyFileSync(from, join(ROOT, 'src/tokens.css'));
  console.log(`src/tokens.css ← ${from}`);
}
