import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile, readdir } from 'node:fs/promises';
import path from 'node:path';
import { parseModel, buildCategories } from '../scripts/snapshot';
import type { Library } from '../lib/types';

test('snapshot preserves all public models and structured relationships', async () => {
  const library: Library = JSON.parse(await readFile('../backend/data/library.json', 'utf8'));
  async function files(dir: string): Promise<string[]> { const entries = await readdir(dir, { withFileTypes: true }); return (await Promise.all(entries.map(e => e.isDirectory() ? files(path.join(dir, e.name)) : Promise.resolve(e.name.endsWith('.md') ? [path.join(dir, e.name)] : [])))).flat().sort(); }
  const paths = await files('../01-Models'); assert.equal(library.models.length, paths.length);
  const originals = await Promise.all(paths.map(async file => parseModel(await readFile(file, 'utf8'), path.relative('..', file).replaceAll('\\', '/'))));
  assert.deepEqual(library.models.map(({ reading, ...model }) => model), originals);
  assert.deepEqual(library.categories, buildCategories(originals));
  assert.equal(library.categories.reduce((sum, c) => sum + c.count, 0), paths.length);
  assert.equal(library.associations.length, 0);
  assert(library.models.every(m => m.sourcePath.startsWith('01-Models/')));
});
