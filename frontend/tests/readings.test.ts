import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { loadReadings, attachReadings, parseReading } from '../scripts/readings';
import type { Library } from '../lib/types';

test('published reading batches bind to their exact public model versions', async () => {
  const readings = await loadReadings('content/model-readings');
  const library: Library = JSON.parse(await readFile('../backend/data/library.json', 'utf8'));
  const batches: { modelIds: string[] }[] = JSON.parse(await readFile('content/reading-batches.json', 'utf8'));
  const published = batches.flatMap(b => b.modelIds);
  assert.equal(new Set(published).size, published.length, 'batch entries must be unique');
  assert.deepEqual(readings.map(r => r.modelId).sort(), published.sort());
  const originals = library.models.map(({ reading, ...model }) => model);
  assert.deepEqual(attachReadings(originals, readings), library.models);
  assert.equal(library.models.filter(m => !m.reading).length, library.models.length - published.length);
  for (const r of readings) {
    const hanCount = (r.body.match(/\p{Script=Han}/gu) || []).length;
    assert(hanCount >= 1200 && hanCount <= 2000, `${r.modelId}: ${hanCount} Chinese characters`);
    assert(!r.body.includes('机制接口'), r.modelId);
    assert(!/[—–]/.test(r.body), r.modelId);
    assert(r.sources.length >= 2);
  }
  assert.throws(() => attachReadings(originals, [{ ...readings[0], modelVersion: 999 }]), /mismatched/);
  assert.throws(() => attachReadings(originals, [{ ...readings[0], modelId: 'unknown' }]), /missing/);
  assert.throws(() => attachReadings(originals, [readings[0], readings[0]]), /Duplicate/);
  assert.deepEqual(attachReadings(originals, []), originals, 'no-reading models retain original data');
});
test('reading parsing rejects broken citations and missing content', async () => {
  const raw = await readFile('content/model-readings/base-rate.md', 'utf8');
  assert.throws(() => parseReading(raw.replace('#source-bayes', '#source-missing'), 'fixture'), /Unknown citation/);
  assert.throws(() => parseReading(raw.replace('<!-- technical-appendix -->', ''), 'fixture'), /appendix/);
});
