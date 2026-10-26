import { readdir, readFile } from 'node:fs/promises';
import path from 'node:path';
import { parse } from 'yaml';
import type { Model, ModelReading } from '../lib/types';

export function parseReading(text: string, sourcePath: string): ModelReading {
  const normalized = text.replace(/\r\n/g, '\n');
  const match = normalized.match(/^---\n([\s\S]*?)\n---\n/);
  if (!match) throw new Error(`Missing reading metadata: ${sourcePath}`);
  const f = parse(match[1]);
  for (const key of ['modelId', 'title', 'summary']) if (typeof f[key] !== 'string' || !f[key].trim()) throw new Error(`Invalid reading ${key}: ${sourcePath}`);
  for (const key of ['modelVersion', 'revision']) if (!Number.isInteger(f[key]) || f[key] < 1) throw new Error(`Invalid reading ${key}: ${sourcePath}`);
  const parts = normalized.slice(match[0].length).split('<!-- technical-appendix -->');
  if (parts.length !== 2 || parts.some(p => !p.trim())) throw new Error(`Reading must include body and appendix: ${sourcePath}`);
  if (!Array.isArray(f.sources) || f.sources.length === 0) throw new Error(`Missing reading sources: ${sourcePath}`);
  const ids = new Set<string>();
  for (const source of f.sources) {
    if (['id', 'label', 'url', 'locator', 'note'].some(key => typeof source[key] !== 'string' || !source[key].trim()) || !/^[a-z][a-z0-9-]*$/.test(source.id) || !/^https?:\/\//.test(source.url) || ids.has(source.id)) throw new Error(`Invalid reading source: ${sourcePath}`);
    ids.add(source.id);
  }
  for (const ref of normalized.matchAll(/\]\(#source-([a-z0-9-]+)\)/g)) if (!ids.has(ref[1])) throw new Error(`Unknown citation ${ref[1]}: ${sourcePath}`);
  return { modelId: f.modelId, modelVersion: f.modelVersion, revision: f.revision, title: f.title, summary: f.summary, body: parts[0].trim(), appendix: parts[1].trim(), sources: f.sources, sourcePath };
}

export function attachReadings(models: Model[], readings: ModelReading[]): Model[] {
  const map = new Map<string, ModelReading>();
  for (const reading of readings) {
    const model = models.find(m => m.id === reading.modelId);
    if (!model || model.version !== reading.modelVersion) throw new Error(`Reading references missing or mismatched model: ${reading.modelId}@${reading.modelVersion}`);
    if (map.has(reading.modelId)) throw new Error(`Duplicate reading: ${reading.modelId}`);
    map.set(reading.modelId, reading);
  }
  return models.map(model => map.has(model.id) ? { ...model, reading: map.get(model.id)! } : model);
}

export async function loadReadings(dir: string): Promise<ModelReading[]> {
  const entries = await readdir(dir, { withFileTypes: true });
  return Promise.all(entries.filter(e => e.isFile() && e.name.endsWith('.md')).sort((a, b) => a.name.localeCompare(b.name)).map(async e => parseReading(await readFile(path.join(dir, e.name), 'utf8'), `frontend/content/model-readings/${e.name}`)));
}
