import { readdir, readFile, mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parse } from 'yaml';
import { unified } from 'unified';
import remarkParse from 'remark-parse';
import { toString } from 'mdast-util-to-string';
import { modes } from '../lib/modes';
import { attachReadings, loadReadings } from './readings';
import type { Category, Library, Model, Section, Source } from '../lib/types';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
async function walk(dir: string): Promise<string[]> {
  const entries = await readdir(dir, { withFileTypes: true });
  const nested = await Promise.all(entries.filter(e => !e.isSymbolicLink()).map(e => e.isDirectory() ? walk(path.join(dir, e.name)) : Promise.resolve(e.name.endsWith('.md') ? [path.join(dir, e.name)] : [])));
  return nested.flat().sort();
}
export function parseModel(text: string, sourcePath: string): Model {
  const normalized = text.replace(/\r\n/g, '\n');
  const match = normalized.match(/^---\n([\s\S]*?)\n---\n/);
  if (!match) throw new Error(`Missing frontmatter: ${sourcePath}`);
  const f = parse(match[1]);
  if (f.type !== 'mental_model' || !f.id || !f.name || !Number.isInteger(f.version)) throw new Error(`Invalid model metadata: ${sourcePath}`);
  const body = normalized.slice(match[0].length);
  const tree = unified().use(remarkParse).parse(body);
  const sections: Section[] = [];
  let current: { title: string; start: number } | undefined;
  for (const node of tree.children) {
    if (node.type === 'heading' && node.depth === 2) {
      if (current) sections.push({ title: current.title, markdown: body.slice(current.start, node.position!.start.offset).trim() });
      current = { title: toString(node), start: node.position!.end.offset! };
    }
  }
  if (current) sections.push({ title: current.title, markdown: body.slice(current.start).trim() });
  const sources: Source[] = [];
  function links(node: typeof tree | typeof tree.children[number]) {
    if (node.type === 'link' && /^https?:\/\//.test(node.url) && !sources.some(s => s.url === node.url)) sources.push({ label: toString(node), url: node.url });
    if ('children' in node) for (const child of node.children) links(child as typeof tree.children[number]);
  }
  links(tree);
  const definitionMarkdown = sections.find(s => s.title === '一句话定义')?.markdown || '';
  const definition = toString(unified().use(remarkParse).parse(definitionMarkdown));
  return { id: f.id, name: f.name, englishName: f.english_name || '', version: f.version, status: f.status, kind: f.kind, formalism: !!f.formalism, categoryPath: [f.source_category || '未分类', f.discipline || '未分类', f.subdiscipline || '未分类'], definition, sections, sources, relations: (f.model_relations || []).map((r: { type: string; target: string; rationale: string }) => ({ type: r.type, target: r.target, rationale: r.rationale })), sourcePath };
}
export function buildCategories(models: Model[]): Category[] {
  const categories: Category[] = [];
  for (const model of models) {
    let siblings = categories;
    model.categoryPath.forEach((name, i) => {
      const id = model.categoryPath.slice(0, i + 1).join('/');
      let item = siblings.find(c => c.id === id);
      if (!item) { item = { id, name, count: 0, children: [] }; siblings.push(item); }
      item.count++; siblings = item.children;
    });
  }
  return categories;
}
export async function generate() {
  const files = await walk(path.join(root, '01-Models'));
  const originalModels = await Promise.all(files.map(async file => parseModel(await readFile(file, 'utf8'), path.relative(root, file).replaceAll('\\', '/'))));
  const models = attachReadings(originalModels, await loadReadings(path.join(root, 'frontend/content/model-readings')));
  const ids = new Set<string>();
  for (const m of models) { if (ids.has(m.id)) throw new Error(`Duplicate ID: ${m.id}`); ids.add(m.id); }
  const library: Library = { schemaVersion: 1, generatedAt: new Date().toISOString(), source: 'Legacy 模型快照 · 01-Models', models, categories: buildCategories(models), modes, associations: [] };
  await mkdir(path.join(root, 'backend/data'), { recursive: true });
  await writeFile(path.join(root, 'backend/data/library.json'), JSON.stringify(library, null, 2) + '\n');
  console.log(`Snapshot: ${models.length} models, ${modes.length} modes, ${modes.reduce((sum, m) => sum + m.variants.length, 0)} variants`);
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) await generate();
