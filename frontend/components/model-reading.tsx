'use client';
import { useState } from 'react';
import type { Model } from '../lib/types';
import Markdown from './markdown';

const statusLabels: Record<string, string> = { active: '有效', candidate: '候选', contested: '有争议', deprecated: '已废弃' };

export default function ModelReadingView({ model, original, onSelect }: { model: Model; original: React.ReactNode; onSelect: (id: string) => void }) {
  const [tab, setTab] = useState<'reading' | 'original'>('reading');
  const reading = model.reading!;
  return <>
    <div className="reading-tabs" role="tablist" aria-label="模型阅读方式">
      <button id="reading-tab" aria-controls="reading-panel" role="tab" aria-selected={tab === 'reading'} onClick={() => setTab('reading')}>新版讲解 · 试读</button>
      <button id="original-tab" aria-controls="original-panel" role="tab" aria-selected={tab === 'original'} onClick={() => setTab('original')}>原始资料</button>
    </div>
    {tab === 'original' ? <div id="original-panel" role="tabpanel" aria-labelledby="original-tab">{original}</div> : <div id="reading-panel" role="tabpanel" aria-labelledby="reading-tab">
      <header className="reading-header"><div className="reading-label">新版讲解 · 试读 <span>稿件 r{reading.revision} · 依据模型 v{reading.modelVersion}</span></div><h2>{reading.title}</h2><p className="reading-deck">{reading.summary}</p><p className="reading-byline">LollaNet 原创讲解 · <span className={`badge ${model.status}`}>{statusLabels[model.status] || model.status}</span></p></header>
      <div className="reading-body"><Markdown text={reading.body} /></div>
      <section className="reading-sources" aria-label="参考来源"><h3>参考来源</h3><ol>{reading.sources.map(s => <li key={s.id} id={`source-${s.id}`}><a href={s.url} target="_blank" rel="noreferrer">{s.label}</a><p>{s.locator}</p><p className="muted">{s.note}</p></li>)}</ol></section>
      <details className="reading-appendix"><summary>技术附录</summary><div className="reading-body"><Markdown text={reading.appendix} /></div></details>
      {model.relations.length > 0 && <section className="reading-related"><h3>相关模型</h3>{model.relations.map((r, i) => <button key={i} className="relation-row" onClick={() => onSelect(r.target.split('@')[0])}><strong>{r.target}</strong><p>{r.rationale}</p></button>)}</section>}
    </div>}
  </>;
}
