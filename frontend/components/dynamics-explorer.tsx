'use client';
import { useId, useMemo } from 'react';
import { ArrowLeft, ArrowUpRight, RotateCcw, SlidersHorizontal } from 'lucide-react';
import { experiments, experimentDefaults, readDynamicsValues, simulate, type DynamicsValues, type Experiment, type ExperimentID } from '../lib/dynamics';
import DynamicsChart, { BasinChart } from './dynamics-plot';
import Markdown from './markdown';

export function ExperimentGrid({ query = '', onSelect }: { query?: string; onSelect: (id: ExperimentID) => void }) {
  const visible = experiments.filter(e => `${e.name} ${e.system} ${e.question} ${e.description}`.toLowerCase().includes(query.toLowerCase()));
  const previews = useMemo(() => experiments.map(e => simulate(e.id, experimentDefaults(e), 0, false).plots[e.id === 'limit-cycle' ? 1 : 0]), []);
  return <>{visible.length ? <div className="experiment-grid">{visible.map(e => <button className="experiment-card" key={e.id} onClick={() => onSelect(e.id)}><div className="mode-card-top"><span>实验 / {String(experiments.indexOf(e) + 1).padStart(2, '0')}</span><ArrowUpRight size={16} /></div><DynamicsChart compact plot={previews[experiments.indexOf(e)]} /><h2>{e.name}</h2><p>{e.question}</p><span className="experiment-system">{e.system}</span></button>)}</div> : <div className="empty-state"><h3>没有匹配的动力学实验</h3><p>可以搜索“延迟”“同步”或具体系统名称。</p></div>}</>;
}

export default function DynamicsExplorer({ experiment, encoded, kickEncoded, onChange, onSelect, onBack }: { experiment: Experiment; encoded: string; kickEncoded: string; onChange: (updates: Record<string, string | null>, replace?: boolean) => void; onSelect: (id: ExperimentID) => void; onBack: () => void }) {
  const id = useId();
  const values = useMemo(() => readDynamicsValues(experiment, encoded), [experiment, encoded]);
  const kick = kickEncoded === '1' ? 1 : 0;
  const result = useMemo(() => {
    try { return { simulation: simulate(experiment.id, values, kick), error: '' }; }
    catch (e) { return { simulation: undefined, error: e instanceof Error ? e.message : '计算失败，请重置参数。' }; }
  }, [experiment, values, kick]);
  function update(updates: DynamicsValues, replace = false) {
    const next = readDynamicsValues(experiment, JSON.stringify({ ...values, ...updates }));
    onChange({ settings: JSON.stringify(next), kick: null }, replace);
  }
  const reset = () => onChange({ settings: null, kick: null });
  return <article className="dynamics-explorer">
    <button className="back-link" onClick={onBack}><ArrowLeft size={15} />全部动力学实验</button>
    <header className="experiment-intro"><span className="badge experimental">教学实验 · 无量纲示意</span><h2>{experiment.question}</h2><p>{experiment.description}</p><span>{experiment.system}</span></header>
    <div className="experiment-presets" aria-label="实验预设">{experiment.presets.map(p => <button key={p.name} onClick={() => update({ ...experimentDefaults(experiment), ...p.values })}>{p.name}</button>)}</div>
    <div className="experiment-layout">
      <section className="experiment-controls" aria-label="实验条件"><div className="plot-heading"><h3><SlidersHorizontal size={15} />条件与参数</h3><button className="icon-button" aria-label="重置实验" title="重置实验" onClick={reset}><RotateCcw size={16} /></button></div>
        <div className="parameter-list">{experiment.parameters.map(p => <div className="parameter" key={p.key}><label htmlFor={`${id}-${p.key}`}>{p.label}</label><input aria-label={p.label} type="number" min={p.min} max={p.max} step={p.step} value={values[p.key]} onChange={event => { if (event.target.value && Number.isFinite(Number(event.target.value))) update({ [p.key]: Number(event.target.value) }, true); }} /><input id={`${id}-${p.key}`} aria-label={`${p.label}滑块`} type="range" min={p.min} max={p.max} step={p.step} value={values[p.key]} onChange={event => update({ [p.key]: Number(event.target.value) }, true)} /><div className="parameter-bounds"><span>{p.min}</span><span>{p.max}</span></div></div>)}</div>
        {experiment.id === 'stability' && <button className="experiment-kick" onClick={() => onChange({ kick: kick ? null : '1' })}>{kick ? '撤销扰动' : '在 t=4 施加一次速度扰动'}</button>}
        <p className="control-note">参数和起点保存在地址中，可刷新或分享当前实验。预设切换可用浏览器后退恢复。</p>
      </section>
      <div className="experiment-results">{result.error ? <div className="error-state" role="alert"><h3>{result.error}</h3><button onClick={reset}>重置实验</button></div> : result.simulation && <>
        <div className="experiment-result" role="status"><strong>当前结果</strong><p>{result.simulation.message}</p>{kick === 1 && experiment.id === 'stability' && <p>主轨迹在 t=4 时速度增加 1，空心点仍标示原始起点。</p>}</div>
        <div className="dynamics-plots">{result.simulation.plots.map(plot => <section className="dynamics-panel" key={plot.title}><h3>{plot.title}</h3><DynamicsChart plot={plot} onSelect={plot.interaction ? point => update({ [plot.interaction!]: point.x }) : undefined} /></section>)}
          {result.simulation.basin && <section className="dynamics-panel"><h3>初始条件的吸引域</h3><BasinChart cells={result.simulation.basin} current={{ x: values.initial, y: values.velocity }} onSelect={point => update({ initial: point.x, velocity: point.y })} /></section>}
        </div>
      </>}</div>
    </div>
    <section className="experiment-explanation"><h3>怎样读这些图</h3><p>{experiment.observation}</p><h3>成立条件与限制</h3><p>{experiment.conditions}</p></section>
    <details className="experiment-equations"><summary>方程与计算说明</summary><Markdown text={experiment.equation.replace(/^\$|\$$/g, '').replace(/^/, '$$\n').replace(/$/, '\n$$')} /><p>所有轨迹在浏览器内计算，时间窗固定，坐标按数据范围适配。有限时间与采样精度影响图形；条件判断依据这里明确给出的方程。</p></details>
    <section className="experiment-sources"><h3>参考来源</h3>{experiment.sources.map(source => <div key={source.url}><a href={source.url} target="_blank" rel="noreferrer">{source.label}<ArrowUpRight size={13} /></a><span>{source.verified ? '已读取相关段落' : '待核验线索'}</span><p>{source.note}</p></div>)}</section>
    <section className="related-experiments"><h3>继续探索</h3>{experiment.related.map(key => <button key={key} onClick={() => onSelect(key)}>{experiments.find(e => e.id === key)!.name}<ArrowUpRight size={14} /></button>)}</section>
  </article>;
}
