'use client';
import { useId, useState } from 'react';
import type { BasinCell, DynamicsPlot, Point } from '../lib/dynamics';

function extent(values: number[]): [number, number] {
  const min = Math.min(...values), max = Math.max(...values), pad = Math.max((max - min) * 0.08, 0.1);
  return [min - pad, max + pad];
}
const label = (n: number) => Math.abs(n) >= 1000 ? n.toExponential(1) : Number(n.toFixed(2)).toString();

export default function DynamicsChart({ plot, compact = false, onSelect }: { plot: DynamicsPlot; compact?: boolean; onSelect?: (point: Point) => void }) {
  const id = useId(), [hover, setHover] = useState<Point>();
  const all = plot.series.flatMap(s => s.points);
  let [xmin, xmax] = extent(all.map(p => p.x)), [ymin, ymax] = extent(all.map(p => p.y));
  const width = 600, height = compact ? 145 : 340, left = compact ? 8 : 68, right = compact ? 8 : 24, top = compact ? 8 : 35, bottom = compact ? 8 : 53;
  const innerWidth = width - left - right, innerHeight = height - top - bottom;
  if (plot.square && !compact) {
    const scale = Math.max((xmax - xmin) / innerWidth, (ymax - ymin) / innerHeight);
    const xc = (xmin + xmax) / 2, yc = (ymin + ymax) / 2;
    xmin = xc - scale * innerWidth / 2; xmax = xc + scale * innerWidth / 2;
    ymin = yc - scale * innerHeight / 2; ymax = yc + scale * innerHeight / 2;
  }
  const x = (n: number) => left + (n - xmin) / (xmax - xmin) * innerWidth;
  const y = (n: number) => top + (ymax - n) / (ymax - ymin) * innerHeight;
  function coordinates(event: React.MouseEvent<SVGSVGElement>): Point {
    const rect = event.currentTarget.getBoundingClientRect();
    const xx = (event.clientX - rect.left) / rect.width * width, yy = (event.clientY - rect.top) / rect.height * height;
    return { x: xmin + Math.max(0, Math.min(1, (xx - left) / innerWidth)) * (xmax - xmin), y: ymax - Math.max(0, Math.min(1, (yy - top) / innerHeight)) * (ymax - ymin) };
  }
  return <div className={`dynamics-chart ${compact ? 'compact' : ''}`}>
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${plot.title}：${plot.xLabel}与${plot.yLabel}`} onPointerLeave={() => setHover(undefined)} onPointerMove={compact ? undefined : event => {
      const point = coordinates(event);
      // Find the nearest rendered sample in both dimensions, including phase trajectories.
      const nearest = all.reduce((best, sample) => Math.hypot(x(sample.x) - x(point.x), y(sample.y) - y(point.y)) < Math.hypot(x(best.x) - x(point.x), y(best.y) - y(point.y)) ? sample : best, all[0]);
      setHover(nearest);
    }} onClick={onSelect ? event => onSelect(coordinates(event)) : undefined} style={onSelect ? { cursor: 'crosshair' } : undefined}>
      <defs><clipPath id={id}><rect x={left} y={top} width={innerWidth} height={innerHeight} /></clipPath></defs>
      {!compact && <>{[0, 1, 2, 3, 4].map(i => {
        const xx = xmin + (xmax - xmin) * i / 4, yy = ymin + (ymax - ymin) * i / 4;
        return <g key={i}><line x1={left} x2={width - right} y1={y(yy)} y2={y(yy)} stroke="#e0e6e3" strokeDasharray="3 4" /><line x1={x(xx)} x2={x(xx)} y1={top} y2={height - bottom} stroke="#edf0ef" /><text x={left - 9} y={y(yy) + 4} textAnchor="end">{label(yy)}</text><text x={x(xx)} y={height - bottom + 20} textAnchor="middle">{label(xx)}</text></g>;
      })}<text x={left} y={17}>{plot.yLabel}</text><text x={width - right} y={height - 7} textAnchor="end">{plot.xLabel}</text></>}
      <g clipPath={`url(#${id})`}>
        {ymin <= 0 && ymax >= 0 && <line x1={left} x2={width - right} y1={y(0)} y2={y(0)} stroke="#becbc4" />}
        {xmin <= 0 && xmax >= 0 && <line x1={x(0)} x2={x(0)} y1={top} y2={height - bottom} stroke="#becbc4" />}
        {plot.series.map(s => <g key={s.name}><path className="trajectory" d={s.points.map((p, i) => `${i ? 'L' : 'M'}${x(p.x).toFixed(2)},${y(p.y).toFixed(2)}`).join(' ')} fill="none" stroke={s.color} strokeWidth={compact ? 3 : 2.2} strokeDasharray={s.dashed ? '6 4' : undefined} />{!compact && <><circle cx={x(s.points[0].x)} cy={y(s.points[0].y)} r="3.5" fill="white" stroke={s.color} /><circle cx={x(s.points.at(-1)!.x)} cy={y(s.points.at(-1)!.y)} r="3.5" fill={s.color} /></>}</g>)}
        {plot.marker && <circle cx={x(plot.marker.x)} cy={y(plot.marker.y)} r="5" fill="#c4513e" stroke="white" strokeWidth="2" />}
        {hover && !compact && <circle cx={x(hover.x)} cy={y(hover.y)} r="4.5" fill="#202c29" stroke="white" strokeWidth="1.5" />}
      </g>
    </svg>
    {!compact && <><div className="dynamics-legend">{plot.series.map(s => <span key={s.name}><i style={{ background: s.color }} />{s.name}{s.dashed && '（虚线）'}</span>)}</div><output className="dynamics-readout">{hover ? `${plot.xLabel}: ${label(hover.x)} · ${plot.yLabel}: ${label(hover.y)}` : '空心点为起点，实心点为终点 · 无量纲示意'}{onSelect && ' · 点击图中位置调整参数'}</output></>}
  </div>;
}

export function BasinChart({ cells, current, onSelect }: { cells: BasinCell[]; current: Point; onSelect: (point: Point) => void }) {
  const width = 600, height = 340, left = 68, top = 35, iw = 508, ih = 252;
  const x = (v: number) => left + (v + 2.1) / 4.2 * iw, y = (v: number) => top + (2.125 - v) / 4.25 * ih;
  return <div className="dynamics-chart"><svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="初始位置与速度的吸引域，青绿为右侧，朱红为左侧，灰色为未确认" style={{ cursor: 'crosshair' }} onClick={event => {
    const rect = event.currentTarget.getBoundingClientRect();
    const xx = (event.clientX - rect.left) / rect.width * width, yy = (event.clientY - rect.top) / rect.height * height;
    onSelect({ x: Math.max(-2, Math.min(2, (xx - left) / iw * 4.2 - 2.1)), y: Math.max(-2, Math.min(2, 2.125 - (yy - top) / ih * 4.25)) });
  }}>
    <text x={left} y={17}>初始速度 v₀</text><text x={width - 24} y={333} textAnchor="end">初始位置 x₀</text>
    {cells.map(cell => <rect key={`${cell.x}-${cell.y}`} x={x(cell.x - 0.1)} y={y(cell.y + 0.125)} width={iw / 21 + 0.2} height={ih / 17 + 0.2} fill={cell.outcome === 1 ? '#188579' : cell.outcome === -1 ? '#c4513e' : '#b6c1bb'} opacity=".8"><title>{`x₀=${cell.x.toFixed(2)}, v₀=${cell.y.toFixed(2)}：${cell.outcome === 1 ? '右侧' : cell.outcome === -1 ? '左侧' : '未确认'}`}</title></rect>)}
    {[-2, -1, 0, 1, 2].map(v => <g key={v}><text x={x(v)} y={307} textAnchor="middle">{v}</text><text x={left - 10} y={y(v) + 4} textAnchor="end">{v}</text></g>)}
    <circle cx={x(current.x)} cy={y(current.y)} r="6" fill="white" stroke="#202c29" strokeWidth="2" />
  </svg><div className="dynamics-legend"><span><i style={{ background: '#c4513e' }} />左侧平衡</span><span><i style={{ background: '#188579' }} />右侧平衡</span><span><i style={{ background: '#b6c1bb' }} />有限时间未确认</span></div><p className="dynamics-readout">有限网格估计 · 点击选取主轨迹起点，也可用参数输入</p></div>;
}
