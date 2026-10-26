'use client';
import { useState } from 'react';
import { defaults, range, samples, type Values } from '../lib/curves';
import type { Variant } from '../lib/types';

export default function Chart({ variant, values = defaults(variant), color = '#188579', compact = false, domain }: { variant: Variant; values?: Values; color?: string; compact?: boolean; domain?: [number, number] }) {
  const [hover, setHover] = useState<number | null>(null);
  const points = samples(variant, values), [lo, hi] = domain || range(points);
  const width = 600, height = compact ? 150 : 300, left = compact ? 8 : 52, right = compact ? 8 : 20, top = compact ? 10 : 24, bottom = compact ? 10 : 42;
  const x = (t: number) => left + t / 10 * (width - left - right);
  const y = (v: number) => top + (hi - v) / (hi - lo) * (height - top - bottom);
  const line = points.map((p, i) => `${i ? 'L' : 'M'}${x(p.t).toFixed(2)},${y(p.y).toFixed(2)}`).join(' ');
  const point = hover === null ? null : points[hover];
  return <div className={compact ? 'chart compact' : 'chart'}>
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${variant.name}示意曲线`} onPointerLeave={() => setHover(null)} onPointerMove={compact ? undefined : e => { const rect = e.currentTarget.getBoundingClientRect(); const t = ((e.clientX - rect.left) / rect.width * width - left) / (width - left - right); setHover(Math.max(0, Math.min(240, Math.round(t * 240)))); }}>
      {!compact && <>{Array.from({ length: 5 }, (_, i) => { const val = lo + (hi - lo) * i / 4; return <g key={i}><line x1={left} x2={width - right} y1={y(val)} y2={y(val)} stroke="#e5e9e8" strokeDasharray="3 4" /><text x={left - 9} y={y(val) + 4} textAnchor="end">{Math.abs(val) >= 100 ? val.toFixed(0) : val.toFixed(1)}</text></g>; })}{[0, 2, 4, 6, 8, 10].map(t => <g key={t}><line x1={x(t)} x2={x(t)} y1={top} y2={height - bottom} stroke="#edf0ef" /><text x={x(t)} y={height - bottom + 20} textAnchor="middle">{t}</text></g>)}<text x={left} y={13}>状态 y · 无量纲</text><text x={width - right} y={height - 3} textAnchor="end">时间 t · 无量纲</text></>}
      <line x1={left} x2={width - right} y1={y(0)} y2={y(0)} stroke="#c6cecb" />
      <path d={line} fill="none" stroke={color} strokeWidth={compact ? 3.5 : 2.8} strokeLinecap="round" />
      {point && <><line x1={x(point.t)} x2={x(point.t)} y1={top} y2={height - bottom} stroke={color} opacity=".4" /><circle cx={x(point.t)} cy={y(point.y)} r="4" fill={color} /></>}
    </svg>
    {!compact && <output className="chart-readout">{point ? `t ${point.t.toFixed(2)} · y ${point.y.toFixed(3)}` : 't 0 – 10 · 定性示意'}</output>}
  </div>;
}
