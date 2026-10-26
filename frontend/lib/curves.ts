import type { Variant } from './types';
export type Values = Record<string, number>;
export const defaults = (v: Variant): Values => Object.fromEntries(v.parameters.map(p => [p.key, p.value]));
export function evaluate(v: Variant, t: number, p: Values): number {
  switch (v.formula) {
    case 'constant': return p.level;
    case 'linear': return p.initial + v.direction * p.slope * t;
    case 'exponential': return p.initial * Math.exp(v.direction * p.rate * t);
    case 'goal': return p.target + (p.initial - p.target) * Math.exp(-p.rate * t);
    case 'logistic': return p.limit / (1 + Math.exp(-v.direction * p.rate * (t - p.midpoint)));
    case 'gompertz': {
      const growth = p.limit * Math.exp(-Math.exp(-p.rate * (t - p.midpoint)));
      return v.direction > 0 ? growth : p.limit - growth;
    }
    case 'peak': return t === 0 ? 0 : p.amplitude * Math.pow(t / p.peakTime, p.recovery) * Math.exp(p.recovery * (1 - t / p.peakTime));
    case 'overshoot': {
      const z = p.recovery, root = Math.sqrt(1 - z * z), wd = Math.PI / p.peakTime, w = wd / root;
      return p.amplitude * (1 - Math.exp(-z * w * t) * (Math.cos(wd * t) + z / root * Math.sin(wd * t)));
    }
    case 'oscillation': return p.baseline + p.amplitude * Math.exp(v.direction * (p.damping || 0) * t) * Math.sin(2 * Math.PI * p.frequency * t);
  }
}
export const samples = (v: Variant, p = defaults(v)) => Array.from({ length: 241 }, (_, i) => ({ t: i / 24, y: evaluate(v, i / 24, p) }));
export function range(points: { y: number }[]): [number, number] {
  const min = Math.min(0, ...points.map(p => p.y)), max = Math.max(0, ...points.map(p => p.y));
  const pad = Math.max((max - min) * 0.12, 0.5);
  return [min - pad, max + pad];
}
