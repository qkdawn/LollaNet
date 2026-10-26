import { test } from 'node:test';
import assert from 'node:assert/strict';
import { modes } from '../lib/modes';
import { defaults, evaluate, samples } from '../lib/curves';

const variants = modes.flatMap(m => m.variants);
test('every formula remains finite at every combination of parameter limits', () => {
  for (const v of variants) {
    const combinations = 2 ** v.parameters.length;
    for (let mask = 0; mask < combinations; mask++) {
      const values = Object.fromEntries(v.parameters.map((p, i) => [p.key, (mask & (1 << i)) ? p.max : p.min]));
      assert(samples(v, values).every(p => Number.isFinite(p.y)), v.id);
    }
  }
});
test('growth, decline and goal approaches have the expected direction', () => {
  for (const v of variants.filter(v => ['linear', 'exponential', 'goal', 'logistic', 'gompertz'].includes(v.formula))) {
    const points = samples(v);
    for (let i = 1; i < points.length; i++) assert((points[i].y - points[i - 1].y) * v.direction >= -1e-10, v.id);
  }
  for (const v of variants.filter(v => v.formula === 'goal')) {
    const p = defaults(v); assert.equal(evaluate(v, 0, p), p.initial);
    assert(Math.abs(evaluate(v, 100, p) - p.target) < 1e-8);
  }
});
test('sigmoid inflection, peak and overshoot initial state', () => {
  const logistic = variants.find(v => v.id === 'logistic-growth')!, p = defaults(logistic);
  assert.equal(evaluate(logistic, p.midpoint, p), p.limit / 2);
  const second = (t: number) => evaluate(logistic, t + 0.01, p) - 2 * evaluate(logistic, t, p) + evaluate(logistic, t - 0.01, p);
  assert(second(p.midpoint - 1) > 0); assert(second(p.midpoint + 1) < 0);
  const gompertz = variants.find(v => v.id === 'gompertz-growth')!, g = defaults(gompertz);
  assert(Math.abs(evaluate(gompertz, g.midpoint, g) - g.limit / Math.E) < 1e-10);
  const peak = variants.find(v => v.formula === 'peak')!, a = defaults(peak);
  assert.equal(evaluate(peak, 0, a), 0); assert.equal(evaluate(peak, a.peakTime, a), a.amplitude);
  const overshoot = variants.find(v => v.formula === 'overshoot')!, o = defaults(overshoot);
  assert.equal(evaluate(overshoot, 0, o), 0); assert(evaluate(overshoot, o.peakTime, o) > o.amplitude);
  assert(Math.abs(evaluate(overshoot, 100, o) - o.amplitude) < 1e-8);
});
test('higher damping reduces the oscillation envelope', () => {
  const v = variants.find(v => v.id === 'oscillation-damped')!, p = defaults(v);
  assert(Math.abs(evaluate(v, 4.5, { ...p, damping: 0.3 })) < Math.abs(evaluate(v, 4.5, { ...p, damping: 0.01 })));
});
