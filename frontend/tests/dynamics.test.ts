import { test } from 'node:test';
import assert from 'node:assert/strict';
import { experiments, experimentDefaults, readDynamicsValues, integrate, simulate, simulateDelay, hopfRadius, relay, responseAmplitude } from '../lib/dynamics';

test('all eight dynamics systems produce finite trajectories across parameter corners', () => {
  assert.equal(experiments.length, 8);
  for (const experiment of experiments) {
    const defaults = experimentDefaults(experiment);
    const configurations = [defaults, ...experiment.presets.map(p => ({ ...defaults, ...p.values }))];
    for (let mask = 0; mask < 2 ** experiment.parameters.length; mask++) configurations.push(Object.fromEntries(experiment.parameters.map((p, i) => [p.key, mask & 1 << i ? p.min : p.max])));
    for (const values of configurations) {
      const result = simulate(experiment.id, values, 0, false);
      assert(result.plots.length > 0);
      for (const plot of result.plots) for (const series of plot.series) {
        assert(series.points.length > 1);
        assert(series.points.every(p => Number.isFinite(p.x) && Number.isFinite(p.y)), `${experiment.id}: ${JSON.stringify(values)}`);
      }
    }
  }
});

test('stability distinguishes asymptotic return, neutral motion and divergence; impulse changes velocity', () => {
  const experiment = experiments[0], defaults = experimentDefaults(experiment);
  const end = (curvature: number, damping: number) => simulate('stability', { ...defaults, curvature, damping }).plots[0].series[0].points.at(-1)!.y;
  assert(Math.abs(end(1, 0.6)) < 0.04);
  assert.equal(end(0, 0.6), 1);
  assert(Math.abs(end(-1, 0.6)) > 100);
  const undamped = integrate((t, [x, v]) => [v, -x], [1, 0], 20);
  assert(undamped.every(({ state: [x, v] }) => Math.abs(x*x + v*v - 1) < 1e-7));
  const disturbed = simulate('stability', defaults, 1).plots[1].series[0].points;
  const at = disturbed.filter((_, i) => i === 100 || i === 101);
  assert.equal(at[0].x, at[1].x);
  assert(Math.abs(at[1].y - at[0].y - 1) < 1e-9);
});

test('Van der Pol trajectories approach the same periodic orbit from different amplitudes', () => {
  const result = simulate('limit-cycle', experimentDefaults(experiments[1]));
  const amplitudes = result.plots[0].series.map(s => Math.max(...s.points.filter(p => p.x > 30).map(p => p.y)));
  assert(amplitudes.every(a => a > 1.9 && a < 2.1));
  assert(Math.max(...amplitudes) - Math.min(...amplitudes) < 0.02);
});

test('double well basin is symmetric and excludes the saddle; strong bias removes one basin', () => {
  const p = experimentDefaults(experiments[2]);
  const basin = simulate('multistability', p).basin!;
  assert.equal(basin.find(c => c.x === -1 && c.y === 0)!.outcome, -1);
  assert.equal(basin.find(c => c.x === 1 && c.y === 0)!.outcome, 1);
  assert.equal(basin.find(c => c.x === 0 && c.y === 0)!.outcome, 0);
  for (const c of basin) {
    const mirror = basin.find(d => Math.abs(d.x + c.x) < 1e-9 && Math.abs(d.y + c.y) < 1e-9)!;
    assert.equal(c.outcome + mirror.outcome, 0);
  }
  const biased = simulate('multistability', { ...p, bias: 0.5 }).basin!;
  assert(!biased.some(c => c.outcome === -1));
});

test('Hopf analytic solution matches integration, critical decay and limiting radius', () => {
  for (const mu of [-0.5, 0, 0.5]) {
    const run = integrate((t, [x, y]) => [(mu-x*x-y*y)*x-y, x+(mu-x*x-y*y)*y], [0.2, 0], 5);
    const [x,y] = run.at(-1)!.state;
    assert(Math.abs(Math.hypot(x,y)-hopfRadius(mu,0.2,5))<1e-8);
  }
  assert(Math.abs(hopfRadius(0.5, 0.2, 100) - Math.sqrt(0.5)) < 1e-9);
  assert.equal(hopfRadius(0, 1, 4), 1/3);
});

test('relay remembers its state between thresholds and switches at the right inputs', () => {
  const defaults = experimentDefaults(experiments[4]);
  const off = relay(defaults), on = relay({ ...defaults, initialOn: 1 });
  assert.equal(off.output[0].y, 0); assert.equal(on.output[0].y, 1);
  const rising = off.input.findIndex(p => p.x > 6 && p.y >= 0.5);
  assert.equal(off.output[rising].y, 1); assert.equal(off.output[rising-1].y, 0);
  const falling = off.input.findIndex(p => p.x > 14 && p.y <= -0.5);
  assert.equal(off.output[falling].y, 0); assert.equal(off.output[falling-1].y, 1);
});

test('delay respects the constant initial history and converges under step refinement', () => {
  const p = { gain: 0.7, delay: 1, history: 1 };
  const samples = simulateDelay(p);
  for (const point of samples.filter(p => p.x <= 1)) assert(Math.abs(point.y - (1 - 0.7 * point.x)) < 1e-10);
  const fine = simulateDelay(p, 0.0025);
  assert(Math.abs(samples.at(-1)!.y - fine.at(-1)!.y) < 1e-6);
  const zero = simulateDelay({ ...p, delay: 0 });
  assert(zero.every(point => Math.abs(point.y-Math.exp(-0.7*point.x))<1e-12));
  const unstable = simulateDelay({ ...p, gain: 0.9, delay: 2.5 });
  assert(Math.max(...unstable.filter(p => p.x > 20).map(p => Math.abs(p.y))) > 2);
});

test('forced response matches the steady solution, and its peak shifts with damping', () => {
  const p = { ...experimentDefaults(experiments[6]), natural: 1, damping: 0.2, drive: 0.8, force: 1 };
  const peak = Math.sqrt(1-2*p.damping*p.damping);
  assert(responseAmplitude(p,peak)>responseAmplitude(p,peak-0.05));
  assert(responseAmplitude(p,peak)>responseAmplitude(p,peak+0.05));
  const a = responseAmplitude(p,p.drive), lag = Math.atan2(2*p.damping*p.natural*p.drive,p.natural*p.natural-p.drive*p.drive);
  const initial: [number,number] = [a*Math.cos(lag), a*p.drive*Math.sin(lag)];
  const run = integrate((t,[x,v])=>[v,p.force*Math.cos(p.drive*t)-2*p.damping*p.natural*v-p.natural*p.natural*x],initial,10);
  assert(run.every(s=>Math.abs(s.state[0]-a*Math.cos(p.drive*s.t-lag))<1e-8));
  const high = { ...p, damping: 0.8 };
  assert(responseAmplitude(high,0)>responseAmplitude(high,0.2));
});

test('coupled phases lock to the predicted nonzero phase difference, and drift without coupling', () => {
  const p = experimentDefaults(experiments[7]);
  const locked = simulate('synchronization', p).plots[1].series[0].points;
  assert(Math.abs(locked.at(-1)!.y-Math.asin((p.omega2-p.omega1)/(2*p.coupling)))<1e-9);
  const free = simulate('synchronization', { ...p, coupling: 0 }).plots[1].series[0].points;
  assert(Math.abs(free.at(-1)!.y-(p.phase+(p.omega2-p.omega1)*40))<1e-9);
});

test('URL settings reject invalid data, clamp ranges and preserve discrete parameter semantics', () => {
  const e = experiments[0];
  assert.deepEqual(readDynamicsValues(e,'broken'),experimentDefaults(e));
  assert.deepEqual(readDynamicsValues(e,'null'),experimentDefaults(e));
  const p = readDynamicsValues(e,'{"curvature":0.3,"initial":999,"damping":"bad","velocity":-999,"unknown":3}');
  assert.equal(p.curvature,0); assert.equal(p.initial,2); assert.equal(p.velocity,-2); assert.equal(p.damping,0.6);
  assert(!('unknown' in p));
});
