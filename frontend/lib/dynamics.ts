import type { Parameter } from './types';

export type ExperimentID = 'stability' | 'limit-cycle' | 'multistability' | 'bifurcation' | 'hysteresis' | 'delay' | 'resonance' | 'synchronization';
export type DynamicsValues = Record<string, number>;
export interface Experiment {
  id: ExperimentID; name: string; system: string; question: string; description: string;
  parameters: Parameter[]; presets: { name: string; values: DynamicsValues }[];
  equation: string; conditions: string; observation: string; related: ExperimentID[];
  sources: { label: string; url: string; note: string; verified: boolean }[];
}
const p = (key: string, label: string, min: number, max: number, step: number, value: number): Parameter => ({ key, label, min, max, step, value });
const position = () => p('initial', '初始位置', -2, 2, 0.05, 1);
const velocity = () => p('velocity', '初始速度', -2, 2, 0.05, 0);
const mit = { label: 'MIT · Underactuated Robotics：相图与稳定性', url: 'https://underactuated.mit.edu/pend.html', note: '已读取相图、阻尼与平衡分析；这里的小球和双势阱是独立教学方程。', verified: true };
const scholar = (article: string, label: string) => ({ label: `Scholarpedia · ${label}`, url: `https://www.scholarpedia.org/article/${article}`, note: '进一步阅读线索，本轮未取得正文。', verified: false });

export const experiments: Experiment[] = [
  { id: 'stability', name: '平衡与稳定性', system: '小球与局部地形', question: '推一下之后，会回来、停在别处，还是越走越远？', description: '比较碗底、平地和山顶。三条轨迹使用不同起点，同一组系统参数。', parameters: [p('curvature', '地形曲率', -1, 1, 1, 1), p('damping', '阻尼', 0, 2, 0.05, 0.6), position(), velocity()], presets: [{ name: '碗底 · 回归', values: { curvature: 1, damping: 0.6 } }, { name: '平地 · 留在新位置', values: { curvature: 0, damping: 0.6, velocity: 0.5 } }, { name: '山顶 · 远离', values: { curvature: -1, damping: 0.6 } }, { name: '无阻尼 · 持续摆动', values: { curvature: 1, damping: 0 } }], equation: String.raw`$\dot x=v,\quad \dot v=-kx-cv,\quad U(x)=kx^2/2$`, conditions: '质量取 1；碗底和山顶使用局部二次势能。山顶发散只说明此教学方程的行为，不能当作真实山坡的完整运动。碗底在有正阻尼时才渐近回归，无阻尼时不会自行停住。', observation: '比较三个起点的最终位置，再看相图是否向原点收拢。平地有一整条平衡位置，不会把小球拉回原位置。', related: ['resonance', 'multistability'], sources: [mit] },
  { id: 'limit-cycle', name: '极限环与自激振荡', system: 'Van der Pol 振荡器', question: '为什么大小不同的起始扰动，最后形成相近的持续振荡？', description: '小振幅时补充能量，大振幅时耗散能量；轨迹逐渐靠近稳定周期运动。', parameters: [p('mu', '非线性强度 μ', 0.1, 3, 0.1, 1), position(), velocity()], presets: [{ name: '近正弦', values: { mu: 0.1 } }, { name: '松弛振荡', values: { mu: 3 } }], equation: String.raw`$\dot x=v,\quad \dot v=\mu(1-x^2)v-x$`, conditions: 'μ 为正；除精确位于原点的轨迹外，本系统趋向稳定极限环。这里只显示有限时间，较弱的非线性可能收敛较慢。没有周期外力，但自激系统仍需能源。', observation: '相图中不同起点逐渐靠近同一闭合轨道。调高 μ，时间曲线出现缓慢运动与快速跳转；有限时间的闭合外观本身并非极限环证明。', related: ['bifurcation', 'synchronization'], sources: [{ label: 'MIT · Algorithms for Limit Cycles', url: 'https://underactuated.mit.edu/limit_cycles.html', note: '已读取极限环稳定性讨论；具体示范采用标准 Van der Pol 方程。', verified: true }, scholar('Van_der_Pol_oscillator', 'Van der Pol oscillator')] },
  { id: 'multistability', name: '多稳态与吸引域', system: '有阻尼的双势阱', question: '同样的规则，哪些起点走向左边，哪些走向右边？', description: '改变势阱偏置，观察两个稳定结果及不同初始位置、速度的落点。', parameters: [p('bias', '势阱偏置', -0.6, 0.6, 0.02, 0), p('damping', '阻尼', 0.2, 2, 0.05, 0.6), position(), velocity()], presets: [{ name: '对称双稳态', values: { bias: 0, damping: 0.6 } }, { name: '向右偏置', values: { bias: 0.2 } }, { name: '只剩一个势阱', values: { bias: 0.5 } }], equation: String.raw`$\dot x=v,\quad \dot v=x-x^3+b-cv,\quad U(x)=x^4/4-x^2/2-bx$`, conditions: '正阻尼下能量逐渐减少。当 |b|<2/(3√3) 时有两个稳定平衡；越过该值后只剩一个。吸引域是有限网格、有限时间的估计，灰色表示尚未确认稳定。', observation: '吸引域图的每个格子对应一个初始位置与速度，不是一个时间采样。点击格子可把它设为主轨迹的起点，观察越过势垒的过程。', related: ['stability', 'hysteresis', 'bifurcation'], sources: [mit] },
  { id: 'bifurcation', name: '分岔与临界转变', system: '超临界 Hopf 正规形', question: '一个参数跨过零，为什么静止会变成持续振荡？', description: '把控制参数与长期振幅放在一张图上，再联动查看当前参数下的轨迹。', parameters: [p('mu', '控制参数 μ', -1, 1, 0.02, 0.3), p('radius', '初始半径', 0.05, 2, 0.05, 0.2), p('omega', '角频率', 0.5, 3, 0.1, 1)], presets: [{ name: '临界前 · 回归', values: { mu: -0.3 } }, { name: '临界处 · 慢回归', values: { mu: 0 } }, { name: '临界后 · 周期运动', values: { mu: 0.3 } }], equation: String.raw`$\dot x=(\mu-x^2-y^2)x-\omega y,\quad \dot y=\omega x+(\mu-x^2-y^2)y$`, conditions: '这是超临界 Hopf 的教学正规形，不代表所有分岔。μ≤0 时轨迹趋向原点；μ>0 时稳定周期轨道半径为 √μ，原点不稳定。', observation: '下方分岔图横轴是 μ，纵轴是长期半径；点击图上的参数位置更新实验。实线是稳定分支，虚线是不稳定原点。', related: ['limit-cycle', 'multistability'], sources: [scholar('Andronov-Hopf_bifurcation', 'Andronov–Hopf bifurcation')] },
  { id: 'hysteresis', name: '滞回与路径依赖', system: '带上下阈值的开关', question: '同样的输入，为什么开关状态还取决于来时的路线？', description: '输入缓慢上升再下降；达到高阈值开，降至低阈值关，中间保留之前的状态。', parameters: [p('center', '阈值中心', -0.5, 0.5, 0.05, 0), p('gap', '阈值间隔', 0.2, 2, 0.1, 1), p('initialOn', '初始开关状态', 0, 1, 1, 0)], presets: [{ name: '窄滞回', values: { gap: 0.2 } }, { name: '宽滞回', values: { gap: 2 } }, { name: '同输入 · 初始开启', values: { initialOn: 1 } }], equation: String.raw`$s(t^+)=\begin{cases}1,&u\ge h\\0,&u\le l\\s(t^-),&l<u<h\end{cases},\quad h=m+d/2,\ l=m-d/2$`, conditions: '输入是预设的外部信号，开关输出并不反过来改变输入；这不是完整恒温房间模型。高输入开启的约定可对应加热输入的检测开关，温控器也可用相反约定。', observation: '第一段输入固定为 0，选择不同初始状态可看到同输入不同输出；随后升降扫描画出回线。阈值间隔越大，记忆区间越宽。', related: ['multistability', 'delay'], sources: [scholar('Hysteresis', 'Hysteresis')] },
  { id: 'delay', name: '延迟动力学', system: '带延迟的负反馈', question: '调节方向正确，为什么仍会过度纠正或失稳？', description: '系统根据一段时间以前的偏差做调整；与没有延迟的同强度反馈并排比较。', parameters: [p('gain', '调节强度 k', 0.1, 1, 0.05, 0.7), p('delay', '反馈延迟 τ', 0, 3, 0.05, 1), p('history', '初始历史偏差', -2, 2, 0.1, 1)], presets: [{ name: '无延迟', values: { delay: 0 } }, { name: '有延迟 · 收敛', values: { gain: 0.7, delay: 1 } }, { name: '有延迟 · 失稳', values: { gain: 0.9, delay: 2.5 } }], equation: String.raw`$\dot x(t)=-k x(t-\tau),\quad x(t)=h\text{ for }t\le0$`, conditions: '这是线性延迟方程，初始条件必须给出一段历史，这里采用恒定历史。k>0 时渐近稳定的条件是 kτ<π/2；临界值及失稳侧不是渐近回归。数值使用固定小步长与历史插值。', observation: '调整 k 或 τ，观察有延迟轨迹与无延迟基准的差别；增加调节强度并不总能提高稳定性。当前阈值提示来自该方程的解析条件。', related: ['stability', 'hysteresis'], sources: [scholar('Delay-differential_equations', 'Delay-differential equations')] },
  { id: 'resonance', name: '受迫响应与共振', system: '受周期驱动的弹簧振子', question: '改变推的节奏，为什么响应幅度会不同？', description: '质量取 1，正弦外力持续驱动；时间曲线包含暂态，频率响应图显示理论稳态。', parameters: [p('natural', '固有角频率 ω₀', 0.5, 2, 0.05, 1), p('damping', '阻尼比 ζ', 0.05, 1, 0.05, 0.15), p('drive', '驱动角频率 Ω', 0.1, 3, 0.05, 1), p('force', '驱动力幅度', 0.1, 2, 0.1, 1), position(), velocity()], presets: [{ name: '低频驱动', values: { drive: 0.3 } }, { name: '接近固有频率', values: { drive: 1, natural: 1 } }, { name: '高阻尼', values: { damping: 0.8 } }], equation: String.raw`$\ddot x+2\zeta\omega_0\dot x+\omega_0^2x=F\cos(\Omega t),\quad A(\Omega)=\frac{F}{\sqrt{(\omega_0^2-\Omega^2)^2+(2\zeta\omega_0\Omega)^2}}$`, conditions: '线性弹簧、正阻尼、持续正弦驱动。位移响应的非零频率峰只在 ζ<1/√2 时存在，峰值频率为 ω₀√(1−2ζ²)，并非总等于 ω₀；时间窗内可能尚未达到稳态。', observation: '点击频率响应图改变驱动频率。区分当前时间窗内的振幅与理论稳态幅度；调高阻尼时，尖锐的共振峰会变平甚至消失。', related: ['stability', 'synchronization'], sources: [{ label: 'OpenStax · Forced Oscillations', url: 'https://openstax.org/books/university-physics-volume-1/pages/15-6-forced-oscillations', note: '已读取驱动、阻尼、稳态振幅及共振讨论。', verified: true }] },
  { id: 'synchronization', name: '耦合与同步', system: '两个相互耦合的相位振荡器', question: '频率原本不同的两个振荡器，什么时候会锁定相位差？', description: '用正弦相位耦合连接两个振荡器，比较各自信号与连续相位差。', parameters: [p('omega1', '振荡器一角频率', 0.5, 2, 0.05, 1), p('omega2', '振荡器二角频率', 0.5, 2, 0.05, 1.4), p('coupling', '耦合强度 K', 0, 1.5, 0.05, 0.5), p('phase', '初始相位差', -3, 3, 0.1, 1)], presets: [{ name: '没有耦合', values: { coupling: 0 } }, { name: '弱耦合 · 漂移', values: { coupling: 0.1 } }, { name: '强耦合 · 锁定', values: { coupling: 0.5 } }], equation: String.raw`$\dot\theta_1=\omega_1+K\sin(\theta_2-\theta_1),\quad\dot\theta_2=\omega_2+K\sin(\theta_1-\theta_2)$`, conditions: '这是对称、吸引性、仅含相位的双振荡器近似，没有模拟节拍器的机械结构。|ω₂−ω₁|<2K 时存在稳定锁相解，等号是临界情况；锁相不意味着相位差必为零。', observation: '两个信号的峰可以保持固定错位。下方相位差若趋向常数，说明频率锁定；持续增长则表示滑相，不能仅凭偶尔峰值重合判断同步。', related: ['limit-cycle', 'resonance'], sources: [scholar('Kuramoto_model', 'Kuramoto model')] },
];

export function experimentDefaults(experiment: Experiment): DynamicsValues {
  return Object.fromEntries(experiment.parameters.map(p => [p.key, p.value]));
}
export function readDynamicsValues(experiment: Experiment, encoded: string): DynamicsValues {
  let input: unknown;
  try { input = JSON.parse(encoded); } catch { input = {}; }
  const values = experimentDefaults(experiment);
  if (input && typeof input === 'object' && !Array.isArray(input)) for (const p of experiment.parameters) {
    const value = (input as Record<string, unknown>)[p.key];
    if (typeof value === 'number' && Number.isFinite(value)) {
      const clamped = Math.max(p.min, Math.min(p.max, value));
      values[p.key] = Number((p.min + Math.round((clamped - p.min) / p.step) * p.step).toFixed(6));
    }
  }
  return values;
}

export interface Point { x: number; y: number }
export interface Series { name: string; color: string; points: Point[]; dashed?: boolean }
export interface DynamicsPlot { title: string; xLabel: string; yLabel: string; series: Series[]; marker?: Point; interaction?: 'mu' | 'drive'; square?: boolean }
export interface BasinCell { x: number; y: number; outcome: -1 | 0 | 1 }
export interface Simulation { plots: DynamicsPlot[]; message: string; basin?: BasinCell[] }
const colors = ['#188579', '#c4513e', '#82648c'];
type State = [number, number];
type Flow = (t: number, state: State) => State;
export interface StateSample { t: number; state: State }
export function integrate(flow: Flow, initial: State, duration: number, step = 0.01): StateSample[] {
  const count = Math.ceil(duration / step), dt = duration / count;
  const output: StateSample[] = [{ t: 0, state: [...initial] }];
  let state: State = [...initial];
  const add = (a: State, b: State, f: number): State => [a[0] + f * b[0], a[1] + f * b[1]];
  for (let i = 0; i < count; i++) {
    const t = i * dt, a = flow(t, state), b = flow(t + dt / 2, add(state, a, dt / 2)), c = flow(t + dt / 2, add(state, b, dt / 2)), d = flow(t + dt, add(state, c, dt));
    state = [state[0] + dt * (a[0] + 2 * b[0] + 2 * c[0] + d[0]) / 6, state[1] + dt * (a[1] + 2 * b[1] + 2 * c[1] + d[1]) / 6];
    if (!state.every(Number.isFinite)) throw new Error('当前参数下积分出现非有限值，请重置参数。');
    if ((i + 1) % 4 === 0 || i === count - 1) output.push({ t: (i + 1) * dt, state: [...state] });
  }
  return output;
}
const series = (name: string, index: number, points: Point[], dashed = false): Series => ({ name, color: colors[index % colors.length], points, dashed });
const timeSeries = (name: string, index: number, samples: StateSample[], axis = 0) => series(name, index, samples.map(s => ({ x: s.t, y: s.state[axis] })));
const phaseSeries = (name: string, index: number, samples: StateSample[]) => series(name, index, samples.map(s => ({ x: s.state[0], y: s.state[1] })));
const timePlot = (title: string, yLabel: string, items: Series[]): DynamicsPlot => ({ title, xLabel: '时间 t', yLabel, series: items });
const phasePlot = (items: Series[], yLabel = '速度 v'): DynamicsPlot => ({ title: '相图', xLabel: '位置 x', yLabel, series: items, square: true });

export function responseAmplitude(p: DynamicsValues, omega: number): number {
  return p.force / Math.hypot(p.natural ** 2 - omega ** 2, 2 * p.damping * p.natural * omega);
}
export function hopfRadius(mu: number, r0: number, t: number): number {
  if (Math.abs(mu) < 1e-10) return r0 / Math.sqrt(1 + 2 * r0 ** 2 * t);
  return r0 * Math.exp(mu * t) / Math.sqrt(1 + r0 ** 2 * Math.expm1(2 * mu * t) / mu);
}
export function simulateDelay(p: DynamicsValues, step = 0.005): Point[] {
  const n = Math.round(30 / step), dt = 30 / n, history = [p.history];
  const past = (time: number): number => {
    if (time <= 0) return p.history;
    const index = time / dt, lower = Math.floor(index), fraction = index - lower;
    return history[lower] + fraction * ((history[lower + 1] ?? history[lower]) - history[lower]);
  };
  for (let i = 0; i < n; i++) {
    if (p.delay === 0) history.push(p.history * Math.exp(-p.gain * (i + 1) * dt));
    else {
      // For a delay shorter than the integration step use a smaller step recursively.
      if (p.delay < dt) return simulateDelay(p, p.delay / 2);
      const first = -p.gain * past(i * dt - p.delay);
      const second = -p.gain * past((i + 1) * dt - p.delay);
      history.push(history[i] + dt * (first + second) / 2);
    }
  }
  return history.flatMap((y, i) => i % Math.max(1, Math.round(0.04 / dt)) === 0 || i === n ? [{ x: i * dt, y }] : []);
}
export function relay(p: DynamicsValues): { input: Point[]; output: Point[]; loop: Point[] } {
  const input: Point[] = [], output: Point[] = [], loop: Point[] = [];
  let on = p.initialOn;
  for (let i = 0; i <= 800; i++) {
    const t = i / 40;
    const u = t < 2 ? 0 : t < 6 ? -(t - 2) / 2 : t < 14 ? -2 + (t - 6) / 2 : 2 - (t - 14) * 2 / 3;
    if (u >= p.center + p.gap / 2) on = 1;
    else if (u <= p.center - p.gap / 2) on = 0;
    input.push({ x: t, y: u }); output.push({ x: t, y: on });
    if (t >= 6) loop.push({ x: u, y: on });
  }
  return { input, output, loop };
}

export function simulate(id: ExperimentID, p: DynamicsValues, kick = 0, includeBasin = true): Simulation {
  switch (id) {
    case 'stability': {
      const starts: State[] = [[p.initial, p.velocity], [-1.5, 0], [0.35, 0.5]];
      const runs = starts.map(s => integrate((t, [x, v]) => [v, -p.curvature * x - p.damping * v], s, 12));
      if (kick) runs[0] = disturbed((t, [x, v]) => [v, -p.curvature * x - p.damping * v], starts[0], 12, kick);
      const names = ['主轨迹', '起点二', '起点三'];
      return { plots: [timePlot('三个起点的位置随时间变化', '位置 x', runs.map((r, i) => timeSeries(names[i], i, r))), phasePlot(runs.map((r, i) => phaseSeries(names[i], i, r))), { title: '局部势能地形', xLabel: '位置 x', yLabel: '势能 U', series: [series('势能', 0, Array.from({ length: 121 }, (_, i) => { const x = -2 + i / 30; return { x, y: p.curvature * x * x / 2 }; }))] }], message: p.curvature < 0 ? '山顶平衡不稳定，偏离会被放大。' : p.curvature === 0 ? '平地没有回复力；阻尼消耗速度，却不会指定最终位置。' : p.damping === 0 ? '碗底有回复力，但无阻尼，能量不消失，轨迹持续摆动。' : '碗底有回复力和正阻尼，最终趋向原点。' };
    }
    case 'limit-cycle': {
      const runs = [[p.initial, p.velocity], [0.2, 0], [2.5, 0]].map(s => integrate((t, [x, v]) => [v, p.mu * (1 - x * x) * v - x], s as State, 40));
      return { plots: [timePlot('不同起点的自激振荡', '位置 x', runs.map((r, i) => timeSeries(['主轨迹', '小扰动', '大扰动'][i], i, r))), phasePlot(runs.map((r, i) => phaseSeries(['主轨迹', '小扰动', '大扰动'][i], i, r)))], message: 'μ>0：振幅小时输入能量，振幅大时耗散。比较末段轨迹是否靠近同一周期运动。' };
    }
    case 'multistability': {
      const flow: Flow = (t, [x, v]) => [v, x - x ** 3 + p.bias - p.damping * v];
      const runs = [[p.initial, p.velocity], [-1, 0], [0.1, 0]].map(s => integrate(flow, s as State, 30));
      const basin: BasinCell[] = [];
      if (includeBasin) for (let iy = 0; iy <= 16; iy++) for (let ix = 0; ix <= 20; ix++) {
        const x = -2 + ix / 5, y = -2 + iy / 4;
        const end = integrate(flow, [x, y], 35, 0.04).at(-1)!.state;
        const settled = Math.abs(end[1]) < 0.03 && Math.abs(end[0] - end[0] ** 3 + p.bias) < 0.03;
        basin.push({ x, y, outcome: settled && Math.abs(end[0]) > 1 / Math.sqrt(3) ? end[0] > 0 ? 1 : -1 : 0 });
      }
      return { basin, plots: [timePlot('不同起点的最终去向', '位置 x', runs.map((r, i) => timeSeries(['主轨迹', '左侧起点', '近势垒起点'][i], i, r))), phasePlot(runs.map((r, i) => phaseSeries(['主轨迹', '左侧起点', '近势垒起点'][i], i, r))), { title: '势能地形', xLabel: '位置 x', yLabel: '势能 U', series: [series('势能', 0, Array.from({ length: 161 }, (_, i) => { const x = -2 + i / 40; return { x, y: x ** 4 / 4 - x * x / 2 - p.bias * x }; }))] }], message: Math.abs(p.bias) < 2 / (3 * Math.sqrt(3)) ? '当前偏置下有两个稳定平衡。速度也影响能否越过势垒，因此只看初始位置还不够。' : '偏置已越过双稳态范围，只剩一个势能最低点。' };
    }
    case 'bifurcation': {
      const runs = [p.radius, 0.1, 1.6].map(r0 => Array.from({ length: 626 }, (_, i) => { const t = i / 25, r = hopfRadius(p.mu, r0, t); return { t, state: [r * Math.cos(p.omega * t), r * Math.sin(p.omega * t)] as State }; }));
      const stable = Array.from({ length: 201 }, (_, i) => { const x = -1 + i / 100; return { x, y: Math.sqrt(Math.max(0, x)) }; });
      const unstable = Array.from({ length: 101 }, (_, i) => ({ x: i / 100, y: 0 }));
      return { plots: [timePlot('控制参数下的时间轨迹', '状态 x', runs.map((r, i) => timeSeries(['主轨迹', '小半径', '大半径'][i], i, r))), phasePlot(runs.map((r, i) => phaseSeries(['主轨迹', '小半径', '大半径'][i], i, r)), '状态 y'), { title: '长期半径的分岔图', xLabel: '控制参数 μ', yLabel: '长期半径 r', series: [series('稳定分支', 0, stable), series('不稳定原点', 1, unstable, true)], marker: { x: p.mu, y: Math.sqrt(Math.max(0, p.mu)) }, interaction: 'mu' }], message: p.mu > 0 ? `μ=${p.mu.toFixed(2)}：原点不稳定，稳定周期轨道半径为 ${Math.sqrt(p.mu).toFixed(3)}。` : p.mu === 0 ? '临界点仍回归原点，但按代数速度而非指数速度衰减。' : 'μ<0：不同初始半径都趋向原点。' };
    }
    case 'hysteresis': {
      const r = relay(p);
      return { plots: [timePlot('外部输入与开关状态', '输入 u / 状态 s', [series('外部输入', 0, r.input), series('开关状态', 1, r.output)]), { title: '升降输入的滞回回线', xLabel: '外部输入 u', yLabel: '开关状态 s', series: [series('升降路径', 1, r.loop)] }], message: `开启阈值 ${(p.center + p.gap / 2).toFixed(2)}，关闭阈值 ${(p.center - p.gap / 2).toFixed(2)}；两个阈值之间保留之前的开关状态。` };
    }
    case 'delay': {
      const points = simulateDelay(p), baseline = points.map(({ x }) => ({ x, y: p.history * Math.exp(-p.gain * x) }));
      const product = p.gain * p.delay;
      return { plots: [timePlot('延迟反馈与无延迟基准', '偏差 x', [series('有延迟', 0, points), series('无延迟基准', 1, baseline)])], message: `kτ=${product.toFixed(3)}，临界值 π/2≈1.571。${Math.abs(product - Math.PI / 2) < 1e-8 ? '当前为临界情况。' : product < Math.PI / 2 ? '当前参数渐近稳定，收敛可能伴随振荡。' : '当前参数不稳定，线性方程的振幅会增长。'}` };
    }
    case 'resonance': {
      const run = integrate((t, [x, v]) => [v, p.force * Math.cos(p.drive * t) - 2 * p.damping * p.natural * v - p.natural ** 2 * x], [p.initial, p.velocity], 40);
      const points = Array.from({ length: 301 }, (_, i) => ({ x: i / 100, y: responseAmplitude(p, i / 100) }));
      return { plots: [timePlot('受迫运动（含暂态）', '位置 x', [timeSeries('响应', 0, run)]), phasePlot([phaseSeries('受迫轨迹', 0, run)]), { title: '理论稳态频率响应', xLabel: '驱动角频率 Ω', yLabel: '稳态幅度 A', series: [series('幅频响应', 0, points)], marker: { x: p.drive, y: responseAmplitude(p, p.drive) }, interaction: 'drive' }], message: `当前理论稳态幅度为 ${responseAmplitude(p, p.drive).toFixed(3)}。${p.damping < 1 / Math.sqrt(2) ? `位移共振峰在 Ω≈${(p.natural * Math.sqrt(1 - 2 * p.damping ** 2)).toFixed(3)}。` : '当前阻尼较高，没有非零频率的位移共振峰。'}` };
    }
    case 'synchronization': {
      const run = integrate((t, [a, b]) => [p.omega1 + p.coupling * Math.sin(b - a), p.omega2 + p.coupling * Math.sin(a - b)], [0, p.phase], 40);
      const delta = Math.abs(p.omega2 - p.omega1), boundary = 2 * p.coupling;
      return { plots: [timePlot('两个振荡器的信号', 'sin θ', [series('振荡器一', 0, run.map(s => ({ x: s.t, y: Math.sin(s.state[0]) }))), series('振荡器二', 1, run.map(s => ({ x: s.t, y: Math.sin(s.state[1]) })))]), timePlot('连续相位差（不折回）', 'θ₂ − θ₁ · 弧度', [series('相位差', 2, run.map(s => ({ x: s.t, y: s.state[1] - s.state[0] })))])], message: p.coupling === 0 ? (delta === 0 ? '没有耦合但固有频率相同，相位差保持初值；这不能作为耦合促成同步的证据。' : '没有耦合且固有频率不同，相位差持续漂移。') : delta < boundary ? `|Δω|=${delta.toFixed(2)}<2K=${boundary.toFixed(2)}，存在稳定锁相解；同频仍可保持非零相位差。` : delta === boundary ? '当前位于锁相边界，不按稳定锁相情形解释。' : '|Δω|>2K，耦合不足以锁定频率，持续出现滑相。' };
    }
  }
}
function disturbed(flow: Flow, initial: State, duration: number, kick: number): StateSample[] {
  const before = integrate(flow, initial, 4), at = before.at(-1)!.state;
  const after = integrate((t, s) => flow(t + 4, s), [at[0], at[1] + kick], duration - 4);
  return [...before, ...after.map(s => ({ ...s, t: s.t + 4 }))];
}
