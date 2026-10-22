import { CartPole, DEG, PHYSICS } from './physics.js';
import { Policy, DEEP_ACTIONS, STATE_LABELS } from './dqn.js';
import {
  CartPoleRenderer, NetworkView, PolicyMap, drawQChart, drawAttribution, drawTrace, stabilityColor, forceColor,
} from './render.js';

const $ = (id) => document.getElementById(id);
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const fmt = (v, d = 1) => (v > 0 ? '+' : '') + v.toFixed(d);
const avg = (a) => a.reduce((s, v) => s + v, 0) / a.length;

// ---------------------------------------------------------------------------
// Datos del entrenamiento (exportados por training/dqn_pendulo.py)
// ---------------------------------------------------------------------------
const load = (f) => fetch(`data/${f}`).then((r) => r.json());
const [modelDeep, modelShallow, histDeep, histShallow] = await Promise.all([
  load('model_deep.json'), load('model_shallow.json'), load('history_deep.json'), load('history_shallow.json'),
]);
const policies = { deep: new Policy(modelDeep), shallow: new Policy(modelShallow), browser: null };

// ---------------------------------------------------------------------------
// Simulación en tiempo real con paso fijo ΔT
// ---------------------------------------------------------------------------
class LiveSim {
  constructor(canvas, { policy, compact = false, autoPush = false, theta0 = 10 * DEG, x0 = 0, onFall } = {}) {
    this.env = new CartPole({ maxSteps: Infinity });
    this.renderer = new CartPoleRenderer(canvas, { compact });
    this.policy = policy;
    this.mode = 'agent';
    this.autoPush = autoPush;
    this.theta0 = theta0;
    this.x0 = x0;
    this.speed = 1;
    this.paused = false;
    this.acc = 0;
    this.manualForce = 0;
    this.onFall = onFall;
    this.best = 0;
    this.reset();
  }

  reset(theta0 = this.theta0, x0 = this.x0) {
    this.env.reset(theta0, 0, x0);
    this.renderer.resetCamera();
    this.fallen = false;
    this.crashed = false;
    this.fallTime = 0;
    this.nextPush = 3 + Math.random() * 3;
    this.history = { theta: [], force: [], x: [], phase: [] };
    this.last = null;
  }

  push(dir, strength = 0.55) {
    if (!this.fallen) this.env.omega += dir * strength;
  }

  control() {
    const s = this.env.state();
    if (this.fallen || this.mode === 'none') return { force: 0 };
    if (this.mode === 'manual') return { force: this.manualForce };
    return this.policy.evaluate(s);
  }

  tick(dtReal) {
    if (this.paused) return;
    this.acc += Math.min(dtReal, 0.1) * this.speed;
    const dt = this.env.cfg.dt;
    while (this.acc >= dt) {
      this.acc -= dt;
      const { force } = this.control();
      this.last = { force };
      const { fell } = this.env.step(force);
      // Los topes del riel detienen el carro siempre, aunque el péndulo ya haya caído
      if (Math.abs(this.env.x) > PHYSICS.xLimit) {
        this.env.x = Math.sign(this.env.x) * PHYSICS.xLimit;
        this.env.v = 0;
        this.crashed = true;
      }
      const tSec = this.env.t * dt;
      if (!this.fallen) {
        const h = this.history;
        h.theta.push(this.env.theta / DEG);
        h.force.push(force);
        h.x.push(this.env.x);
        h.phase.push([this.env.theta, this.env.omega]);
        if (h.theta.length > 300) { h.theta.shift(); h.force.shift(); h.x.shift(); }
        if (h.phase.length > 60) h.phase.shift();
        this.best = Math.max(this.best, tSec);
        if (this.autoPush && tSec > this.nextPush) {
          this.push(Math.random() < 0.5 ? -1 : 1, 0.35 + Math.random() * 0.3);
          this.nextPush = tSec + 3 + Math.random() * 3;
        }
      }
      if (fell && !this.fallen) {
        this.fallen = true;
        this.fallTime = 0;
        this.onFall?.(true, this.crashed);
      }
      if (this.fallen) {
        this.fallTime += dt;
        // El péndulo termina de caer y se reinicia
        if (Math.abs(this.env.theta) > 80 * DEG) { this.env.theta = Math.sign(this.env.theta) * 80 * DEG; this.env.omega = 0; }
        if (this.fallTime > 1.3) {
          this.onFall?.(false);
          if (this.randomStart) this.reset((Math.random() * 2 - 1) * 10 * DEG, (Math.random() * 2 - 1) * 1.0);
          else this.reset();
        }
      }
    }
  }

  draw(decision = null) {
    this.renderer.draw({
      x: this.env.x, theta: this.env.theta, force: this.last?.force ?? 0,
      fallen: this.fallen, crashed: this.crashed, decision,
    });
  }
}

// Bucle de animación compartido; sólo se dibuja lo visible
const visible = new Set();
const io = new IntersectionObserver((entries) => {
  for (const e of entries) e.isIntersecting ? visible.add(e.target) : visible.delete(e.target);
});
const frameHooks = [];
let prevT = performance.now();
function frame(t) {
  const dt = (t - prevT) / 1000;
  prevT = t;
  for (const hook of frameHooks) hook(dt);
  requestAnimationFrame(frame);
}

// ---------------------------------------------------------------------------
// Héroe
// ---------------------------------------------------------------------------
const heroCanvas = $('hero-canvas');
const hero = new LiveSim(heroCanvas, { policy: policies.deep, autoPush: true, theta0: 9 * DEG, x0: -0.8 });
hero.randomStart = true;
io.observe(heroCanvas);
frameHooks.push((dt) => {
  if (!visible.has(heroCanvas)) return;
  hero.tick(dt);
  hero.draw();
  $('hero-theta').textContent = `${fmt(hero.env.theta / DEG)}°`;
  $('hero-force').textContent = `${fmt(hero.last?.force ?? 0)} N`;
  $('hero-x').textContent = `${fmt(hero.env.x, 2)} m`;
  $('hero-time').textContent = `${(hero.env.t * hero.env.cfg.dt).toFixed(1)} s`;
});

// ---------------------------------------------------------------------------
// Paso 3: tira de acciones
// ---------------------------------------------------------------------------
$('action-strip').innerHTML = DEEP_ACTIONS.map((f) => `<i title="${f} N" style="background:${forceColor(f)}"></i>`).join('');

// ---------------------------------------------------------------------------
// Paso 5 y demo: diagramas de la red
// ---------------------------------------------------------------------------
const netCanvas = $('net-canvas');
const netView = new NetworkView(netCanvas, policies.deep.net, { labels: STATE_LABELS });
io.observe(netCanvas);
const demoNetCanvas = $('demo-net');
const demoNet = new NetworkView(demoNetCanvas, policies.deep.net, { labels: STATE_LABELS });
io.observe(demoNetCanvas);

// ---------------------------------------------------------------------------
// Paso 8: resultados y gráficas
// ---------------------------------------------------------------------------
function greedyResponse(policy, seconds = 5, theta0 = 10 * DEG) {
  const env = new CartPole({ maxSteps: Infinity });
  env.reset(theta0, 0, 0);
  const out = { t: [], theta: [], force: [], x: [], fellAt: null, crashed: false };
  const n = Math.round(seconds / env.cfg.dt);
  for (let i = 0; i < n; i++) {
    const { force } = policy.evaluate(env.state());
    out.t.push(+(env.t * env.cfg.dt).toFixed(2));
    out.theta.push(env.theta / DEG);
    out.force.push(force);
    out.x.push(env.x);
    if (env.step(force).fell) { out.fellAt = env.t * env.cfg.dt; out.crashed = env.crashed; break; }
  }
  return out;
}

const responses = { deep: greedyResponse(policies.deep), shallow: greedyResponse(policies.shallow) };

// Métricas de la respuesta greedy; "régimen" = últimos 3 s
function responseStats(r) {
  const n = 150;
  const th = r.theta.slice(-n), F = r.force.slice(-n);
  const settle = r.t.find((_, i) => r.theta.slice(i).every((v) => Math.abs(v) < 1.5));
  let dF = 0;
  for (let i = 1; i < F.length; i++) dF += Math.abs(F[i] - F[i - 1]);
  return {
    rms: Math.sqrt(avg(th.map((v) => v * v))),
    peak: Math.max(...th.map(Math.abs)),
    effort: avg(F.map(Math.abs)),
    chatter: dF / (F.length - 1),
    levels: new Set(r.force).size,
    usedSteady: [...new Set(F)].sort((a, b) => a - b),
    xMax: Math.max(...r.x.map(Math.abs)),
    xEnd: Math.abs(r.x.at(-1)),
    settle,
  };
}
const firstEpisode = (h, thr = 250) => h.episode.find((_, i) => h.avg[i] >= thr);
const stats = { deep: responseStats(responses.deep), shallow: responseStats(responses.shallow) };

function diagnostics() {
  const S = stats.shallow, D = stats.deep;
  const rows = [
    ['Error RMS de θ en régimen', S.rms, D.rms, (v) => `${v.toFixed(2)}°`, 'min'],
    ['|θ| máximo en régimen', S.peak, D.peak, (v) => `${v.toFixed(2)}°`, 'min'],
    ['Desplazamiento máximo |x|', S.xMax, D.xMax, (v) => `${v.toFixed(2)} m`, 'min'],
    ['|x| al final (5 s)', S.xEnd, D.xEnd, (v) => `${v.toFixed(2)} m`, 'min'],
    ['Fuerza media |F|', S.effort, D.effort, (v) => `${v.toFixed(2)} N`, 'min'],
    ['Cambio medio de fuerza por paso', S.chatter, D.chatter, (v) => `${v.toFixed(2)} N`, 'min'],
    ['Niveles de fuerza usados', S.levels, D.levels, (v) => `${v}`, null],
    ['Episodios hasta promedio ≥ 250', firstEpisode(histShallow), firstEpisode(histDeep), (v) => `${v ?? '—'}`, 'min'],
    ['Mejor evaluación greedy', modelShallow.best_eval, modelDeep.best_eval, (v) => v.toFixed(1), 'max'],
  ];
  $('diag-table').innerHTML =
    '<thead><tr><th>Métrica</th><th class="num">Superficial</th><th class="num">Profunda</th></tr></thead><tbody>' +
    rows.map(([label, a, b, f, better]) => {
      const winA = better && (better === 'min' ? a < b : a > b);
      const winB = better && (better === 'min' ? b < a : b > a);
      return `<tr><td>${label}</td><td class="num${winA ? ' win' : ''}">${f(a)}</td><td class="num${winB ? ' win' : ''}">${f(b)}</td></tr>`;
    }).join('') + '</tbody>';

  $('compare-analysis').innerHTML =
    `Recompensa promedio (ventana de 20 episodios). La red superficial converge antes (episodio ${firstEpisode(histShallow)} contra ${firstEpisode(histDeep)}): ` +
    `explora 3 acciones en lugar de 41. La profunda tarda más, pero su política es ${(S.rms / D.rms).toFixed(1)} veces más precisa en régimen ` +
    `y deja el carro a ${D.xEnd.toFixed(2)} m del centro (la superficial, a ${S.xEnd.toFixed(2)} m).`;
}

function kpis() {
  const conv = firstEpisode(histDeep);
  const settle = stats.deep.settle;
  const items = [
    ['Recompensa máx. promedio', Math.max(...histDeep.avg).toFixed(1), 'de 300 posibles (ventana 20)'],
    ['Convergencia', conv ? `ep. ${conv}` : '—', 'promedio ≥ 250 por primera vez'],
    ['Tiempo de asentamiento', settle != null ? `${settle.toFixed(2)} s` : '—', '|θ| < 1.5° desde θ₀ = 10°'],
    ['Error RMS en régimen', `${stats.deep.rms.toFixed(2)}°`, `últimos 3 s · superficial: ${stats.shallow.rms.toFixed(2)}°`],
  ];
  $('kpis').innerHTML = items
    .map(([l, v, s]) => `<div class="card kpi"><div class="label">${l}</div><div class="value">${v}</div><div class="sub">${s}</div></div>`)
    .join('');

  const firstHigh = histDeep.episode.find((_, i) => histDeep.reward[i] >= 250);
  $('deep-analysis').innerHTML =
    `Durante los primeros episodios ε≈1 y el agente actúa casi al azar: el episodio termina en ~${Math.round(avg(histDeep.steps.slice(0, 50)))} pasos. ` +
    `El primer episodio con recompensa ≥ 250 llega en el episodio ${firstHigh}; a partir del ${conv} el promedio se mantiene cerca del máximo. ` +
    `Se conserva el modelo con mejor evaluación greedy (episodio ${modelDeep.best_episode}: ${modelDeep.best_eval.toFixed(1)} puntos promedio desde seis ángulos ` +
    `y ${modelDeep.best_long_ok ?? 4}/4 arranques que aguantan 30 s sin caer ni chocar).`;
}

function chartDefaults() {
  const C = window.Chart;
  C.defaults.font.family = 'Inter, system-ui, sans-serif';
  C.defaults.font.size = 11;
  C.defaults.color = cssVar('--muted');
  C.defaults.borderColor = cssVar('--border');
  C.defaults.animation = false;
  C.defaults.plugins.legend.display = false;
  C.defaults.plugins.tooltip.backgroundColor = 'rgba(11,16,32,0.92)';
  C.defaults.plugins.tooltip.titleColor = '#e8ecff';
  C.defaults.plugins.tooltip.bodyColor = '#e8ecff';
  C.defaults.plugins.tooltip.padding = 10;
  C.defaults.plugins.tooltip.cornerRadius = 8;
  C.defaults.elements.point.radius = 0;
  C.defaults.maintainAspectRatio = false;
  C.defaults.interaction = { mode: 'index', intersect: false };
}

const axis = (title, extra = {}) => ({ title: { display: !!title, text: title }, grid: { color: cssVar('--border') }, ...extra });
const charts = {};

function buildCharts() {
  if (!window.Chart) return;
  chartDefaults();
  const accent = cssVar('--accent'), accent2 = cssVar('--accent-2'), warn = cssVar('--warn');

  $('legend-deep').innerHTML =
    `<span style="--c:${accent2}66">recompensa por episodio</span><span style="--c:${accent}">promedio 20 episodios</span><span style="--c:${warn}">evaluación greedy</span>`;

  const evalPts = histDeep.eval.map(([x, y]) => ({ x, y }));
  charts.deep = new Chart($('chart-deep'), {
    type: 'line',
    data: {
      datasets: [
        { label: 'Recompensa', data: histDeep.episode.map((x, i) => ({ x, y: histDeep.reward[i] })), borderColor: `${accent2}55`, borderWidth: 1 },
        { label: 'Promedio (20)', data: histDeep.episode.map((x, i) => ({ x, y: histDeep.avg[i] })), borderColor: accent, borderWidth: 2.5 },
        { label: 'Evaluación greedy', data: evalPts, borderColor: warn, backgroundColor: warn, showLine: false, pointRadius: 2.5 },
      ],
    },
    options: {
      parsing: false,
      scales: { x: axis('Episodio', { type: 'linear', min: 1, max: histDeep.episode.length }), y: axis('Recompensa total', { min: -20, max: 310 }) },
    },
  });

  charts.compare = new Chart($('chart-compare'), {
    type: 'line',
    data: {
      datasets: [
        { label: 'Profunda · 41 acciones', data: histDeep.episode.map((x, i) => ({ x, y: histDeep.avg[i] })), borderColor: accent, borderWidth: 2 },
        { label: 'Superficial · 3 acciones', data: histShallow.episode.map((x, i) => ({ x, y: histShallow.avg[i] })), borderColor: cssVar('--danger'), borderWidth: 2 },
      ],
    },
    options: {
      parsing: false,
      plugins: { legend: { display: true, labels: { boxWidth: 12, boxHeight: 3 } } },
      scales: { x: axis('Episodio', { type: 'linear', min: 1, max: histDeep.episode.length }), y: axis('Recompensa promedio', { min: -20, max: 310 }) },
    },
  });

  const lossSmooth = smooth(histDeep.loss.map((v) => v ?? NaN), 25);
  charts.eps = new Chart($('chart-eps'), {
    type: 'line',
    data: {
      datasets: [
        { label: 'ε', data: histDeep.episode.map((x, i) => ({ x, y: histDeep.epsilon[i] })), borderColor: accent2, borderWidth: 2, yAxisID: 'y' },
        { label: 'Pérdida TD', data: histDeep.episode.map((x, i) => ({ x, y: lossSmooth[i] })), borderColor: warn, borderWidth: 1.6, yAxisID: 'y2' },
      ],
    },
    options: {
      parsing: false,
      plugins: { legend: { display: true, labels: { boxWidth: 12, boxHeight: 3 } } },
      scales: {
        x: axis('Episodio', { type: 'linear', min: 1, max: histDeep.episode.length }),
        y: axis('ε', { min: 0, max: 1 }),
        y2: axis('Pérdida (MSE)', { position: 'right', type: 'logarithmic', grid: { display: false } }),
      },
    },
  });

  charts.theta = new Chart($('chart-theta'), {
    type: 'line',
    data: { datasets: [
      { label: 'θ', data: [], borderColor: accent2, borderWidth: 2.2 },
      { label: 'límite', data: [{ x: 0, y: 12 }, { x: 5, y: 12 }], borderColor: `${cssVar('--danger')}88`, borderDash: [5, 5], borderWidth: 1 },
      { label: 'límite', data: [{ x: 0, y: -12 }, { x: 5, y: -12 }], borderColor: `${cssVar('--danger')}88`, borderDash: [5, 5], borderWidth: 1 },
    ] },
    options: { parsing: false, scales: { x: axis('Tiempo (s)', { type: 'linear', min: 0, max: 5 }), y: axis('θ (°)', { min: -13, max: 13 }) } },
  });
  charts.force = new Chart($('chart-force'), {
    type: 'line',
    data: { datasets: [{ label: 'F', data: [], borderColor: warn, borderWidth: 1.4, stepped: true }] },
    options: { parsing: false, scales: { x: axis('Tiempo (s)', { type: 'linear', min: 0, max: 5 }), y: axis('F (N)', { min: -11, max: 11 }) } },
  });
  showResponse('deep');

  charts.live = new Chart($('chart-live'), {
    type: 'line',
    data: { datasets: [
      { label: 'Recompensa', data: [], borderColor: `${accent2}55`, borderWidth: 1 },
      { label: 'Promedio (20)', data: [], borderColor: accent, borderWidth: 2.4 },
      { label: 'Evaluación greedy', data: [], borderColor: warn, backgroundColor: warn, showLine: false, pointRadius: 3 },
    ] },
    options: {
      parsing: false,
      plugins: { legend: { display: true, labels: { boxWidth: 12, boxHeight: 3 } } },
      scales: { x: axis('Episodio', { type: 'linear', min: 1, max: 600 }), y: axis('Recompensa total', { min: -20, max: 310 }) },
    },
  });
}

function smooth(arr, k) {
  const out = [];
  for (let i = 0; i < arr.length; i++) {
    const w = arr.slice(Math.max(0, i - k + 1), i + 1).filter((v) => Number.isFinite(v));
    out.push(w.length ? avg(w) : null);
  }
  return out;
}

function showResponse(key) {
  const r = responses[key];
  if (charts.theta) {
    charts.theta.data.datasets[0].data = r.t.map((x, i) => ({ x, y: r.theta[i] }));
    charts.theta.data.datasets[0].borderColor = key === 'deep' ? cssVar('--accent-2') : cssVar('--danger');
    charts.force.data.datasets[0].data = r.t.map((x, i) => ({ x, y: r.force[i] }));
    charts.theta.update();
    charts.force.update();
  }
  const st = stats[key];
  const settle = st.settle != null ? `en ${st.settle.toFixed(2)} s` : '';
  const levels = st.usedSteady.map((f) => `${fmt(f, 1)}`).join(', ');
  $('resp-analysis').innerHTML = r.fellAt != null
    ? `<b>${key === 'deep' ? 'Red profunda' : 'Red superficial'}:</b> ${r.crashed ? 'el carro choca con el tope' : 'el péndulo cae'} a los ${r.fellAt.toFixed(2)} s.`
    : key === 'deep'
      ? `<b>Red profunda:</b> empuja fuerte para recuperar el ángulo, llega a |θ| < 1.5° ${settle} y lo mantiene con un error RMS de ${st.rms.toFixed(2)}°. ` +
        `En régimen usa las fuerzas {${levels}} N, y regresa el carro al centro: termina a ${st.xEnd.toFixed(2)} m.`
      : `<b>Red superficial:</b> también recupera la vertical ${settle}, pero sólo puede aplicar −10, 0 o +10 N. Corrige con golpes de fuerza máxima, ` +
        `el ángulo oscila con un error RMS de ${st.rms.toFixed(2)}° (pico de ${st.peak.toFixed(2)}°) y el carro termina a ${st.xEnd.toFixed(2)} m del centro.`;
}

document.querySelectorAll('#resp-toggle button').forEach((b) =>
  b.addEventListener('click', () => {
    document.querySelectorAll('#resp-toggle button').forEach((x) => x.classList.toggle('on', x === b));
    showResponse(b.dataset.model);
  }),
);

// Mapa estático de la política (carro quieto en el centro)
const staticMap = new PolicyMap($('policy-canvas'), { nx: 120, ny: 72 });
function drawPolicyMap() {
  staticMap.update(policies.deep, 0, 0);
  staticMap.draw();
}

// ---------------------------------------------------------------------------
// ¿Por qué decide eso la red? Atribución con gradientes integrados
// ---------------------------------------------------------------------------
// f(s) = max Q(s, F>0) − max Q(s, F<0): qué tanto prefiere la red empujar a la
// derecha que a la izquierda. La atribución de la entrada i es s̃ᵢ · el gradiente
// ∂f/∂s̃ᵢ promediado en el camino de 0 (equilibrio) a s̃, y en total suman
// f(s) − f(0). Positivo = empuja la decisión a la derecha, negativo = a la izquierda.
function explain(policy, s) {
  const info = policy.evaluate(s);
  const { q, a, input } = info;
  const actions = policy.actions;
  const a0 = actions.indexOf(0) >= 0 ? actions.indexOf(0) : Math.floor(actions.length / 2);
  let second = -Infinity, aR = -1, aL = -1;
  q.forEach((v, i) => {
    if (i !== a && v > second) second = v;
    if (actions[i] > 0 && (aR < 0 || v > q[aR])) aR = i;
    if (actions[i] < 0 && (aL < 0 || v > q[aL])) aL = i;
  });
  const g = new Float64Array(q.length);
  g[aR] = 1;
  g[aL] = -1;
  const attr = new Array(input.length).fill(0);
  const K = 12;
  for (let k = 0; k < K; k++) {
    const alpha = (k + 0.5) / K;
    const grad = policy.net.inputGrad(input.map((v) => v * alpha), g);
    for (let i = 0; i < input.length; i++) attr[i] += (input[i] * grad[i]) / K;
  }
  return { ...info, a0, qStar: q[a], q0: q[a0], margin: q[a] - second, attr };
}

const describe = {
  x: (v) => (Math.abs(v) < 0.05 ? 'el carro está en el centro' : `el carro está ${v > 0 ? 'a la derecha' : 'a la izquierda'} del centro (x = ${fmt(v, 2)} m)`),
  'ẋ': (v) => (Math.abs(v) < 0.03 ? 'el carro casi no se mueve' : `el carro se mueve hacia la ${v > 0 ? 'derecha' : 'izquierda'} (ẋ = ${fmt(v, 2)} m/s)`),
  'θ': (v) => (Math.abs(v) < 0.2 * DEG ? 'el péndulo está vertical' : `el péndulo se inclina a la ${v > 0 ? 'derecha' : 'izquierda'} (θ = ${fmt(v / DEG)}°)`),
  'θ̇': (v) => (Math.abs(v) < 0.02 ? 'el péndulo casi no gira' : `el péndulo gira hacia la ${v > 0 ? 'derecha' : 'izquierda'} (θ̇ = ${fmt(v, 2)} rad/s)`),
};
const physValue = [(v) => `${fmt(v, 2)} m`, (v) => `${fmt(v, 2)} m/s`, (v) => `${fmt(v / DEG, 1)}°`, (v) => `${fmt(v, 2)}/s`];

function decisionText(ex, s, manual) {
  const F = ex.policy.actions[ex.a];
  const order = ex.attr.map((v, i) => [i, v]).sort((p, q) => Math.abs(q[1]) - Math.abs(p[1]));
  const total = order.reduce((t, [, v]) => t + Math.abs(v), 0) || 1;
  const big = order.filter(([, v]) => Math.abs(v) / total > 0.12);
  const name = (i) => describe[STATE_LABELS[i]](s[i]);
  if (ex.a === ex.a0) {
    const head = manual ? 'La red no empujaría' : 'La red no empuja';
    const r = big.filter(([, v]) => v > 0)[0], l = big.filter(([, v]) => v < 0)[0];
    if (r && l) return `${head}: ${name(r[0])} pide empujar a la derecha, pero ${name(l[0])} pide lo contrario, y se compensan.`;
    return `${head}: con este estado, ninguna fuerza espera más recompensa que dejar el carro libre.`;
  }
  const dir = Math.sign(F);
  const pro = big.filter(([, v]) => Math.sign(v) === dir).slice(0, 2);
  const con = big.filter(([, v]) => Math.sign(v) === -dir).slice(0, 1);
  let txt = `${manual ? 'La red empujaría' : 'La red empuja'} <b>${fmt(F)} N</b> hacia la ${F > 0 ? 'derecha' : 'izquierda'}`;
  if (pro.length) txt += ` porque ${pro.map(([i]) => name(i)).join(' y ')}`;
  txt += '.';
  if (con.length) txt += ` En contra pesa que ${name(con[0][0])}, por eso no empuja más fuerte.`;
  return txt;
}

// ---------------------------------------------------------------------------
// Paso 9: demo interactiva
// ---------------------------------------------------------------------------
const demoCanvas = $('demo-canvas');
const demo = new LiveSim(demoCanvas, {
  policy: policies.deep,
  onFall: (on, crashed) => {
    $('demo-overlay').hidden = !on;
    if (on) $('overlay-title').textContent = crashed ? '¡Chocó con el tope!' : '¡Cayó!';
  },
});
io.observe(demoCanvas);

const ctrlSelect = $('ctrl-select');
function setDemoPolicy(policy) {
  demo.policy = policy;
  netView.setNetwork(policy.net);
  demoNet.setNetwork(policy.net);
}
ctrlSelect.addEventListener('change', () => {
  const v = ctrlSelect.value;
  demo.mode = v === 'manual' || v === 'none' ? v : 'agent';
  // En modo manual o sin control, el panel muestra lo que haría la red profunda
  setDemoPolicy(policies[v] ?? policies.deep);
  demo.best = 0;
  demo.reset();
});
$('theta0').addEventListener('input', (e) => {
  demo.theta0 = +e.target.value * DEG;
  $('theta0-val').textContent = `${e.target.value}°`;
});
$('theta0').addEventListener('change', () => demo.reset());
$('x0').addEventListener('input', (e) => {
  demo.x0 = +e.target.value;
  $('x0-val').textContent = `${fmt(+e.target.value, 1)} m`;
});
$('x0').addEventListener('change', () => demo.reset());
$('speed').addEventListener('input', (e) => {
  demo.speed = +e.target.value;
  $('speed-val').textContent = `${e.target.value}×`;
});
$('push-left').addEventListener('click', () => demo.push(-1));
$('push-right').addEventListener('click', () => demo.push(1));
$('reset').addEventListener('click', () => demo.reset());
$('pause').addEventListener('click', (e) => {
  demo.paused = !demo.paused;
  e.target.textContent = demo.paused ? 'Continuar' : 'Pausa';
});
$('auto-push').addEventListener('change', (e) => { demo.autoPush = e.target.checked; });

// Clic / toque: empujón (o fuerza manual mientras se mantiene presionado)
demoCanvas.addEventListener('pointerdown', (e) => {
  const r = demoCanvas.getBoundingClientRect();
  const dir = e.clientX - r.left < r.width / 2 ? -1 : 1;
  if (demo.mode === 'manual') demo.manualForce = 10 * dir;
  else demo.push(dir);
});
const release = () => { if (demo.mode === 'manual') demo.manualForce = 0; };
demoCanvas.addEventListener('pointerup', release);
demoCanvas.addEventListener('pointerleave', release);

const keys = new Set();
window.addEventListener('keydown', (e) => {
  if (!visible.has(demoCanvas) || !['ArrowLeft', 'ArrowRight'].includes(e.key)) return;
  e.preventDefault();
  keys.add(e.key);
  if (demo.mode === 'manual') demo.manualForce = (keys.has('ArrowRight') ? 10 : 0) - (keys.has('ArrowLeft') ? 10 : 0);
  else if (!e.repeat) demo.push(e.key === 'ArrowLeft' ? -1 : 1);
});
window.addEventListener('keyup', (e) => {
  keys.delete(e.key);
  if (demo.mode === 'manual') demo.manualForce = (keys.has('ArrowRight') ? 10 : 0) - (keys.has('ArrowLeft') ? 10 : 0);
});

const qCanvas = $('q-canvas'), attrCanvas = $('attr-canvas');
const thetaTrace = $('theta-trace'), forceTrace = $('force-trace'), xTrace = $('x-trace');
const liveMap = new PolicyMap($('live-map'), { nx: 60, ny: 36 });
const brainEls = [qCanvas, attrCanvas, $('live-map'), $('dec-text')];
brainEls.forEach((el) => io.observe(el));
let lastText = 0, lastMap = 0;

frameHooks.push((dt) => {
  const demoVisible = visible.has(demoCanvas);
  const brainVisible = brainEls.some((el) => visible.has(el)) || visible.has(demoNetCanvas);
  const netVisible = visible.has(netCanvas);
  if (!demoVisible && !netVisible && !brainVisible) return;
  demo.tick(dt);
  const s = demo.env.state();
  const ex = explain(demo.policy, s);
  ex.policy = demo.policy;
  if (netVisible) netView.draw(ex.acts, ex.a);
  if (visible.has(demoNetCanvas)) demoNet.draw(ex.acts, ex.a);
  if (demoVisible) {
    demo.draw(demo.fallen ? null : { q: ex.q, actions: demo.policy.actions, a: ex.a });
    const e = demo.env;
    $('d-theta').textContent = `${fmt(e.theta / DEG)}°`;
    $('d-theta').style.color = demo.fallen ? cssVar('--danger') : stabilityColor(e.theta);
    $('d-omega').textContent = fmt(e.omega, 2);
    $('d-force').textContent = `${fmt(demo.last?.force ?? 0)} N`;
    $('d-x').textContent = `${fmt(e.x, 2)} m`;
    $('d-x').style.color = Math.abs(e.x) > PHYSICS.xLimit - 0.4 ? cssVar('--danger') : '';
    $('d-time').textContent = `${(e.t * e.cfg.dt).toFixed(1)} s`;
    $('d-best').textContent = `${demo.best.toFixed(1)} s`;
  }
  if (!brainVisible && !demoVisible) return;

  drawQChart(qCanvas, ex.q, demo.policy.actions, ex.a);
  drawAttribution(attrCanvas, STATE_LABELS.map((label, i) => ({ label, value: ex.attr[i], detail: physValue[i](s[i]) })));
  const now = performance.now();
  if (now - lastMap > 200) {
    liveMap.update(demo.policy, s[0], s[1]);
    lastMap = now;
  }
  liveMap.draw(demo.history.phase);
  drawTrace(thetaTrace, demo.history.theta, { min: -13, max: 13, color: cssVar('--accent-2'), label: '±13°', bands: [12, -12] });
  drawTrace(xTrace, demo.history.x, { min: -2.6, max: 2.6, color: cssVar('--accent'), label: '±2.6 m', bands: [2.4, -2.4] });
  drawTrace(forceTrace, demo.history.force, { min: -11, max: 11, color: cssVar('--warn'), step: true, label: '±10 N' });

  // Texto y métricas a ~6 Hz para que se puedan leer
  if (now - lastText > 160) {
    lastText = now;
    const F = demo.policy.actions[ex.a];
    const manual = demo.mode !== 'agent';
    $('dec-force').textContent = `${fmt(F)} N`;
    $('dec-force').style.color = Math.abs(F) < 2 ? cssVar('--text') : forceColor(F);
    $('dec-dir').textContent = manual ? 'sugerencia (no se aplica)' : F === 0 ? 'sin empujar' : F > 0 ? 'empuja a la derecha →' : '← empuja a la izquierda';
    $('dec-text').innerHTML = demo.fallen ? 'El episodio terminó; reiniciando…' : decisionText(ex, s, manual);
    $('dec-q').textContent = ex.qStar.toFixed(2);
    $('dec-adv').textContent = ex.a === ex.a0 ? '—' : fmt(ex.qStar - ex.q0, 3);
    $('dec-margin').textContent = ex.margin.toFixed(3);
  }
});

// ---------------------------------------------------------------------------
// Paso 10: entrenamiento en el navegador (Web Worker)
// ---------------------------------------------------------------------------
let worker = null;
let liveHist = { window: [] };
let trainState = 'idle'; // idle | running | paused | done
const previewCanvas = $('preview-canvas');
const preview = new LiveSim(previewCanvas, { policy: null, compact: true, theta0: 8 * DEG });
preview.randomStart = true;
io.observe(previewCanvas);
frameHooks.push((dt) => {
  if (!preview.policy || !visible.has(previewCanvas)) return;
  preview.tick(dt);
  preview.draw();
});

$('t-episodes').addEventListener('input', (e) => { $('t-ep-val').textContent = e.target.value; });

function setTrainState(s) {
  trainState = s;
  $('t-start').textContent = s === 'running' || s === 'paused' ? 'Reiniciar' : 'Entrenar';
  $('t-pause').disabled = !(s === 'running' || s === 'paused');
  $('t-pause').textContent = s === 'paused' ? 'Continuar' : 'Pausa';
}

const mmss = (sec) => `${Math.floor(sec / 60)}:${String(Math.round(sec % 60)).padStart(2, '0')}`;

$('t-start').addEventListener('click', () => {
  worker?.terminate();
  const shallow = $('t-arch').value === 'shallow';
  const episodes = +$('t-episodes').value;
  const config = {
    episodes, lr: +$('t-lr').value, seed: +$('t-seed').value, shallow,
    hidden: shallow ? [24] : [64, 64], trainEvery: 2,
  };
  try {
    worker = new Worker(new URL('./train-worker.js', import.meta.url), { type: 'module' });
  } catch (err) {
    $('t-status').textContent = `Tu navegador no pudo iniciar el entrenamiento (${err.message}). Prueba con una versión reciente de Chrome, Edge, Firefox o Safari.`;
    return;
  }
  worker.onerror = (err) => {
    $('t-status').textContent = `Error en el entrenamiento: ${err.message ?? 'desconocido'}`;
    setTrainState('idle');
  };
  liveHist = { window: [] };
  if (charts.live) {
    charts.live.data.datasets.forEach((d) => { d.data = []; });
    charts.live.options.scales.x.max = episodes;
    charts.live.update();
  }
  $('t-best').textContent = '';
  const t0 = performance.now();
  worker.onmessage = ({ data }) => {
    if (data.type !== 'progress') return;
    for (const ep of data.batch) {
      liveHist.window.push(ep.reward);
      if (liveHist.window.length > 20) liveHist.window.shift();
      charts.live?.data.datasets[0].data.push({ x: ep.episode, y: ep.reward });
      charts.live?.data.datasets[1].data.push({ x: ep.episode, y: avg(liveHist.window) });
      if (ep.eval) charts.live?.data.datasets[2].data.push({ x: ep.episode, y: ep.eval.score });
    }
    charts.live?.update();
    const last = data.batch.at(-1);
    if (last) {
      const elapsed = (performance.now() - t0) / 1000;
      // Estimación con el ritmo de pasos actual y la duración del último episodio
      const stepsPerSec = data.steps / Math.max(elapsed, 1e-3);
      const eta = ((episodes - last.episode) * Math.max(last.steps, 150)) / stepsPerSec;
      $('t-bar').style.width = `${(100 * last.episode) / episodes}%`;
      $('t-status').textContent =
        `Episodio ${last.episode}/${episodes} · ε = ${last.epsilon.toFixed(3)} · promedio ${avg(liveHist.window).toFixed(1)} · ` +
        `${Math.round(stepsPerSec).toLocaleString('es-MX')} pasos/s · ${mmss(elapsed)}` + (data.done ? '' : ` · faltan ~${mmss(eta)}`);
    }
    if (data.best) {
      policies.browser = new Policy(data.best);
      preview.policy = policies.browser;
      preview.reset();
      const b = data.best;
      $('preview-label').textContent = `Mejor modelo hasta ahora: episodio ${b.best_episode}`;
      $('t-best').innerHTML = `Mejor modelo: episodio <b>${b.best_episode}</b> · ${b.best_eval.toFixed(1)} puntos · aguanta 30 s en ` +
        `<b>${b.best_long_ok}/${b.long_total}</b> arranques de prueba`;
      $('t-use').disabled = false;
      ctrlSelect.querySelector('option[value="browser"]').disabled = false;
      if (ctrlSelect.value === 'browser') setDemoPolicy(policies.browser);
    }
    if (data.done) {
      setTrainState('done');
      $('t-status').textContent += ' · ¡terminado!';
      const ok = policies.browser?.meta.best_long_ok === policies.browser?.meta.long_total;
      if (!ok) $('t-best').innerHTML += '<br><span class="neg-txt">Todavía no es robusto: prueba con más episodios u otra semilla.</span>';
    }
  };
  worker.postMessage({ type: 'start', config });
  setTrainState('running');
  $('t-status').textContent = 'Llenando la memoria de repetición…';
});

$('t-pause').addEventListener('click', () => {
  if (!worker) return;
  if (trainState === 'running') { worker.postMessage({ type: 'pause' }); setTrainState('paused'); }
  else if (trainState === 'paused') { worker.postMessage({ type: 'resume' }); setTrainState('running'); }
});

$('t-use').addEventListener('click', () => {
  ctrlSelect.value = 'browser';
  ctrlSelect.dispatchEvent(new Event('change'));
  $('demo').scrollIntoView({ behavior: 'smooth' });
});

// ---------------------------------------------------------------------------
// Navegación activa, KaTeX y arranque
// ---------------------------------------------------------------------------
const navLinks = [...document.querySelectorAll('.topbar nav a')];
const navIO = new IntersectionObserver((entries) => {
  for (const e of entries) {
    if (!e.isIntersecting) continue;
    navLinks.forEach((a) => a.classList.toggle('active', a.getAttribute('href') === `#${e.target.id}`));
  }
}, { rootMargin: '-45% 0px -50% 0px' });
document.querySelectorAll('section[id]').forEach((s) => navIO.observe(s));

window.renderMathInElement?.(document.body, {
  delimiters: [
    { left: '$$', right: '$$', display: true },
    { left: '$', right: '$', display: false },
  ],
  throwOnError: false,
});

kpis();
diagnostics();
buildCharts();
drawPolicyMap();
let resizeTimer;
window.addEventListener('resize', () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(drawPolicyMap, 150);
});
requestAnimationFrame(frame);
