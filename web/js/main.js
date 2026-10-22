import { CartPole, DEG } from './physics.js';
import { Policy, DEEP_ACTIONS } from './dqn.js';
import { CartPoleRenderer, NetworkView, drawQBars, drawTrace, fitCanvas, stabilityColor } from './render.js';

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
  constructor(canvas, { policy, compact = false, autoPush = false, theta0 = 10 * DEG, onFall } = {}) {
    this.env = new CartPole({ maxSteps: Infinity });
    this.renderer = new CartPoleRenderer(canvas, { compact });
    this.policy = policy;
    this.mode = 'agent';
    this.autoPush = autoPush;
    this.theta0 = theta0;
    this.speed = 1;
    this.paused = false;
    this.acc = 0;
    this.manualForce = 0;
    this.onFall = onFall;
    this.history = { theta: [], force: [] };
    this.last = null;
    this.best = 0;
    this.reset();
  }

  reset(theta0 = this.theta0) {
    this.env.reset(theta0, 0);
    this.renderer.resetCamera(0);
    this.fallen = false;
    this.fallTime = 0;
    this.nextPush = 3 + Math.random() * 3;
    this.history = { theta: [], force: [] };
    this.last = null;
  }

  push(dir, strength = 0.55) {
    if (!this.fallen) this.env.omega += dir * strength;
  }

  control() {
    const s = this.env.state();
    if (this.fallen || this.mode === 'none') return { force: 0 };
    if (this.mode === 'manual') return { force: this.manualForce, ...(this.policy ? this.policy.evaluate(s) : {}), manual: true };
    return this.policy.evaluate(s);
  }

  tick(dtReal) {
    if (this.paused) return;
    this.acc += Math.min(dtReal, 0.1) * this.speed;
    const dt = this.env.cfg.dt;
    while (this.acc >= dt) {
      this.acc -= dt;
      const out = this.control();
      const force = out.manual ? this.manualForce : out.force;
      this.last = { ...out, force };
      const { fell } = this.env.step(force);
      const tSec = this.env.t * dt;
      if (!this.fallen) {
        this.history.theta.push(this.env.theta / DEG);
        this.history.force.push(force);
        if (this.history.theta.length > 300) { this.history.theta.shift(); this.history.force.shift(); }
        this.best = Math.max(this.best, tSec);
        if (this.autoPush && tSec > this.nextPush) {
          this.push(Math.random() < 0.5 ? -1 : 1, 0.35 + Math.random() * 0.3);
          this.nextPush = tSec + 3 + Math.random() * 3;
        }
      }
      if (fell && !this.fallen) {
        this.fallen = true;
        this.fallTime = 0;
        this.onFall?.(true);
      }
      if (this.fallen) {
        this.fallTime += dt;
        // Deja caer el péndulo un momento y reinicia
        if (Math.abs(this.env.theta) > 80 * DEG) { this.env.theta = Math.sign(this.env.theta) * 80 * DEG; this.env.omega = 0; }
        if (this.fallTime > 1.3) {
          this.onFall?.(false);
          const th0 = this.randomStart ? (Math.random() * 2 - 1) * 10 * DEG : this.theta0;
          this.reset(th0);
        }
      }
    }
  }

  draw() {
    this.renderer.draw({ x: this.env.x, theta: this.env.theta, force: this.last?.force ?? 0, fallen: this.fallen });
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
const hero = new LiveSim(heroCanvas, { policy: policies.deep, autoPush: true, theta0: 9 * DEG });
hero.randomStart = true;
io.observe(heroCanvas);
frameHooks.push((dt) => {
  if (!visible.has(heroCanvas)) return;
  hero.tick(dt);
  hero.draw();
  $('hero-theta').textContent = `${fmt(hero.env.theta / DEG)}°`;
  $('hero-force').textContent = `${fmt(hero.last?.force ?? 0)} N`;
  $('hero-time').textContent = `${(hero.env.t * hero.env.cfg.dt).toFixed(1)} s`;
});

// ---------------------------------------------------------------------------
// Paso 3: tira de acciones
// ---------------------------------------------------------------------------
$('action-strip').innerHTML = DEEP_ACTIONS.map((f) => `<i title="${f} N" style="background:${forceColor(f)}"></i>`).join('');

function forceColor(f, alpha = 1) {
  const t = Math.max(-1, Math.min(1, f / 10));
  const mid = [27, 35, 80], pos = [46, 230, 197], neg = [255, 93, 115];
  const target = t >= 0 ? pos : neg;
  const k = Math.abs(t);
  const c = mid.map((v, i) => Math.round(v + (target[i] - v) * k));
  return `rgba(${c[0]},${c[1]},${c[2]},${alpha})`;
}

// ---------------------------------------------------------------------------
// Paso 5: red en vivo
// ---------------------------------------------------------------------------
const netCanvas = $('net-canvas');
const netView = new NetworkView(netCanvas, policies.deep.net);
io.observe(netCanvas);

// ---------------------------------------------------------------------------
// Paso 8: resultados y gráficas
// ---------------------------------------------------------------------------
function greedyResponse(policy, seconds = 5, theta0 = 10 * DEG) {
  const env = new CartPole({ maxSteps: Infinity });
  env.reset(theta0, 0);
  const out = { t: [], theta: [], force: [], fellAt: null };
  const n = Math.round(seconds / env.cfg.dt);
  for (let i = 0; i < n; i++) {
    const { force } = policy.evaluate(env.state());
    out.t.push(+(env.t * env.cfg.dt).toFixed(2));
    out.theta.push(env.theta / DEG);
    out.force.push(force);
    if (env.step(force).fell) { out.fellAt = env.t * env.cfg.dt; break; }
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
    ['Fuerza media |F|', S.effort, D.effort, (v) => `${v.toFixed(2)} N`, 'min'],
    ['Cambio medio de fuerza por paso', S.chatter, D.chatter, (v) => `${v.toFixed(2)} N`, 'min'],
    ['Niveles de fuerza usados', S.levels, D.levels, (v) => `${v}`, null],
    ['Episodios hasta promedio ≥ 250', firstEpisode(histShallow), firstEpisode(histDeep), (v) => `${v}`, 'min'],
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
    `explora 3 acciones en lugar de 41. La profunda tarda más, pero su política es ${(S.rms / D.rms).toFixed(1)} veces más precisa en régimen.`;
}

function kpis() {
  const conv = histDeep.episode.find((_, i) => histDeep.avg[i] >= 250);
  const settle = stats.deep.settle;
  const rms = stats.deep.rms;
  const items = [
    ['Recompensa máx. promedio', Math.max(...histDeep.avg).toFixed(1), 'de 300 posibles (ventana 20)'],
    ['Convergencia', conv ? `ep. ${conv}` : '—', 'promedio ≥ 250 por primera vez'],
    ['Tiempo de asentamiento', settle != null ? `${settle.toFixed(2)} s` : '—', '|θ| < 1.5° desde θ₀ = 10°'],
    ['Error RMS en régimen', `${rms.toFixed(2)}°`, `últimos 3 s · superficial: ${stats.shallow.rms.toFixed(2)}°`],
  ];
  $('kpis').innerHTML = items
    .map(([l, v, s]) => `<div class="card kpi"><div class="label">${l}</div><div class="value">${v}</div><div class="sub">${s}</div></div>`)
    .join('');

  const firstHigh = histDeep.episode.find((_, i) => histDeep.reward[i] >= 250);
  $('deep-analysis').innerHTML =
    `Durante los primeros episodios ε≈1 y el agente actúa casi al azar: el péndulo cae en ~${Math.round(avg(histDeep.steps.slice(0, 50)))} pasos. ` +
    `El primer episodio con recompensa ≥ 250 llega en el episodio ${firstHigh}; a partir del ${conv} el promedio se mantiene cerca del máximo. ` +
    `Se conserva el modelo con mejor evaluación greedy (episodio ${modelDeep.best_episode}, ${modelDeep.best_eval.toFixed(1)} puntos promedio desde seis ángulos iniciales).`;
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
    ? `<b>${key === 'deep' ? 'Red profunda' : 'Red superficial'}:</b> el péndulo cae a los ${r.fellAt.toFixed(2)} s.`
    : key === 'deep'
      ? `<b>Red profunda:</b> empuja con +10 N para recuperar el ángulo, llega a |θ| < 1.5° ${settle} y después lo mantiene con un error RMS de ${st.rms.toFixed(2)}°. ` +
        `En régimen usa las fuerzas {${levels}} N: empujes pequeños e intermedios en lugar de golpes de fuerza máxima.`
      : `<b>Red superficial:</b> también recupera la vertical ${settle}, pero sólo puede aplicar −10, 0 o +10 N. Corrige con golpes de fuerza máxima, ` +
        `cada golpe la pasa de largo y el ángulo oscila con un error RMS de ${st.rms.toFixed(2)}° (pico de ${st.peak.toFixed(2)}°).`;
}

document.querySelectorAll('#resp-toggle button').forEach((b) =>
  b.addEventListener('click', () => {
    document.querySelectorAll('#resp-toggle button').forEach((x) => x.classList.toggle('on', x === b));
    showResponse(b.dataset.model);
  }),
);

// Mapa de la política: acción greedy sobre una malla (θ, θ̇)
function drawPolicyMap() {
  const canvas = $('policy-canvas');
  const { ctx, w, h } = fitCanvas(canvas);
  const padL = 34, padB = 22, padR = 14, padT = 8, gw = w - padL - padR, gh = h - padB - padT;
  const nx = 90, ny = 56;
  const cw = gw / nx, ch = gh / ny;
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < ny; j++) {
      const th = (-12 + (24 * (i + 0.5)) / nx) * DEG;
      const om = 2 - (4 * (j + 0.5)) / ny;
      const { force } = policies.deep.evaluate([th, om]);
      ctx.fillStyle = forceColor(force);
      ctx.fillRect(padL + i * cw, padT + j * ch, cw + 0.6, ch + 0.6);
    }
  }
  ctx.fillStyle = cssVar('--muted');
  ctx.font = '10px "JetBrains Mono", monospace';
  ctx.textAlign = 'center';
  for (const d of [-12, -6, 0, 6, 12]) ctx.fillText(`${d}°`, padL + ((d + 12) / 24) * gw, h - 6);
  ctx.textAlign = 'right';
  for (const v of [2, 1, 0, -1, -2]) ctx.fillText(`${v}`, padL - 6, padT + ((2 - v) / 4) * gh + 4);
  ctx.save();
  ctx.translate(10, padT + gh / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = 'center';
  ctx.fillText('θ̇ (rad/s)', 0, 0);
  ctx.restore();
}

// ---------------------------------------------------------------------------
// Paso 9: demo interactiva
// ---------------------------------------------------------------------------
const demoCanvas = $('demo-canvas');
const demo = new LiveSim(demoCanvas, {
  policy: policies.deep,
  onFall: (on) => { $('demo-overlay').hidden = !on; },
});
io.observe(demoCanvas);

const ctrlSelect = $('ctrl-select');
ctrlSelect.addEventListener('change', () => {
  const v = ctrlSelect.value;
  demo.mode = v === 'manual' || v === 'none' ? v : 'agent';
  if (policies[v]) demo.policy = policies[v];
  else demo.policy = policies.deep;
  netView.setNetwork(demo.policy.net);
  demo.best = 0;
  demo.reset();
});
$('theta0').addEventListener('input', (e) => {
  demo.theta0 = +e.target.value * DEG;
  $('theta0-val').textContent = `${e.target.value}°`;
});
$('theta0').addEventListener('change', () => demo.reset());
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

const qCanvas = $('q-canvas'), thetaTrace = $('theta-trace'), forceTrace = $('force-trace');
frameHooks.push((dt) => {
  const demoVisible = visible.has(demoCanvas);
  const netVisible = visible.has(netCanvas);
  if (!demoVisible && !netVisible) return;
  demo.tick(dt);
  const s = demo.env.state();
  const info = demo.policy.evaluate(s);
  if (netVisible) netView.draw(info.acts, info.a);
  if (!demoVisible) return;
  demo.draw();
  const e = demo.env;
  $('d-theta').textContent = `${fmt(e.theta / DEG)}°`;
  $('d-theta').style.color = demo.fallen ? cssVar('--danger') : stabilityColor(e.theta);
  $('d-omega').textContent = fmt(e.omega, 2);
  $('d-force').textContent = `${fmt(demo.last?.force ?? 0)} N`;
  $('d-x').textContent = `${fmt(e.x, 2)} m`;
  $('d-time').textContent = `${(e.t * e.cfg.dt).toFixed(1)} s`;
  $('d-best').textContent = `${demo.best.toFixed(1)} s`;
  drawQBars(qCanvas, info.q, demo.policy.actions, info.a);
  drawTrace(thetaTrace, demo.history.theta, { min: -13, max: 13, color: cssVar('--accent-2'), label: '±13°' });
  drawTrace(forceTrace, demo.history.force, { min: -11, max: 11, color: cssVar('--warn'), step: true, label: '±10 N' });
});

// ---------------------------------------------------------------------------
// Paso 10: entrenamiento en el navegador (Web Worker)
// ---------------------------------------------------------------------------
let worker = null;
let liveHist = { reward: [], window: [] };
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

$('t-start').addEventListener('click', () => {
  worker?.terminate();
  const shallow = $('t-arch').value === 'shallow';
  const episodes = +$('t-episodes').value;
  const config = { episodes, lr: +$('t-lr').value, seed: +$('t-seed').value, shallow, hidden: shallow ? [24] : [64, 64] };
  worker = new Worker(new URL('./train-worker.js', import.meta.url), { type: 'module' });
  liveHist = { reward: [], window: [] };
  if (charts.live) {
    charts.live.data.datasets.forEach((d) => { d.data = []; });
    charts.live.options.scales.x.max = episodes;
    charts.live.update();
  }
  const t0 = performance.now();
  worker.onmessage = ({ data }) => {
    if (data.type !== 'progress') return;
    for (const ep of data.batch) {
      liveHist.window.push(ep.reward);
      if (liveHist.window.length > 20) liveHist.window.shift();
      const a = avg(liveHist.window);
      charts.live?.data.datasets[0].data.push({ x: ep.episode, y: ep.reward });
      charts.live?.data.datasets[1].data.push({ x: ep.episode, y: a });
    }
    charts.live?.update();
    const last = data.batch.at(-1);
    if (last) {
      $('t-bar').style.width = `${(100 * last.episode) / episodes}%`;
      $('t-status').textContent =
        `Episodio ${last.episode}/${episodes} · ε = ${last.epsilon.toFixed(3)} · promedio ${avg(liveHist.window).toFixed(1)} · ` +
        `${data.steps.toLocaleString('es-MX')} pasos · ${((performance.now() - t0) / 1000).toFixed(0)} s`;
    }
    if (data.model) {
      policies.browser = new Policy(data.model);
      preview.policy = policies.browser;
      $('preview-label').textContent = `Política greedy con los pesos del episodio ${last?.episode ?? '—'}`;
      $('t-use').disabled = false;
      ctrlSelect.querySelector('option[value="browser"]').disabled = false;
      if (ctrlSelect.value === 'browser') { demo.policy = policies.browser; netView.setNetwork(demo.policy.net); }
    }
    if (data.done) {
      setTrainState('done');
      $('t-status').textContent += ' · ¡terminado!';
    }
  };
  worker.postMessage({ type: 'start', config });
  setTrainState('running');
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
