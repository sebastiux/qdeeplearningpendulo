// Agente DQN: red en línea, red objetivo, memoria de repetición y política ε-greedy.
import { MLP, Adam, argmax } from './nn.js';
import { CartPole, DEG } from './physics.js';

export const DEEP_ACTIONS = Array.from({ length: 41 }, (_, i) => Math.round((-10 + 0.5 * i) * 10) / 10);
// s = [x, ẋ, θ, θ̇] → todas las entradas en ~[-1, 1]
export const STATE_DIM = 4;
export const INPUT_SCALE = [1 / 2.4, 0.5, 1 / (12 * DEG), 0.5];
export const STATE_LABELS = ['x', 'ẋ', 'θ', 'θ̇'];

export const DEFAULTS = {
  hidden: [64, 64],
  actions: DEEP_ACTIONS,
  episodes: 2000,
  lr: 0.0005,
  gamma: 0.99,
  batch: 64,
  buffer: 50000,
  warmup: 1000,
  targetSync: 500,
  trainEvery: 1,       // pasos de ambiente entre actualizaciones de la red
  epsStart: 1,
  epsEnd: 0.01,
  epsDecay: 0.996,
  evalEvery: 25,
};

export function scaleInput(s, scale = INPUT_SCALE) {
  return s.map((v, i) => v * scale[i]);
}

// Política greedy a partir de un modelo exportado (Python o navegador)
export class Policy {
  constructor(model) {
    this.net = model.network instanceof MLP ? model.network : MLP.fromJSON(model.network);
    this.actions = model.actions;
    this.scale = model.input_scale ?? INPUT_SCALE;
    this.meta = model;
  }

  evaluate(s) {
    const input = scaleInput(s, this.scale);
    const { q, acts } = this.net.forward(input);
    const a = argmax(q);
    return { a, force: this.actions[a], q, acts, input };
  }
}

// Arranques de evaluación (θ0, x0), los mismos que usa el entrenador de Python
const EVAL_STARTS = [-10, -6, -3, 3, 6, 10].map((d) => [d * DEG, 0]);
const LONG_STARTS = [[10 * DEG, 0], [-10 * DEG, 0], [2 * DEG, 1.5], [-2 * DEG, -1.5]];
const LONG_STEPS = 1500;

// Evaluación greedy: recompensa media en 300 pasos y arranques que aguantan 30 s
export function evaluatePolicy(policy) {
  const env = new CartPole();
  let score = 0;
  for (const [th0, x0] of EVAL_STARTS) {
    env.reset(th0, 0, x0);
    for (;;) {
      const { reward, fell, truncated } = env.step(policy.evaluate(env.state()).force);
      score += reward;
      if (fell || truncated) break;
    }
  }
  const long = new CartPole({ maxSteps: Infinity });
  let longOk = 0;
  for (const [th0, x0] of LONG_STARTS) {
    long.reset(th0, 0, x0);
    let fell = false;
    for (let i = 0; i < LONG_STEPS && !fell; i++) fell = long.step(policy.evaluate(long.state()).force).fell;
    if (!fell) longOk++;
  }
  return { score: score / EVAL_STARTS.length, longOk, longTotal: LONG_STARTS.length };
}

class ReplayBuffer {
  constructor(capacity, dim) {
    this.cap = capacity;
    this.dim = dim;
    this.s = new Float64Array(capacity * dim);  // estados ya normalizados
    this.s2 = new Float64Array(capacity * dim);
    this.a = new Int32Array(capacity);
    this.r = new Float64Array(capacity);
    this.d = new Float64Array(capacity);
    this.n = 0;
    this.i = 0;
  }

  push(s, a, r, s2, done) {
    const i = this.i, D = this.dim;
    for (let k = 0; k < D; k++) {
      this.s[i * D + k] = s[k] * INPUT_SCALE[k];
      this.s2[i * D + k] = s2[k] * INPUT_SCALE[k];
    }
    this.a[i] = a; this.r[i] = r; this.d[i] = done ? 1 : 0;
    this.i = (i + 1) % this.cap;
    this.n = Math.min(this.n + 1, this.cap);
  }
}

export class DQNTrainer {
  constructor(cfg, rng) {
    this.cfg = { ...DEFAULTS, ...cfg };
    this.rng = rng;
    const sizes = [STATE_DIM, ...this.cfg.hidden, this.cfg.actions.length];
    this.q = new MLP(sizes, rng);
    this.qTarget = this.q.clone();
    this.opt = new Adam(this.q, this.cfg.lr);
    this.mem = new ReplayBuffer(this.cfg.buffer, STATE_DIM);
    this.env = new CartPole({}, rng);
    this.eps = this.cfg.epsStart;
    this.steps = 0;
    this.episode = 0;
    this.best = null; // { net, episode, score, longOk }
    const B = this.cfg.batch;
    this.S = new Float64Array(B * STATE_DIM);
    this.S2 = new Float64Array(B * STATE_DIM);
    this.idx = new Int32Array(B);
  }

  act(s) {
    if (this.rng() < this.eps) return Math.floor(this.rng() * this.cfg.actions.length);
    return argmax(this.q.forward(scaleInput(s)).q);
  }

  learn() {
    const { batch: B, gamma, actions } = this.cfg;
    const nA = actions.length, D = STATE_DIM;
    const mem = this.mem, S = this.S, S2 = this.S2, idx = this.idx;
    for (let k = 0; k < B; k++) {
      const j = Math.floor(this.rng() * mem.n);
      idx[k] = j;
      for (let c = 0; c < D; c++) {
        S[k * D + c] = mem.s[j * D + c];
        S2[k * D + c] = mem.s2[j * D + c];
      }
    }
    const qNext = this.qTarget.forwardBatch(S2, B).at(-1);
    const acts = this.q.forwardBatch(S, B);
    const qAll = acts.at(-1);
    const grad = new Float64Array(B * nA);
    let loss = 0;
    for (let k = 0; k < B; k++) {
      const j = idx[k];
      let mx = -Infinity;
      for (let a = 0; a < nA; a++) mx = Math.max(mx, qNext[k * nA + a]);
      // Target Q = r + γ·max_a' Q(s',a';w⁻)
      const target = mem.r[j] + gamma * mx * (1 - mem.d[j]);
      const td = qAll[k * nA + mem.a[j]] - target;
      grad[k * nA + mem.a[j]] = (2 * td) / B;
      loss += td * td;
    }
    this.opt.step(this.q.backward(acts, grad, B));
    return loss / B;
  }

  // Ejecuta un episodio completo de entrenamiento y devuelve sus estadísticas
  runEpisode() {
    const { actions, warmup, targetSync, trainEvery, epsEnd, epsDecay, evalEvery } = this.cfg;
    const env = this.env;
    let s = env.reset();
    let total = 0, lossSum = 0, lossN = 0;
    for (;;) {
      const a = this.act(s);
      const { state: s2, reward, fell, truncated } = env.step(actions[a]);
      this.mem.push(s, a, reward, s2, fell);
      s = s2;
      total += reward;
      this.steps++;
      if (this.mem.n >= warmup && this.steps % trainEvery === 0) { lossSum += this.learn(); lossN++; }
      if (this.steps % targetSync === 0) this.qTarget.copyFrom(this.q);
      if (fell || truncated) break;
    }
    this.eps = Math.max(epsEnd, this.eps * epsDecay);
    this.episode++;
    const out = { episode: this.episode, reward: total, steps: env.t, epsilon: this.eps, loss: lossN ? lossSum / lossN : null };

    // Igual que en Python: se conserva el mejor modelo según la evaluación greedy
    if (this.episode % evalEvery === 0 || this.episode === this.cfg.episodes) {
      const ev = evaluatePolicy(new Policy({ network: this.q, actions, input_scale: INPUT_SCALE }));
      out.eval = ev;
      if (!this.best || ev.longOk > this.best.longOk || (ev.longOk === this.best.longOk && ev.score >= this.best.score)) {
        this.best = { net: this.q.clone(), episode: this.episode, ...ev };
        out.improved = true;
      }
    }
    return out;
  }
}
