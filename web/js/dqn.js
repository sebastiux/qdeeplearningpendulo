// Agente DQN: red en línea, red objetivo, memoria de repetición y política ε-greedy.
import { MLP, Adam, argmax } from './nn.js';
import { CartPole, DEG } from './physics.js';

export const DEEP_ACTIONS = Array.from({ length: 41 }, (_, i) => Math.round((-10 + 0.5 * i) * 10) / 10);
export const INPUT_SCALE = [1 / (12 * DEG), 0.5];

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
  epsStart: 1,
  epsEnd: 0.01,
  epsDecay: 0.996,
  rewardShaping: 1,
};

export function scaleInput(s, scale = INPUT_SCALE) {
  return [s[0] * scale[0], s[1] * scale[1]];
}

// Política greedy a partir de un modelo exportado (Python o navegador)
export class Policy {
  constructor(model) {
    this.net = model.network instanceof MLP ? model.network : MLP.fromJSON(model.network);
    this.actions = model.actions;
    this.scale = model.input_scale ?? INPUT_SCALE;
  }

  evaluate(s) {
    const { q, acts } = this.net.forward(scaleInput(s, this.scale));
    const a = argmax(q);
    return { a, force: this.actions[a], q, acts };
  }
}

class ReplayBuffer {
  constructor(capacity, rng) {
    this.cap = capacity;
    this.s = new Float64Array(capacity * 2);
    this.s2 = new Float64Array(capacity * 2);
    this.a = new Int32Array(capacity);
    this.r = new Float64Array(capacity);
    this.d = new Float64Array(capacity);
    this.n = 0;
    this.i = 0;
    this.rng = rng;
  }

  push(s, a, r, s2, done) {
    const i = this.i;
    this.s[2 * i] = s[0]; this.s[2 * i + 1] = s[1];
    this.s2[2 * i] = s2[0]; this.s2[2 * i + 1] = s2[1];
    this.a[i] = a; this.r[i] = r; this.d[i] = done ? 1 : 0;
    this.i = (i + 1) % this.cap;
    this.n = Math.min(this.n + 1, this.cap);
  }
}

export class DQNTrainer {
  constructor(cfg, rng) {
    this.cfg = { ...DEFAULTS, ...cfg };
    this.rng = rng;
    const sizes = [2, ...this.cfg.hidden, this.cfg.actions.length];
    this.q = new MLP(sizes, rng);
    this.qTarget = this.q.clone();
    this.opt = new Adam(this.q, this.cfg.lr);
    this.mem = new ReplayBuffer(this.cfg.buffer, rng);
    this.env = new CartPole({ rewardShaping: this.cfg.rewardShaping }, rng);
    this.eps = this.cfg.epsStart;
    this.steps = 0;
    this.episode = 0;
  }

  act(s) {
    if (this.rng() < this.eps) return Math.floor(this.rng() * this.cfg.actions.length);
    return argmax(this.q.forward(scaleInput(s)).q);
  }

  learn() {
    const { batch: B, gamma, actions } = this.cfg;
    const nA = actions.length;
    const mem = this.mem;
    const S = new Float64Array(B * 2), S2 = new Float64Array(B * 2);
    const idx = new Int32Array(B);
    for (let k = 0; k < B; k++) {
      const j = Math.floor(this.rng() * mem.n);
      idx[k] = j;
      S[2 * k] = mem.s[2 * j] * INPUT_SCALE[0];
      S[2 * k + 1] = mem.s[2 * j + 1] * INPUT_SCALE[1];
      S2[2 * k] = mem.s2[2 * j] * INPUT_SCALE[0];
      S2[2 * k + 1] = mem.s2[2 * j + 1] * INPUT_SCALE[1];
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
    const { actions, warmup, targetSync, epsEnd, epsDecay } = this.cfg;
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
      if (this.mem.n >= warmup) { lossSum += this.learn(); lossN++; }
      if (this.steps % targetSync === 0) this.qTarget.copyFrom(this.q);
      if (fell || truncated) break;
    }
    this.eps = Math.max(epsEnd, this.eps * epsDecay);
    this.episode++;
    return { episode: this.episode, reward: total, steps: env.t, epsilon: this.eps, loss: lossN ? lossSum / lossN : null };
  }
}
