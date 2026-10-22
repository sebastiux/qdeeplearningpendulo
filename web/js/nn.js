// Perceptrón multicapa (ReLU + salida lineal) con retropropagación y Adam.
// Es el mismo modelo de training/dqn_pendulo.py, reescrito para el navegador.

export class MLP {
  constructor(sizes, rng = Math.random) {
    this.sizes = sizes;
    this.W = []; // W[l] es una matriz (n_in × n_out) aplanada por filas
    this.b = [];
    for (let l = 0; l < sizes.length - 1; l++) {
      const nIn = sizes[l], nOut = sizes[l + 1];
      const std = Math.sqrt(2 / nIn); // inicialización He
      const W = new Float64Array(nIn * nOut);
      for (let i = 0; i < W.length; i++) W[i] = gaussian(rng) * std;
      this.W.push(W);
      this.b.push(new Float64Array(nOut));
    }
  }

  static fromJSON(net) {
    const mlp = new MLP(net.sizes);
    mlp.W = net.W.map((rows) => Float64Array.from(rows.flat()));
    mlp.b = net.b.map((row) => Float64Array.from(row));
    return mlp;
  }

  toJSON() {
    return {
      sizes: this.sizes,
      activation: 'relu',
      W: this.W.map((W, l) => {
        const nOut = this.sizes[l + 1];
        const rows = [];
        for (let i = 0; i < this.sizes[l]; i++) rows.push(Array.from(W.subarray(i * nOut, (i + 1) * nOut)));
        return rows;
      }),
      b: this.b.map((b) => Array.from(b)),
    };
  }

  clone() {
    const c = Object.create(MLP.prototype);
    c.sizes = [...this.sizes];
    c.W = this.W.map((w) => w.slice());
    c.b = this.b.map((b) => b.slice());
    return c;
  }

  copyFrom(other) {
    for (let l = 0; l < this.W.length; l++) {
      this.W[l].set(other.W[l]);
      this.b[l].set(other.b[l]);
    }
  }

  // Propagación hacia adelante de un lote X (n × sizes[0]); devuelve todas las activaciones.
  forwardBatch(X, n) {
    const acts = [X];
    let h = X;
    const last = this.W.length - 1;
    for (let l = 0; l <= last; l++) {
      const nIn = this.sizes[l], nOut = this.sizes[l + 1];
      const W = this.W[l], b = this.b[l];
      const out = new Float64Array(n * nOut);
      for (let r = 0; r < n; r++) {
        const o = r * nOut;
        for (let j = 0; j < nOut; j++) out[o + j] = b[j];
        for (let i = 0; i < nIn; i++) {
          const x = h[r * nIn + i];
          if (x === 0) continue;
          const wi = i * nOut;
          for (let j = 0; j < nOut; j++) out[o + j] += x * W[wi + j];
        }
        if (l !== last) for (let j = 0; j < nOut; j++) if (out[o + j] < 0) out[o + j] = 0;
      }
      acts.push(out);
      h = out;
    }
    return acts;
  }

  // Un solo estado: devuelve { q, acts } con las activaciones por capa (para visualizar).
  forward(x) {
    const acts = this.forwardBatch(Float64Array.from(x), 1);
    return { q: acts[acts.length - 1], acts };
  }

  // Retropropagación: grad = dL/d(salida) (n × nOut). Devuelve gradientes por capa.
  backward(acts, grad, n) {
    const L = this.W.length;
    const gW = new Array(L), gb = new Array(L);
    let delta = grad;
    for (let l = L - 1; l >= 0; l--) {
      const nIn = this.sizes[l], nOut = this.sizes[l + 1];
      const A = acts[l];
      const GW = new Float64Array(nIn * nOut), GB = new Float64Array(nOut);
      for (let r = 0; r < n; r++) {
        const d = r * nOut;
        for (let j = 0; j < nOut; j++) GB[j] += delta[d + j];
        for (let i = 0; i < nIn; i++) {
          const a = A[r * nIn + i];
          if (a === 0) continue;
          const wi = i * nOut;
          for (let j = 0; j < nOut; j++) GW[wi + j] += a * delta[d + j];
        }
      }
      gW[l] = GW;
      gb[l] = GB;
      if (l > 0) {
        const W = this.W[l];
        const prev = new Float64Array(n * nIn);
        for (let r = 0; r < n; r++) {
          for (let i = 0; i < nIn; i++) {
            if (A[r * nIn + i] <= 0) continue; // derivada de ReLU
            let s = 0;
            const wi = i * nOut, d = r * nOut;
            for (let j = 0; j < nOut; j++) s += W[wi + j] * delta[d + j];
            prev[r * nIn + i] = s;
          }
        }
        delta = prev;
      }
    }
    return { gW, gb };
  }
}

export class Adam {
  constructor(net, lr, { beta1 = 0.9, beta2 = 0.999, eps = 1e-8, clip = 10 } = {}) {
    Object.assign(this, { net, lr, beta1, beta2, eps, clip, t: 0 });
    this.mW = net.W.map((w) => new Float64Array(w.length));
    this.vW = net.W.map((w) => new Float64Array(w.length));
    this.mb = net.b.map((b) => new Float64Array(b.length));
    this.vb = net.b.map((b) => new Float64Array(b.length));
  }

  step({ gW, gb }) {
    let sq = 0;
    for (const g of [...gW, ...gb]) for (let i = 0; i < g.length; i++) sq += g[i] * g[i];
    const norm = Math.sqrt(sq);
    const scale = norm > this.clip ? this.clip / norm : 1;
    this.t++;
    const c1 = 1 - this.beta1 ** this.t, c2 = 1 - this.beta2 ** this.t;
    const upd = (P, G, M, V) => {
      for (let i = 0; i < P.length; i++) {
        const g = G[i] * scale;
        M[i] = this.beta1 * M[i] + (1 - this.beta1) * g;
        V[i] = this.beta2 * V[i] + (1 - this.beta2) * g * g;
        P[i] -= (this.lr * (M[i] / c1)) / (Math.sqrt(V[i] / c2) + this.eps);
      }
    };
    for (let l = 0; l < gW.length; l++) {
      upd(this.net.W[l], gW[l], this.mW[l], this.vW[l]);
      upd(this.net.b[l], gb[l], this.mb[l], this.vb[l]);
    }
  }
}

function gaussian(rng) {
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export function argmax(arr) {
  let best = 0;
  for (let i = 1; i < arr.length; i++) if (arr[i] > arr[best]) best = i;
  return best;
}
