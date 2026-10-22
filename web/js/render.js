// Dibujo en canvas: carro-péndulo, valores Q, trazas en vivo y la red neuronal.
import { DEG } from './physics.js';

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

export function fitCanvas(canvas) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const { width, height } = canvas.getBoundingClientRect();
  const w = Math.max(1, Math.round(width * dpr)), h = Math.max(1, Math.round(height * dpr));
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w: width, h: height };
}

function lerpColor(a, b, t) {
  const pa = a.match(/\w\w/g).map((x) => parseInt(x, 16));
  const pb = b.match(/\w\w/g).map((x) => parseInt(x, 16));
  const c = pa.map((v, i) => Math.round(v + (pb[i] - v) * t));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

export function stabilityColor(theta) {
  const t = Math.min(1, Math.abs(theta) / (12 * DEG));
  return t < 0.5 ? lerpColor('2ee6c5', 'ffc857', t * 2) : lerpColor('ffc857', 'ff5d73', (t - 0.5) * 2);
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

export class CartPoleRenderer {
  constructor(canvas, { poleLength = 1, compact = false } = {}) {
    this.canvas = canvas;
    this.poleLength = poleLength; // 2L en metros
    this.compact = compact;
    this.camX = 0;
    this.trail = [];
  }

  resetCamera(x = 0) {
    this.camX = x;
    this.trail = [];
  }

  draw({ x, theta, force, maxForce = 10, fallen = false }) {
    const { ctx, w, h } = fitCanvas(this.canvas);
    const groundY = h * (this.compact ? 0.78 : 0.76);
    const ppm = Math.min(h * (this.compact ? 0.5 : 0.44), w * 0.42) / this.poleLength; // píxeles por metro
    this.camX += (x - this.camX) * 0.08;
    const sx = (wx) => w / 2 + (wx - this.camX) * ppm;

    // Fondo
    const bg = ctx.createLinearGradient(0, 0, 0, h);
    bg.addColorStop(0, css('--sim-top'));
    bg.addColorStop(1, css('--sim-bottom'));
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, w, h);

    // Cuadrícula sutil que se desplaza con la cámara
    ctx.strokeStyle = css('--sim-grid');
    ctx.lineWidth = 1;
    const step = 0.25 * ppm;
    const off = ((-this.camX * ppm) % step + step) % step;
    ctx.beginPath();
    for (let gx = off + ((w / 2) % step); gx < w; gx += step) { ctx.moveTo(gx, 0); ctx.lineTo(gx, groundY); }
    for (let gy = groundY; gy > 0; gy -= step) { ctx.moveTo(0, gy); ctx.lineTo(w, gy); }
    ctx.stroke();

    // Riel y marcas de distancia
    ctx.fillStyle = css('--sim-rail');
    ctx.fillRect(0, groundY, w, 3);
    ctx.font = `${this.compact ? 10 : 11}px "JetBrains Mono", monospace`;
    ctx.textAlign = 'center';
    const span = w / ppm / 2 + 1;
    for (let m = Math.floor((this.camX - span) * 2) / 2; m <= this.camX + span; m += 0.5) {
      const px = sx(m);
      const major = Math.abs(m - Math.round(m)) < 1e-6;
      ctx.fillStyle = css('--sim-tick');
      ctx.fillRect(px - 0.5, groundY + 4, 1, major ? 10 : 5);
      if (major && !this.compact) ctx.fillText(`${m.toFixed(0)} m`, px, groundY + 28);
    }

    const cartW = 0.5 * ppm, cartH = 0.22 * ppm, wheelR = 0.055 * ppm;
    const cx = sx(x);
    const cy = groundY - wheelR * 2 - cartH / 2 + 1;
    const pivotY = cy - cartH / 2;
    const tipX = cx + Math.sin(theta) * this.poleLength * ppm;
    const tipY = pivotY - Math.cos(theta) * this.poleLength * ppm;
    const color = fallen ? css('--danger') : stabilityColor(theta);

    // Cono de seguridad ±12°
    const lim = 12 * DEG, R = this.poleLength * ppm * 1.08;
    ctx.beginPath();
    ctx.moveTo(cx, pivotY);
    ctx.arc(cx, pivotY, R, -Math.PI / 2 - lim, -Math.PI / 2 + lim);
    ctx.closePath();
    const cone = ctx.createRadialGradient(cx, pivotY, 0, cx, pivotY, R);
    cone.addColorStop(0, 'rgba(46,230,197,0.00)');
    cone.addColorStop(1, 'rgba(46,230,197,0.10)');
    ctx.fillStyle = cone;
    ctx.fill();
    ctx.setLineDash([4, 6]);
    ctx.strokeStyle = 'rgba(46,230,197,0.25)';
    ctx.beginPath();
    ctx.moveTo(cx, pivotY);
    ctx.lineTo(cx, pivotY - R);
    ctx.stroke();
    ctx.setLineDash([]);

    // Estela de la punta del péndulo
    this.trail.push([x + Math.sin(theta) * this.poleLength, Math.cos(theta)]);
    if (this.trail.length > 40) this.trail.shift();
    for (let i = 1; i < this.trail.length; i++) {
      const [ax, ay] = this.trail[i - 1], [bx, by] = this.trail[i];
      ctx.strokeStyle = `rgba(124,140,255,${(i / this.trail.length) * 0.35})`;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(sx(ax), pivotY - ay * this.poleLength * ppm);
      ctx.lineTo(sx(bx), pivotY - by * this.poleLength * ppm);
      ctx.stroke();
    }

    // Sombra
    ctx.fillStyle = 'rgba(0,0,0,0.28)';
    ctx.beginPath();
    ctx.ellipse(cx, groundY + 2, cartW * 0.6, 5, 0, 0, Math.PI * 2);
    ctx.fill();

    // Ruedas (giran con el desplazamiento)
    for (const dx of [-cartW * 0.3, cartW * 0.3]) {
      const wx = cx + dx, wy = groundY - wheelR;
      ctx.fillStyle = css('--sim-wheel');
      ctx.beginPath();
      ctx.arc(wx, wy, wheelR, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = css('--sim-wheel-spoke');
      ctx.lineWidth = 1.5;
      const rot = (x * ppm) / wheelR;
      ctx.beginPath();
      for (let k = 0; k < 3; k++) {
        const a = rot + (k * Math.PI) / 3;
        ctx.moveTo(wx - Math.cos(a) * wheelR * 0.75, wy - Math.sin(a) * wheelR * 0.75);
        ctx.lineTo(wx + Math.cos(a) * wheelR * 0.75, wy + Math.sin(a) * wheelR * 0.75);
      }
      ctx.stroke();
    }

    // Carro
    const cg = ctx.createLinearGradient(0, cy - cartH / 2, 0, cy + cartH / 2);
    cg.addColorStop(0, css('--cart-top'));
    cg.addColorStop(1, css('--cart-bottom'));
    ctx.fillStyle = cg;
    roundRect(ctx, cx - cartW / 2, cy - cartH / 2, cartW, cartH, 8);
    ctx.fill();
    ctx.strokeStyle = 'rgba(255,255,255,0.14)';
    ctx.lineWidth = 1;
    ctx.stroke();
    if (!this.compact) {
      ctx.fillStyle = 'rgba(255,255,255,0.85)';
      ctx.font = `600 ${Math.max(10, cartH * 0.34)}px Inter, sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('M', cx, cy + 1);
      ctx.textBaseline = 'alphabetic';
    }

    // Flecha de fuerza
    if (Math.abs(force) > 1e-9) {
      const len = (force / maxForce) * cartW * 0.9;
      const ay = cy;
      const ax = force > 0 ? cx - cartW / 2 - 6 : cx + cartW / 2 + 6;
      const fx = force > 0 ? ax - Math.abs(len) : ax + Math.abs(len);
      ctx.strokeStyle = css('--force');
      ctx.fillStyle = css('--force');
      ctx.lineWidth = 3;
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.moveTo(fx, ay);
      ctx.lineTo(ax, ay);
      ctx.stroke();
      const dir = Math.sign(force);
      ctx.beginPath();
      ctx.moveTo(ax + dir * 2, ay);
      ctx.lineTo(ax - dir * 8, ay - 6);
      ctx.lineTo(ax - dir * 8, ay + 6);
      ctx.closePath();
      ctx.fill();
      ctx.lineCap = 'butt';
      if (!this.compact) {
        ctx.font = '11px "JetBrains Mono", monospace';
        ctx.textAlign = 'center';
        ctx.fillText(`${force > 0 ? '+' : ''}${force.toFixed(1)} N`, (fx + ax) / 2, ay - 12);
      }
    }

    // Barra del péndulo con brillo
    ctx.save();
    ctx.shadowColor = color;
    ctx.shadowBlur = 18;
    ctx.strokeStyle = color;
    ctx.lineWidth = Math.max(4, ppm * 0.028);
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(cx, pivotY);
    ctx.lineTo(tipX, tipY);
    ctx.stroke();
    // Masa m
    const bobR = Math.max(8, ppm * 0.055);
    const bg2 = ctx.createRadialGradient(tipX - bobR / 3, tipY - bobR / 3, 1, tipX, tipY, bobR);
    bg2.addColorStop(0, '#ffffff');
    bg2.addColorStop(0.35, color);
    bg2.addColorStop(1, color);
    ctx.fillStyle = bg2;
    ctx.beginPath();
    ctx.arc(tipX, tipY, bobR, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();

    // Pivote
    ctx.fillStyle = css('--sim-pivot');
    ctx.beginPath();
    ctx.arc(cx, pivotY, 5, 0, Math.PI * 2);
    ctx.fill();

    // Arco del ángulo
    if (!this.compact && Math.abs(theta) > 1.5 * DEG) {
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      const r = this.poleLength * ppm * 0.32;
      ctx.arc(cx, pivotY, r, -Math.PI / 2, -Math.PI / 2 + theta, theta < 0);
      ctx.stroke();
      ctx.fillStyle = color;
      ctx.font = '12px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      const la = -Math.PI / 2 + theta / 2;
      ctx.fillText(`θ ${(theta / DEG).toFixed(1)}°`, cx + Math.cos(la) * (r + 34), pivotY + Math.sin(la) * (r + 14));
    }
  }
}

// Barras de valores Q(s, a) para cada acción; resalta la acción elegida.
export function drawQBars(canvas, q, actions, chosen) {
  const { ctx, w, h } = fitCanvas(canvas);
  ctx.clearRect(0, 0, w, h);
  const n = q.length;
  let lo = Infinity, hi = -Infinity;
  for (const v of q) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  const range = Math.max(1e-6, hi - lo);
  const pad = 22, gap = n > 10 ? 1.5 : 8;
  const bw = (w - gap * (n - 1)) / n;
  for (let i = 0; i < n; i++) {
    const t = (q[i] - lo) / range;
    const bh = 6 + t * (h - pad - 10);
    const bx = i * (bw + gap);
    const by = h - pad - bh;
    ctx.fillStyle = i === chosen ? css('--accent') : `rgba(124,140,255,${0.18 + 0.5 * t})`;
    roundRect(ctx, bx, by, bw, bh, Math.min(3, bw / 2));
    ctx.fill();
  }
  ctx.fillStyle = css('--muted');
  ctx.font = '10px "JetBrains Mono", monospace';
  ctx.textAlign = 'center';
  const labels = n > 10 ? [0, 10, 20, 30, 40] : actions.map((_, i) => i);
  for (const i of labels) ctx.fillText(`${actions[i] > 0 ? '+' : ''}${actions[i]}`, i * (bw + gap) + bw / 2, h - 6);
}

// Traza deslizante (θ o F en el tiempo)
export function drawTrace(canvas, values, { min, max, color, zero = true, step = false, label = '' }) {
  const { ctx, w, h } = fitCanvas(canvas);
  ctx.clearRect(0, 0, w, h);
  const y = (v) => h - 6 - ((v - min) / (max - min)) * (h - 12);
  if (zero) {
    ctx.strokeStyle = css('--sim-grid-strong');
    ctx.setLineDash([3, 4]);
    ctx.beginPath();
    ctx.moveTo(0, y(0));
    ctx.lineTo(w, y(0));
    ctx.stroke();
    ctx.setLineDash([]);
  }
  if (values.length > 1) {
    const dx = w / (values.length - 1);
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    values.forEach((v, i) => {
      const px = i * dx, py = y(Math.max(min, Math.min(max, v)));
      if (i === 0) ctx.moveTo(px, py);
      else if (step) { ctx.lineTo(px, y(Math.max(min, Math.min(max, values[i - 1])))); ctx.lineTo(px, py); }
      else ctx.lineTo(px, py);
    });
    ctx.stroke();
  }
  if (label) {
    ctx.fillStyle = css('--muted');
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    ctx.fillText(label, 6, 12);
  }
}

// Diagrama de la red con las activaciones del estado actual
export class NetworkView {
  constructor(canvas, net) {
    this.canvas = canvas;
    this.setNetwork(net);
  }

  setNetwork(net) {
    this.net = net;
    this.edgeCache = null;
  }

  layout(w, h) {
    const sizes = this.net.sizes;
    const padX = 34, padY = 14;
    return sizes.map((n, l) => {
      const x = padX + (l * (w - 2 * padX)) / (sizes.length - 1);
      const spacing = Math.min(22, (h - 2 * padY) / Math.max(1, n - 1));
      const top = h / 2 - (spacing * (n - 1)) / 2;
      return Array.from({ length: n }, (_, i) => [x, top + i * spacing]);
    });
  }

  draw(acts, chosen) {
    const { ctx, w, h } = fitCanvas(this.canvas);
    const pos = this.layout(w, h);
    const key = `${w}x${h}`;
    if (!this.edgeCache || this.edgeCache.key !== key) {
      // Conexiones estáticas (opacidad ∝ |w|) en un canvas fuera de pantalla
      const off = document.createElement('canvas');
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      off.width = w * dpr;
      off.height = h * dpr;
      const o = off.getContext('2d');
      o.scale(dpr, dpr);
      this.net.W.forEach((W, l) => {
        const nIn = this.net.sizes[l], nOut = this.net.sizes[l + 1];
        let mx = 0;
        for (const v of W) mx = Math.max(mx, Math.abs(v));
        for (let i = 0; i < nIn; i++) {
          for (let j = 0; j < nOut; j++) {
            const v = W[i * nOut + j];
            const a = Math.abs(v) / mx;
            if (a < 0.18) continue;
            o.strokeStyle = v > 0 ? `rgba(46,230,197,${a * 0.22})` : `rgba(255,93,115,${a * 0.22})`;
            o.lineWidth = 0.6;
            o.beginPath();
            o.moveTo(...pos[l][i]);
            o.lineTo(...pos[l + 1][j]);
            o.stroke();
          }
        }
      });
      this.edgeCache = { key, off };
    }
    ctx.clearRect(0, 0, w, h);
    ctx.drawImage(this.edgeCache.off, 0, 0, w, h);

    pos.forEach((layer, l) => {
      const a = acts?.[l];
      let mx = 1e-6, mn = Infinity;
      if (a) for (const v of a) { mx = Math.max(mx, Math.abs(v)); mn = Math.min(mn, v); }
      const spacing = layer.length > 1 ? layer[1][1] - layer[0][1] : 20;
      const r = Math.max(1.6, Math.min(5, spacing * 0.42));
      layer.forEach(([x, y], i) => {
        let t = a ? Math.abs(a[i]) / mx : 0;
        if (l === pos.length - 1 && a) t = (a[i] - mn) / Math.max(1e-6, mx - mn);
        const isChosen = l === pos.length - 1 && i === chosen;
        ctx.fillStyle = isChosen ? css('--accent') : `rgba(124,140,255,${0.15 + 0.85 * t})`;
        ctx.beginPath();
        ctx.arc(x, y, isChosen ? r + 2 : r, 0, Math.PI * 2);
        ctx.fill();
      });
    });
    ctx.fillStyle = css('--muted');
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    const names = ['entrada', ...this.net.sizes.slice(1, -1).map((n) => `oculta ${n}`), `salida ${this.net.sizes.at(-1)}`];
    pos.forEach((layer, l) => ctx.fillText(names[l], layer[0][0], h - 2));
  }
}
