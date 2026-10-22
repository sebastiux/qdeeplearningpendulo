// Dibujo en canvas: carro-péndulo en su riel, valores Q, atribución de la
// decisión, mapa de la política, trazas en vivo y la red neuronal.
import { DEG, PHYSICS } from './physics.js';

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const MONO = '"JetBrains Mono", monospace';

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

// Escala divergente para fuerzas: rojo (−10 N, izquierda) → azul oscuro (0) → turquesa (+10 N, derecha)
export function forceRGB(f) {
  const t = Math.max(-1, Math.min(1, f / 10));
  const mid = [27, 35, 80], pos = [46, 230, 197], neg = [255, 93, 115];
  const target = t >= 0 ? pos : neg;
  const k = Math.abs(t);
  return mid.map((v, i) => Math.round(v + (target[i] - v) * k));
}
export const forceColor = (f, alpha = 1) => {
  const c = forceRGB(f);
  return `rgba(${c[0]},${c[1]},${c[2]},${alpha})`;
};

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

const RAIL = PHYSICS.xLimit;   // topes en ±2.4 m
const MARGIN = 0.45;           // espacio visible más allá de los topes

export class CartPoleRenderer {
  constructor(canvas, { poleLength = 1, compact = false } = {}) {
    this.canvas = canvas;
    this.poleLength = poleLength; // 2L en metros
    this.compact = compact;
    this.camX = 0;
    this.trail = [];
  }

  resetCamera() {
    this.trail = [];
  }

  // decision = { q, actions, a } para dibujar la franja de valores Q bajo el riel
  draw({ x, theta, force, maxForce = 10, fallen = false, crashed = false, decision = null }) {
    const { ctx, w, h } = fitCanvas(this.canvas);
    const showStrip = !this.compact && decision;
    const groundY = h * (this.compact ? 0.8 : showStrip ? 0.7 : 0.76);
    // Escala: todo el riel si cabe; en pantallas angostas, al menos 4 m visibles y la cámara sigue al carro
    const fullPpm = w / (2 * (RAIL + MARGIN));
    const ppm = Math.min((h * (this.compact ? 0.52 : 0.44)) / this.poleLength, w >= 640 ? fullPpm : Math.max(fullPpm, w / 4));
    const halfView = w / ppm / 2;
    const maxCam = Math.max(0, RAIL + MARGIN - halfView);
    const target = Math.max(-maxCam, Math.min(maxCam, x));
    this.camX += (target - this.camX) * 0.12;
    const sx = (wx) => w / 2 + (wx - this.camX) * ppm;

    // Fondo
    const bg = ctx.createLinearGradient(0, 0, 0, h);
    bg.addColorStop(0, css('--sim-top'));
    bg.addColorStop(1, css('--sim-bottom'));
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, w, h);

    // Cuadrícula (cada 0.25 m)
    ctx.strokeStyle = css('--sim-grid');
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let m = -RAIL - MARGIN; m <= RAIL + MARGIN + 1e-6; m += 0.25) { const px = sx(m); ctx.moveTo(px, 0); ctx.lineTo(px, groundY); }
    for (let gy = groundY; gy > 0; gy -= 0.25 * ppm) { ctx.moveTo(0, gy); ctx.lineTo(w, gy); }
    ctx.stroke();

    // Zonas de peligro junto a los topes (último 0.4 m)
    for (const side of [-1, 1]) {
      const x0 = sx(side * RAIL), x1 = sx(side * (RAIL - 0.4));
      const g = ctx.createLinearGradient(x0, 0, x1, 0);
      g.addColorStop(0, 'rgba(255,93,115,0.20)');
      g.addColorStop(1, 'rgba(255,93,115,0)');
      ctx.fillStyle = g;
      ctx.fillRect(Math.min(x0, x1), 0, Math.abs(x1 - x0), groundY);
    }

    // Riel finito con topes
    const railL = sx(-RAIL - 0.12), railR = sx(RAIL + 0.12);
    ctx.fillStyle = css('--sim-rail');
    roundRect(ctx, railL, groundY - 1, railR - railL, 5, 2.5);
    ctx.fill();
    for (const side of [-1, 1]) {
      const px = sx(side * (RAIL + 0.06));
      const hit = crashed && Math.sign(x) === side;
      ctx.fillStyle = hit ? css('--danger') : css('--sim-stop');
      roundRect(ctx, px - 5, groundY - 0.3 * ppm, 10, 0.3 * ppm + 6, 3);
      ctx.fill();
      if (hit) {
        ctx.save();
        ctx.shadowColor = css('--danger');
        ctx.shadowBlur = 20;
        ctx.fill();
        ctx.restore();
      }
    }
    // Soportes del riel
    ctx.fillStyle = css('--sim-rail');
    for (const side of [-1, 1]) ctx.fillRect(sx(side * (RAIL + 0.06)) - 2, groundY + 4, 4, 10);

    // Marcas de distancia
    ctx.font = `${this.compact ? 9 : 11}px ${MONO}`;
    ctx.textAlign = 'center';
    for (let m = -2; m <= 2; m += 0.5) {
      const px = sx(m);
      const major = Number.isInteger(m);
      ctx.fillStyle = css('--sim-tick');
      ctx.fillRect(px - 0.5, groundY + 6, 1, major ? 9 : 5);
      if (major && !this.compact) ctx.fillText(`${m} m`, px, groundY + 28);
    }
    if (!this.compact) {
      ctx.fillStyle = css('--danger');
      ctx.font = `10px ${MONO}`;
      ctx.fillText('tope −2.4 m', sx(-RAIL), groundY + 44);
      ctx.fillText('tope +2.4 m', sx(RAIL), groundY + 44);
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
    cg.addColorStop(0, crashed ? '#ff8a9a' : css('--cart-top'));
    cg.addColorStop(1, crashed ? '#c43a50' : css('--cart-bottom'));
    ctx.fillStyle = cg;
    roundRect(ctx, cx - cartW / 2, cy - cartH / 2, cartW, cartH, Math.min(8, cartH / 3));
    ctx.fill();
    ctx.strokeStyle = 'rgba(255,255,255,0.14)';
    ctx.lineWidth = 1;
    ctx.stroke();
    if (!this.compact && cartH > 18) {
      ctx.fillStyle = 'rgba(255,255,255,0.85)';
      ctx.font = `600 ${Math.max(10, cartH * 0.34)}px Inter, sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('M', cx, cy + 1);
      ctx.textBaseline = 'alphabetic';
    }

    // Flecha de fuerza (color según dirección, como en la escala de fuerzas)
    if (Math.abs(force) > 1e-9) {
      const len = Math.max(10, (Math.abs(force) / maxForce) * cartW * 1.1);
      const ay = cy;
      const dir = Math.sign(force);
      const ax = dir > 0 ? cx - cartW / 2 - 4 : cx + cartW / 2 + 4; // punta tocando el carro
      const fx = ax - dir * len;
      const fc = forceColor(force);
      ctx.strokeStyle = fc;
      ctx.fillStyle = fc;
      ctx.lineWidth = 3.5;
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.moveTo(fx, ay);
      ctx.lineTo(ax - dir * 6, ay);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(ax - dir * 10, ay - 6);
      ctx.lineTo(ax - dir * 10, ay + 6);
      ctx.closePath();
      ctx.fill();
      ctx.lineCap = 'butt';
      if (!this.compact) {
        ctx.font = `600 11px ${MONO}`;
        ctx.textAlign = 'center';
        ctx.fillText(`${force > 0 ? '+' : ''}${force.toFixed(1)} N`, (fx + ax) / 2, ay - 12);
      }
    }

    // Barra del péndulo con brillo
    ctx.save();
    ctx.shadowColor = color;
    ctx.shadowBlur = 18;
    ctx.strokeStyle = color;
    ctx.lineWidth = Math.max(3, ppm * 0.028);
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(cx, pivotY);
    ctx.lineTo(tipX, tipY);
    ctx.stroke();
    const bobR = Math.max(6, ppm * 0.055);
    const bg2 = ctx.createRadialGradient(tipX - bobR / 3, tipY - bobR / 3, 1, tipX, tipY, bobR);
    bg2.addColorStop(0, '#ffffff');
    bg2.addColorStop(0.35, color);
    bg2.addColorStop(1, color);
    ctx.fillStyle = bg2;
    ctx.beginPath();
    ctx.arc(tipX, tipY, bobR, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();

    ctx.fillStyle = css('--sim-pivot');
    ctx.beginPath();
    ctx.arc(cx, pivotY, 4, 0, Math.PI * 2);
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
      ctx.font = `12px ${MONO}`;
      ctx.textAlign = 'center';
      const la = -Math.PI / 2 + theta / 2;
      ctx.fillText(`θ ${(theta / DEG).toFixed(1)}°`, cx + Math.cos(la) * (r + 34), pivotY + Math.sin(la) * (r + 14));
    }

    if (showStrip) this.drawStrip(ctx, w, h, groundY, decision);
  }

  // Franja con Q(s, a) de cada fuerza: la decisión de la red, dentro de la escena
  drawStrip(ctx, w, h, groundY, { q, actions, a }) {
    const n = q.length;
    const top = groundY + 58, sh = Math.min(26, h - top - 22);
    if (sh < 10) return;
    const width = Math.min(w - 40, 560), left = (w - width) / 2;
    let lo = Infinity, hi = -Infinity;
    for (const v of q) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    const range = Math.max(1e-9, hi - lo);
    const cw = width / n;
    for (let i = 0; i < n; i++) {
      const t = (q[i] - lo) / range;                 // 0 = peor acción, 1 = mejor
      const c = forceRGB(actions[i]);
      ctx.fillStyle = `rgba(${c[0]},${c[1]},${c[2]},${0.12 + 0.88 * t ** 3})`;
      ctx.fillRect(left + i * cw + 0.5, top, cw - 1, sh);
    }
    const ax = left + (a + 0.5) * cw;
    ctx.strokeStyle = css('--text');
    ctx.lineWidth = 2;
    ctx.strokeRect(left + a * cw, top - 2, cw, sh + 4);
    ctx.fillStyle = css('--text');
    ctx.beginPath();
    ctx.moveTo(ax, top - 4);
    ctx.lineTo(ax - 5, top - 11);
    ctx.lineTo(ax + 5, top - 11);
    ctx.closePath();
    ctx.fill();
    ctx.font = `10px ${MONO}`;
    ctx.fillStyle = css('--muted');
    ctx.textAlign = 'left';
    ctx.fillText(`${actions[0]} N`, left, top + sh + 13);
    ctx.textAlign = 'right';
    ctx.fillText(`+${actions[n - 1]} N`, left + width, top + sh + 13);
    ctx.textAlign = 'center';
    ctx.fillText('Q(s, a) de cada fuerza · más brillante = mejor', w / 2, top + sh + 13);
  }
}

// Valores Q(s, a) sobre el eje de fuerzas, con la acción elegida y la de no empujar (F = 0)
export function drawQChart(canvas, q, actions, chosen) {
  const { ctx, w, h } = fitCanvas(canvas);
  ctx.clearRect(0, 0, w, h);
  const n = q.length;
  const padL = 44, padR = 8, padT = 26, padB = 22;
  const gw = w - padL - padR, gh = h - padT - padB;
  let lo = Infinity, hi = -Infinity;
  for (const v of q) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  const span = Math.max(1e-6, hi - lo);
  const y0 = lo - span * 0.15, y1 = hi + span * 0.1;
  const Y = (v) => padT + gh - ((v - y0) / (y1 - y0)) * gh;
  const bw = gw / n;
  const zero = actions.indexOf(0);

  // Eje Y (valor Q)
  ctx.font = `10px ${MONO}`;
  ctx.fillStyle = css('--muted');
  ctx.strokeStyle = css('--border');
  ctx.textAlign = 'right';
  for (let k = 0; k <= 3; k++) {
    const v = y0 + ((y1 - y0) * k) / 3;
    ctx.beginPath();
    ctx.moveTo(padL, Y(v));
    ctx.lineTo(w - padR, Y(v));
    ctx.stroke();
    ctx.fillText(v.toFixed(span < 1 ? 2 : 1), padL - 6, Y(v) + 3);
  }

  for (let i = 0; i < n; i++) {
    const x = padL + i * bw;
    const t = (q[i] - lo) / span;
    const c = forceRGB(actions[i]);
    ctx.fillStyle = i === chosen ? css('--text') : `rgba(${c[0]},${c[1]},${c[2]},${0.25 + 0.75 * t})`;
    const top = Y(q[i]);
    roundRect(ctx, x + Math.min(1, bw * 0.15), top, Math.max(1, bw - Math.min(2, bw * 0.3)), padT + gh - top, Math.min(3, bw / 3));
    ctx.fill();
  }
  // F = 0 (no empujar) como referencia
  if (zero >= 0 && zero !== chosen) {
    ctx.strokeStyle = css('--muted');
    ctx.setLineDash([3, 3]);
    ctx.strokeRect(padL + zero * bw + 0.5, Y(q[zero]), bw - 1, padT + gh - Y(q[zero]));
    ctx.setLineDash([]);
  }
  // Etiqueta de la elegida
  const cxp = padL + (chosen + 0.5) * bw;
  ctx.fillStyle = css('--text');
  ctx.font = `600 11px ${MONO}`;
  ctx.textAlign = cxp < w * 0.25 ? 'left' : cxp > w * 0.75 ? 'right' : 'center';
  ctx.fillText(`F* = ${actions[chosen] > 0 ? '+' : ''}${actions[chosen]} N`, cxp, padT - 10);

  // Eje X (fuerza)
  ctx.font = `10px ${MONO}`;
  ctx.fillStyle = css('--muted');
  ctx.textAlign = 'center';
  const ticks = n > 10 ? [0, 10, 20, 30, 40] : actions.map((_, i) => i);
  for (const i of ticks) ctx.fillText(`${actions[i] > 0 ? '+' : ''}${actions[i]}`, padL + (i + 0.5) * bw, h - 6);
}

// Atribución de la decisión por variable de estado (barras divergentes)
export function drawAttribution(canvas, items) {
  const { ctx, w, h } = fitCanvas(canvas);
  ctx.clearRect(0, 0, w, h);
  const rowH = h / items.length;
  const labelW = 58, valW = 64;
  const mid = labelW + (w - labelW - valW) / 2, half = (w - labelW - valW) / 2 - 4;
  const mx = Math.max(1e-9, ...items.map((it) => Math.abs(it.value)));
  ctx.strokeStyle = css('--border');
  ctx.beginPath();
  ctx.moveTo(mid, 0);
  ctx.lineTo(mid, h);
  ctx.stroke();
  items.forEach((it, i) => {
    const y = i * rowH + rowH / 2;
    ctx.fillStyle = css('--text');
    ctx.font = `600 12px ${MONO}`;
    ctx.textAlign = 'left';
    ctx.fillText(it.label, 4, y + 4);
    const len = (Math.abs(it.value) / mx) * half;
    ctx.fillStyle = it.value >= 0 ? css('--accent') : css('--danger');
    roundRect(ctx, it.value >= 0 ? mid : mid - len, y - rowH * 0.22, Math.max(1, len), rowH * 0.44, 3);
    ctx.fill();
    ctx.fillStyle = css('--muted');
    ctx.font = `10px ${MONO}`;
    ctx.textAlign = 'right';
    ctx.fillText(it.detail, w - 2, y + 4);
  });
}

// Traza deslizante en el tiempo
export function drawTrace(canvas, values, { min, max, color, zero = true, step = false, label = '', bands = [] }) {
  const { ctx, w, h } = fitCanvas(canvas);
  ctx.clearRect(0, 0, w, h);
  const y = (v) => h - 6 - ((v - min) / (max - min)) * (h - 12);
  for (const b of bands) {
    ctx.strokeStyle = 'rgba(255,93,115,0.5)';
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(0, y(b));
    ctx.lineTo(w, y(b));
    ctx.stroke();
  }
  if (zero) {
    ctx.strokeStyle = css('--sim-grid-strong');
    ctx.setLineDash([3, 4]);
    ctx.beginPath();
    ctx.moveTo(0, y(0));
    ctx.lineTo(w, y(0));
    ctx.stroke();
  }
  ctx.setLineDash([]);
  if (values.length > 1) {
    const dx = w / 299;
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
    ctx.font = `10px ${MONO}`;
    ctx.textAlign = 'left';
    ctx.fillText(label, 6, 12);
  }
}

// Mapa de la política sobre (θ, θ̇) para un x, ẋ fijos, con el estado actual encima
export class PolicyMap {
  constructor(canvas, { nx = 64, ny = 40 } = {}) {
    this.canvas = canvas;
    this.nx = nx;
    this.ny = ny;
    this.img = null;
    this.key = null;
  }

  compute(policy, x, v) {
    const { nx, ny } = this;
    const off = this.img ?? document.createElement('canvas');
    off.width = nx;
    off.height = ny;
    const o = off.getContext('2d');
    const data = o.createImageData(nx, ny);
    for (let j = 0; j < ny; j++) {
      const om = 2 - (4 * (j + 0.5)) / ny;
      for (let i = 0; i < nx; i++) {
        const th = (-12 + (24 * (i + 0.5)) / nx) * DEG;
        const c = forceRGB(policy.evaluate([x, v, th, om]).force);
        const k = (j * nx + i) * 4;
        data.data[k] = c[0]; data.data[k + 1] = c[1]; data.data[k + 2] = c[2]; data.data[k + 3] = 255;
      }
    }
    o.putImageData(data, 0, 0);
    this.img = off;
  }

  // Recalcula sólo si cambió el modelo o el corte (x, ẋ) lo suficiente
  update(policy, x, v) {
    const key = `${Math.round(x * 10)}:${Math.round(v * 10)}`;
    if (this.policy !== policy || this.key !== key) {
      this.compute(policy, x, v);
      this.policy = policy;
      this.key = key;
    }
  }

  draw(trail = []) {
    const { ctx, w, h } = fitCanvas(this.canvas);
    const padL = 34, padB = 22, padR = 14, padT = 8;
    const gw = w - padL - padR, gh = h - padB - padT;
    ctx.clearRect(0, 0, w, h);
    if (this.img) {
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(this.img, padL, padT, gw, gh);
    }
    const px = (th) => padL + ((th / DEG + 12) / 24) * gw;
    const py = (om) => padT + ((2 - om) / 4) * gh;
    ctx.strokeStyle = 'rgba(255,255,255,0.25)';
    ctx.setLineDash([3, 4]);
    ctx.beginPath();
    ctx.moveTo(px(0), padT); ctx.lineTo(px(0), padT + gh);
    ctx.moveTo(padL, py(0)); ctx.lineTo(padL + gw, py(0));
    ctx.stroke();
    ctx.setLineDash([]);
    if (trail.length) {
      ctx.lineWidth = 2;
      for (let i = 1; i < trail.length; i++) {
        ctx.strokeStyle = `rgba(255,255,255,${(i / trail.length) * 0.8})`;
        ctx.beginPath();
        ctx.moveTo(px(trail[i - 1][0]), py(trail[i - 1][1]));
        ctx.lineTo(px(trail[i][0]), py(trail[i][1]));
        ctx.stroke();
      }
      const [th, om] = trail.at(-1);
      const cx = Math.max(padL, Math.min(padL + gw, px(th))), cy = Math.max(padT, Math.min(padT + gh, py(om)));
      ctx.fillStyle = '#fff';
      ctx.strokeStyle = '#0b1020';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(cx, cy, 6, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }
    ctx.fillStyle = css('--muted');
    ctx.font = `10px ${MONO}`;
    ctx.textAlign = 'center';
    for (const d of [-12, -6, 0, 6, 12]) ctx.fillText(`${d}°`, px(d * DEG), h - 6);
    ctx.textAlign = 'right';
    for (const v of [2, 1, 0, -1, -2]) ctx.fillText(`${v}`, padL - 6, py(v) + 4);
    ctx.save();
    ctx.translate(10, padT + gh / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.fillText('θ̇ (rad/s)', 0, 0);
    ctx.restore();
  }
}

// Diagrama de la red. En vivo dibuja la contribución de cada neurona a la acción elegida:
// grosor ∝ |activación × peso|, turquesa = la favorece, rojo = la frena.
export class NetworkView {
  constructor(canvas, net, { labels = [] } = {}) {
    this.canvas = canvas;
    this.labels = labels;
    this.setNetwork(net);
  }

  setNetwork(net) {
    this.net = net;
    this.edgeCache = null;
  }

  layout(w, h) {
    const sizes = this.net.sizes;
    const padX = 40, padY = 14, padB = 18;
    return sizes.map((n, l) => {
      const x = padX + (l * (w - 2 * padX)) / (sizes.length - 1);
      const spacing = Math.min(l === 0 ? 40 : 22, (h - padY - padB) / Math.max(1, n - 1));
      const top = (h - padB) / 2 - (spacing * (n - 1)) / 2;
      return Array.from({ length: n }, (_, i) => [x, top + i * spacing]);
    });
  }

  draw(acts, chosen) {
    const { ctx, w, h } = fitCanvas(this.canvas);
    const pos = this.layout(w, h);
    const key = `${w}x${h}`;
    const L = this.net.W.length;
    if (!this.edgeCache || this.edgeCache.key !== key) {
      // Conexiones estáticas tenues (opacidad ∝ |w|) en un canvas fuera de pantalla
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
            if (a < 0.25) continue;
            o.strokeStyle = v > 0 ? `rgba(46,230,197,${a * 0.1})` : `rgba(255,93,115,${a * 0.1})`;
            o.lineWidth = 0.5;
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

    // Camino de la decisión: contribuciones hacia la salida elegida, capa por capa
    if (acts && chosen != null) {
      let targets = new Map([[chosen, 1]]); // neurona destino → importancia
      for (let l = L - 1; l >= 0; l--) {
        const W = this.net.W[l], nIn = this.net.sizes[l], nOut = this.net.sizes[l + 1];
        const a = acts[l];
        const contrib = [];
        for (const [j, imp] of targets) {
          for (let i = 0; i < nIn; i++) {
            const c = a[i] * W[i * nOut + j];
            if (c !== 0) contrib.push([i, j, c * imp]);
          }
        }
        contrib.sort((p, q) => Math.abs(q[2]) - Math.abs(p[2]));
        const top = contrib.slice(0, l === L - 1 ? 24 : 40);
        const mx = Math.max(1e-9, ...top.map((c) => Math.abs(c[2])));
        for (const [i, j, c] of top) {
          const t = Math.abs(c) / mx;
          ctx.strokeStyle = c > 0 ? `rgba(46,230,197,${0.15 + 0.75 * t})` : `rgba(255,93,115,${0.15 + 0.75 * t})`;
          ctx.lineWidth = 0.6 + 2.6 * t;
          ctx.beginPath();
          ctx.moveTo(...pos[l][i]);
          ctx.lineTo(...pos[l + 1][j]);
          ctx.stroke();
        }
        const next = new Map();
        for (const [i, , c] of top.slice(0, 10)) next.set(i, (next.get(i) ?? 0) + Math.abs(c) / mx);
        targets = next;
      }
    }

    pos.forEach((layer, l) => {
      const a = acts?.[l];
      const isOut = l === pos.length - 1;
      let mx = -Infinity, mn = Infinity, amx = 1e-6;
      if (a) for (const v of a) { mx = Math.max(mx, v); mn = Math.min(mn, v); amx = Math.max(amx, Math.abs(v)); }
      const spacing = layer.length > 1 ? layer[1][1] - layer[0][1] : 20;
      const r = Math.max(1.6, Math.min(l === 0 ? 7 : 5, spacing * 0.42));
      layer.forEach(([x, y], i) => {
        let t = 0;
        if (a) t = isOut ? (a[i] - mn) / Math.max(1e-9, mx - mn) : Math.abs(a[i]) / amx;
        const isChosen = isOut && i === chosen;
        if (l === 0 && a) ctx.fillStyle = a[i] >= 0 ? `rgba(46,230,197,${0.3 + 0.7 * t})` : `rgba(255,93,115,${0.3 + 0.7 * t})`;
        else ctx.fillStyle = isChosen ? css('--text') : isOut ? `rgba(124,140,255,${0.1 + 0.9 * t ** 3})` : `rgba(124,140,255,${0.12 + 0.88 * t})`;
        ctx.beginPath();
        ctx.arc(x, y, isChosen ? r + 2.5 : r, 0, Math.PI * 2);
        ctx.fill();
        if (l === 0 && this.labels[i]) {
          ctx.fillStyle = css('--text');
          ctx.font = `600 11px ${MONO}`;
          ctx.textAlign = 'right';
          ctx.fillText(this.labels[i], x - r - 5, y + 4);
        }
      });
    });
    ctx.fillStyle = css('--muted');
    ctx.font = `10px ${MONO}`;
    ctx.textAlign = 'center';
    const names = ['entrada', ...this.net.sizes.slice(1, -1).map((n) => `oculta ${n}`), `salida ${this.net.sizes.at(-1)}`];
    pos.forEach((layer, l) => ctx.fillText(names[l], layer[0][0], h - 3));
  }
}
