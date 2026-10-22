// Péndulo invertido sobre un carro: misma dinámica e integración de Euler
// que training/dqn_pendulo.py, para que los pesos exportados funcionen igual.

export const DEG = Math.PI / 180;

export const PHYSICS = {
  M: 1.0,               // masa del carro [kg]
  m: 0.1,               // masa del péndulo [kg]
  L: 0.5,               // mitad de la longitud de la barra [m]
  g: 9.8,               // gravedad [m/s²]
  dt: 0.02,             // ΔT [s]
  thetaFail: 12 * DEG,  // ángulo de caída
  maxSteps: 300,
  rewardAlive: 1,
  rewardFall: -10,
  rewardShaping: 1,     // r = 1 − k·(θ/θ_caída)² mientras sigue de pie
  rewardEffort: 0.1,    // costo de esfuerzo −c·(F/F_max)²
  forceMax: 10,
  theta0Range: 10 * DEG,
  omega0Range: 0.05,
};

export class CartPole {
  constructor(cfg = {}, rng = Math.random) {
    this.cfg = { ...PHYSICS, ...cfg };
    this.rng = rng;
    this.reset();
  }

  reset(theta0 = null, omega0 = null) {
    const c = this.cfg;
    this.theta = theta0 ?? (this.rng() * 2 - 1) * c.theta0Range;
    this.omega = omega0 ?? (this.rng() * 2 - 1) * c.omega0Range;
    this.x = 0;
    this.v = 0;
    this.t = 0;
    this.lastForce = 0;
    this.thetaAcc = 0;
    return this.state();
  }

  state() {
    return [this.theta, this.omega];
  }

  step(F) {
    const c = this.cfg;
    const th = this.theta, om = this.omega;
    const sin = Math.sin(th), cos = Math.cos(th);
    const total = c.M + c.m;

    // θ̈ = [g·sinθ − cosθ·(F + m·L·θ̇²·sinθ)/(M+m)] / [L·(4/3 − m·cos²θ/(M+m))]
    const temp = (F + c.m * c.L * om * om * sin) / total;
    const thetaAcc = (c.g * sin - cos * temp) / (c.L * (4 / 3 - (c.m * cos * cos) / total));
    // ẍ = [F + m·L·(θ̇²·sinθ − θ̈·cosθ)] / (M+m)
    const xAcc = (F + c.m * c.L * (om * om * sin - thetaAcc * cos)) / total;

    // Euler: velocidades primero, después posiciones
    this.omega = om + thetaAcc * c.dt;
    this.theta = th + this.omega * c.dt;
    this.v += xAcc * c.dt;
    this.x += this.v * c.dt;
    this.t += 1;
    this.lastForce = F;
    this.thetaAcc = thetaAcc;

    const fell = Math.abs(this.theta) > c.thetaFail;
    const truncated = this.t >= c.maxSteps;
    const reward = fell
      ? c.rewardFall
      : c.rewardAlive * (1 - c.rewardShaping * (this.theta / c.thetaFail) ** 2) - c.rewardEffort * (F / c.forceMax) ** 2;
    return { state: this.state(), reward, fell, truncated };
  }
}

// Generador pseudoaleatorio reproducible (mulberry32)
export function seededRandom(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
