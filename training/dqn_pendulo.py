"""
Deep Q-Learning (DQN) para el péndulo invertido sobre un carro.

Implementación desde cero con NumPy (forward, retropropagación y Adam
escritos a mano) siguiendo las especificaciones del curso
"Aprendizaje reforzado - Deep Q-learning" (Universidad Iberoamericana):

  * Estado        s = [x, ẋ, θ, θ̇]                 (continuo, R^4; el carro corre en un riel de ±2.4 m)
  * Acciones      F ∈ {-10.0, -9.5, ..., +9.5, +10.0} N  -> 41 acciones
  * Recompensa    r = +1 en la vertical (decae con θ², x² y el esfuerzo F²), r = -10 si cae o choca
  * Dinámica      ecuaciones no lineales acopladas + integración de Euler (ΔT = 0.02 s)
  * Red Q         4 -> 64 -> 64 -> 41  (dos capas ocultas, ReLU)
  * Optimización  Adam, α = 0.0005, pérdida MSE contra el Target de Bellman
  * Entrenamiento 2000 episodios, máximo 300 pasos por episodio

También entrena la versión "superficial" del Ejercicio 1
(4 -> 24 -> 3, acciones {-10, 0, +10} N, 500 episodios) para compararlas.

Uso:
    python training/dqn_pendulo.py            # entrena ambas y exporta a web/data
    python training/dqn_pendulo.py --only deep
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "web" / "data"


# ---------------------------------------------------------------------------
# 1. Ambiente: péndulo invertido sobre un carro
# ---------------------------------------------------------------------------
@dataclass
class CartPoleConfig:
    M: float = 1.0           # masa del carro [kg]
    m: float = 0.1           # masa del péndulo [kg]
    L: float = 0.5           # mitad de la longitud de la barra [m]
    g: float = 9.8           # gravedad [m/s^2]
    dt: float = 0.02         # paso de integración ΔT [s]
    theta_fail: float = math.radians(12.0)   # ángulo de caída
    x_limit: float = 2.4     # el carro choca con el tope del riel en |x| > 2.4 m
    max_steps: int = 300
    reward_alive: float = 1.0
    reward_fall: float = -10.0
    # "Vertical" = θ≈0: la recompensa de +1 decae como 1 − k·(θ/θ_caída)².
    # Con k = 0 se recupera la recompensa binaria del enunciado, pero el agente
    # aprende a sobrevivir inclinado ~11° acelerando el carro indefinidamente.
    reward_shaping: float = 1.0
    # Costo de esfuerzo −c·(F/F_max)²: premia las fuerzas suaves (control fino).
    reward_effort: float = 0.1
    force_max: float = 10.0
    # Costo de posición −k·(x/x_lím)²: mantiene el carro cerca del centro del riel.
    reward_position: float = 0.25
    theta0_range: float = math.radians(10.0) # θ0 ~ U(-10°, 10°) al entrenar
    omega0_range: float = 0.05
    x0_range: float = 1.0                     # x0 ~ U(-1, 1) m al entrenar


class CartPole:
    def __init__(self, cfg: CartPoleConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.reset()

    def reset(self, theta0: float | None = None, omega0: float | None = None, x0: float | None = None):
        c = self.cfg
        self.theta = self.rng.uniform(-c.theta0_range, c.theta0_range) if theta0 is None else theta0
        self.omega = self.rng.uniform(-c.omega0_range, c.omega0_range) if omega0 is None else omega0
        self.x = self.rng.uniform(-c.x0_range, c.x0_range) if x0 is None else x0
        self.v = 0.0
        self.t = 0
        return self.state()

    def state(self):
        return np.array([self.x, self.v, self.theta, self.omega], dtype=np.float64)

    def step(self, F: float):
        c = self.cfg
        th, om = self.theta, self.omega
        sin, cos = math.sin(th), math.cos(th)
        total = c.M + c.m

        # Ecuación de movimiento: aceleración angular
        temp = (F + c.m * c.L * om * om * sin) / total
        theta_acc = (c.g * sin - cos * temp) / (c.L * (4.0 / 3.0 - c.m * cos * cos / total))
        # Aceleración horizontal del carro (acoplada a θ̈)
        x_acc = (F + c.m * c.L * (om * om * sin - theta_acc * cos)) / total

        # Integración de Euler: primero velocidades, luego posiciones
        self.omega = om + theta_acc * c.dt
        self.theta = th + self.omega * c.dt
        self.v = self.v + x_acc * c.dt
        self.x = self.x + self.v * c.dt
        self.t += 1

        # Termina si el péndulo cae o si el carro choca con un extremo del riel
        fell = abs(self.theta) > c.theta_fail or abs(self.x) > c.x_limit
        truncated = self.t >= c.max_steps
        if fell:
            reward = c.reward_fall
        else:
            reward = (c.reward_alive * (1.0 - c.reward_shaping * (self.theta / c.theta_fail) ** 2)
                      - c.reward_effort * (F / c.force_max) ** 2
                      - c.reward_position * (self.x / c.x_limit) ** 2)
        return self.state(), reward, fell, truncated


# ---------------------------------------------------------------------------
# 2. Red neuronal feed-forward con retropropagación manual
# ---------------------------------------------------------------------------
class MLP:
    """Perceptrón multicapa: capas ocultas ReLU, salida lineal (valores Q)."""

    def __init__(self, sizes: list[int], rng: np.random.Generator):
        self.sizes = sizes
        self.W, self.b = [], []
        for n_in, n_out in zip(sizes[:-1], sizes[1:]):
            # Inicialización He (adecuada para ReLU)
            self.W.append(rng.normal(0.0, math.sqrt(2.0 / n_in), size=(n_in, n_out)))
            self.b.append(np.zeros(n_out))

    def forward(self, x: np.ndarray, keep: bool = False):
        acts = [x]
        h = x
        last = len(self.W) - 1
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            h = z if i == last else np.maximum(z, 0.0)
            acts.append(h)
        return (h, acts) if keep else h

    def backward(self, acts: list[np.ndarray], grad_out: np.ndarray):
        """Retropropagación: devuelve gradientes dL/dW y dL/db por capa."""
        gW = [None] * len(self.W)
        gb = [None] * len(self.b)
        delta = grad_out
        for i in reversed(range(len(self.W))):
            gW[i] = acts[i].T @ delta
            gb[i] = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ self.W[i].T) * (acts[i] > 0.0)  # derivada de ReLU
        return gW, gb

    def copy_from(self, other: "MLP"):
        self.W = [w.copy() for w in other.W]
        self.b = [b.copy() for b in other.b]

    def to_json(self):
        return {
            "sizes": self.sizes,
            "activation": "relu",
            "W": [np.round(w, 7).tolist() for w in self.W],
            "b": [np.round(b, 7).tolist() for b in self.b],
        }


class Adam:
    def __init__(self, net: MLP, lr: float, beta1=0.9, beta2=0.999, eps=1e-8, clip=10.0):
        self.net, self.lr, self.b1, self.b2, self.eps, self.clip = net, lr, beta1, beta2, eps, clip
        self.mW = [np.zeros_like(w) for w in net.W]
        self.vW = [np.zeros_like(w) for w in net.W]
        self.mb = [np.zeros_like(b) for b in net.b]
        self.vb = [np.zeros_like(b) for b in net.b]
        self.t = 0

    def step(self, gW, gb):
        # Recorte de la norma global del gradiente (estabilidad)
        norm = math.sqrt(sum(float((g * g).sum()) for g in gW + gb))
        scale = self.clip / norm if norm > self.clip else 1.0
        self.t += 1
        c1 = 1.0 - self.b1 ** self.t
        c2 = 1.0 - self.b2 ** self.t
        for i in range(len(gW)):
            for P, G, m, v in ((self.net.W, gW, self.mW, self.vW), (self.net.b, gb, self.mb, self.vb)):
                g = G[i] * scale
                m[i] = self.b1 * m[i] + (1 - self.b1) * g
                v[i] = self.b2 * v[i] + (1 - self.b2) * g * g
                P[i] -= self.lr * (m[i] / c1) / (np.sqrt(v[i] / c2) + self.eps)


# ---------------------------------------------------------------------------
# 3. Agente DQN
# ---------------------------------------------------------------------------
@dataclass
class DQNConfig:
    name: str = "deep"
    hidden: list[int] = field(default_factory=lambda: [64, 64])
    actions: list[float] = field(default_factory=lambda: [round(-10 + 0.5 * i, 1) for i in range(41)])
    episodes: int = 2000
    lr: float = 0.0005
    gamma: float = 0.99
    batch: int = 64
    buffer: int = 50_000
    warmup: int = 1_000
    target_sync: int = 500         # pasos entre copias a la red objetivo
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay: float = 0.996       # ε ← ε · decay al final de cada episodio
    eval_every: int = 25
    seed: int = 7


# Normalización de la entrada: x en [-2.4, 2.4] m, ẋ ~ [-2, 2] m/s,
# θ en [-12°, 12°] y θ̇ ~ [-2, 2] rad/s → todas ~[-1, 1]
STATE_DIM = 4
INPUT_SCALE = np.array([1.0 / 2.4, 0.5, 1.0 / math.radians(12.0), 0.5])


class ReplayBuffer:
    def __init__(self, capacity: int, rng: np.random.Generator):
        self.s = np.zeros((capacity, STATE_DIM))
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity)
        self.s2 = np.zeros((capacity, STATE_DIM))
        self.d = np.zeros(capacity)
        self.cap, self.n, self.i, self.rng = capacity, 0, 0, rng

    def push(self, s, a, r, s2, done):
        self.s[self.i], self.a[self.i], self.r[self.i], self.s2[self.i], self.d[self.i] = s, a, r, s2, done
        self.i = (self.i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, k):
        idx = self.rng.integers(0, self.n, size=k)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx]


class DQNAgent:
    def __init__(self, cfg: DQNConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        sizes = [STATE_DIM, *cfg.hidden, len(cfg.actions)]
        self.q = MLP(sizes, rng)          # red Q en línea  Q(s, a; w)
        self.q_target = MLP(sizes, rng)   # red objetivo    Q(s', a'; w⁻)
        self.q_target.copy_from(self.q)
        self.opt = Adam(self.q, cfg.lr)
        self.memory = ReplayBuffer(cfg.buffer, rng)
        self.eps = cfg.eps_start
        self.steps = 0

    def act(self, s, greedy=False):
        if not greedy and self.rng.random() < self.eps:
            return int(self.rng.integers(len(self.cfg.actions)))
        q = self.q.forward((s * INPUT_SCALE)[None, :])[0]
        return int(np.argmax(q))

    def learn(self):
        s, a, r, s2, d = self.memory.sample(self.cfg.batch)
        s, s2 = s * INPUT_SCALE, s2 * INPUT_SCALE
        # Target Q = r + γ · max_a' Q(s', a'; w⁻)   (sin bootstrap si cayó)
        q_next = self.q_target.forward(s2).max(axis=1)
        target = r + self.cfg.gamma * q_next * (1.0 - d)
        q_all, acts = self.q.forward(s, keep=True)
        rows = np.arange(len(a))
        td = q_all[rows, a] - target
        # Loss(w) = E[(Target Q − Q(s,a;w))²]  ->  dL/dQ(s,a) = 2·(Q − Target)/N
        grad = np.zeros_like(q_all)
        grad[rows, a] = 2.0 * td / len(a)
        gW, gb = self.q.backward(acts, grad)
        self.opt.step(gW, gb)
        return float((td ** 2).mean())


# ---------------------------------------------------------------------------
# 4. Evaluación y entrenamiento
# ---------------------------------------------------------------------------
# (θ0, x0): ángulos iniciales con el carro centrado y dos arranques cerca de los topes
EVAL_STARTS = [(math.radians(d), 0.0) for d in (-10, -6, -3, 3, 6, 10)]
LONG_STARTS = [(math.radians(10), 0.0), (math.radians(-10), 0.0), (math.radians(2), 1.5), (math.radians(-2), -1.5)]
LONG_STEPS = 1500  # 30 s: verifica que el carro no termine chocando con el riel


def rollout(agent: DQNAgent, env: CartPole, theta0: float, steps: int, x0: float = 0.0):
    s = env.reset(theta0=theta0, omega0=0.0, x0=x0)
    total, log = 0.0, {"t": [], "theta": [], "omega": [], "force": [], "x": []}
    for _ in range(steps):
        a = agent.act(s, greedy=True)
        F = agent.cfg.actions[a]
        log["t"].append(round(env.t * env.cfg.dt, 3))
        log["theta"].append(round(math.degrees(env.theta), 4))
        log["omega"].append(round(env.omega, 5))
        log["x"].append(round(env.x, 4))
        log["force"].append(F)
        s, r, fell, _ = env.step(F)
        total += r
        if fell:
            break
    return total, log


def evaluate(agent: DQNAgent, env: CartPole, long_env: CartPole):
    """Recompensa greedy promedio, error angular medio |θ| (2ª mitad) y cuántos
    arranques sobreviven 30 s sin caer ni chocar."""
    scores, errors = [], []
    for th0, x0 in EVAL_STARTS:
        total, log = rollout(agent, env, th0, env.cfg.max_steps, x0)
        scores.append(total)
        tail = log["theta"][len(log["theta"]) // 2:]
        survived = len(log["theta"]) == env.cfg.max_steps
        errors.append(float(np.mean(np.abs(tail))) if survived else 12.0)
    long_ok = sum(len(rollout(agent, long_env, th0, LONG_STEPS, x0)[1]["t"]) == LONG_STEPS
                  for th0, x0 in LONG_STARTS)
    return float(np.mean(scores)), float(np.mean(errors)), long_ok


def train(cfg: DQNConfig, phys: CartPoleConfig):
    rng = np.random.default_rng(cfg.seed)
    env = CartPole(phys, rng)
    eval_env = CartPole(phys, np.random.default_rng(0))
    long_env = CartPole(CartPoleConfig(**{**phys.__dict__, "max_steps": 10**9}), np.random.default_rng(0))
    agent = DQNAgent(cfg, rng)

    history = {"episode": [], "reward": [], "avg": [], "steps": [], "epsilon": [], "loss": [], "eval": []}
    window = deque(maxlen=20)
    best = (-1, -1e9, 1e9, None, -1)  # (long_ok, score, err, red, episodio)
    t0 = time.time()

    for ep in range(1, cfg.episodes + 1):
        s = env.reset()
        total, losses = 0.0, []
        while True:
            a = agent.act(s)
            s2, r, fell, trunc = env.step(cfg.actions[a])
            agent.memory.push(s, a, r, s2, float(fell))
            s = s2
            total += r
            agent.steps += 1
            if agent.memory.n >= cfg.warmup:
                losses.append(agent.learn())
            if agent.steps % cfg.target_sync == 0:
                agent.q_target.copy_from(agent.q)
            if fell or trunc:
                break
        agent.eps = max(cfg.eps_end, agent.eps * cfg.eps_decay)

        window.append(total)
        history["episode"].append(ep)
        history["reward"].append(round(total, 2))
        history["avg"].append(round(float(np.mean(window)), 2))
        history["steps"].append(env.t)
        history["epsilon"].append(round(agent.eps, 4))
        history["loss"].append(round(float(np.mean(losses)), 5) if losses else None)

        if ep % cfg.eval_every == 0:
            score, err, long_ok = evaluate(agent, eval_env, long_env)
            history["eval"].append([ep, round(score, 2), round(err, 3), long_ok])
            # Mejor modelo: más arranques que aguantan 30 s, luego mayor recompensa
            if (long_ok, score) >= (best[0], best[1]):
                best = (long_ok, score, err, agent.q.to_json(), ep)
            print(f"[{cfg.name}] ep {ep:4d} | avg20 {np.mean(window):7.1f} | eval {score:6.1f} "
                  f"| |θ| {err:5.2f}° | 30s {long_ok}/{len(LONG_STARTS)} "
                  f"| ε {agent.eps:.3f} | {time.time() - t0:5.0f}s", flush=True)

    # Respuesta greedy final desde θ0 = 10° (5 s de simulación, como en las diapositivas)
    best_agent = DQNAgent(cfg, np.random.default_rng(0))
    best_agent.q.W = [np.array(w) for w in best[3]["W"]]
    best_agent.q.b = [np.array(b) for b in best[3]["b"]]
    _, response = rollout(best_agent, long_env, math.radians(10.0), 250)

    model = {
        "name": cfg.name,
        "network": best[3],
        "actions": cfg.actions,
        "input_scale": INPUT_SCALE.tolist(),
        "best_episode": best[4],
        "best_eval": best[1],
        "best_abs_theta": best[2],
        "best_long_ok": best[0],
        "hyperparameters": {
            "hidden": cfg.hidden, "episodes": cfg.episodes, "lr": cfg.lr, "gamma": cfg.gamma,
            "batch": cfg.batch, "buffer": cfg.buffer, "warmup": cfg.warmup,
            "target_sync": cfg.target_sync, "eps_start": cfg.eps_start, "eps_end": cfg.eps_end,
            "eps_decay": cfg.eps_decay, "seed": cfg.seed,
        },
        "physics": phys.__dict__,
        "train_seconds": round(time.time() - t0, 1),
    }
    return model, history, response


CONFIGS = {
    "deep": DQNConfig(),
    "shallow": DQNConfig(name="shallow", hidden=[24], actions=[-10.0, 0.0, 10.0], episodes=500,
                         lr=0.001, eps_decay=0.99, seed=3),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=list(CONFIGS), default=None)
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--effort", type=float, default=None, help="costo de esfuerzo c")
    p.add_argument("--position", type=float, default=None, help="costo de posición k")
    p.add_argument("--out", type=Path, default=OUT_DIR)
    args = p.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    phys = CartPoleConfig()
    if args.effort is not None:
        phys.reward_effort = args.effort
    if args.position is not None:
        phys.reward_position = args.position
    for key, cfg in CONFIGS.items():
        if args.only and key != args.only:
            continue
        if args.episodes:
            cfg.episodes = args.episodes
        if args.seed is not None:
            cfg.seed = args.seed
        model, history, response = train(cfg, phys)
        (out / f"model_{key}.json").write_text(json.dumps(model, separators=(",", ":")))
        (out / f"history_{key}.json").write_text(json.dumps(history, separators=(",", ":")))
        (out / f"response_{key}.json").write_text(json.dumps(response, separators=(",", ":")))
        print(f"[{key}] mejor modelo: episodio {model['best_episode']} (eval {model['best_eval']:.1f}, "
              f"30 s {model['best_long_ok']}/{len(LONG_STARTS)})")


if __name__ == "__main__":
    main()
