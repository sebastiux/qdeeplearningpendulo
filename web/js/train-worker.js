// Web Worker: entrena un DQN sin bloquear la interfaz y reporta el progreso.
import { DQNTrainer, DEEP_ACTIONS, INPUT_SCALE } from './dqn.js';
import { seededRandom } from './physics.js';

let trainer = null;
let running = false;
let total = 0;

self.onmessage = ({ data }) => {
  if (data.type === 'start') {
    const { episodes, lr, hidden, seed, shallow, trainEvery } = data.config;
    const actions = shallow ? [-10, 0, 10] : DEEP_ACTIONS;
    // ε llega a 0.01 al 60 % de los episodios (con 2000 episodios ≈ 0.996, como en Python)
    const epsDecay = 0.01 ** (1 / (0.6 * episodes));
    trainer = new DQNTrainer({ lr, hidden, actions, epsDecay, episodes, trainEvery }, seededRandom(seed));
    total = episodes;
    running = true;
    loop();
  } else if (data.type === 'pause') {
    running = false;
  } else if (data.type === 'resume' && trainer) {
    running = true;
    loop();
  }
};

const pack = (net, extra = {}) => ({ name: 'navegador', network: net.toJSON(), actions: trainer.cfg.actions, input_scale: INPUT_SCALE, ...extra });

function loop() {
  if (!running || !trainer) return;
  const batch = [];
  let improved = false;
  const t0 = performance.now();
  // Bloques de ~50 ms para poder recibir mensajes (pausa)
  while (performance.now() - t0 < 50 && trainer.episode < total) {
    const ep = trainer.runEpisode();
    improved ||= !!ep.improved;
    batch.push(ep);
  }
  const done = trainer.episode >= total;
  const b = trainer.best;
  self.postMessage({
    type: 'progress',
    batch,
    steps: trainer.steps,
    done,
    // El mejor modelo encontrado hasta ahora (no el último, que puede estar en una mala racha)
    best: improved || done ? pack(b.net, { best_episode: b.episode, best_eval: b.score, best_long_ok: b.longOk, long_total: b.longTotal }) : null,
  });
  if (done) running = false;
  else setTimeout(loop, 0);
}
