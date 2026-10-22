// Web Worker: entrena un DQN sin bloquear la interfaz y reporta el progreso.
import { DQNTrainer, DEEP_ACTIONS, INPUT_SCALE } from './dqn.js';
import { seededRandom } from './physics.js';

let trainer = null;
let running = false;
let total = 0;

self.onmessage = ({ data }) => {
  if (data.type === 'start') {
    const { episodes, lr, hidden, seed, shallow } = data.config;
    const actions = shallow ? [-10, 0, 10] : DEEP_ACTIONS;
    trainer = new DQNTrainer(
      { lr, hidden, actions, epsDecay: shallow ? 0.99 : 0.996 },
      seededRandom(seed),
    );
    total = episodes;
    running = true;
    loop();
  } else if (data.type === 'pause') {
    running = false;
  } else if (data.type === 'resume' && trainer) {
    running = true;
    loop();
  } else if (data.type === 'stop') {
    running = false;
    trainer = null;
  }
};

function snapshot() {
  return {
    name: 'navegador',
    network: trainer.q.toJSON(),
    actions: trainer.cfg.actions,
    input_scale: INPUT_SCALE,
  };
}

function loop() {
  if (!running || !trainer) return;
  const batch = [];
  const t0 = performance.now();
  // Trabajar en bloques de ~40 ms para poder recibir mensajes (pausa/stop)
  while (performance.now() - t0 < 40 && trainer.episode < total) batch.push(trainer.runEpisode());
  const done = trainer.episode >= total;
  const withModel = done || trainer.episode % 10 < batch.length;
  self.postMessage({ type: 'progress', batch, steps: trainer.steps, model: withModel ? snapshot() : null, done });
  if (done) running = false;
  else setTimeout(loop, 0);
}
