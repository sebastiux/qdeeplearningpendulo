// Prueba de humo: el modelo exportado por Python debe equilibrar el péndulo
// usando la física y la red reescritas en JavaScript (las mismas del sitio).
import { readFileSync } from 'node:fs';
import { CartPole, DEG } from '../web/js/physics.js';
import { Policy } from '../web/js/dqn.js';

const load = (f) => JSON.parse(readFileSync(new URL(`../web/data/${f}`, import.meta.url)));
let failed = false;

for (const name of ['deep', 'shallow']) {
  const policy = new Policy(load(`model_${name}.json`));
  const env = new CartPole({ maxSteps: Infinity });
  for (const deg of [-10, -5, 5, 10]) {
    env.reset(deg * DEG, 0);
    let fell = false;
    for (let i = 0; i < 1500 && !fell; i++) fell = env.step(policy.evaluate(env.state()).force).fell;
    const ok = !fell && Math.abs(env.theta) < 3 * DEG;
    console.log(`${ok ? 'ok  ' : 'FAIL'} ${name.padEnd(7)} θ0=${String(deg).padStart(3)}° → θ(30 s)=${(env.theta / DEG).toFixed(2)}°`);
    if (!ok) failed = true;
  }
}

process.exit(failed ? 1 : 0);
