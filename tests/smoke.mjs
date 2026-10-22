// Prueba de humo: el modelo exportado por Python debe equilibrar el péndulo sin
// chocar con los topes del riel, usando la física y la red reescritas en
// JavaScript (las mismas del sitio).
import { readFileSync } from 'node:fs';
import { CartPole, DEG } from '../web/js/physics.js';
import { Policy } from '../web/js/dqn.js';

const load = (f) => JSON.parse(readFileSync(new URL(`../web/data/${f}`, import.meta.url)));
let failed = false;

for (const name of ['deep', 'shallow']) {
  const policy = new Policy(load(`model_${name}.json`));
  const env = new CartPole({ maxSteps: Infinity });
  for (const [deg, x0] of [[-10, 0], [10, 0], [5, -1.5], [-5, 1.5]]) {
    env.reset(deg * DEG, 0, x0);
    let fell = false, xMax = 0;
    for (let i = 0; i < 1500 && !fell; i++) {
      fell = env.step(policy.evaluate(env.state()).force).fell;
      xMax = Math.max(xMax, Math.abs(env.x));
    }
    const ok = !fell && Math.abs(env.theta) < 3 * DEG;
    console.log(`${ok ? 'ok  ' : 'FAIL'} ${name.padEnd(7)} θ0=${String(deg).padStart(3)}° x0=${String(x0).padStart(4)} m → ` +
      `θ(30 s)=${(env.theta / DEG).toFixed(2)}°, x(30 s)=${env.x.toFixed(2)} m, |x|máx=${xMax.toFixed(2)} m${env.crashed ? ' (chocó)' : ''}`);
    if (!ok) failed = true;
  }
}

process.exit(failed ? 1 : 0);
