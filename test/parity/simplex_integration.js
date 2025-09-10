import { fromSeed, random as simplexRandom } from '../../src/simplex.js';

const seed = Number(process.argv[2]);
const { data } = fromSeed(seed);
const r = simplexRandom(0.25, seed, 1);
const dt = 1e-3;
const d0 = (simplexRandom(dt, seed, 1) - simplexRandom(0.0, seed, 1)) / dt;
const d1 = (simplexRandom(1 + dt, seed, 1) - simplexRandom(1.0, seed, 1)) / dt;

console.log(JSON.stringify({
  perm: Array.from(data.perm.slice(0, 10)),
  perm_grad: Array.from(data.perm_grad.slice(0, 10)),
  random: r,
  d0,
  d1
}));
