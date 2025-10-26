import crypto from 'crypto';
import { fromSeed, random as simplexRandom, simplex } from '../../js/noisemaker/simplex.js';

const seed = Number(process.argv[2]);
const { data } = fromSeed(seed);
const r = simplexRandom(0.25, seed, 1);
const dt = 1e-3;
const d0 = (simplexRandom(dt, seed, 1) - simplexRandom(0.0, seed, 1)) / dt;
const d1 = (simplexRandom(1 + dt, seed, 1) - simplexRandom(1.0, seed, 1)) / dt;

const tensor = simplex([128, 128, 3], { seed, time: 0, speed: 1 });
const arr = tensor.read();
const buf = Buffer.from(new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength));
const hash = crypto.createHash('sha256').update(buf).digest('hex');

console.log(JSON.stringify({
  perm: Array.from(data.perm.slice(0, 10)),
  perm_grad: Array.from(data.perm_grad.slice(0, 10)),
  random: r,
  d0,
  d1,
  hash
}));
