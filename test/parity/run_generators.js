import { setSeed } from '../../src/rng.js';
import { setSeed as setValueSeed } from '../../src/value.js';
import { basic, multires } from '../../src/generators.js';

const [,, name, seedStr] = process.argv;
const seed = parseInt(seedStr, 10);
setSeed(seed);
setValueSeed(seed);
let tensor;
if (name === 'basic') {
  tensor = await basic(2, [128, 128, 3]);
} else if (name === 'multires') {
  setSeed(seed);
  setValueSeed(seed);
  tensor = await multires(2, [128, 128, 3], {
    octaves: 2,
    hueRotation: 0,
    postEffects: [],
    finalEffects: [],
  });
} else {
  throw new Error(`Unknown generator ${name}`);
}
const arr = await tensor.read();
const buf = Buffer.from(new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength));
console.log(buf.toString('base64'));
