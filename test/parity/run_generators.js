import { setSeed, resetCallCount, getCallCount } from '../../src/rng.js';
import { setSeed as setValueSeed } from '../../src/value.js';
import { basic, multires } from '../../src/generators.js';

const [,, name, seedStr, optionsB64] = process.argv;
const seed = parseInt(seedStr, 10);
const options = optionsB64
  ? JSON.parse(Buffer.from(optionsB64, 'base64').toString('utf8'))
  : {};
setSeed(seed);
setValueSeed(seed);
resetCallCount();
let tensor;
if (name === 'basic') {
  tensor = await basic(2, [128, 128, 3], options);
} else if (name === 'multires') {
  tensor = await multires(2, [128, 128, 3], {
    octaves: 2,
    hueRotation: 0,
    postEffects: [],
    finalEffects: [],
    ...options,
  });
} else {
  throw new Error(`Unknown generator ${name}`);
}
const arr = await tensor.read();
const buf = Buffer.from(new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength));
console.log(
  JSON.stringify({ tensor: buf.toString('base64'), callCount: getCallCount() }),
);
