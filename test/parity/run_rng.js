import { setSeed, random, randomInt, choice } from '../../src/rng.js';

const [,, fn, seed, ...args] = process.argv;
setSeed(Number(seed));
let out;
switch (fn) {
  case 'random': {
    const count = Number(args[0] || 1);
    out = Array.from({length: count}, () => random());
    break;
  }
  case 'randomInt': {
    const min = Number(args[0]);
    const max = Number(args[1]);
    const count = Number(args[2] || 1);
    out = Array.from({length: count}, () => randomInt(min, max));
    break;
  }
  case 'choice': {
    const count = Number(args[0] || 1);
    const arr = Array.from({length: 10}, (_, i) => i);
    out = Array.from({length: count}, () => choice(arr));
    break;
  }
  default:
    throw new Error(`Unknown function: ${fn}`);
}
console.log(JSON.stringify(out));
