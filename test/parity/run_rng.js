import {
  setSeed,
  random,
  randomInt,
  choice,
  resetCallCount,
  getCallCount,
  getSeed,
  Random,
} from '../../js/noisemaker/rng.js';

// Arguments: scope ('global' or 'class'), function name, seed, [args]
const [,, scope, fn, seedStr, ...args] = process.argv;
const seed = Number(seedStr);

let values;
if (scope === 'class') {
  // Instance-based RNG
  resetCallCount();
  const rng = new Random(seed);
  switch (fn) {
    case 'random': {
      const count = Number(args[0] || 1);
      values = Array.from({ length: count }, () => rng.random());
      break;
    }
    case 'randomInt': {
      const min = Number(args[0]);
      const max = Number(args[1]);
      const count = Number(args[2] || 1);
      values = Array.from({ length: count }, () => rng.randomInt(min, max));
      break;
    }
    case 'choice': {
      const count = Number(args[0] || 1);
      const arr = Array.from({ length: 10 }, (_, i) => i);
      values = Array.from({ length: count }, () => rng.choice(arr));
      break;
    }
    default:
      throw new Error(`Unknown function: ${fn}`);
  }
  const callCount = getCallCount();
  const state = rng.state >>> 0;
  console.log(JSON.stringify({ values, callCount, seed: state }));
} else {
  // Global RNG
  setSeed(seed);
  resetCallCount();
  switch (fn) {
    case 'random': {
      const count = Number(args[0] || 1);
      values = Array.from({ length: count }, () => random());
      break;
    }
    case 'randomInt': {
      const min = Number(args[0]);
      const max = Number(args[1]);
      const count = Number(args[2] || 1);
      values = Array.from({ length: count }, () => randomInt(min, max));
      break;
    }
    case 'choice': {
      const count = Number(args[0] || 1);
      const arr = Array.from({ length: 10 }, (_, i) => i);
      values = Array.from({ length: count }, () => choice(arr));
      break;
    }
    default:
      throw new Error(`Unknown function: ${fn}`);
  }
  const callCount = getCallCount();
  const state = getSeed();
  console.log(JSON.stringify({ values, callCount, seed: state }));
}
