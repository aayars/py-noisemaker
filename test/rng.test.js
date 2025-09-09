import assert from 'assert';
import { Random, setSeed, getSeed, random } from '../src/rng.js';

// setSeed/getSeed maintain deterministic sequences
setSeed(123);
assert.strictEqual(getSeed(), 123);
const a = random();
setSeed(123);
assert.strictEqual(random(), a);

// Random instances with same seed generate same sequence
const r1 = new Random(42);
const r2 = new Random(42);
for (let i = 0; i < 5; i++) {
  assert.strictEqual(r1.random(), r2.random());
}

console.log('RNG tests passed');
