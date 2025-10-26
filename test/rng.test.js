import assert from 'assert';
import { setSeed, random, randomInt, choice, Random } from '../js/noisemaker/rng.js';
import fs from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const fixturesDir = join(__dirname, 'fixtures', 'rng');

for (const file of fs.readdirSync(fixturesDir)) {
  const m = file.match(/seed_(\d+)\.json/);
  if (!m) continue;
  const seed = Number(m[1]);
  const expected = JSON.parse(fs.readFileSync(join(fixturesDir, file)));

  setSeed(seed);
  for (let i = 0; i < expected.length; i++) {
    const v = random();
    assert.ok(Math.abs(v - expected[i]) < 1e-9, `seed ${seed} index ${i}`);
  }

  setSeed(seed);
  for (let i = 0; i < expected.length; i++) {
    const v = randomInt(0, 99);
    const exp = Math.floor(expected[i] * 100);
    assert.strictEqual(v, exp, `randomInt seed ${seed} index ${i}`);
  }

  const seq = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
  setSeed(seed);
  for (let i = 0; i < expected.length; i++) {
    const v = choice(seq);
    const exp = seq[Math.floor(expected[i] * seq.length)];
    assert.strictEqual(v, exp, `choice seed ${seed} index ${i}`);
  }

  const rng = new Random(seed);
  for (let i = 0; i < expected.length; i++) {
    const v = rng.random();
    assert.ok(Math.abs(v - expected[i]) < 1e-9, `class seed ${seed} index ${i}`);
  }

  const rngInt = new Random(seed);
  for (let i = 0; i < expected.length; i++) {
    const v = rngInt.randomInt(0, 99);
    const exp = Math.floor(expected[i] * 100);
    assert.strictEqual(v, exp, `class randomInt seed ${seed} index ${i}`);
  }

  const rngChoice = new Random(seed);
  for (let i = 0; i < expected.length; i++) {
    const v = rngChoice.choice(seq);
    const exp = seq[Math.floor(expected[i] * seq.length)];
    assert.strictEqual(v, exp, `class choice seed ${seed} index ${i}`);
  }
}

console.log('RNG tests passed');
