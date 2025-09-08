import assert from 'assert';
import { random, simplex } from '../src/simplex.js';
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import path from 'path';

const shape = [2, 2];
const time = 0.25;
const seed = 12345;
const speed = 1.0;

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const fixture = JSON.parse(readFileSync(path.join(__dirname, 'fixtures', 'simplex.json'), 'utf8'));
const { tensor: pyTensor, random: pyRandom } = fixture;

const jsRandom = random(time, seed, speed);
const tensor = simplex([...shape], { time, seed, speed });
const jsVals = Array.from(tensor.read()).slice(0, shape[0] * shape[1]);

assert.ok(Math.abs(jsRandom - pyRandom) < 1e-6);
for (let y = 0; y < shape[0]; y++) {
  for (let x = 0; x < shape[1]; x++) {
    const idx = y * shape[1] + x;
    assert.ok(Math.abs(jsVals[idx] - pyTensor[y][x]) < 1e-6);
  }
}

console.log('Simplex tests passed');
