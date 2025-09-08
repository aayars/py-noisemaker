import assert from 'assert';
import { random, simplex } from '../src/simplex.js';
import { spawnSync } from 'child_process';

const shape = [2, 2];
const time = 0.25;
const seed = 12345;
const speed = 1.0;

const jsRandom = random(time, seed, speed);
const tensor = simplex([...shape], { time, seed, speed });
const jsVals = Array.from(tensor.read()).slice(0, shape[0] * shape[1]);

const pyCode = `
from opensimplex import OpenSimplex
import math, json
seed=${seed}
time=${time}
speed=${speed}
angle=math.tau*time
z=math.cos(angle)*speed
simp=OpenSimplex(seed)
rows=[[ (simp.noise3d(x,y,z)+1)/2 for x in range(${shape[1]})] for y in range(${shape[0]})]
print(json.dumps(rows))
rand=(OpenSimplex(seed).noise2d(math.cos(angle)*speed, math.sin(angle)*speed)+1)/2
print(rand)
`;
const py = spawnSync('python', ['-'], { input: pyCode, encoding: 'utf8' });
const [pyTensorJson, pyRandStr] = py.stdout.trim().split('\n');
const pyTensor = JSON.parse(pyTensorJson);
const pyRandom = parseFloat(pyRandStr);

assert.ok(Math.abs(jsRandom - pyRandom) < 1e-6);
for (let y = 0; y < shape[0]; y++) {
  for (let x = 0; x < shape[1]; x++) {
    const idx = y * shape[1] + x;
    assert.ok(Math.abs(jsVals[idx] - pyTensor[y][x]) < 1e-6);
  }
}

console.log('Simplex tests passed');
