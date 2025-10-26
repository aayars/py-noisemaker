import { simplex } from '../js/noisemaker/simplex.js';
import { spawnSync } from 'child_process';

const shape = [2, 2];
const time = 0.25;
const seed = 12345;
const speed = 1.0;
const js = simplex([...shape], { time, seed, speed });
const jsVals = Array.from(js.read()).slice(0, shape[0]*shape[1]);

const pyCode = `
from opensimplex import OpenSimplex
import math, json
seed=${seed}
angle=math.tau*${time}
z=math.cos(angle)*${speed}
simp=OpenSimplex(seed)
rows=[[ (simp.noise3d(x,y,z)+1)/2 for x in range(${shape[1]})] for y in range(${shape[0]})]
print(json.dumps(rows))
`;
const py = spawnSync('python', ['-'], { input: pyCode, encoding: 'utf8' });
if (py.status !== 0) {
  console.error(py.stderr);
  process.exit(py.status);
}
const pyVals = JSON.parse(py.stdout.trim()).flat();

let diff = 0;
for (let i = 0; i < jsVals.length; i++) {
  diff += Math.abs(jsVals[i] - pyVals[i]);
}
diff /= jsVals.length;
console.log('mean absolute diff', diff);
if (diff > 1e-6) {
  throw new Error('Regression mismatch');
}
console.log('Regression tests passed');
