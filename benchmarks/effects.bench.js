import { Tensor } from '../src/tensor.js';
import { aberration, invert } from '../src/effects.js';

const shape = [256, 256, 3];
const data = new Float32Array(shape[0] * shape[1] * shape[2]).fill(0.5);
const tensor = Tensor.fromArray(null, data, shape);

console.time('aberration');
aberration(tensor, shape, 0, 1);
console.timeEnd('aberration');

console.time('invert');
invert(tensor, shape, 0, 1);
console.timeEnd('invert');

console.log('Benchmarks complete');
