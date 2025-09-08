import assert from 'assert';
import { PointDistribution, ValueMask } from '../src/constants.js';
import { pointCloud } from '../src/points.js';
import { setSeed } from '../src/simplex.js';

// deterministic seed for tests
setSeed(1);

function within(arr, max) {
  return arr.every(v => v >= 0 && v < max);
}

let x, y;

// grid
[x, y] = pointCloud(2, { distrib: PointDistribution.square, shape: [64, 64] });
assert.strictEqual(x.length, 3);
assert.strictEqual(y.length, 3);
assert.ok(within(x, 64) && within(y, 64));

// waffle
[x, y] = pointCloud(2, { distrib: PointDistribution.waffle, shape: [64, 64] });
assert.strictEqual(x.length, 3);

// chess
[x, y] = pointCloud(2, { distrib: PointDistribution.chess, shape: [64, 64] });
assert.strictEqual(x.length, 2);

// hex grids
[x, y] = pointCloud(2, { distrib: PointDistribution.h_hex, shape: [64, 64] });
assert.strictEqual(x.length, 3);
[x, y] = pointCloud(2, { distrib: PointDistribution.v_hex, shape: [64, 64] });
assert.strictEqual(x.length, 4);

// spiral
[x, y] = pointCloud(2, { distrib: PointDistribution.spiral, shape: [64, 64] });
assert.strictEqual(x.length, 3);

// circular variants
[x, y] = pointCloud(2, { distrib: PointDistribution.circular, shape: [64, 64] });
assert.strictEqual(x.length, 4);
[x, y] = pointCloud(2, { distrib: PointDistribution.concentric, shape: [64, 64] });
assert.strictEqual(x.length, 4);
[x, y] = pointCloud(2, { distrib: PointDistribution.rotating, shape: [64, 64] });
assert.strictEqual(x.length, 4);

// mask-driven
[x, y] = pointCloud(2, { distrib: ValueMask.chess, shape: [64, 64] });
assert.strictEqual(x.length, 2);
assert.ok(within(x, 64) && within(y, 64));

// drift and generations
[x, y] = pointCloud(1, {
  distrib: PointDistribution.square,
  shape: [32, 32],
  generations: 2,
  drift: 0.5,
});
assert.ok(x.length > 1);

console.log('All points tests passed');
