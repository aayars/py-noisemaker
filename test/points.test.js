import assert from 'assert';
import { PointDistribution, ValueMask } from '../js/noisemaker/constants.js';
import { pointCloud } from '../js/noisemaker/points.js';
import { setSeed as setSimplexSeed } from '../js/noisemaker/simplex.js';
import { setSeed as setUtilSeed } from '../js/noisemaker/util.js';

// deterministic seeds for tests
setSimplexSeed(1);
setUtilSeed(1);

function within(arr, max) {
  return arr.every(v => v >= 0 && v < max);
}

let x, y;

// grid
[x, y] = pointCloud(2, { distrib: PointDistribution.square, shape: [64, 64] });
// A 2x2 grid should produce four points.
assert.strictEqual(x.length, 4);
assert.strictEqual(y.length, 4);
assert.ok(within(x, 64) && within(y, 64));

// waffle
[x, y] = pointCloud(2, { distrib: PointDistribution.waffle, shape: [64, 64] });
assert.strictEqual(x.length, 3);

// chess
[x, y] = pointCloud(2, { distrib: PointDistribution.chess, shape: [64, 64] });
assert.strictEqual(x.length, 2);

// hex grids
[x, y] = pointCloud(2, { distrib: PointDistribution.h_hex, shape: [64, 64] });
// Horizontal hex grid also produces four points for n=2.
assert.strictEqual(x.length, 4);
[x, y] = pointCloud(2, { distrib: PointDistribution.v_hex, shape: [64, 64] });
assert.strictEqual(x.length, 4);

// spiral
[x, y] = pointCloud(2, { distrib: PointDistribution.spiral, shape: [64, 64] });
assert.strictEqual(x.length, 4);

// circular variants
[x, y] = pointCloud(2, { distrib: PointDistribution.circular, shape: [64, 64] });
// Circular distributions include the origin plus four points for n=2.
assert.strictEqual(x.length, 5);
[x, y] = pointCloud(2, { distrib: PointDistribution.concentric, shape: [64, 64] });
assert.strictEqual(x.length, 5);
[x, y] = pointCloud(2, { distrib: PointDistribution.rotating, shape: [64, 64] });
assert.strictEqual(x.length, 5);

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

// deterministic layout via seed
let sx1, sy1, sx2, sy2;
[sx1, sy1] = pointCloud(2, { distrib: PointDistribution.spiral, shape: [64, 64], seed: 123 });
[sx2, sy2] = pointCloud(2, { distrib: PointDistribution.spiral, shape: [64, 64], seed: 123 });
assert.deepStrictEqual(sx1, sx2);
assert.deepStrictEqual(sy1, sy2);

console.log('All points tests passed');
