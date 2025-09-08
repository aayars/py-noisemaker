import assert from 'assert';
import {
  DistanceMetric,
  ValueDistribution,
  isAbsolute,
  isSigned,
  isCenterDistribution,
  isScan,
  isNativeSize
} from '../src/constants.js';

// enumeration integrity
assert.ok(Object.isFrozen(DistanceMetric), 'DistanceMetric should be frozen');
assert.strictEqual(DistanceMetric.euclidean, 1);

// ensure mutation fails
assert.throws(() => {
  DistanceMetric.euclidean = 99;
}, TypeError);
assert.strictEqual(DistanceMetric.euclidean, 1);

// predicate tests
assert.ok(isAbsolute(DistanceMetric.euclidean));
assert.ok(!isAbsolute(DistanceMetric.triangular));
assert.ok(isSigned(DistanceMetric.triangular));
assert.ok(!isSigned(DistanceMetric.euclidean));

assert.ok(isCenterDistribution(ValueDistribution.center_hexagon));
assert.ok(!isCenterDistribution(ValueDistribution.uniform));
assert.ok(isScan(ValueDistribution.scan_left));
assert.ok(!isScan(ValueDistribution.center_hexagon));
assert.ok(isNativeSize(ValueDistribution.scan_down));
assert.ok(isNativeSize(ValueDistribution.center_circle));
assert.ok(!isNativeSize(ValueDistribution.exp));

console.log('All constants tests passed');
