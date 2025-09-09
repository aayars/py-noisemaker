import assert from 'assert';
import {
  DistanceMetric,
  ValueDistribution,
  PointDistribution,
  ValueMask,
  VoronoiDiagramType,
  WormBehavior,
  ColorSpace,
  isAbsolute,
  isSigned,
  distanceMetricAll,
  distanceMetricAbsoluteMembers,
  distanceMetricSignedMembers,
  isCenterDistribution,
  isScan,
  isNativeSize,
  isGrid,
  isCircular,
  gridMembers,
  circularMembers,
  isNoise,
  isProcedural,
  glyphMembers,
  flowMembers,
  isFlowMember,
  wormBehaviorAll,
  isColor,
  colorMembers
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
assert.ok(distanceMetricAll().includes(DistanceMetric.euclidean));
assert.ok(distanceMetricAll().includes(DistanceMetric.triangular));
assert.ok(!distanceMetricAll().includes(DistanceMetric.none));
assert.ok(distanceMetricAbsoluteMembers().includes(DistanceMetric.euclidean));
assert.ok(!distanceMetricAbsoluteMembers().includes(DistanceMetric.triangular));
assert.ok(distanceMetricSignedMembers().includes(DistanceMetric.triangular));
assert.ok(!distanceMetricSignedMembers().includes(DistanceMetric.euclidean));

assert.ok(isCenterDistribution(ValueDistribution.center_hexagon));
assert.ok(!isCenterDistribution(ValueDistribution.uniform));
assert.ok(isScan(ValueDistribution.scan_left));
assert.ok(!isScan(ValueDistribution.center_hexagon));
assert.ok(isNativeSize(ValueDistribution.scan_down));
assert.ok(isNativeSize(ValueDistribution.center_circle));
assert.ok(!isNativeSize(ValueDistribution.exp));

assert.ok(isGrid(PointDistribution.square));
assert.ok(!isGrid(PointDistribution.spiral));
assert.ok(gridMembers.includes(PointDistribution.chess));
assert.ok(!gridMembers.includes(PointDistribution.random));

assert.ok(isCircular(PointDistribution.circular));
assert.ok(isCircular(PointDistribution.rotating));
assert.ok(!isCircular(PointDistribution.square));
assert.ok(circularMembers.includes(PointDistribution.rotating));
assert.ok(!circularMembers.includes(PointDistribution.square));

assert.ok(isNoise(ValueDistribution.uniform));
assert.ok(!isNoise(ValueDistribution.ones));

assert.ok(isProcedural(ValueMask.sparse));
assert.ok(isProcedural(ValueMask.emoji));
assert.ok(!isProcedural(ValueMask.square));
assert.ok(glyphMembers.includes(ValueMask.emoji));
assert.ok(glyphMembers.includes(ValueMask.bank_ocr));
assert.ok(!glyphMembers.includes(ValueMask.square));

assert.ok(flowMembers.includes(VoronoiDiagramType.flow));
assert.ok(isFlowMember(VoronoiDiagramType.color_flow));
assert.ok(!isFlowMember(VoronoiDiagramType.range));

assert.ok(wormBehaviorAll.includes(WormBehavior.random));
assert.ok(!wormBehaviorAll.includes(WormBehavior.none));

assert.ok(isColor(ColorSpace.rgb));
assert.ok(!isColor(ColorSpace.grayscale));
assert.ok(colorMembers.includes(ColorSpace.hsv));
assert.ok(!colorMembers.includes(ColorSpace.grayscale));

console.log('All constants tests passed');
