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
  isNativeSize,
  isGrid,
  isCircular,
  gridMembers,
  circularMembers,
  isNoise,
  valueMaskConv2dMembers,
  isValueMaskConv2d,
  valueMaskGridMembers,
  isValueMaskGrid,
  valueMaskRgbMembers,
  isValueMaskRgb,
  valueMaskProceduralMembers,
  valueMaskNonproceduralMembers,
  isValueMaskProcedural,
  valueMaskGlyphMembers,
  isValueMaskGlyph,
  flowMembers,
  isFlowMember,
  wormBehaviorAll,
  isColor,
  colorSpaceMembers
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
const centerMembers = Object.values(ValueDistribution).filter((v) =>
  isCenterDistribution(v)
);
centerMembers.forEach((v) => {
  assert.ok(isNativeSize(v));
});
assert.ok(!isNativeSize(ValueDistribution.exp));
assert.ok(!isNativeSize(ValueDistribution.column_index));

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
assert.ok(isNoise(ValueDistribution.exp));
assert.ok(!isNoise(ValueDistribution.ones));

assert.ok(isValueMaskProcedural(ValueMask.sparse));
assert.ok(isValueMaskProcedural(ValueMask.emoji));
assert.ok(!isValueMaskProcedural(ValueMask.square));
assert.ok(valueMaskProceduralMembers.includes(ValueMask.sparse));
assert.ok(valueMaskNonproceduralMembers.includes(ValueMask.square));

assert.ok(valueMaskConv2dMembers.includes(ValueMask.conv2d_blur));
assert.ok(isValueMaskConv2d(ValueMask.conv2d_blur));
assert.ok(!isValueMaskConv2d(ValueMask.square));

assert.ok(valueMaskGridMembers.includes(ValueMask.chess));
assert.ok(!valueMaskGridMembers.includes(ValueMask.alphanum_0));
assert.ok(isValueMaskGrid(ValueMask.square));
assert.ok(!isValueMaskGrid(ValueMask.alphanum_0));

assert.ok(valueMaskRgbMembers.includes(ValueMask.rgb));
assert.ok(!valueMaskRgbMembers.includes(ValueMask.sparse));
assert.ok(isValueMaskRgb(ValueMask.rgb));
assert.ok(!isValueMaskRgb(ValueMask.square));

assert.ok(valueMaskGlyphMembers.includes(ValueMask.emoji));
assert.ok(isValueMaskGlyph(ValueMask.bank_ocr));
assert.ok(!isValueMaskGlyph(ValueMask.square));

assert.ok(flowMembers.includes(VoronoiDiagramType.flow));
assert.ok(isFlowMember(VoronoiDiagramType.color_flow));
assert.ok(!isFlowMember(VoronoiDiagramType.range));

assert.ok(wormBehaviorAll.includes(WormBehavior.random));
assert.ok(!wormBehaviorAll.includes(WormBehavior.none));

assert.ok(isColor(ColorSpace.rgb));
assert.ok(!isColor(ColorSpace.grayscale));
const colorMembers = colorSpaceMembers();
assert.ok(colorMembers.includes(ColorSpace.hsv));
assert.ok(!colorMembers.includes(ColorSpace.grayscale));

console.log('All constants tests passed');
