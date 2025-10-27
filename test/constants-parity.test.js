import assert from 'assert';
import { spawnSync } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';
import {
  DistanceMetric,
  InterpolationType,
  PointDistribution,
  ValueDistribution,
  ValueMask,
  VoronoiDiagramType,
  WormBehavior,
  ColorSpace,
  distanceMetricAll,
  distanceMetricAbsoluteMembers,
  distanceMetricSignedMembers,
  gridMembers,
  circularMembers,
  valueMaskProceduralMembers,
  valueMaskNonproceduralMembers,
  valueMaskConv2dMembers,
  valueMaskGridMembers,
  valueMaskRgbMembers,
  valueMaskGlyphMembers,
  flowMembers,
  wormBehaviorAll,
  colorSpaceMembers,
} from '../js/noisemaker/constants.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(__dirname, '..');

function getPythonData() {
  const py = `
import json
from noisemaker.constants import (DistanceMetric, InterpolationType, PointDistribution, ValueDistribution, ValueMask, VoronoiDiagramType, WormBehavior, ColorSpace)

def enum_to_dict(e):
    return {m.name: m.value for m in e}

data = {
    'DistanceMetric': enum_to_dict(DistanceMetric),
    'InterpolationType': enum_to_dict(InterpolationType),
    'PointDistribution': enum_to_dict(PointDistribution),
    'ValueDistribution': enum_to_dict(ValueDistribution),
    'ValueMask': enum_to_dict(ValueMask),
    'VoronoiDiagramType': enum_to_dict(VoronoiDiagramType),
    'WormBehavior': enum_to_dict(WormBehavior),
    'ColorSpace': enum_to_dict(ColorSpace),
    'groups': {
        'DistanceMetric': {
            'all': [m.value for m in DistanceMetric.all()],
            'absolute_members': [m.value for m in DistanceMetric.absolute_members()],
            'signed_members': [m.value for m in DistanceMetric.signed_members()],
        },
        'PointDistribution': {
            'grid_members': [m.value for m in PointDistribution.grid_members()],
            'circular_members': [m.value for m in PointDistribution.circular_members()],
        },
        'ValueMask': {
            'procedural_members': [m.value for m in ValueMask.procedural_members()],
            'nonprocedural_members': [m.value for m in ValueMask.nonprocedural_members()],
            'conv2d_members': [m.value for m in ValueMask if m.name.startswith('conv2d')],
            'grid_members': [m.value for m in ValueMask if m.value < ValueMask.alphanum_0.value],
            'rgb_members': [m.value for m in ValueMask if m.value >= ValueMask.rgb.value and m.value < ValueMask.sparse.value],
            'glyph_members': [m.value for m in ValueMask.glyph_members()],
        },
        'VoronoiDiagramType': {
            'flow_members': [m.value for m in VoronoiDiagramType.flow_members()],
        },
        'WormBehavior': {
            'all': [m.value for m in WormBehavior.all()],
        },
        'ColorSpace': {
            'color_members': [m.value for m in ColorSpace.color_members()],
        },
    }
}
print(json.dumps(data))
`;
  const res = spawnSync('python3', ['-c', py], { cwd: repoRoot, encoding: 'utf8' });
  if (res.status !== 0) {
    throw new Error(res.stderr);
  }
  return JSON.parse(res.stdout);
}

const pyData = getPythonData();
const enums = {
  DistanceMetric,
  InterpolationType,
  PointDistribution,
  ValueDistribution,
  ValueMask,
  VoronoiDiagramType,
  WormBehavior,
  ColorSpace,
};

for (const [name, jsEnum] of Object.entries(enums)) {
  const pyEnum = pyData[name];
  for (const [k, v] of Object.entries(pyEnum)) {
    assert.strictEqual(jsEnum[k], v, `Mismatch for ${name}.${k}`);
  }
  for (const [k, v] of Object.entries(jsEnum)) {
    assert.strictEqual(pyEnum[k], v, `Extra JS member ${name}.${k}`);
  }
}

const sort = (arr) => arr.slice().sort((a, b) => a - b);
const groups = pyData.groups;

assert.deepStrictEqual(sort(distanceMetricAll()), sort(groups.DistanceMetric.all), 'DistanceMetric.all mismatch');
assert.deepStrictEqual(sort(distanceMetricAbsoluteMembers()), sort(groups.DistanceMetric.absolute_members), 'DistanceMetric.absolute_members mismatch');
assert.deepStrictEqual(sort(distanceMetricSignedMembers()), sort(groups.DistanceMetric.signed_members), 'DistanceMetric.signed_members mismatch');

assert.deepStrictEqual(sort(gridMembers), sort(groups.PointDistribution.grid_members), 'PointDistribution.grid_members mismatch');
assert.deepStrictEqual(sort(circularMembers), sort(groups.PointDistribution.circular_members), 'PointDistribution.circular_members mismatch');

assert.deepStrictEqual(sort(valueMaskProceduralMembers), sort(groups.ValueMask.procedural_members), 'ValueMask.procedural_members mismatch');
assert.deepStrictEqual(sort(valueMaskNonproceduralMembers), sort(groups.ValueMask.nonprocedural_members), 'ValueMask.nonprocedural_members mismatch');
assert.deepStrictEqual(sort(valueMaskConv2dMembers), sort(groups.ValueMask.conv2d_members), 'ValueMask.conv2d_members mismatch');
assert.deepStrictEqual(sort(valueMaskGridMembers), sort(groups.ValueMask.grid_members), 'ValueMask.grid_members mismatch');
assert.deepStrictEqual(sort(valueMaskRgbMembers), sort(groups.ValueMask.rgb_members), 'ValueMask.rgb_members mismatch');
assert.deepStrictEqual(sort(valueMaskGlyphMembers), sort(groups.ValueMask.glyph_members), 'ValueMask.glyph_members mismatch');

assert.deepStrictEqual(sort(flowMembers), sort(groups.VoronoiDiagramType.flow_members), 'VoronoiDiagramType.flow_members mismatch');
assert.deepStrictEqual(sort(wormBehaviorAll), sort(groups.WormBehavior.all), 'WormBehavior.all mismatch');
assert.deepStrictEqual(sort(colorSpaceMembers()), sort(groups.ColorSpace.color_members), 'ColorSpace.color_members mismatch');

console.log('constants parity ok');
