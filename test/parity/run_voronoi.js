import { setSeed } from '../../src/rng.js';
import { setSeed as setValueSeed } from '../../src/value.js';
import { basic } from '../../src/generators.js';
import { voronoi as voronoiEffect } from '../../src/effects.js';

const [,, encoded] = process.argv;
const params = JSON.parse(Buffer.from(encoded, 'base64').toString('utf8'));

const seed = params.seed;
setSeed(seed);
setValueSeed(seed);

const base = basic(2, [128, 128, 3], { hueRotation: 0 });

const tensor = voronoiEffect(
  base,
  [128, 128, 3],
  0,
  1,
  params.diagram_type,
  params.nth,
  params.dist_metric,
  params.sdf_sides,
  params.alpha,
  params.with_refract,
  params.refract_y_from_offset,
  params.point_freq,
  params.point_generations,
  params.point_distrib,
  params.point_drift,
  params.point_corners,
  null,
  params.downsample,
);

const arr = tensor.read();
const buf = Buffer.from(new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength));
console.log(buf.toString('base64'));
