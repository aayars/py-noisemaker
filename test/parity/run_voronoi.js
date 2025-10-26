import { setSeed } from '../../js/noisemaker/rng.js';
import { setSeed as setValueSeed } from '../../js/noisemaker/value.js';
import { basic } from '../../js/noisemaker/generators.js';
import { voronoi as voronoiEffect } from '../../js/noisemaker/effects.js';
import { Context } from '../../js/noisemaker/context.js';

const DEBUG = false; // Set true to diagnose shader issues.

const [,, encoded] = process.argv;
const params = JSON.parse(Buffer.from(encoded, 'base64').toString('utf8'));

const seed = params.seed;
setSeed(seed);
setValueSeed(seed);

const ctx = new Context(null, DEBUG);
await ctx.initWebGPU();
const base = await basic(2, [128, 128, 3], { hueRotation: 0, ctx });

const tensor = await voronoiEffect(
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
  params.inverse,
  params.refract_y_from_offset,
  params.point_freq,
  params.point_generations,
  params.point_distrib,
  params.point_drift,
  params.point_corners,
  null,
  params.downsample,
);

const arr = await tensor.read();
const buf = Buffer.from(new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength));
console.log(buf.toString('base64'));
