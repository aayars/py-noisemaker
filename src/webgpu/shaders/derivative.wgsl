// Compute shader implementing the derivative effect distance calculation.
//
// The shader consumes the X and Y derivative textures produced by the CPU or
// GPU convolution path and writes the per-channel distance magnitude into a
// linear storage buffer.  The logic mirrors the CPU implementation in
// `value.js`/`effects.js`, including support for the different distance metrics
// exposed through the `DistanceMetric` enum.

struct Params {
  width : f32,
  height : f32,
  channels : f32,
  metric : f32,
  padding : vec4<f32>,
};

struct OutputBuffer {
  values : array<f32>,
};

@group(0) @binding(0) var dx_texture : texture_2d<f32>;
@group(0) @binding(1) var dy_texture : texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> output_buffer : OutputBuffer;
@group(0) @binding(3) var<uniform> params : Params;

const PI : f32 = 3.14159265358979323846;
const TAU : f32 = 6.28318530717958647692;
const SQRT2 : f32 = 1.41421356237309504880;
const SDF_SIDES : f32 = 5.0;

fn compute_distance(dx : f32, dy : f32, metric : u32) -> f32 {
  let abs_dx = abs(dx);
  let abs_dy = abs(dy);

  switch metric {
    case 2u: { // DistanceMetric.manhattan
      return abs_dx + abs_dy;
    }
    case 3u: { // DistanceMetric.chebyshev
      return max(abs_dx, abs_dy);
    }
    case 4u: { // DistanceMetric.octagram
      let cross = (abs_dx + abs_dy) / SQRT2;
      return max(cross, max(abs_dx, abs_dy));
    }
    case 101u: { // DistanceMetric.triangular
      return max(abs_dx - dy * 0.5, dy);
    }
    case 102u: { // DistanceMetric.hexagram
      let a = max(abs_dx - dy * 0.5, dy);
      let b = max(abs_dx + dy * 0.5, -dy);
      return max(a, b);
    }
    case 201u: { // DistanceMetric.sdf
      let angle = atan2(dx, -dy) + PI;
      let r = TAU / SDF_SIDES;
      let sector = floor(0.5 + angle / r);
      let offset = sector * r - angle;
      let radius = sqrt(max(dx * dx + dy * dy, 0.0));
      return cos(offset) * radius;
    }
    default: { // DistanceMetric.euclidean and fall-through
      let sum = dx * dx + dy * dy;
      return sqrt(max(sum, 0.0));
    }
  }
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
  let width = u32(params.width);
  let height = u32(params.height);
  if (global_id.x >= width || global_id.y >= height) {
    return;
  }

  let channel_count = max(u32(floor(params.channels + 0.5)), 1u);
  let metric = u32(floor(params.metric + 0.5));
  let coords = vec2<i32>(i32(global_id.x), i32(global_id.y));

  let dx_sample = textureLoad(dx_texture, coords, 0);
  let dy_sample = textureLoad(dy_texture, coords, 0);

  let base_index = (global_id.y * width + global_id.x) * channel_count;

  for (var channel : u32 = 0u; channel < channel_count; channel = channel + 1u) {
    let dx_value = dx_sample[channel];
    let dy_value = dy_sample[channel];
    let distance = compute_distance(dx_value, dy_value, metric);
    output_buffer.values[base_index + channel] = distance;
  }
}
