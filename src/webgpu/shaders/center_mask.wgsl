// Compute shader for the center_mask effect. The shader mirrors the CPU
// implementation by computing a radial blend factor based on the requested
// distance metric and power, then interpolating between the center and edge
// tensors.

struct Params {
  width : f32,
  height : f32,
  channels : f32,
  metric : f32,
  power : f32,
  max_distance : f32,
  sdf_sides : f32,
  padding : f32,
};

@group(0) @binding(0) var center_texture : texture_2d<f32>;
@group(0) @binding(1) var edge_texture : texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(3) var<uniform> params : Params;

const PI : f32 = 3.14159265358979323846;
const TAU : f32 = 6.28318530717958647692;
const SQRT2 : f32 = 1.41421356237309504880;
const EPSILON : f32 = 1e-6;

fn as_u32(value : f32) -> u32 {
  return u32(max(value, 0.0));
}

fn sample_channel(value : vec4<f32>, index : u32) -> f32 {
  switch index {
    case 0u: { return value.x; }
    case 1u: { return value.y; }
    case 2u: { return value.z; }
    default: { return value.w; }
  }
}

fn compute_distance(dx : f32, dy : f32, metric : u32, sdf_sides : f32) -> f32 {
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
      let sides = max(sdf_sides, 3.0);
      let r = TAU / sides;
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
  let width : u32 = as_u32(params.width);
  let height : u32 = as_u32(params.height);
  if (global_id.x >= width || global_id.y >= height) {
    return;
  }

  let channel_count : u32 = max(as_u32(params.channels), 1u);
  let coords : vec2<i32> = vec2<i32>(i32(global_id.x), i32(global_id.y));
  let center_sample : vec4<f32> = textureLoad(center_texture, coords, 0);
  let edge_sample : vec4<f32> = textureLoad(edge_texture, coords, 0);

  let cx : f32 = (params.width - 1.0) * 0.5;
  let cy : f32 = (params.height - 1.0) * 0.5;
  let fx : f32 = f32(global_id.x);
  let fy : f32 = f32(global_id.y);
  let dx : f32 = abs(fx - cx);
  let dy : f32 = abs(fy - cy);

  let metric : u32 = as_u32(round(params.metric));
  let sdf_sides : f32 = params.sdf_sides;
  let dist : f32 = compute_distance(dx, dy, metric, sdf_sides);
  let max_dist : f32 = max(params.max_distance, EPSILON);
  var t : f32 = dist / max_dist;
  t = pow(t, params.power);
  let inverse : f32 = 1.0 - t;

  let base_index : u32 = (global_id.y * width + global_id.x) * channel_count;
  var channel : u32 = 0u;
  loop {
    if (channel >= channel_count) {
      break;
    }
    let center_value : f32 = sample_channel(center_sample, channel);
    let edge_value : f32 = sample_channel(edge_sample, channel);
    output_buffer[base_index + channel] = center_value * inverse + edge_value * t;
    channel = channel + 1u;
  }
}
