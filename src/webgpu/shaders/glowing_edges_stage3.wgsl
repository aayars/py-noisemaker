struct Stage3Params {
  width : f32,
  height : f32,
  channels : f32,
  padding : f32,
};

struct EdgeBuffer {
  values : array<f32>,
};

struct GlowBuffer {
  values : array<f32>,
};

@group(0) @binding(0) var<storage, read> edge_buffer : EdgeBuffer;
@group(0) @binding(1) var<storage, read_write> glow_buffer : GlowBuffer;
@group(0) @binding(2) var<uniform> params : Stage3Params;

const GAUSS_WEIGHTS : array<f32, 25> = array<f32, 25>(
  1.0 / 36.0, 4.0 / 36.0, 6.0 / 36.0, 4.0 / 36.0, 1.0 / 36.0,
  4.0 / 36.0, 16.0 / 36.0, 24.0 / 36.0, 16.0 / 36.0, 4.0 / 36.0,
  6.0 / 36.0, 24.0 / 36.0, 36.0 / 36.0, 24.0 / 36.0, 6.0 / 36.0,
  4.0 / 36.0, 16.0 / 36.0, 24.0 / 36.0, 16.0 / 36.0, 4.0 / 36.0,
  1.0 / 36.0, 4.0 / 36.0, 6.0 / 36.0, 4.0 / 36.0, 1.0 / 36.0,
);

fn as_u32(value : f32) -> u32 {
  return u32(max(value, 0.0));
}

fn clamp_coord(v : i32, limit : u32) -> i32 {
  if (limit == 0u) {
    return 0;
  }
  let max_index = i32(limit) - 1;
  return clamp(v, 0, max_index);
}

fn edge_index(x : i32, y : i32, channel : u32, width : u32, height : u32, channels : u32) -> u32 {
  if (width == 0u || height == 0u || channels == 0u) {
    return 0u;
  }
  let xi = clamp_coord(x, width);
  let yi = clamp_coord(y, height);
  let safe_channel = min(channel, channels - 1u);
  return (u32(yi) * width + u32(xi)) * channels + safe_channel;
}

fn sample_edge(x : i32, y : i32, channel : u32, width : u32, height : u32, channels : u32) -> f32 {
  let idx = edge_index(x, y, channel, width, height, channels);
  return edge_buffer.values[idx];
}

fn gaussian_blur_edge(x : i32, y : i32, channel : u32, width : u32, height : u32, channels : u32) -> f32 {
  var sum : f32 = 0.0;
  var ky : i32 = -2;
  loop {
    if (ky > 2) { break; }
    var kx : i32 = -2;
    loop {
      if (kx > 2) { break; }
      let weight_idx = u32((ky + 2) * 5 + (kx + 2));
      let sample = sample_edge(x + kx, y + ky, channel, width, height, channels);
      sum = sum + GAUSS_WEIGHTS[weight_idx] * sample;
      kx = kx + 1;
    }
    ky = ky + 1;
  }
  return sum;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let width = as_u32(params.width);
  let height = as_u32(params.height);
  if (gid.x >= width || gid.y >= height) {
    return;
  }

  let channel_count = max(as_u32(params.channels), 1u);
  let x = i32(gid.x);
  let y = i32(gid.y);
  let base_index = (gid.y * width + gid.x) * channel_count;

  for (var channel : u32 = 0u; channel < channel_count; channel = channel + 1u) {
    let edge_val = edge_buffer.values[base_index + channel];
    let blur_val = gaussian_blur_edge(x, y, channel, width, height, channel_count);
    let combined = clamp(edge_val + blur_val * 0.5, 0.0, 1.0);
    glow_buffer.values[base_index + channel] = combined;
  }
}
