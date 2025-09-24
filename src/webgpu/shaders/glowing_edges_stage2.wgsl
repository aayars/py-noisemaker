struct Stage2Params {
  width : f32,
  height : f32,
  channels : f32,
  metric : f32,
};

struct PosterBuffer {
  values : array<f32>,
};

struct EdgeBuffer {
  values : array<f32>,
};

@group(0) @binding(0) var<storage, read> poster_buffer : PosterBuffer;
@group(0) @binding(1) var input_tex : texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> edge_buffer : EdgeBuffer;
@group(0) @binding(3) var<uniform> params : Stage2Params;

const GAUSS_WEIGHTS : array<f32, 25> = array<f32, 25>(
  1.0 / 36.0, 4.0 / 36.0, 6.0 / 36.0, 4.0 / 36.0, 1.0 / 36.0,
  4.0 / 36.0, 16.0 / 36.0, 24.0 / 36.0, 16.0 / 36.0, 4.0 / 36.0,
  6.0 / 36.0, 24.0 / 36.0, 36.0 / 36.0, 24.0 / 36.0, 6.0 / 36.0,
  4.0 / 36.0, 16.0 / 36.0, 24.0 / 36.0, 16.0 / 36.0, 4.0 / 36.0,
  1.0 / 36.0, 4.0 / 36.0, 6.0 / 36.0, 4.0 / 36.0, 1.0 / 36.0,
);

const SOBEL_X : array<f32, 9> = array<f32, 9>(
  0.5, 0.0, -0.5,
  1.0, 0.0, -1.0,
  0.5, 0.0, -0.5,
);

const SOBEL_Y : array<f32, 9> = array<f32, 9>(
  0.5, 1.0, 0.5,
  0.0, 0.0, 0.0,
  -0.5, -1.0, -0.5,
);

const PI : f32 = 3.14159265358979323846;
const TAU : f32 = 6.28318530717958647692;
const SQRT2 : f32 = 1.41421356237309504880;
const SDF_SIDES : f32 = 5.0;

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

fn poster_index(x : i32, y : i32, width : u32, height : u32) -> u32 {
  if (width == 0u || height == 0u) {
    return 0u;
  }
  let xi = clamp_coord(x, width);
  let yi = clamp_coord(y, height);
  return u32(yi) * width + u32(xi);
}

fn sample_poster(x : i32, y : i32, width : u32, height : u32) -> f32 {
  let idx = poster_index(x, y, width, height);
  return poster_buffer.values[idx];
}

fn gaussian_blur(x : i32, y : i32, width : u32, height : u32) -> f32 {
  var sum : f32 = 0.0;
  var ky : i32 = -2;
  loop {
    if (ky > 2) {
      break;
    }
    var kx : i32 = -2;
    loop {
      if (kx > 2) {
        break;
      }
      let index = u32((ky + 2) * 5 + (kx + 2));
      let value = sample_poster(x + kx, y + ky, width, height);
      sum = sum + GAUSS_WEIGHTS[index] * value;
      kx = kx + 1;
    }
    ky = ky + 1;
  }
  return sum;
}

fn compute_distance(dx : f32, dy : f32, metric : u32) -> f32 {
  let abs_dx = abs(dx);
  let abs_dy = abs(dy);
  switch metric {
    case 2u: { // Manhattan
      return abs_dx + abs_dy;
    }
    case 3u: { // Chebyshev
      return max(abs_dx, abs_dy);
    }
    case 4u: { // Octagram
      let cross = (abs_dx + abs_dy) / SQRT2;
      return max(cross, max(abs_dx, abs_dy));
    }
    case 101u: { // Triangular
      return max(abs_dx - dy * 0.5, dy);
    }
    case 102u: { // Hexagram
      let a = max(abs_dx - dy * 0.5, dy);
      let b = max(abs_dx + dy * 0.5, -dy);
      return max(a, b);
    }
    case 201u: { // SDF
      let angle = atan2(dx, -dy) + PI;
      let r = TAU / SDF_SIDES;
      let sector = floor(0.5 + angle / r);
      let offset = sector * r - angle;
      let radius = sqrt(max(dx * dx + dy * dy, 0.0));
      return cos(offset) * radius;
    }
    default: {
      let sum = dx * dx + dy * dy;
      return sqrt(max(sum, 0.0));
    }
  }
}

fn gradient_scale(metric : u32) -> f32 {
  switch metric {
    case 2u: { return 4.0; }
    case 3u: { return 2.0; }
    case 4u: { return 4.0; }
    case 101u: { return 4.0; }
    case 102u: { return 4.0; }
    case 201u: { return 4.0; }
    default: { return 2.82842712474619009760; }
  }
}

fn fetch_channel(texel : vec4<f32>, channel : u32) -> f32 {
  let idx = min(channel, 3u);
  switch idx {
    case 0u: { return texel.x; }
    case 1u: { return texel.y; }
    case 2u: { return texel.z; }
    default: { return texel.w; }
  }
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let width = as_u32(params.width);
  let height = as_u32(params.height);
  if (gid.x >= width || gid.y >= height) {
    return;
  }

  let channel_count = max(as_u32(params.channels), 1u);
  let metric = as_u32(params.metric);
  let x = i32(gid.x);
  let y = i32(gid.y);

  var blurred : array<f32, 9>;
  var index : u32 = 0u;
  var offset_y : i32 = -1;
  loop {
    if (offset_y > 1) { break; }
    var offset_x : i32 = -1;
    loop {
      if (offset_x > 1) { break; }
      blurred[index] = gaussian_blur(x + offset_x, y + offset_y, width, height);
      index = index + 1u;
      offset_x = offset_x + 1;
    }
    offset_y = offset_y + 1;
  }

  var gx : f32 = 0.0;
  var gy : f32 = 0.0;
  for (var i : u32 = 0u; i < 9u; i = i + 1u) {
    gx = gx + SOBEL_X[i] * blurred[i];
    gy = gy + SOBEL_Y[i] * blurred[i];
  }

  let distance = compute_distance(gx, gy, metric);
  let scale = max(gradient_scale(metric), 1e-5);
  let normalized = clamp(distance / scale, 0.0, 1.0);
  let inverted = clamp(1.0 - normalized, 0.0, 1.0);
  let edge_factor = min(inverted * 8.0, 1.0);

  let coords = vec2<i32>(x, y);
  let texel = textureLoad(input_tex, coords, 0);
  let base_index = (gid.y * width + gid.x) * channel_count;

  for (var channel : u32 = 0u; channel < channel_count; channel = channel + 1u) {
    let src_val = fetch_channel(texel, channel);
    let bright_val = min(src_val * 1.25, 1.0);
    edge_buffer.values[base_index + channel] = edge_factor * bright_val;
  }
}
