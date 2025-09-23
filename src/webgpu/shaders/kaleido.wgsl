struct KaleidoParams {
  dims0 : vec4<f32>;
  dims1 : vec4<f32>;
};

@group(0) @binding(0) var inputTexture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> outputBuffer : array<f32>;
@group(0) @binding(2) var<storage, read> radiusBuffer : array<f32>;
@group(0) @binding(3) var<storage, read> fadeBuffer : array<f32>;
@group(0) @binding(4) var<uniform> params : KaleidoParams;

const PI : f32 = 3.141592653589793;
const TAU : f32 = 6.283185307179586;

fn positive_mod(value : f32, modulus : f32) -> f32 {
  if (modulus == 0.0) {
    return 0.0;
  }
  let m : f32 = value - modulus * floor(value / modulus);
  if (m < 0.0) {
    return m + modulus;
  }
  return m;
}

fn wrap_trunc(value : f32, limit : f32) -> u32 {
  if (limit <= 0.0) {
    return 0u;
  }
  let truncated : f32 = trunc(value);
  var wrapped : f32 = truncated - limit * floor(truncated / limit);
  if (wrapped < 0.0) {
    wrapped = wrapped + limit;
  }
  let max_index : f32 = max(limit - 1.0, 0.0);
  return u32(clamp(wrapped, 0.0, max_index));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let width_f : f32 = params.dims0.x;
  let height_f : f32 = params.dims0.y;
  let channels_f : f32 = params.dims0.z;
  let sides_f : f32 = max(params.dims0.w, 1.0);
  let blend_edges : f32 = params.dims1.x;

  let width : u32 = u32(width_f);
  let height : u32 = u32(height_f);
  if (gid.x >= width || gid.y >= height) {
    return;
  }

  let channels : u32 = u32(channels_f);
  let step_angle : f32 = TAU / sides_f;

  let x : u32 = gid.x;
  let y : u32 = gid.y;
  let index : u32 = y * width + x;

  var denom_x : f32 = width_f - 1.0;
  if (width <= 1u) {
    denom_x = 1.0;
  }
  var denom_y : f32 = height_f - 1.0;
  if (height <= 1u) {
    denom_y = 1.0;
  }

  let xi : f32 = f32(x) / denom_x - 0.5;
  let yi : f32 = f32(y) / denom_y - 0.5;
  let radius : f32 = radiusBuffer[index];

  var angle : f32 = atan2(yi, xi) + PI / 2.0;
  angle = positive_mod(angle, step_angle);
  angle = abs(angle - step_angle * 0.5);

  var nx : f32 = radius * width_f * sin(angle);
  var ny : f32 = radius * height_f * cos(angle);

  if (blend_edges != 0.0) {
    let fade : f32 = fadeBuffer[index];
    nx = mix(nx, f32(x), fade);
    ny = mix(ny, f32(y), fade);
  }

  let src_x : u32 = wrap_trunc(nx, width_f);
  let src_y : u32 = wrap_trunc(ny, height_f);
  let texel : vec4<f32> = textureLoad(inputTexture, vec2<i32>(i32(src_x), i32(src_y)), 0);

  let base : u32 = index * channels;
  if (channels > 0u) {
    outputBuffer[base] = texel.x;
  }
  if (channels > 1u) {
    outputBuffer[base + 1u] = texel.y;
  }
  if (channels > 2u) {
    outputBuffer[base + 2u] = texel.z;
  }
  if (channels > 3u) {
    outputBuffer[base + 3u] = texel.w;
  }
}
