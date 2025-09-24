struct VHSParams {
  dims : vec4<f32>,
  extras : vec4<f32>,
};

@group(0) @binding(0) var inputTexture : texture_2d<f32>;
@group(0) @binding(1) var scanTexture : texture_2d<f32>;
@group(0) @binding(2) var gradTexture : texture_2d<f32>;
@group(0) @binding(3) var<storage, read_write> outputBuffer : array<f32>;
@group(0) @binding(4) var<uniform> params : VHSParams;

fn clamp01(value : f32) -> f32 {
  return clamp(value, 0.0, 1.0);
}

fn compute_g(raw_value : f32) -> f32 {
  var g : f32 = raw_value - 0.5;
  g = max(g, 0.0);
  g = min(g * 2.0, 1.0);
  return g;
}

fn wrap_index(value : i32, width : i32) -> i32 {
  if (width <= 0) {
    return 0;
  }
  var wrapped : i32 = value % width;
  if (wrapped < 0) {
    wrapped = wrapped + width;
  }
  return wrapped;
}

fn channel_value(texel : vec4<f32>, channel : u32) -> f32 {
  if (channel == 0u) {
    return texel.x;
  }
  if (channel == 1u) {
    return texel.y;
  }
  if (channel == 2u) {
    return texel.z;
  }
  return texel.w;
}

fn expand_scalar(value : f32) -> vec4<f32> {
  return vec4<f32>(value, value, value, value);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let width : u32 = u32(params.dims.x);
  let height : u32 = u32(params.dims.y);
  let channels : u32 = u32(params.dims.z);

  if (gid.x >= width || gid.y >= height || channels == 0u) {
    return;
  }

  let coords : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
  let grad_texel : vec4<f32> = textureLoad(gradTexture, coords, 0);
  let scan_texel : vec4<f32> = textureLoad(scanTexture, coords, 0);

  let g : f32 = compute_g(grad_texel.x);
  let noise : f32 = clamp01(scan_texel.x);

  let shift_amount : i32 = i32(floor(noise * f32(width) * g * g));
  let src_x : i32 = wrap_index(i32(gid.x) - shift_amount, i32(width));
  let sample_coords : vec2<i32> = vec2<i32>(src_x, i32(gid.y));

  let src_texel : vec4<f32> = textureLoad(inputTexture, sample_coords, 0);
  let src_grad_texel : vec4<f32> = textureLoad(gradTexture, sample_coords, 0);
  let src_scan_texel : vec4<f32> = textureLoad(scanTexture, sample_coords, 0);

  let blend_strength : f32 = compute_g(src_grad_texel.x);
  let blend_noise : f32 = clamp01(src_scan_texel.x);

  let blended : vec4<f32> = src_texel * (1.0 - blend_strength) +
    expand_scalar(blend_noise) * blend_strength;

  let base_index : u32 = (gid.y * width + gid.x) * channels;

  var channel : u32 = 0u;
  loop {
    if (channel >= channels) {
      break;
    }
    outputBuffer[base_index + channel] = channel_value(blended, channel);
    channel = channel + 1u;
  }
}
