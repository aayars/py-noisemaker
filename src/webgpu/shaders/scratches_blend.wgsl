// Final blending stage for scratches. Elevates the input tensor wherever the
// mask indicates bright streaks, matching the CPU implementation's channel-wise
// maximum with the amplified mask value.

struct ScratchesBlendParams {
  size : vec4<f32>; // width, height, channels, unused
};

@group(0) @binding(0) var inputTexture : texture_2d<f32>;
@group(0) @binding(1) var maskTexture : texture_2d<f32>;
@group(0) @binding(2) var outputTexture : texture_storage_2d<rgba32float, write>;
@group(0) @binding(3) var<uniform> params : ScratchesBlendParams;

fn asDimension(value : f32) -> u32 {
  return u32(max(round(value), 0.0));
}

fn clamp01(value : f32) -> f32 {
  return clamp(value, 0.0, 1.0);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let width = asDimension(params.size.x);
  let height = asDimension(params.size.y);
  if (gid.x >= width || gid.y >= height) {
    return;
  }

  let channels = clamp(asDimension(params.size.z), 1u, 4u);
  let coord = vec2<i32>(i32(gid.x), i32(gid.y));

  let srcSample = textureLoad(inputTexture, coord, 0);
  let maskSample = textureLoad(maskTexture, coord, 0).x;
  let boost = clamp01(maskSample * 8.0);

  var components = array<f32, 4>(srcSample.x, srcSample.y, srcSample.z, srcSample.w);
  for (var ch : u32 = 0u; ch < channels; ch = ch + 1u) {
    components[ch] = clamp01(max(components[ch], boost));
  }

  let result = vec4<f32>(components[0], components[1], components[2], components[3]);
  textureStore(outputTexture, coord, result);
}
