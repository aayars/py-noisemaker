// Applies the fibers overlay by blending the base tensor with a brightness
// tensor using a worm mask scaled by a constant factor. The shader mirrors the
// CPU implementation's per-channel blending logic while supporting arbitrary
// channel counts for the inputs.

struct FibersParams {
  size : vec4<f32>;      // width, height, baseChannels, brightnessChannels
  factors : vec4<f32>;   // maskChannels, maskScale, unused, unused
};

@group(0) @binding(0) var baseTexture : texture_2d<f32>;
@group(0) @binding(1) var brightnessTexture : texture_2d<f32>;
@group(0) @binding(2) var maskTexture : texture_2d<f32>;
@group(0) @binding(3) var outputTexture : texture_storage_2d<rgba32float, write>;
@group(0) @binding(4) var<uniform> params : FibersParams;

fn toDimension(value : f32) -> u32 {
  return u32(max(round(value), 0.0));
}

fn getComponent(value : vec4<f32>, index : u32) -> f32 {
  switch index {
    case 0u: { return value.x; }
    case 1u: { return value.y; }
    case 2u: { return value.z; }
    default: { return value.w; }
  }
}

fn sampleChannel(sample : vec4<f32>, channel : u32, channelCount : u32) -> f32 {
  let capped = min(channel, max(channelCount, 1u) - 1u);
  return getComponent(sample, capped);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let width = toDimension(params.size.x);
  let height = toDimension(params.size.y);
  if (gid.x >= width || gid.y >= height) {
    return;
  }

  let baseChannels = clamp(toDimension(params.size.z), 1u, 4u);
  let brightnessChannels = clamp(toDimension(params.size.w), 1u, 4u);
  let maskChannels = clamp(toDimension(params.factors.x), 1u, 4u);
  let maskScale = params.factors.y;

  let coord = vec2<i32>(i32(gid.x), i32(gid.y));
  let baseSample = textureLoad(baseTexture, coord, 0);
  let brightnessSample = textureLoad(brightnessTexture, coord, 0);
  let maskSample = textureLoad(maskTexture, coord, 0);

  var result = baseSample;
  for (var channel : u32 = 0u; channel < baseChannels; channel = channel + 1u) {
    let baseValue = sampleChannel(baseSample, channel, baseChannels);
    let brightnessValue = sampleChannel(brightnessSample, channel, brightnessChannels);
    let maskValue = sampleChannel(maskSample, channel, maskChannels);
    let scaledMask = clamp(maskValue * maskScale, 0.0, 1.0);
    result[channel] = baseValue * (1.0 - scaledMask) + brightnessValue * scaledMask;
  }

  textureStore(outputTexture, coord, result);
}
