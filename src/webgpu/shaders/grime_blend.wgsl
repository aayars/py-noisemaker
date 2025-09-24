// Final blending stage for the grime effect. Applies the gate mask, mixes in
// additional noise, multiplies by the speckle map, and blends the result with
// the original tensor using the scaled derivative mask. Matches the sequence of
// CPU operations performed at the end of `grimeCPU`.

struct BlendParams {
  size : vec4<f32>; // width, height, channels, unused
};

@group(0) @binding(0) var inputTexture : texture_2d<f32>;
@group(0) @binding(1) var maskTexture : texture_2d<f32>;
@group(0) @binding(2) var noiseTexture : texture_2d<f32>;
@group(0) @binding(3) var speckTexture : texture_2d<f32>;
@group(0) @binding(4) var outputTexture : texture_storage_2d<rgba32float, write>;
@group(0) @binding(5) var<uniform> params : BlendParams;

fn clamp01(value : f32) -> f32 {
  return clamp(value, 0.0, 1.0);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let width = u32(round(params.size.x));
  let height = u32(round(params.size.y));
  if (gid.x >= width || gid.y >= height) {
    return;
  }

  let channels = clamp(u32(round(params.size.z)), 1u, 4u);
  let coord = vec2<i32>(i32(gid.x), i32(gid.y));

  let srcSample = textureLoad(inputTexture, coord, 0);
  let maskSample = textureLoad(maskTexture, coord, 0).x;
  let noiseSample = textureLoad(noiseTexture, coord, 0).x;
  let speckSample = textureLoad(speckTexture, coord, 0).x;

  let gate = clamp(maskSample * maskSample * 0.075, 0.0, 1.0);
  let maskBlend = clamp(maskSample * 0.75, 0.0, 1.0);
  let noiseMix = 0.075;

  var result = srcSample;
  var components = array<f32, 4>(srcSample.x, srcSample.y, srcSample.z, srcSample.w);

  for (var ch : u32 = 0u; ch < channels; ch = ch + 1u) {
    let baseValue = components[ch];
    let dusty = mix(baseValue, 0.25, gate);
    let noisy = mix(dusty, noiseSample, noiseMix);
    let speckled = noisy * speckSample;
    let finalValue = mix(baseValue, speckled, maskBlend);
    components[ch] = clamp01(finalValue);
  }

  result = vec4<f32>(components[0], components[1], components[2], components[3]);
  textureStore(outputTexture, coord, result);
}
