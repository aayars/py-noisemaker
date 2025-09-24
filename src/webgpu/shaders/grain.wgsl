struct GrainParams {
  sizeAlpha : vec4<f32>;
  timeSpeedSeed : vec4<f32>;
};

@group(0) @binding(0) var inputTex : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> outputBuffer : array<f32>;
@group(0) @binding(2) var<uniform> params : GrainParams;

fn clamp01(value : f32) -> f32 {
  return clamp(value, 0.0, 1.0);
}

fn asU32(value : f32) -> u32 {
  return u32(max(value, 0.0));
}

fn hashNoise(coord : vec2<f32>, seed : f32, timeVal : f32, speedVal : f32) -> f32 {
  let sx = floor(coord.x);
  let sy = floor(coord.y);
  let phase = sx * 12.9898 + sy * 78.233 + seed * 37.719 + timeVal * speedVal * 0.1;
  return fract(sin(phase) * 43758.5453);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let width : u32 = asU32(params.sizeAlpha.x);
  let height : u32 = asU32(params.sizeAlpha.y);
  if (gid.x >= width || gid.y >= height) {
    return;
  }

  let channelCount : u32 = max(asU32(params.sizeAlpha.z), 1u);
  let alpha : f32 = clamp(params.sizeAlpha.w, 0.0, 1.0);
  if (alpha <= 0.0) {
    let texel = textureLoad(inputTex, vec2<i32>(i32(gid.x), i32(gid.y)), 0);
    let pixelIndex : u32 = (gid.y * width + gid.x) * channelCount;
    if (channelCount > 0u) {
      outputBuffer[pixelIndex] = texel.x;
    }
    if (channelCount > 1u) {
      outputBuffer[pixelIndex + 1u] = texel.y;
    }
    if (channelCount > 2u) {
      outputBuffer[pixelIndex + 2u] = texel.z;
    }
    if (channelCount > 3u) {
      outputBuffer[pixelIndex + 3u] = texel.w;
    }
    if (channelCount > 4u) {
      var ch : u32 = 4u;
      loop {
        if (ch >= channelCount) {
          break;
        }
        outputBuffer[pixelIndex + ch] = texel.w;
        ch = ch + 1u;
      }
    }
    return;
  }

  let coords = vec2<i32>(i32(gid.x), i32(gid.y));
  let texel = textureLoad(inputTex, coords, 0);
  let pixelIndex : u32 = (gid.y * width + gid.x) * channelCount;

  let noise = hashNoise(
    vec2<f32>(f32(gid.x), f32(gid.y)),
    params.timeSpeedSeed.z,
    params.timeSpeedSeed.x,
    params.timeSpeedSeed.y,
  );

  var components = array<f32, 4>(texel.x, texel.y, texel.z, texel.w);

  if (channelCount > 0u) {
    outputBuffer[pixelIndex] = clamp01(mix(components[0], noise, alpha));
  }
  if (channelCount > 1u) {
    outputBuffer[pixelIndex + 1u] = clamp01(mix(components[1], noise, alpha));
  }
  if (channelCount > 2u) {
    outputBuffer[pixelIndex + 2u] = clamp01(mix(components[2], noise, alpha));
  }
  if (channelCount > 3u) {
    outputBuffer[pixelIndex + 3u] = clamp01(mix(components[3], noise, alpha));
  }
  if (channelCount > 4u) {
    var ch : u32 = 4u;
    let fallback = clamp01(mix(components[3], noise, alpha));
    loop {
      if (ch >= channelCount) {
        break;
      }
      outputBuffer[pixelIndex + ch] = fallback;
      ch = ch + 1u;
    }
  }
}
