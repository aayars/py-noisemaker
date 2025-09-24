struct PosterizeParams {
  width: f32;
  height: f32;
  channels: f32;
  levels: f32;
};

@group(0) @binding(0) var inputTex : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> outputBuffer : array<f32>;
@group(0) @binding(2) var<uniform> params : PosterizeParams;

fn asU32(value : f32) -> u32 {
  return u32(max(value, 0.0));
}

fn srgbToLinear(value : f32) -> f32 {
  if (value <= 0.04045) {
    return value / 12.92;
  }
  return pow((value + 0.055) / 1.055, 2.4);
}

fn linearToSrgb(value : f32) -> f32 {
  let threshold : f32 = 0.0031308;
  if (value <= threshold) {
    return value * 12.92;
  }
  return 1.055 * pow(value, 1.0 / 2.4) - 0.055;
}

fn quantizeValue(value : f32, factor : f32, invFactor : f32, halfStep : f32) -> f32 {
  let scaled : f32 = value * factor;
  let shifted : f32 = scaled + halfStep;
  let quantized : f32 = floor(shifted);
  return quantized * invFactor;
}

fn processChannel(
  value : f32,
  convert : bool,
  hasLevels : bool,
  factor : f32,
  invFactor : f32,
  halfStep : f32,
) -> f32 {
  if (!hasLevels) {
    return value;
  }
  var working : f32 = value;
  if (convert) {
    working = srgbToLinear(working);
  }
  var quantized : f32 = quantizeValue(working, factor, invFactor, halfStep);
  if (convert) {
    quantized = linearToSrgb(quantized);
  }
  return quantized;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let width : u32 = asU32(params.width);
  let height : u32 = asU32(params.height);
  if (gid.x >= width || gid.y >= height) {
    return;
  }

  let channelCount : u32 = asU32(params.channels);
  if (channelCount == 0u) {
    return;
  }

  let coords : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
  let texel : vec4<f32> = textureLoad(inputTex, coords, 0);
  let pixelIndex : u32 = gid.y * width + gid.x;
  let baseIndex : u32 = pixelIndex * channelCount;

  let factor : f32 = params.levels;
  let hasLevels : bool = factor != 0.0;
  var invFactor : f32 = 0.0;
  if (hasLevels) {
    invFactor = 1.0 / factor;
  }
  let halfStep : f32 = invFactor * 0.5;
  let convertColor : bool = channelCount == 3u;

  if (channelCount > 0u) {
    outputBuffer[baseIndex] = processChannel(
      texel.x,
      convertColor,
      hasLevels,
      factor,
      invFactor,
      halfStep,
    );
  }
  if (channelCount > 1u) {
    outputBuffer[baseIndex + 1u] = processChannel(
      texel.y,
      convertColor,
      hasLevels,
      factor,
      invFactor,
      halfStep,
    );
  }
  if (channelCount > 2u) {
    outputBuffer[baseIndex + 2u] = processChannel(
      texel.z,
      convertColor,
      hasLevels,
      factor,
      invFactor,
      halfStep,
    );
  }

  var fallback : f32 = 0.0;
  if (channelCount > 3u) {
    fallback = processChannel(
      texel.w,
      false,
      hasLevels,
      factor,
      invFactor,
      halfStep,
    );
    outputBuffer[baseIndex + 3u] = fallback;
  }
  if (channelCount > 4u) {
    var ch : u32 = 4u;
    loop {
      if (ch >= channelCount) {
        break;
      }
      outputBuffer[baseIndex + ch] = fallback;
      ch = ch + 1u;
    }
  }
}
