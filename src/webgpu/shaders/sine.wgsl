struct SineParams {
  width: f32,
  height: f32,
  channels: f32,
  amount: f32,
  rgbMode: f32,
  padding0: f32,
  padding1: f32,
  padding2: f32,
};

@group(0) @binding(0) var inputTex : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> outputBuffer : array<f32>;
@group(0) @binding(2) var<uniform> params : SineParams;

fn asU32(value : f32) -> u32 {
  return u32(max(value, 0.0));
}

fn clamp01(value : f32) -> f32 {
  return clamp(value, 0.0, 1.0);
}

fn applySine(value : f32, amount : f32) -> f32 {
  return clamp01((sin(value * amount) + 1.0) * 0.5);
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
  let amount : f32 = params.amount;
  let useRgb : bool = params.rgbMode > 0.5;

  let sinR : f32 = applySine(texel.x, amount);
  let sinG : f32 = applySine(texel.y, amount);
  let sinB : f32 = applySine(texel.z, amount);
  let sinA : f32 = applySine(texel.w, amount);

  if (channelCount >= 1u) {
    if (channelCount <= 2u || useRgb) {
      outputBuffer[baseIndex] = sinR;
    } else {
      outputBuffer[baseIndex] = clamp01(texel.x);
    }
  }

  if (channelCount >= 2u) {
    if (channelCount == 2u) {
      outputBuffer[baseIndex + 1u] = clamp01(texel.y);
    } else if (useRgb) {
      outputBuffer[baseIndex + 1u] = sinG;
    } else {
      outputBuffer[baseIndex + 1u] = clamp01(texel.y);
    }
  }

  if (channelCount >= 3u) {
    outputBuffer[baseIndex + 2u] = sinB;
  }

  if (channelCount >= 4u) {
    outputBuffer[baseIndex + 3u] = clamp01(texel.w);
  }

  if (channelCount > 4u) {
    var ch : u32 = 4u;
    let fallback : f32 = channelCount > 3u ? clamp01(texel.w) : sinA;
    loop {
      if (ch >= channelCount) {
        break;
      }
      outputBuffer[baseIndex + ch] = fallback;
      ch = ch + 1u;
    }
  }
}
