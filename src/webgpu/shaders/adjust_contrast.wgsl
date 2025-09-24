struct AdjustContrastParams {
  width: f32,
  height: f32,
  channels: f32,
  amount: f32,
  mean_r: f32,
  mean_g: f32,
  mean_b: f32,
  mean_a: f32,
};

@group(0) @binding(0) var inputTex : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> outputBuffer : array<f32>;
@group(0) @binding(2) var<uniform> params : AdjustContrastParams;

fn clamp01(value : f32) -> f32 {
  return clamp(value, 0.0, 1.0);
}

fn asU32(value : f32) -> u32 {
  return u32(max(value, 0.0));
}

fn applyContrast(value : f32, mean : f32, amount : f32) -> f32 {
  return clamp01((value - mean) * amount + mean);
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

  if (channelCount > 0u) {
    outputBuffer[baseIndex] = applyContrast(texel.x, params.mean_r, amount);
  }
  if (channelCount > 1u) {
    outputBuffer[baseIndex + 1u] = applyContrast(texel.y, params.mean_g, amount);
  }
  if (channelCount > 2u) {
    outputBuffer[baseIndex + 2u] = applyContrast(texel.z, params.mean_b, amount);
  }
  if (channelCount > 3u) {
    outputBuffer[baseIndex + 3u] = applyContrast(texel.w, params.mean_a, amount);
  }

  if (channelCount > 4u) {
    var ch : u32 = 4u;
    let fallbackValue : f32 = applyContrast(texel.w, params.mean_a, amount);
    loop {
      if (ch >= channelCount) {
        break;
      }
      outputBuffer[baseIndex + ch] = fallbackValue;
      ch = ch + 1u;
    }
  }
}
