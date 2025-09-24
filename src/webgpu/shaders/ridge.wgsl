struct RidgeParams {
  width: f32;
  height: f32;
  channels: f32;
};

@group(0) @binding(0) var inputTex : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> outputBuffer : array<f32>;
@group(0) @binding(2) var<uniform> params : RidgeParams;

fn asU32(value : f32) -> u32 {
  return u32(max(value, 0.0));
}

fn ridgeTransform(value : f32) -> f32 {
  return 1.0 - abs(value * 2.0 - 1.0);
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

  if (channelCount > 0u) {
    outputBuffer[baseIndex] = ridgeTransform(texel.x);
  }
  if (channelCount > 1u) {
    outputBuffer[baseIndex + 1u] = ridgeTransform(texel.y);
  }
  if (channelCount > 2u) {
    outputBuffer[baseIndex + 2u] = ridgeTransform(texel.z);
  }

  var fallback : f32 = 0.0;
  if (channelCount > 3u) {
    fallback = ridgeTransform(texel.w);
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
