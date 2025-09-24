struct WobbleParams {
  sizeChannels : vec4<f32>;
  offset : vec4<f32>;
};

@group(0) @binding(0) var inputTex : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> outputBuffer : array<f32>;
@group(0) @binding(2) var<uniform> params : WobbleParams;

fn asU32(value : f32) -> u32 {
  return u32(max(value, 0.0));
}

fn wrapIndex(coord : i32, offset : i32, size : i32) -> i32 {
  if (size <= 0) {
    return 0;
  }
  let modValue = (coord + offset) % size;
  return (modValue + size) % size;
}

fn storeTexel(baseIndex : u32, channelCount : u32, texel : vec4<f32>) {
  if (channelCount > 0u) {
    outputBuffer[baseIndex] = texel.x;
  }
  if (channelCount > 1u) {
    outputBuffer[baseIndex + 1u] = texel.y;
  }
  if (channelCount > 2u) {
    outputBuffer[baseIndex + 2u] = texel.z;
  }
  if (channelCount > 3u) {
    outputBuffer[baseIndex + 3u] = texel.w;
  }
  if (channelCount > 4u) {
    var ch : u32 = 4u;
    let fallback = texel.w;
    loop {
      if (ch >= channelCount) {
        break;
      }
      outputBuffer[baseIndex + ch] = fallback;
      ch = ch + 1u;
    }
  }
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let width : u32 = asU32(params.sizeChannels.x);
  let height : u32 = asU32(params.sizeChannels.y);
  if (gid.x >= width || gid.y >= height) {
    return;
  }

  let channelCount : u32 = asU32(params.sizeChannels.z);
  if (channelCount == 0u) {
    return;
  }

  let xOff : i32 = i32(params.offset.x);
  let yOff : i32 = i32(params.offset.y);
  let wrappedX : i32 = wrapIndex(i32(gid.x), xOff, i32(width));
  let wrappedY : i32 = wrapIndex(i32(gid.y), yOff, i32(height));
  let texel : vec4<f32> = textureLoad(inputTex, vec2<i32>(wrappedX, wrappedY), 0);

  let pixelIndex : u32 = gid.y * width + gid.x;
  let baseIndex : u32 = pixelIndex * channelCount;
  storeTexel(baseIndex, channelCount, texel);
}
