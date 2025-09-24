struct DegaussParams {
  width : f32,
  height : f32,
  channels : f32,
  pad0 : f32,
  pad1 : f32,
  pad2 : f32,
  pad3 : f32,
  pad4 : f32,
};

@group(0) @binding(0) var redTexture : texture_2d<f32>;
@group(0) @binding(1) var greenTexture : texture_2d<f32>;
@group(0) @binding(2) var blueTexture : texture_2d<f32>;
@group(0) @binding(3) var alphaTexture : texture_2d<f32>;
@group(0) @binding(4) var<storage, read_write> outputBuffer : array<f32>;
@group(0) @binding(5) var<uniform> params : DegaussParams;

fn asU32(value : f32) -> u32 {
  return u32(max(value, 0.0));
}

fn sampleChannel(tex : texture_2d<f32>, coords : vec2<i32>) -> f32 {
  return textureLoad(tex, coords, 0).x;
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
  let r : f32 = sampleChannel(redTexture, coords);
  let g : f32 = sampleChannel(greenTexture, coords);
  let b : f32 = sampleChannel(blueTexture, coords);
  let a : f32 = sampleChannel(alphaTexture, coords);

  let pixelIndex : u32 = gid.y * width + gid.x;
  let baseIndex : u32 = pixelIndex * channelCount;

  if (channelCount > 0u) {
    outputBuffer[baseIndex] = r;
  }
  if (channelCount > 1u) {
    outputBuffer[baseIndex + 1u] = g;
  }
  if (channelCount > 2u) {
    outputBuffer[baseIndex + 2u] = b;
  }
  if (channelCount > 3u) {
    outputBuffer[baseIndex + 3u] = a;
  }

  if (channelCount > 4u) {
    var idx : u32 = 4u;
    loop {
      if (idx >= channelCount) {
        break;
      }
      outputBuffer[baseIndex + idx] = a;
      idx = idx + 1u;
    }
  }
}
