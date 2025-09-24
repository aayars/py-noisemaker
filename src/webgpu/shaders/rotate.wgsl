struct RotateParams {
  width : f32,
  height : f32,
  channels : f32,
  angle : f32,
  pad0 : f32,
  pad1 : f32,
  pad2 : f32,
  pad3 : f32,
};

@group(0) @binding(0) var inputTexture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> outputBuffer : array<f32>;
@group(0) @binding(2) var<uniform> params : RotateParams;

fn asU32(value : f32) -> u32 {
  return u32(max(value, 0.0));
}

fn wrapIndex(value : i32, size : i32) -> i32 {
  if (size <= 0) {
    return 0;
  }
  var wrapped : i32 = value % size;
  if (wrapped < 0) {
    wrapped = wrapped + size;
  }
  return wrapped;
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

  let widthF : f32 = params.width;
  let heightF : f32 = params.height;
  if (widthF <= 0.0 || heightF <= 0.0) {
    return;
  }

  let cosAngle : f32 = cos(params.angle);
  let sinAngle : f32 = sin(params.angle);

  let dx : f32 = f32(gid.x) / widthF - 0.5;
  let dy : f32 = f32(gid.y) / heightF - 0.5;

  let sampleXF : f32 = (cosAngle * dx + sinAngle * dy + 0.5) * widthF;
  let sampleYF : f32 = (-sinAngle * dx + cosAngle * dy + 0.5) * heightF;

  let sx : i32 = wrapIndex(i32(trunc(sampleXF)), i32(width));
  let sy : i32 = wrapIndex(i32(trunc(sampleYF)), i32(height));
  let coords : vec2<i32> = vec2<i32>(sx, sy);
  let sample : vec4<f32> = textureLoad(inputTexture, coords, 0);

  let pixelIndex : u32 = (gid.y * width + gid.x) * channelCount;

  if (channelCount > 0u) {
    outputBuffer[pixelIndex] = sample.x;
  }
  if (channelCount > 1u) {
    outputBuffer[pixelIndex + 1u] = sample.y;
  }
  if (channelCount > 2u) {
    outputBuffer[pixelIndex + 2u] = sample.z;
  }
  if (channelCount > 3u) {
    outputBuffer[pixelIndex + 3u] = sample.w;
  }
  if (channelCount > 4u) {
    var ch : u32 = 4u;
    let fallback : f32 = sample.w;
    loop {
      if (ch >= channelCount) {
        break;
      }
      outputBuffer[pixelIndex + ch] = fallback;
      ch = ch + 1u;
    }
  }
}
