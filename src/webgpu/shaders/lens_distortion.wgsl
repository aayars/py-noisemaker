struct LensDistortionParams {
  width : f32,
  height : f32,
  channels : f32,
  displacement : f32,
  pad0 : f32,
  pad1 : f32,
  pad2 : f32,
  pad3 : f32,
};

@group(0) @binding(0) var inputTex : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> outputBuffer : array<f32>;
@group(0) @binding(2) var<uniform> params : LensDistortionParams;

const MAX_DIST : f32 = sqrt(0.5 * 0.5 + 0.5 * 0.5);

fn asU32(value : f32) -> u32 {
  return u32(max(value, 0.0));
}

fn wrapIndex(value : f32, size : u32) -> i32 {
  if (size == 0u) {
    return 0;
  }
  var idx : i32 = i32(value);
  let sizeI : i32 = i32(size);
  var wrapped : i32 = idx % sizeI;
  if (wrapped < 0) {
    wrapped = wrapped + sizeI;
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

  let widthF : f32 = f32(width);
  let heightF : f32 = f32(height);
  let displacement : f32 = params.displacement;
  let zoom : f32 = select(0.0, displacement * -0.25, displacement < 0.0);

  let xIndex : f32 = f32(gid.x) / widthF;
  let yIndex : f32 = f32(gid.y) / heightF;
  let xDist : f32 = xIndex - 0.5;
  let yDist : f32 = yIndex - 0.5;
  var centerDist : f32 = 1.0 - sqrt(xDist * xDist + yDist * yDist) / MAX_DIST;
  centerDist = clamp(centerDist, 0.0, 1.0);
  let centerDistSq : f32 = centerDist * centerDist;

  let xOff : f32 = (
    xIndex -
    xDist * zoom -
    xDist * centerDistSq * displacement
  ) * widthF;
  let yOff : f32 = (
    yIndex -
    yDist * zoom -
    yDist * centerDistSq * displacement
  ) * heightF;

  let xi : i32 = wrapIndex(xOff, width);
  let yi : i32 = wrapIndex(yOff, height);
  let sampleCoords : vec2<i32> = vec2<i32>(xi, yi);
  let texel : vec4<f32> = textureLoad(inputTex, sampleCoords, 0);

  let pixelIndex : u32 = gid.y * width + gid.x;
  let baseIndex : u32 = pixelIndex * channelCount;

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
    loop {
      if (ch >= channelCount) {
        break;
      }
      outputBuffer[baseIndex + ch] = texel.w;
      ch = ch + 1u;
    }
  }
}
