struct ConvolutionParams {
  sizeKernel : vec4<f32>;
  extras : vec4<f32>;
};

@group(0) @binding(0) var inputTex : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> outputBuffer : array<f32>;
@group(0) @binding(2) var<storage, read> kernelBuffer : array<f32>;
@group(0) @binding(3) var<uniform> params : ConvolutionParams;

fn asU32(value : f32) -> u32 {
  return u32(max(value, 0.0));
}

fn asI32(value : f32) -> i32 {
  return i32(max(value, 0.0));
}

fn writeChannels(baseIndex : u32, channelCount : u32, accum : vec4<f32>, fallback : f32) {
  if (channelCount == 0u) {
    return;
  }
  outputBuffer[baseIndex] = accum.x;
  if (channelCount == 1u) {
    return;
  }
  outputBuffer[baseIndex + 1u] = accum.y;
  if (channelCount == 2u) {
    return;
  }
  outputBuffer[baseIndex + 2u] = accum.z;
  if (channelCount == 3u) {
    return;
  }
  outputBuffer[baseIndex + 3u] = accum.w;
  if (channelCount <= 4u) {
    return;
  }
  var channel : u32 = 4u;
  loop {
    if (channel >= channelCount) {
      break;
    }
    outputBuffer[baseIndex + channel] = fallback;
    channel = channel + 1u;
  }
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let width : u32 = asU32(params.sizeKernel.x + 0.5);
  let height : u32 = asU32(params.sizeKernel.y + 0.5);
  if (gid.x >= width || gid.y >= height) {
    return;
  }

  let channels : u32 = max(asU32(params.sizeKernel.z + 0.5), 1u);
  let kernelWidth : u32 = max(asU32(params.sizeKernel.w + 0.5), 1u);
  let kernelHeight : u32 = max(asU32(params.extras.x + 0.5), 1u);

  let tileHeight : u32 = height * 2u;
  let tileWidth : u32 = width * 2u;
  let outHeight : u32 = tileHeight + 1u - kernelHeight;
  let outWidth : u32 = tileWidth + 1u - kernelWidth;

  let heightI : i32 = asI32(params.sizeKernel.y + 0.5);
  let widthI : i32 = asI32(params.sizeKernel.x + 0.5);
  let outHeightI : i32 = i32(outHeight);
  let outWidthI : i32 = i32(outWidth);
  var cropY : i32 = 0;
  var cropX : i32 = 0;
  var padY : i32 = 0;
  var padX : i32 = 0;
  let diffY : i32 = outHeightI - heightI;
  if (diffY > 0) {
    cropY = diffY / 2;
  } else if (diffY < 0) {
    padY = (-diffY) / 2;
  }
  let diffX : i32 = outWidthI - widthI;
  if (diffX > 0) {
    cropX = diffX / 2;
  } else if (diffX < 0) {
    padX = (-diffX) / 2;
  }

  let yI : i32 = i32(gid.y);
  let xI : i32 = i32(gid.x);
  let srcY : i32 = yI + cropY - padY;
  let srcX : i32 = xI + cropX - padX;

  let channelCount : u32 = channels;
  let baseIndex : u32 = (gid.y * width + gid.x) * channelCount;

  if (srcY < 0 || srcY >= outHeightI || srcX < 0 || srcX >= outWidthI) {
    if (channelCount == 0u) {
      return;
    }
    outputBuffer[baseIndex] = 0.0;
    if (channelCount == 1u) { return; }
    outputBuffer[baseIndex + 1u] = 0.0;
    if (channelCount == 2u) { return; }
    outputBuffer[baseIndex + 2u] = 0.0;
    if (channelCount == 3u) { return; }
    outputBuffer[baseIndex + 3u] = 0.0;
    if (channelCount <= 4u) { return; }
    var extra : u32 = 4u;
    loop {
      if (extra >= channelCount) {
        break;
      }
      outputBuffer[baseIndex + extra] = 0.0;
      extra = extra + 1u;
    }
    return;
  }

  let halfHeight : u32 = height / 2u;
  let halfWidth : u32 = width / 2u;

  var accum : vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  var extraAccum : f32 = 0.0;
  var kernelIndex : u32 = 0u;

  for (var ky : u32 = 0u; ky < kernelHeight; ky = ky + 1u) {
    let yy : u32 = u32(srcY) + ky;
    let sampleY : u32 = (yy + halfHeight) % height;
    for (var kx : u32 = 0u; kx < kernelWidth; kx = kx + 1u) {
      let xx : u32 = u32(srcX) + kx;
      let sampleX : u32 = (xx + halfWidth) % width;
      let texel : vec4<f32> = textureLoad(
        inputTex,
        vec2<i32>(i32(sampleX), i32(sampleY)),
        0,
      );
      let weight : f32 = kernelBuffer[kernelIndex];
      accum = accum + texel * weight;
      if (channelCount > 4u) {
        extraAccum = extraAccum + texel.w * weight;
      }
      kernelIndex = kernelIndex + 1u;
    }
  }

  let fallback : f32 = extraAccum;
  if (channelCount <= 4u) {
    fallback = accum.w;
  }
  writeChannels(baseIndex, channelCount, accum, fallback);
}
