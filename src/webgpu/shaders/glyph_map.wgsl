struct GlyphMapParams {
  width: f32;
  height: f32;
  channels: f32;
  glyphWidth: f32;
  glyphHeight: f32;
  glyphCount: f32;
  colorize: f32;
  zoom: f32;
  alpha: f32;
  _pad0: f32;
  _pad1: f32;
  _pad2: f32;
};

@group(0) @binding(0) var inputTexture : texture_2d<f32>;
@group(0) @binding(1) var glyphTexture : texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> outputBuffer : array<f32>;
@group(0) @binding(3) var<uniform> params : GlyphMapParams;

fn clamp01(value : f32) -> f32 {
  return clamp(value, 0.0, 1.0);
}

fn asU32(value : f32) -> u32 {
  if (value <= 0.0) {
    return 0u;
  }
  return u32(floor(value + 0.5));
}

fn readChannel(texel : vec4<f32>, index : u32) -> f32 {
  switch index {
    case 0u: { return texel.x; }
    case 1u: { return texel.y; }
    case 2u: { return texel.z; }
    default: { return texel.w; }
  }
}

fn computeLuminance(texel : vec4<f32>, channelCount : u32) -> f32 {
  if (channelCount <= 1u) {
    return texel.x;
  }
  if (channelCount == 2u) {
    return texel.x;
  }
  let rgb : vec3<f32> = vec3<f32>(texel.x, texel.y, texel.z);
  return clamp01(dot(rgb, vec3<f32>(0.2126, 0.7152, 0.0722)));
}

fn mixChannel(a : f32, b : f32, t : f32) -> f32 {
  return a + (b - a) * t;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let width : u32 = asU32(params.width);
  let height : u32 = asU32(params.height);
  if (width == 0u || height == 0u) {
    return;
  }
  if (gid.x >= width || gid.y >= height) {
    return;
  }

  let channelCount : u32 = max(asU32(params.channels), 1u);
  let glyphWidth : u32 = max(asU32(params.glyphWidth), 1u);
  let glyphHeight : u32 = max(asU32(params.glyphHeight), 1u);
  let glyphCount : u32 = asU32(params.glyphCount);
  let alphaFactor : f32 = clamp01(params.alpha);
  let colorize : bool = params.colorize > 0.5;
  let zoom : f32 = max(params.zoom, 1.0e-5);

  let pixelIndex : u32 = gid.y * width + gid.x;
  let baseIndex : u32 = pixelIndex * channelCount;
  let srcCoords : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
  let srcTexel : vec4<f32> = textureLoad(inputTexture, srcCoords, 0);

  if (glyphCount == 0u) {
    for (var ch : u32 = 0u; ch < channelCount; ch = ch + 1u) {
      outputBuffer[baseIndex + ch] = readChannel(srcTexel, ch);
    }
    return;
  }

  let invZoom : f32 = 1.0 / zoom;
  let inWidth : u32 = max(asU32(f32(width) * invZoom), 1u);
  let inHeight : u32 = max(asU32(f32(height) * invZoom), 1u);

  var uvWidth : u32 = inWidth / glyphWidth;
  if (uvWidth == 0u) {
    uvWidth = 1u;
  }
  var uvHeight : u32 = inHeight / glyphHeight;
  if (uvHeight == 0u) {
    uvHeight = 1u;
  }

  let approxWidth : u32 = max(glyphWidth * uvWidth, 1u);
  let approxHeight : u32 = max(glyphHeight * uvHeight, 1u);

  let widthF : f32 = f32(width);
  let heightF : f32 = f32(height);
  let approxWidthF : f32 = f32(approxWidth);
  let approxHeightF : f32 = f32(approxHeight);

  let approxXF : f32 = (f32(gid.x) + 0.5) / max(widthF, 1.0) * approxWidthF;
  let approxYF : f32 = (f32(gid.y) + 0.5) / max(heightF, 1.0) * approxHeightF;
  let approxX : u32 = min(u32(floor(approxXF)), approxWidth - 1u);
  let approxY : u32 = min(u32(floor(approxYF)), approxHeight - 1u);

  let glyphLocalX : u32 = approxX % glyphWidth;
  let glyphLocalY : u32 = approxY % glyphHeight;
  let cellX : u32 = min(approxX / glyphWidth, max(uvWidth, 1u) - 1u);
  let cellY : u32 = min(approxY / glyphHeight, max(uvHeight, 1u) - 1u);

  let uvWidthF : f32 = f32(max(uvWidth, 1u));
  let uvHeightF : f32 = f32(max(uvHeight, 1u));
  let cellCenterXF : f32 = (f32(cellX) + 0.5) / uvWidthF;
  let cellCenterYF : f32 = (f32(cellY) + 0.5) / uvHeightF;

  let inWidthF : f32 = f32(inWidth);
  let inHeightF : f32 = f32(inHeight);
  let sampleInXF : f32 = clamp(cellCenterXF * inWidthF, 0.0, max(inWidthF - 1.0, 0.0));
  let sampleInYF : f32 = clamp(cellCenterYF * inHeightF, 0.0, max(inHeightF - 1.0, 0.0));

  let sampleOrigXF : f32 = clamp((sampleInXF + 0.5) * zoom, 0.0, max(widthF - 1.0, 0.0));
  let sampleOrigYF : f32 = clamp((sampleInYF + 0.5) * zoom, 0.0, max(heightF - 1.0, 0.0));

  let sampleCoords : vec2<i32> = vec2<i32>(i32(sampleOrigXF), i32(sampleOrigYF));
  let sampleTexel : vec4<f32> = textureLoad(inputTexture, sampleCoords, 0);
  let luminanceValue : f32 = computeLuminance(sampleTexel, channelCount);
  let clampedValue : f32 = clamp01(luminanceValue);

  var glyphIndex : i32 = i32(floor(clampedValue * f32(glyphCount)));
  let maxGlyphIndex : i32 = i32(glyphCount) - 1;
  if (glyphIndex > maxGlyphIndex) {
    glyphIndex = maxGlyphIndex;
  }
  if (glyphIndex < 0) {
    glyphIndex = 0;
  }

  let glyphSampleY : i32 = glyphIndex * i32(glyphHeight) + i32(glyphLocalY);
  let glyphSampleCoords : vec2<i32> = vec2<i32>(i32(glyphLocalX), glyphSampleY);
  let glyphTexel : vec4<f32> = textureLoad(glyphTexture, glyphSampleCoords, 0);
  let glyphValue : f32 = clamp01(glyphTexel.x);

  for (var ch : u32 = 0u; ch < channelCount; ch = ch + 1u) {
    var effectChannel : f32;
    if (colorize) {
      effectChannel = glyphValue * readChannel(sampleTexel, ch);
    } else {
      effectChannel = glyphValue;
    }
    let resultChannel : f32 = mixChannel(readChannel(srcTexel, ch), effectChannel, alphaFactor);
    outputBuffer[baseIndex + ch] = resultChannel;
  }
}
