struct ShadowParams {
  width : f32;
  height : f32;
  channels : f32;
  alpha : f32;
  stage : f32;
  refChannels : f32;
  shadeBlend : f32;
  flags : f32;
};

@group(0) @binding(0) var srcTex : texture_2d<f32>;
@group(0) @binding(1) var refTex : texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> gradientBuffer : array<f32>;
@group(0) @binding(3) var<storage, read_write> shadeBuffer : array<f32>;
@group(0) @binding(4) var<storage, read_write> outputBuffer : array<f32>;
@group(0) @binding(5) var<storage, read_write> reduceBuffer : array<f32>;
@group(0) @binding(6) var<uniform> params : ShadowParams;

const SOBEL_X : array<f32, 9> = array<f32, 9>(
  1.0, 0.0, -1.0,
  2.0, 0.0, -2.0,
  1.0, 0.0, -1.0,
);

const SOBEL_Y : array<f32, 9> = array<f32, 9>(
  1.0, 2.0, 1.0,
  0.0, 0.0, 0.0,
  -1.0, -2.0, -1.0,
);

const SHARPEN_KERNEL : array<f32, 9> = array<f32, 9>(
  0.0, -1.0, 0.0,
  -1.0, 5.0, -1.0,
  0.0, -1.0, 0.0,
);

const FLT_MAX : f32 = 3.40282347e38;
const FLT_MIN : f32 = -3.40282347e38;

fn asU32(value : f32) -> u32 {
  return u32(max(value, 0.0));
}

fn wrapCoord(value : i32, size : i32) -> i32 {
  if (size <= 0) {
    return 0;
  }
  var modVal : i32 = value % size;
  if (modVal < 0) {
    modVal = modVal + size;
  }
  return modVal;
}

fn srgbToLinear(c : f32) -> f32 {
  if (c <= 0.04045) {
    return c / 12.92;
  }
  let base : f32 = max((c + 0.055) / 1.055, 0.0);
  return pow(base, 2.4);
}

fn cbrtSafe(v : f32) -> f32 {
  return pow(max(v, 0.0), 1.0 / 3.0);
}

fn computeOklabL(rgb : vec3<f32>) -> f32 {
  let clamped : vec3<f32> = clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0));
  let lin : vec3<f32> = vec3<f32>(
    srgbToLinear(clamped.x),
    srgbToLinear(clamped.y),
    srgbToLinear(clamped.z),
  );
  let lVal : f32 = 0.4121656120 * lin.x + 0.5362752080 * lin.y + 0.0514575653 * lin.z;
  let mVal : f32 = 0.2118591070 * lin.x + 0.6807189584 * lin.y + 0.1074065790 * lin.z;
  let sVal : f32 = 0.0883097947 * lin.x + 0.2818474174 * lin.y + 0.6302613616 * lin.z;
  let lC : f32 = cbrtSafe(lVal);
  let mC : f32 = cbrtSafe(mVal);
  let sC : f32 = cbrtSafe(sVal);
  return 0.2104542553 * lC + 0.7936177850 * mC - 0.0040720468 * sC;
}

fn getReferenceValue(x : i32, y : i32, width : i32, height : i32, refChannels : i32) -> f32 {
  if (width <= 0 || height <= 0) {
    return 0.0;
  }
  let xi : i32 = wrapCoord(x, width);
  let yi : i32 = wrapCoord(y, height);
  let sample : vec4<f32> = textureLoad(refTex, vec2<i32>(xi, yi), 0);
  if (refChannels <= 1) {
    return sample.x;
  }
  if (refChannels == 2) {
    return sample.x;
  }
  return computeOklabL(sample.xyz);
}

fn sampleNormalizedRef(x : i32, y : i32, width : i32, height : i32) -> f32 {
  if (width <= 0 || height <= 0) {
    return 0.0;
  }
  let xi : i32 = wrapCoord(x, width);
  let yi : i32 = wrapCoord(y, height);
  let idx : u32 = u32(yi * width + xi);
  return shadeBuffer[idx];
}

fn sampleShadeNormalized(
  x : i32,
  y : i32,
  width : i32,
  height : i32,
  minShade : f32,
  invShadeRange : f32,
) -> f32 {
  if (width <= 0 || height <= 0) {
    return 0.0;
  }
  let xi : i32 = wrapCoord(x, width);
  let yi : i32 = wrapCoord(y, height);
  let idx : u32 = u32(yi * width + xi) * 2u;
  let raw : f32 = gradientBuffer[idx];
  if (invShadeRange == 0.0) {
    return 0.0;
  }
  return clamp((raw - minShade) * invShadeRange, 0.0, 1.0);
}

fn getChannelValue(color : vec4<f32>, index : u32) -> f32 {
  switch index {
    case 0u: {
      return color.x;
    }
    case 1u: {
      return color.y;
    }
    case 2u: {
      return color.z;
    }
    default: {
      return color.w;
    }
  }
}

fn mixf(a : f32, b : f32, t : f32) -> f32 {
  return a * (1.0 - t) + b * t;
}

fn clampVec3(v : vec3<f32>) -> vec3<f32> {
  return clamp(v, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn rgbToHsv(rgb : vec3<f32>) -> vec3<f32> {
  let clamped : vec3<f32> = clampVec3(rgb);
  let r : f32 = clamped.x;
  let g : f32 = clamped.y;
  let b : f32 = clamped.z;
  let maxVal : f32 = max(max(r, g), b);
  let minVal : f32 = min(min(r, g), b);
  let delta : f32 = maxVal - minVal;
  var hue : f32 = 0.0;
  if (delta != 0.0) {
    if (maxVal == r) {
      hue = (g - b) / delta;
      if (hue < 0.0) {
        hue = hue + 6.0;
      }
    } else if (maxVal == g) {
      hue = (b - r) / delta + 2.0;
    } else {
      hue = (r - g) / delta + 4.0;
    }
    hue = hue / 6.0;
    if (hue < 0.0) {
      hue = hue + 1.0;
    }
  }
  var saturation : f32 = 0.0;
  if (maxVal != 0.0) {
    saturation = delta / maxVal;
  }
  return vec3<f32>(hue, saturation, maxVal);
}

fn hsvToRgb(hsv : vec3<f32>) -> vec3<f32> {
  let H : f32 = clamp(hsv.x, 0.0, 1.0);
  let S : f32 = clamp(hsv.y, 0.0, 1.0);
  let V : f32 = clamp(hsv.z, 0.0, 1.0);
  let dh : f32 = H * 6.0;
  let dr : f32 = clamp(abs(dh - 3.0) - 1.0, 0.0, 1.0);
  let dg : f32 = clamp(-abs(dh - 2.0) + 2.0, 0.0, 1.0);
  let db : f32 = clamp(-abs(dh - 4.0) + 2.0, 0.0, 1.0);
  let oneMinusS : f32 = 1.0 - S;
  let sr : f32 = S * dr;
  let sg : f32 = S * dg;
  let sb : f32 = S * db;
  let r : f32 = (oneMinusS + sr) * V;
  let g : f32 = (oneMinusS + sg) * V;
  let b : f32 = (oneMinusS + sb) * V;
  return clamp(vec3<f32>(r, g, b), vec3<f32>(0.0), vec3<f32>(1.0));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let stage : u32 = asU32(params.stage);
  if (stage == 0u) {
    if (gid.x != 0u || gid.y != 0u || gid.z != 0u) {
      return;
    }
    let width : u32 = asU32(params.width);
    let height : u32 = asU32(params.height);
    let widthI : i32 = i32(width);
    let heightI : i32 = i32(height);
    let refChannels : i32 = i32(max(params.refChannels, 1.0));

    var minRef : f32 = FLT_MAX;
    var maxRef : f32 = FLT_MIN;

    var y : i32 = 0;
    loop {
      if (y >= heightI) {
        break;
      }
      var x : i32 = 0;
      loop {
        if (x >= widthI) {
          break;
        }
        let value : f32 = getReferenceValue(x, y, widthI, heightI, refChannels);
        let idx : u32 = u32(y * widthI + x);
        shadeBuffer[idx] = value;
        minRef = min(minRef, value);
        maxRef = max(maxRef, value);
        x = x + 1;
      }
      y = y + 1;
    }

    if (minRef == FLT_MAX && maxRef == FLT_MIN) {
      minRef = 0.0;
      maxRef = 0.0;
    }
    reduceBuffer[0] = minRef;
    reduceBuffer[1] = maxRef;

    var invRangeRef : f32 = 0.0;
    let rangeRef : f32 = maxRef - minRef;
    if (rangeRef != 0.0) {
      invRangeRef = 1.0 / rangeRef;
    }

    let pixelCount : u32 = width * height;
    var idxNorm : u32 = 0u;
    loop {
      if (idxNorm >= pixelCount) {
        break;
      }
      var norm : f32 = 0.0;
      if (invRangeRef != 0.0) {
        norm = (shadeBuffer[idxNorm] - minRef) * invRangeRef;
      }
      shadeBuffer[idxNorm] = clamp(norm, 0.0, 1.0);
      idxNorm = idxNorm + 1u;
    }

    var minShade : f32 = FLT_MAX;
    var maxShade : f32 = FLT_MIN;

    y = 0;
    loop {
      if (y >= heightI) {
        break;
      }
      var x : i32 = 0;
      loop {
        if (x >= widthI) {
          break;
        }
        var gx : f32 = 0.0;
        var gy : f32 = 0.0;
        var kernelIndex : u32 = 0u;
        var ky : i32 = -1;
        loop {
          if (ky > 1) {
            break;
          }
          var kx : i32 = -1;
          loop {
            if (kx > 1) {
              break;
            }
            let sample : f32 = sampleNormalizedRef(x + kx, y + ky, widthI, heightI);
            let weightX : f32 = SOBEL_X[kernelIndex];
            let weightY : f32 = SOBEL_Y[kernelIndex];
            gx = gx + sample * weightX;
            gy = gy + sample * weightY;
            kernelIndex = kernelIndex + 1u;
            kx = kx + 1;
          }
          ky = ky + 1;
        }
        let magnitude : f32 = sqrt(gx * gx + gy * gy);
        let baseIdx : u32 = u32(y * widthI + x) * 2u;
        gradientBuffer[baseIdx] = magnitude;
        gradientBuffer[baseIdx + 1u] = 0.0;
        minShade = min(minShade, magnitude);
        maxShade = max(maxShade, magnitude);
        x = x + 1;
      }
      y = y + 1;
    }

    if (minShade == FLT_MAX && maxShade == FLT_MIN) {
      minShade = 0.0;
      maxShade = 0.0;
    }
    reduceBuffer[2] = minShade;
    reduceBuffer[3] = maxShade;
    return;
  }

  if (stage != 1u) {
    return;
  }

  let width : u32 = asU32(params.width);
  let height : u32 = asU32(params.height);
  if (gid.x >= width || gid.y >= height) {
    return;
  }

  let channelCount : u32 = max(asU32(params.channels), 1u);
  let pixelIndex : u32 = gid.y * width + gid.x;
  let baseIndex : u32 = pixelIndex * channelCount;
  let minShade : f32 = reduceBuffer[2];
  let maxShade : f32 = reduceBuffer[3];
  let shadeRange : f32 = maxShade - minShade;
  var invShadeRange : f32 = 0.0;
  if (shadeRange != 0.0) {
    invShadeRange = 1.0 / shadeRange;
  }
  let widthI : i32 = i32(width);
  let heightI : i32 = i32(height);
  let shadeRaw : f32 = gradientBuffer[pixelIndex * 2u];
  var shadeNorm : f32 = 0.0;
  if (invShadeRange != 0.0) {
    shadeNorm = clamp((shadeRaw - minShade) * invShadeRange, 0.0, 1.0);
  }

  var conv : f32 = 0.0;
  var kernelIndex : u32 = 0u;
  var ky : i32 = -1;
  loop {
    if (ky > 1) {
      break;
    }
    var kx : i32 = -1;
    loop {
      if (kx > 1) {
        break;
      }
      let weight : f32 = SHARPEN_KERNEL[kernelIndex];
      if (weight != 0.0) {
        let sampleShade : f32 = sampleShadeNormalized(
          i32(gid.x) + kx,
          i32(gid.y) + ky,
          widthI,
          heightI,
          minShade,
          invShadeRange,
        );
        conv = conv + sampleShade * weight;
      }
      kernelIndex = kernelIndex + 1u;
      kx = kx + 1;
    }
    ky = ky + 1;
  }

  let shadeBlendAlpha : f32 = clamp(params.shadeBlend, 0.0, 1.0);
  let convClamped : f32 = clamp(conv, 0.0, 1.0);
  let finalShade : f32 = mixf(shadeNorm, convClamped, shadeBlendAlpha);
  let highlight : f32 = clamp(finalShade * finalShade, 0.0, 1.0);
  let alphaVal : f32 = clamp(params.alpha, 0.0, 1.0);

  let coords : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
  let srcColor : vec4<f32> = textureLoad(srcTex, coords, 0);

  var shadedChannels : array<f32, 4>;
  var channel : u32 = 0u;
  loop {
    if (channel >= channelCount || channel >= 4u) {
      break;
    }
    let srcVal : f32 = getChannelValue(srcColor, channel);
    let dark : f32 = (1.0 - srcVal) * (1.0 - highlight);
    let lit : f32 = 1.0 - dark;
    let shaded : f32 = clamp(lit * finalShade, 0.0, 1.0);
    shadedChannels[channel] = shaded;
    channel = channel + 1u;
  }

  if (channelCount == 1u) {
    let srcVal : f32 = getChannelValue(srcColor, 0u);
    let mixed : f32 = mixf(srcVal, shadedChannels[0], alphaVal);
    outputBuffer[baseIndex] = clamp(mixed, 0.0, 1.0);
    return;
  }

  if (channelCount == 2u) {
    let src0 : f32 = getChannelValue(srcColor, 0u);
    let src1 : f32 = getChannelValue(srcColor, 1u);
    let mixed : f32 = mixf(src0, shadedChannels[0], alphaVal);
    outputBuffer[baseIndex] = clamp(mixed, 0.0, 1.0);
    outputBuffer[baseIndex + 1u] = clamp(src1, 0.0, 1.0);
    return;
  }

  let srcRgb : vec3<f32> = vec3<f32>(
    getChannelValue(srcColor, 0u),
    getChannelValue(srcColor, 1u),
    getChannelValue(srcColor, 2u),
  );
  let shadeR : f32 = shadedChannels[0];
  let shadeG : f32 = shadedChannels[min(1u, channelCount - 1u)];
  let shadeB : f32 = shadedChannels[min(2u, channelCount - 1u)];
  let baseHsv : vec3<f32> = rgbToHsv(srcRgb);
  let shadeHsv : vec3<f32> = rgbToHsv(vec3<f32>(shadeR, shadeG, shadeB));
  let finalValue : f32 = mixf(baseHsv.z, shadeHsv.z, alphaVal);
  let finalRgb : vec3<f32> = hsvToRgb(vec3<f32>(baseHsv.x, baseHsv.y, finalValue));
  outputBuffer[baseIndex] = finalRgb.x;
  outputBuffer[baseIndex + 1u] = finalRgb.y;
  outputBuffer[baseIndex + 2u] = finalRgb.z;
  if (channelCount > 3u) {
    let alphaOut : f32 = clamp(getChannelValue(srcColor, 0u), 0.0, 1.0);
    outputBuffer[baseIndex + 3u] = alphaOut;
    var extra : u32 = 4u;
    loop {
      if (extra >= channelCount) {
        break;
      }
      outputBuffer[baseIndex + extra] = finalRgb.z;
      extra = extra + 1u;
    }
  }
}
