struct CRTParams {
  sizeDisp : vec4<f32>;
  hueSatVig : vec4<f32>;
  extras : vec4<f32>;
};

@group(0) @binding(0) var inputTex : texture_2d<f32>;
@group(0) @binding(1) var scanTex : texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> outputBuffer : array<f32>;
@group(0) @binding(3) var<uniform> params : CRTParams;

const PI : f32 = 3.141592653589793;

fn clamp01(value : f32) -> f32 {
  return clamp(value, 0.0, 1.0);
}

fn wrapUnit(value : f32) -> f32 {
  let wrapped = value - floor(value);
  if (wrapped < 0.0) {
    return wrapped + 1.0;
  }
  return wrapped;
}

fn rgbToHsv(rgb : vec3<f32>) -> vec3<f32> {
  let cMax = max(max(rgb.x, rgb.y), rgb.z);
  let cMin = min(min(rgb.x, rgb.y), rgb.z);
  let delta = cMax - cMin;

  var hue : f32 = 0.0;
  if (delta > 0.0) {
    if (cMax == rgb.x) {
      var segment = (rgb.y - rgb.z) / delta;
      if (segment < 0.0) {
        segment = segment + 6.0;
      }
      hue = segment;
    } else if (cMax == rgb.y) {
      hue = ((rgb.z - rgb.x) / delta) + 2.0;
    } else {
      hue = ((rgb.x - rgb.y) / delta) + 4.0;
    }
    hue = wrapUnit(hue / 6.0);
  }

  let saturation = select(0.0, delta / cMax, cMax != 0.0);
  return vec3<f32>(hue, saturation, cMax);
}

fn hsvToRgb(hsv : vec3<f32>) -> vec3<f32> {
  let H = hsv.x;
  let S = hsv.y;
  let V = hsv.z;

  let dH = H * 6.0;
  let rComp = clamp01(abs(dH - 3.0) - 1.0);
  let gComp = clamp01(-abs(dH - 2.0) + 2.0);
  let bComp = clamp01(-abs(dH - 4.0) + 2.0);

  let oneMinusS = 1.0 - S;
  let sr = S * rComp;
  let sg = S * gComp;
  let sb = S * bComp;

  let r = clamp01((oneMinusS + sr) * V);
  let g = clamp01((oneMinusS + sg) * V);
  let b = clamp01((oneMinusS + sb) * V);

  return vec3<f32>(r, g, b);
}

fn adjustHue(color : vec3<f32>, amount : f32) -> vec3<f32> {
  var hsv = rgbToHsv(color);
  hsv.x = wrapUnit(hsv.x + amount);
  return hsvToRgb(hsv);
}

fn adjustSaturation(color : vec3<f32>, amount : f32) -> vec3<f32> {
  var hsv = rgbToHsv(color);
  hsv.y = clamp01(hsv.y * amount);
  return hsvToRgb(hsv);
}

fn blend_linear(a : f32, b : f32, t : f32) -> f32 {
  return mix(a, b, clamp(t, 0.0, 1.0));
}

fn blend_cosine(a : f32, b : f32, g : f32) -> f32 {
  let clamped : f32 = clamp(g, 0.0, 1.0);
  let weight : f32 = (1.0 - cos(clamped * PI)) * 0.5;
  return mix(a, b, weight);
}

fn clamp_index(value : f32, max_index : f32) -> u32 {
  if (value <= 0.0) {
    return 0u;
  }
  if (value >= max_index) {
    return u32(max_index);
  }
  return u32(floor(value));
}

fn process_sample_rgb(coords : vec2<i32>) -> vec3<f32> {
  let sample = textureLoad(inputTex, coords, 0).xyz;
  let scan = textureLoad(scanTex, coords, 0).x;
  let scan_vec = vec3<f32>(scan, scan, scan);
  let blended = mix(sample, (sample + scan_vec) * scan, vec3<f32>(0.05));
  return clamp(blended, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn apply_vignette(value : f32, brightness : f32, mask : f32, alpha : f32) -> f32 {
  let edge_mix = mix(value, brightness, mask);
  return mix(value, edge_mix, alpha);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let width : u32 = u32(max(params.sizeDisp.x, 0.0));
  let height : u32 = u32(max(params.sizeDisp.y, 0.0));
  if (gid.x >= width || gid.y >= height) {
    return;
  }

  let channelCount : u32 = u32(max(params.sizeDisp.z, 0.0));
  if (channelCount == 0u) {
    return;
  }

  let coords : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
  let texel : vec4<f32> = textureLoad(inputTex, coords, 0);
  let scan_val : f32 = textureLoad(scanTex, coords, 0).x;
  let pixelIndex : u32 = gid.y * width + gid.x;
  let baseIndex : u32 = pixelIndex * channelCount;

  var components = array<f32, 4>(texel.x, texel.y, texel.z, texel.w);
  let baseChannels : u32 = min(channelCount, 4u);
  for (var ch : u32 = 0u; ch < baseChannels; ch = ch + 1u) {
    let value = components[ch];
    let scanContribution = (value + scan_val) * scan_val;
    components[ch] = clamp01(mix(value, scanContribution, 0.05));
  }

  let widthF : f32 = params.sizeDisp.x;
  let heightF : f32 = params.sizeDisp.y;
  if (widthF <= 0.0 || heightF <= 0.0) {
    return;
  }

  let px : f32 = f32(gid.x) + 0.5;
  let py : f32 = f32(gid.y) + 0.5;
  let dx : f32 = (px - widthF * 0.5) / widthF;
  let dy : f32 = (py - heightF * 0.5) / heightF;
  let maxDx : f32 = abs((widthF * 0.5 - 0.5) / widthF);
  let maxDy : f32 = abs((heightF * 0.5 - 0.5) / heightF);
  let maxDist : f32 = sqrt(maxDx * maxDx + maxDy * maxDy);
  let dist : f32 = sqrt(dx * dx + dy * dy);
  var vignetteMask : f32 = 0.0;
  if (maxDist > 0.0) {
    vignetteMask = clamp(dist / maxDist, 0.0, 1.0);
  }
  let aberrationMask : f32 = pow(vignetteMask, 3.0);

  if (channelCount >= 3u) {
    let widthMinusOne : f32 = max(widthF - 1.0, 0.0);
    let xFloat : f32 = f32(gid.x);
    var gradient : f32 = 0.0;
    if (width > 1u && widthMinusOne > 0.0) {
      gradient = xFloat / widthMinusOne;
    }

    let dispPixels : f32 = clamp(round(params.sizeDisp.w), 0.0, widthF);

    var redOffset : f32 = min(widthMinusOne, xFloat + dispPixels);
    redOffset = blend_linear(redOffset, xFloat, gradient);
    redOffset = blend_cosine(xFloat, redOffset, aberrationMask);
    let redX : u32 = clamp_index(redOffset, widthMinusOne);

    var blueOffset : f32 = max(0.0, xFloat - dispPixels);
    blueOffset = blend_linear(xFloat, blueOffset, gradient);
    blueOffset = blend_cosine(xFloat, blueOffset, aberrationMask);
    let blueX : u32 = clamp_index(blueOffset, widthMinusOne);

    let greenOffset : f32 = blend_cosine(xFloat, xFloat, aberrationMask);
    let greenX : u32 = clamp_index(greenOffset, widthMinusOne);

    let redCoords : vec2<i32> = vec2<i32>(i32(redX), i32(gid.y));
    let greenCoords : vec2<i32> = vec2<i32>(i32(greenX), i32(gid.y));
    let blueCoords : vec2<i32> = vec2<i32>(i32(blueX), i32(gid.y));

    var redSample : vec3<f32> = process_sample_rgb(redCoords);
    var greenSample : vec3<f32> = process_sample_rgb(greenCoords);
    var blueSample : vec3<f32> = process_sample_rgb(blueCoords);

    redSample = adjustHue(redSample, params.hueSatVig.x);
    greenSample = adjustHue(greenSample, params.hueSatVig.x);
    blueSample = adjustHue(blueSample, params.hueSatVig.x);

    var aberrated : vec3<f32> = vec3<f32>(
      redSample.x,
      greenSample.y,
      blueSample.z,
    );

    aberrated = adjustHue(aberrated, -params.hueSatVig.x);
    aberrated = adjustHue(aberrated, params.hueSatVig.y);
    aberrated = adjustSaturation(aberrated, params.hueSatVig.z);

    components[0] = clamp01(aberrated.x);
    components[1] = clamp01(aberrated.y);
    components[2] = clamp01(aberrated.z);
  }

  let brightness : f32 = 0.0;
  let alpha : f32 = clamp01(params.hueSatVig.w);
  let vignetteValue : f32 = vignetteMask;

  if (channelCount > 0u) {
    components[0] = clamp01(apply_vignette(components[0], brightness, vignetteValue, alpha));
  }
  if (channelCount > 1u) {
    components[1] = clamp01(apply_vignette(components[1], brightness, vignetteValue, alpha));
  }
  if (channelCount > 2u) {
    components[2] = clamp01(apply_vignette(components[2], brightness, vignetteValue, alpha));
  }
  if (channelCount > 3u) {
    components[3] = clamp01(apply_vignette(components[3], brightness, vignetteValue, alpha));
  }

  if (channelCount > 0u) {
    outputBuffer[baseIndex] = components[0];
  }
  if (channelCount > 1u) {
    outputBuffer[baseIndex + 1u] = components[1];
  }
  if (channelCount > 2u) {
    outputBuffer[baseIndex + 2u] = components[2];
  }
  if (channelCount > 3u) {
    outputBuffer[baseIndex + 3u] = components[3];
  }
  if (channelCount > 4u) {
    var ch : u32 = 4u;
    let fallback : f32 = components[3];
    loop {
      if (ch >= channelCount) {
        break;
      }
      outputBuffer[baseIndex + ch] = fallback;
      ch = ch + 1u;
    }
  }
}
