struct AdjustHueParams {
  width: f32,
  height: f32,
  channels: f32,
  amount: f32,
};

@group(0) @binding(0) var inputTex : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> outputBuffer : array<f32>;
@group(0) @binding(2) var<uniform> params : AdjustHueParams;

fn asU32(value : f32) -> u32 {
  return u32(max(value, 0.0));
}

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

fn writeChannels(baseIndex : u32, channelCount : u32, texel : vec4<f32>) {
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

  if (channelCount < 3u) {
    writeChannels(baseIndex, channelCount, texel);
    return;
  }

  let rgb : vec3<f32> = texel.xyz;
  var hsv : vec3<f32> = rgbToHsv(rgb);
  hsv.x = wrapUnit(hsv.x + params.amount);
  let adjusted : vec3<f32> = hsvToRgb(hsv);

  outputBuffer[baseIndex] = adjusted.x;
  outputBuffer[baseIndex + 1u] = adjusted.y;
  outputBuffer[baseIndex + 2u] = adjusted.z;

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
