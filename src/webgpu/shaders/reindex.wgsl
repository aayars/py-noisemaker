struct ReindexParams {
  width : f32,
  height : f32,
  channels : f32,
  displacement : f32,
  modulo : f32,
  pad0 : f32,
  pad1 : f32,
  pad2 : f32,
};

@group(0) @binding(0) var inputTex : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> outputBuffer : array<f32>;
@group(0) @binding(2) var<uniform> params : ReindexParams;

fn asU32(value : f32) -> u32 {
  return u32(max(value, 0.0));
}

fn clamp01(value : f32) -> f32 {
  return clamp(value, 0.0, 1.0);
}

fn srgbToLinear(value : f32) -> f32 {
  if (value <= 0.04045) {
    return value / 12.92;
  }
  return pow((value + 0.055) / 1.055, 2.4);
}

fn cubeRoot(value : f32) -> f32 {
  if (value == 0.0) {
    return 0.0;
  }
  let signValue = select(-1.0, 1.0, value >= 0.0);
  return signValue * pow(abs(value), 1.0 / 3.0);
}

fn oklabL(rgb : vec3<f32>) -> f32 {
  let rLin = srgbToLinear(clamp(rgb.x, 0.0, 1.0));
  let gLin = srgbToLinear(clamp(rgb.y, 0.0, 1.0));
  let bLin = srgbToLinear(clamp(rgb.z, 0.0, 1.0));
  let l = 0.4121656120 * rLin + 0.5362752080 * gLin + 0.0514575653 * bLin;
  let m = 0.2118591070 * rLin + 0.6807189584 * gLin + 0.1074065790 * bLin;
  let s = 0.0883097947 * rLin + 0.2818474174 * gLin + 0.6302613616 * bLin;
  let lC = cubeRoot(l);
  let mC = cubeRoot(m);
  let sC = cubeRoot(s);
  return clamp(0.2104542553 * lC + 0.7936177850 * mC - 0.0040720468 * sC, 0.0, 1.0);
}

fn fetchRgb(texel : vec4<f32>, channelCount : u32) -> vec3<f32> {
  if (channelCount <= 1u) {
    let v = clamp01(texel.x);
    return vec3<f32>(v, v, v);
  }
  if (channelCount == 2u) {
    let r = clamp01(texel.x);
    let g = clamp01(texel.y);
    return vec3<f32>(r, g, g);
  }
  let r = clamp01(texel.x);
  let g = clamp01(texel.y);
  let b = clamp01(texel.z);
  return vec3<f32>(r, g, b);
}

fn wrapFloat(value : f32, range : f32) -> f32 {
  if (range <= 0.0) {
    return 0.0;
  }
  let scaled = floor(value / range);
  return value - range * scaled;
}

fn fetchComponent(texel : vec4<f32>, channel : u32) -> f32 {
  switch channel {
    case 0u: {
      return texel.x;
    }
    case 1u: {
      return texel.y;
    }
    case 2u: {
      return texel.z;
    }
    default: {
      return texel.w;
    }
  }
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
  let coords : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
  let texel : vec4<f32> = textureLoad(inputTex, coords, 0);
  let luminance : f32 = oklabL(fetchRgb(texel, channelCount));
  let modRange : f32 = max(params.modulo, 1.0);
  let displacement : f32 = params.displacement;
  let offset : f32 = luminance * displacement * modRange + luminance;
  let widthF : f32 = max(params.width, 1.0);
  let heightF : f32 = max(params.height, 1.0);
  let wrappedX : f32 = wrapFloat(offset, widthF);
  let wrappedY : f32 = wrapFloat(offset, heightF);
  let sampleX : i32 = i32(clamp(floor(wrappedX), 0.0, widthF - 1.0));
  let sampleY : i32 = i32(clamp(floor(wrappedY), 0.0, heightF - 1.0));
  let sampleCoords : vec2<i32> = vec2<i32>(sampleX, sampleY);
  let sampled : vec4<f32> = textureLoad(inputTex, sampleCoords, 0);

  let pixelIndex : u32 = gid.y * width + gid.x;
  let baseIndex : u32 = pixelIndex * channelCount;
  if (channelCount > 0u) {
    outputBuffer[baseIndex] = fetchComponent(sampled, 0u);
  }
  if (channelCount > 1u) {
    outputBuffer[baseIndex + 1u] = fetchComponent(sampled, 1u);
  }
  if (channelCount > 2u) {
    outputBuffer[baseIndex + 2u] = fetchComponent(sampled, 2u);
  }
  if (channelCount > 3u) {
    outputBuffer[baseIndex + 3u] = fetchComponent(sampled, 3u);
  }
  if (channelCount > 4u) {
    var ch : u32 = 4u;
    let fallback : f32 = fetchComponent(sampled, 3u);
    loop {
      if (ch >= channelCount) {
        break;
      }
      outputBuffer[baseIndex + ch] = fallback;
      ch = ch + 1u;
    }
  }
}
