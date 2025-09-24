struct FxaaParams {
  width : f32,
  height : f32,
  channels : f32,
  _pad : f32,
};

@group(0) @binding(0) var inputTex : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> outputBuffer : array<f32>;
@group(0) @binding(2) var<uniform> params : FxaaParams;

const EPSILON : f32 = 1e-10;
const LUMA_WEIGHTS : vec3<f32> = vec3<f32>(0.299, 0.587, 0.114);

fn asU32(value : f32) -> u32 {
  return u32(max(value, 0.0));
}

fn reflectCoord(coord : i32, limit : i32) -> i32 {
  if (limit <= 1) {
    return 0;
  }
  let period : i32 = 2 * limit - 2;
  var wrapped : i32 = coord % period;
  if (wrapped < 0) {
    wrapped = wrapped + period;
  }
  if (wrapped < limit) {
    return wrapped;
  }
  return period - wrapped;
}

fn loadTexel(x : i32, y : i32, width : i32, height : i32) -> vec4<f32> {
  let rx : i32 = reflectCoord(x, width);
  let ry : i32 = reflectCoord(y, height);
  return textureLoad(inputTex, vec2<i32>(rx, ry), 0);
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
  let pixelIndex : u32 = gid.y * width + gid.x;
  let baseIndex : u32 = pixelIndex * channelCount;

  let xi : i32 = i32(gid.x);
  let yi : i32 = i32(gid.y);
  let widthI : i32 = i32(width);
  let heightI : i32 = i32(height);

  if (channelCount == 1u) {
    let center : f32 = loadTexel(xi, yi, widthI, heightI).x;
    let north : f32 = loadTexel(xi, yi - 1, widthI, heightI).x;
    let south : f32 = loadTexel(xi, yi + 1, widthI, heightI).x;
    let west : f32 = loadTexel(xi - 1, yi, widthI, heightI).x;
    let east : f32 = loadTexel(xi + 1, yi, widthI, heightI).x;

    let wC : f32 = 1.0;
    let wN : f32 = exp(-abs(center - north));
    let wS : f32 = exp(-abs(center - south));
    let wW : f32 = exp(-abs(center - west));
    let wE : f32 = exp(-abs(center - east));
    let sumWeights : f32 = wC + wN + wS + wW + wE + EPSILON;

    outputBuffer[baseIndex] =
      (center * wC + north * wN + south * wS + west * wW + east * wE) / sumWeights;
    return;
  }

  if (channelCount == 2u) {
    let centerTex : vec4<f32> = loadTexel(xi, yi, widthI, heightI);
    let north : f32 = loadTexel(xi, yi - 1, widthI, heightI).x;
    let south : f32 = loadTexel(xi, yi + 1, widthI, heightI).x;
    let west : f32 = loadTexel(xi - 1, yi, widthI, heightI).x;
    let east : f32 = loadTexel(xi + 1, yi, widthI, heightI).x;

    let centerLum : f32 = centerTex.x;
    let wC : f32 = 1.0;
    let wN : f32 = exp(-abs(centerLum - north));
    let wS : f32 = exp(-abs(centerLum - south));
    let wW : f32 = exp(-abs(centerLum - west));
    let wE : f32 = exp(-abs(centerLum - east));
    let sumWeights : f32 = wC + wN + wS + wW + wE + EPSILON;

    outputBuffer[baseIndex] =
      (centerLum * wC + north * wN + south * wS + west * wW + east * wE) / sumWeights;
    outputBuffer[baseIndex + 1u] = centerTex.y;
    return;
  }

  let centerTex : vec4<f32> = loadTexel(xi, yi, widthI, heightI);
  let northTex : vec4<f32> = loadTexel(xi, yi - 1, widthI, heightI);
  let southTex : vec4<f32> = loadTexel(xi, yi + 1, widthI, heightI);
  let westTex : vec4<f32> = loadTexel(xi - 1, yi, widthI, heightI);
  let eastTex : vec4<f32> = loadTexel(xi + 1, yi, widthI, heightI);

  let centerLum : f32 = dot(centerTex.xyz, LUMA_WEIGHTS);
  let northLum : f32 = dot(northTex.xyz, LUMA_WEIGHTS);
  let southLum : f32 = dot(southTex.xyz, LUMA_WEIGHTS);
  let westLum : f32 = dot(westTex.xyz, LUMA_WEIGHTS);
  let eastLum : f32 = dot(eastTex.xyz, LUMA_WEIGHTS);

  let wC : f32 = 1.0;
  let wN : f32 = exp(-abs(centerLum - northLum));
  let wS : f32 = exp(-abs(centerLum - southLum));
  let wW : f32 = exp(-abs(centerLum - westLum));
  let wE : f32 = exp(-abs(centerLum - eastLum));
  let sumWeights : f32 = wC + wN + wS + wW + wE + EPSILON;

  let filteredRgb : vec3<f32> =
    (centerTex.xyz * wC + northTex.xyz * wN + southTex.xyz * wS + westTex.xyz * wW + eastTex.xyz * wE) /
    sumWeights;

  outputBuffer[baseIndex] = filteredRgb.x;
  if (channelCount > 1u) {
    outputBuffer[baseIndex + 1u] = filteredRgb.y;
  }
  if (channelCount > 2u) {
    outputBuffer[baseIndex + 2u] = filteredRgb.z;
  }
  if (channelCount > 3u) {
    outputBuffer[baseIndex + 3u] = centerTex.w;
  }
  if (channelCount > 4u) {
    var ch : u32 = 4u;
    let fallback : f32 = centerTex.w;
    loop {
      if (ch >= channelCount) {
        break;
      }
      outputBuffer[baseIndex + ch] = fallback;
      ch = ch + 1u;
    }
  }
}
