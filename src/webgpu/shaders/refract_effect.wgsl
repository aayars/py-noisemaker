const PI : f32 = 3.14159265358979323846;
const TAU : f32 = 6.28318530717958647692;

struct RefractEffectParams {
  width : f32,
  height : f32,
  channels : f32,
  displacement : f32,
  quadDirectional : f32,
  splineOrder : f32,
  usePolar : f32,
  pad0 : f32,
};

@group(0) @binding(0) var inputTex : texture_2d<f32>;
@group(0) @binding(1) var referenceXTex : texture_2d<f32>;
@group(0) @binding(2) var referenceYTex : texture_2d<f32>;
@group(0) @binding(3) var<storage, read_write> outputBuffer : array<f32>;
@group(0) @binding(4) var<uniform> params : RefractEffectParams;

fn asU32(value : f32) -> u32 {
  return u32(max(value, 0.0));
}

fn wrapCoord(coord : i32, limit : i32) -> i32 {
  if (limit <= 0) {
    return 0;
  }
  var wrapped : i32 = coord % limit;
  if (wrapped < 0) {
    wrapped = wrapped + limit;
  }
  return wrapped;
}

fn wrapFloat(value : f32, limit : f32) -> f32 {
  if (limit == 0.0) {
    return 0.0;
  }
  let div : f32 = floor(value / limit);
  var result : f32 = value - div * limit;
  if (result < 0.0) {
    result = result + limit;
  }
  return result;
}

fn storeTexel(baseIndex : u32, channelCount : u32, texel : vec4<f32>) {
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
    let fallback : f32 = texel.w;
    loop {
      if (ch >= channelCount) {
        break;
      }
      outputBuffer[baseIndex + ch] = fallback;
      ch = ch + 1u;
    }
  }
}

fn applySpline(value : f32, splineOrder : f32) -> f32 {
  var clamped : f32 = clamp(value, 0.0, 1.0);
  let order : i32 = i32(splineOrder + 0.5);
  if (order == 2) {
    clamped = 0.5 - cos(clamped * PI) * 0.5;
  }
  return clamped;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let width : u32 = asU32(params.width);
  let height : u32 = asU32(params.height);
  if (gid.x >= width || gid.y >= height) {
    return;
  }

  let channelCount : u32 = max(asU32(params.channels), 1u);
  let pixelIndex : u32 = gid.y * width + gid.x;
  let baseIndex : u32 = pixelIndex * channelCount;

  let widthF : f32 = params.width;
  let heightF : f32 = params.height;

  let usePolar : bool = params.usePolar > 0.5;
  var refX : f32 = textureLoad(referenceXTex, vec2<i32>(i32(gid.x), i32(gid.y)), 0).x;
  var refY : f32;
  if (usePolar) {
    refX = clamp(refX, 0.0, 1.0);
    let angle : f32 = refX * TAU;
    let cosVal : f32 = cos(angle);
    let sinVal : f32 = sin(angle);
    refX = clamp(cosVal * 0.5 + 0.5, 0.0, 1.0);
    refY = clamp(sinVal * 0.5 + 0.5, 0.0, 1.0);
  } else {
    refX = clamp(refX, 0.0, 1.0);
    refY = clamp(
      textureLoad(referenceYTex, vec2<i32>(i32(gid.x), i32(gid.y)), 0).x,
      0.0,
      1.0,
    );
  }

  let quadDirectional : bool = params.quadDirectional > 0.5;
  if (quadDirectional) {
    refX = clamp(refX * 2.0 - 1.0, -1.0, 1.0);
    refY = clamp(refY * 2.0 - 1.0, -1.0, 1.0);
  } else {
    refX = clamp(refX * 2.0, -2.0, 2.0);
    refY = clamp(refY * 2.0, -2.0, 2.0);
  }

  let scaleX : f32 = params.displacement * widthF;
  let scaleY : f32 = params.displacement * heightF;

  let sampleX : f32 = f32(gid.x) + refX * scaleX;
  let sampleY : f32 = f32(gid.y) + refY * scaleY;

  let wrappedX : f32 = wrapFloat(sampleX, widthF);
  let wrappedY : f32 = wrapFloat(sampleY, heightF);

  var x0 : i32 = i32(floor(wrappedX));
  var y0 : i32 = i32(floor(wrappedY));

  let widthI : i32 = i32(width);
  let heightI : i32 = i32(height);

  if (x0 < 0) {
    x0 = 0;
  } else if (x0 >= widthI) {
    x0 = widthI - 1;
  }

  if (y0 < 0) {
    y0 = 0;
  } else if (y0 >= heightI) {
    y0 = heightI - 1;
  }

  let x1 : i32 = wrapCoord(x0 + 1, widthI);
  let y1 : i32 = wrapCoord(y0 + 1, heightI);

  var fx : f32 = wrappedX - f32(x0);
  var fy : f32 = wrappedY - f32(y0);

  fx = clamp(fx, 0.0, 1.0);
  fy = clamp(fy, 0.0, 1.0);

  let tx : f32 = applySpline(fx, params.splineOrder);
  let ty : f32 = applySpline(fy, params.splineOrder);

  let tex00 : vec4<f32> = textureLoad(inputTex, vec2<i32>(x0, y0), 0);
  let tex10 : vec4<f32> = textureLoad(inputTex, vec2<i32>(x1, y0), 0);
  let tex01 : vec4<f32> = textureLoad(inputTex, vec2<i32>(x0, y1), 0);
  let tex11 : vec4<f32> = textureLoad(inputTex, vec2<i32>(x1, y1), 0);

  let mixX0 : vec4<f32> = mix(tex00, tex10, vec4<f32>(tx));
  let mixX1 : vec4<f32> = mix(tex01, tex11, vec4<f32>(tx));
  let result : vec4<f32> = mix(mixX0, mixX1, vec4<f32>(ty));

  storeTexel(baseIndex, channelCount, result);
}
