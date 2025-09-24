const PI : f32 = 3.14159265358979323846;

struct WarpParams {
  width : f32,
  height : f32,
  channels : f32,
  displacement : f32,
  signedRange : f32,
  splineOrder : f32,
  pad0 : f32,
  pad1 : f32,
};

@group(0) @binding(0) var inputTex : texture_2d<f32>;
@group(0) @binding(1) var flowXTex : texture_2d<f32>;
@group(0) @binding(2) var flowYTex : texture_2d<f32>;
@group(0) @binding(3) var<storage, read_write> outputBuffer : array<f32>;
@group(0) @binding(4) var<uniform> params : WarpParams;

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

fn cubicInterpolate(a : vec4<f32>, b : vec4<f32>, c : vec4<f32>, d : vec4<f32>, t : f32) -> vec4<f32> {
  let t2 : f32 = t * t;
  let t3 : f32 = t2 * t;
  let a0 : vec4<f32> = d - c - a + b;
  let a1 : vec4<f32> = a - b - a0;
  let a2 : vec4<f32> = c - a;
  let a3 : vec4<f32> = b;
  return a0 * t3 + a1 * t2 + a2 * t + a3;
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
  let signedRange : bool = params.signedRange > 0.5;

  let baseScaleX : f32 = params.displacement * widthF;
  let baseScaleY : f32 = params.displacement * heightF;
  let scaleX : f32 = signedRange ? baseScaleX : baseScaleX * 2.0;
  let scaleY : f32 = signedRange ? baseScaleY : baseScaleY * 2.0;

  var flowX : f32 = textureLoad(flowXTex, vec2<i32>(i32(gid.x), i32(gid.y)), 0).x;
  var flowY : f32 = textureLoad(flowYTex, vec2<i32>(i32(gid.x), i32(gid.y)), 0).x;

  if (signedRange) {
    flowX = flowX * 2.0 - 1.0;
    flowY = flowY * 2.0 - 1.0;
  }

  let sampleX : f32 = f32(gid.x) + flowX * scaleX;
  let sampleY : f32 = f32(gid.y) + flowY * scaleY;

  let wrappedX : f32 = wrapFloat(sampleX, widthF);
  let wrappedY : f32 = wrapFloat(sampleY, heightF);

  let widthI : i32 = i32(width);
  let heightI : i32 = i32(height);

  let order : i32 = i32(params.splineOrder + 0.5);

  if (order == 0) {
    let ix : i32 = wrapCoord(i32(round(wrappedX)), widthI);
    let iy : i32 = wrapCoord(i32(round(wrappedY)), heightI);
    let texel : vec4<f32> = textureLoad(inputTex, vec2<i32>(ix, iy), 0);
    storeTexel(baseIndex, channelCount, texel);
    return;
  }

  var x0 : i32 = i32(floor(wrappedX));
  var y0 : i32 = i32(floor(wrappedY));

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

  let fxRaw : f32 = wrappedX - f32(x0);
  let fyRaw : f32 = wrappedY - f32(y0);

  let fx : f32 = clamp(fxRaw, 0.0, 1.0);
  let fy : f32 = clamp(fyRaw, 0.0, 1.0);

  if (order == 3) {
    var cols : array<vec4<f32>, 4>;
    for (var m : i32 = -1; m < 3; m = m + 1) {
      var row : array<vec4<f32>, 4>;
      for (var n : i32 = -1; n < 3; n = n + 1) {
        let sx : i32 = wrapCoord(x0 + n, widthI);
        let sy : i32 = wrapCoord(y0 + m, heightI);
        row[n + 1] = textureLoad(inputTex, vec2<i32>(sx, sy), 0);
      }
      cols[m + 1] = cubicInterpolate(row[0], row[1], row[2], row[3], fx);
    }
    let result : vec4<f32> = cubicInterpolate(cols[0], cols[1], cols[2], cols[3], fy);
    storeTexel(baseIndex, channelCount, result);
    return;
  }

  let x1 : i32 = wrapCoord(x0 + 1, widthI);
  let y1 : i32 = wrapCoord(y0 + 1, heightI);

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
