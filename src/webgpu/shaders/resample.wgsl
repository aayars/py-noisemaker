const PI : f32 = 3.14159265358979323846;

struct ResampleParams {
  srcWidth: f32,
  srcHeight: f32,
  srcChannels: f32,
  splineOrder: f32,
  dstWidth: f32,
  dstHeight: f32,
  dstChannels: f32,
  padding: f32,
};

@group(0) @binding(0) var inputTex : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> outputBuffer : array<f32>;
@group(0) @binding(2) var<uniform> params : ResampleParams;

fn asU32(value : f32) -> u32 {
  return u32(max(value, 0.0));
}

fn asI32(value : f32) -> i32 {
  return i32(max(value, 0.0));
}

fn wrapCoord(coord : i32, limit : i32) -> i32 {
  var wrapped = coord % limit;
  if (wrapped < 0) {
    wrapped = wrapped + limit;
  }
  return wrapped;
}

fn sampleWrapped(x : i32, y : i32) -> vec4<f32> {
  let width : i32 = asI32(params.srcWidth);
  let height : i32 = asI32(params.srcHeight);
  let sx : i32 = wrapCoord(x, width);
  let sy : i32 = wrapCoord(y, height);
  return textureLoad(inputTex, vec2<i32>(sx, sy), 0);
}

fn blendLinear(a : vec4<f32>, b : vec4<f32>, t : f32) -> vec4<f32> {
  return mix(a, b, vec4<f32>(t));
}

fn blendCosine(a : vec4<f32>, b : vec4<f32>, t : f32) -> vec4<f32> {
  let g : f32 = (1.0 - cos(t * PI)) * 0.5;
  return mix(a, b, vec4<f32>(g));
}

fn cubicInterpolate(a : vec4<f32>, b : vec4<f32>, c : vec4<f32>, d : vec4<f32>, t : f32) -> vec4<f32> {
  let t2 : f32 = t * t;
  let a0 : vec4<f32> = d - c - a + b;
  let a1 : vec4<f32> = a - b - a0;
  let a2 : vec4<f32> = c - a;
  let a3 : vec4<f32> = b;
  return ((a0 * t) * t2) + (a1 * t2) + (a2 * t) + a3;
}

fn selectChannel(value : vec4<f32>, index : u32) -> f32 {
  switch index {
    case 0u: {
      return value.x;
    }
    case 1u: {
      return value.y;
    }
    case 2u: {
      return value.z;
    }
    default: {
      return value.w;
    }
  }
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let dstWidth : u32 = asU32(params.dstWidth);
  let dstHeight : u32 = asU32(params.dstHeight);
  if (gid.x >= dstWidth || gid.y >= dstHeight) {
    return;
  }

  let srcWidth : f32 = params.srcWidth;
  let srcHeight : f32 = params.srcHeight;
  let scaleX : f32 = srcWidth / max(params.dstWidth, 1.0);
  let scaleY : f32 = srcHeight / max(params.dstHeight, 1.0);

  let gx : f32 = f32(gid.x) * scaleX;
  let gy : f32 = f32(gid.y) * scaleY;
  let x0 : i32 = i32(floor(gx));
  let y0 : i32 = i32(floor(gy));
  let xf : f32 = gx - f32(x0);
  let yf : f32 = gy - f32(y0);

  let order : i32 = i32(round(params.splineOrder));
  var result : vec4<f32>;
  if (order <= 0) {
    result = sampleWrapped(x0, y0);
  } else if (order == 1 || order == 2) {
    let v00 : vec4<f32> = sampleWrapped(x0, y0);
    let v10 : vec4<f32> = sampleWrapped(x0 + 1, y0);
    let v01 : vec4<f32> = sampleWrapped(x0, y0 + 1);
    let v11 : vec4<f32> = sampleWrapped(x0 + 1, y0 + 1);
    if (order == 2) {
      let mx0 : vec4<f32> = blendCosine(v00, v10, xf);
      let mx1 : vec4<f32> = blendCosine(v01, v11, xf);
      result = blendCosine(mx0, mx1, yf);
    } else {
      let mx0 : vec4<f32> = blendLinear(v00, v10, xf);
      let mx1 : vec4<f32> = blendLinear(v01, v11, xf);
      result = blendLinear(mx0, mx1, yf);
    }
  } else {
    var rows : array<vec4<f32>, 4>;
    for (var m : i32 = -1; m < 3; m = m + 1) {
      let row : vec4<f32> = cubicInterpolate(
        sampleWrapped(x0 - 1, y0 + m),
        sampleWrapped(x0, y0 + m),
        sampleWrapped(x0 + 1, y0 + m),
        sampleWrapped(x0 + 2, y0 + m),
        xf,
      );
      rows[u32(m + 1)] = row;
    }
    result = cubicInterpolate(rows[0], rows[1], rows[2], rows[3], yf);
  }

  let dstChannels : u32 = max(asU32(params.dstChannels), 1u);
  let srcChannels : u32 = max(asU32(params.srcChannels), 1u);
  let pixelIndex : u32 = gid.y * dstWidth + gid.x;
  let baseIndex : u32 = pixelIndex * dstChannels;
  for (var ch : u32 = 0u; ch < dstChannels; ch = ch + 1u) {
    var srcIndex : u32 = ch;
    if (srcIndex >= srcChannels) {
      srcIndex = srcChannels - 1u;
    }
    outputBuffer[baseIndex + ch] = selectChannel(result, srcIndex);
  }
}
