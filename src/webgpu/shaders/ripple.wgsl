const TAU : f32 = 6.28318530717958647692;

struct RippleParams {
  width : f32,
  height : f32,
  channels : f32,
  displacement : f32,
  kink : f32,
  rand : f32,
  pad0 : f32,
  pad1 : f32,
};

@group(0) @binding(0) var inputTex : texture_2d<f32>;
@group(0) @binding(1) var referenceTex : texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> outputBuffer : array<f32>;
@group(0) @binding(3) var<uniform> params : RippleParams;

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

  let refTexel : vec4<f32> = textureLoad(referenceTex, vec2<i32>(i32(gid.x), i32(gid.y)), 0);
  let angle : f32 = refTexel.x * TAU * params.kink * params.rand;

  let fx : f32 = f32(gid.x) + cos(angle) * params.displacement * widthF;
  let fy : f32 = f32(gid.y) + sin(angle) * params.displacement * heightF;

  let x0 : i32 = i32(floor(fx));
  let y0 : i32 = i32(floor(fy));
  let x1 : i32 = x0 + 1;
  let y1 : i32 = y0 + 1;

  let sx : f32 = fx - f32(x0);
  let sy : f32 = fy - f32(y0);

  let widthI : i32 = i32(width);
  let heightI : i32 = i32(height);

  let x0w : i32 = wrapCoord(x0, widthI);
  let x1w : i32 = wrapCoord(x1, widthI);
  let y0w : i32 = wrapCoord(y0, heightI);
  let y1w : i32 = wrapCoord(y1, heightI);

  let c00 : vec4<f32> = textureLoad(inputTex, vec2<i32>(x0w, y0w), 0);
  let c10 : vec4<f32> = textureLoad(inputTex, vec2<i32>(x1w, y0w), 0);
  let c01 : vec4<f32> = textureLoad(inputTex, vec2<i32>(x0w, y1w), 0);
  let c11 : vec4<f32> = textureLoad(inputTex, vec2<i32>(x1w, y1w), 0);

  let sxVec : vec4<f32> = vec4<f32>(sx);
  let syVec : vec4<f32> = vec4<f32>(sy);
  let mixX0 : vec4<f32> = mix(c00, c10, sxVec);
  let mixX1 : vec4<f32> = mix(c01, c11, sxVec);
  let result : vec4<f32> = mix(mixX0, mixX1, syVec);

  storeTexel(baseIndex, channelCount, result);
}
