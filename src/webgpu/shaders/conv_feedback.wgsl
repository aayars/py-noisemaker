struct ConvFeedbackParams {
  size : vec4<f32>;
};

@group(0) @binding(0) var inputTex : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> outputBuffer : array<f32>;
@group(0) @binding(2) var<uniform> params : ConvFeedbackParams;

fn asU32(value : f32) -> u32 {
  return u32(max(value, 0.0));
}

fn combineValue(value : f32) -> f32 {
  let up = max((value - 0.5) * 2.0, 0.0);
  let down = min(value * 2.0, 1.0);
  return up + (1.0 - down);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let width : u32 = asU32(params.size.x + 0.5);
  let height : u32 = asU32(params.size.y + 0.5);
  if (gid.x >= width || gid.y >= height) {
    return;
  }
  let channels : u32 = max(asU32(params.size.z + 0.5), 1u);
  let pixelIndex : u32 = gid.y * width + gid.x;
  let baseIndex : u32 = pixelIndex * channels;
  let texel : vec4<f32> = textureLoad(
    inputTex,
    vec2<i32>(i32(gid.x), i32(gid.y)),
    0,
  );
  var combined : vec4<f32> = vec4<f32>(
    combineValue(texel.x),
    combineValue(texel.y),
    combineValue(texel.z),
    combineValue(texel.w),
  );
  if (channels == 0u) {
    return;
  }
  outputBuffer[baseIndex] = combined.x;
  if (channels == 1u) {
    return;
  }
  outputBuffer[baseIndex + 1u] = combined.y;
  if (channels == 2u) {
    return;
  }
  outputBuffer[baseIndex + 2u] = combined.z;
  if (channels == 3u) {
    return;
  }
  outputBuffer[baseIndex + 3u] = combined.w;
  if (channels <= 4u) {
    return;
  }
  var extra : u32 = 4u;
  loop {
    if (extra >= channels) {
      break;
    }
    outputBuffer[baseIndex + extra] = combined.w;
    extra = extra + 1u;
  }
}
