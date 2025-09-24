struct ScaleParams {
  width: f32,
  height: f32,
  channels: f32,
  factor: f32,
};

@group(0) @binding(0) var inputTex : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> outputBuffer : array<f32>;
@group(0) @binding(2) var<uniform> params : ScaleParams;

fn asU32(value : f32) -> u32 {
  return u32(max(value, 0.0));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let width : u32 = asU32(params.width);
  let height : u32 = asU32(params.height);
  if (gid.x >= width || gid.y >= height) {
    return;
  }

  let channels : u32 = max(asU32(params.channels), 1u);
  let coords : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
  let texel : vec4<f32> = textureLoad(inputTex, coords, 0);
  let factor : f32 = params.factor;
  let pixelIndex : u32 = gid.y * width + gid.x;
  let baseIndex : u32 = pixelIndex * channels;

  for (var ch : u32 = 0u; ch < channels; ch = ch + 1u) {
    var value : f32;
    switch ch {
      case 0u: { value = texel.x; }
      case 1u: { value = texel.y; }
      case 2u: { value = texel.z; }
      default: { value = texel.w; }
    }
    outputBuffer[baseIndex + ch] = value * factor;
  }
}
