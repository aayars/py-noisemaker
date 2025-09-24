struct CropParams {
  srcWidth: f32,
  srcHeight: f32,
  channels: f32,
  offsetX: f32,
  offsetY: f32,
  outSize: f32,
};

@group(0) @binding(0) var inputTex : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> outputBuffer : array<f32>;
@group(0) @binding(2) var<uniform> params : CropParams;

fn asU32(value : f32) -> u32 {
  return u32(max(value, 0.0));
}

fn clampCoord(value : i32, minValue : i32, maxValue : i32) -> i32 {
  return max(minValue, min(value, maxValue));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let outSize : u32 = asU32(params.outSize);
  if (gid.x >= outSize || gid.y >= outSize) {
    return;
  }

  let srcWidth : i32 = i32(asU32(params.srcWidth));
  let srcHeight : i32 = i32(asU32(params.srcHeight));
  let channels : u32 = max(asU32(params.channels), 1u);
  let offsetX : i32 = i32(asU32(params.offsetX));
  let offsetY : i32 = i32(asU32(params.offsetY));

  let sampleX : i32 = clampCoord(i32(gid.x) + offsetX, 0, srcWidth - 1);
  let sampleY : i32 = clampCoord(i32(gid.y) + offsetY, 0, srcHeight - 1);
  let texel : vec4<f32> = textureLoad(inputTex, vec2<i32>(sampleX, sampleY), 0);

  let pixelIndex : u32 = gid.y * outSize + gid.x;
  let baseIndex : u32 = pixelIndex * channels;

  for (var ch : u32 = 0u; ch < channels; ch = ch + 1u) {
    var value : f32;
    switch ch {
      case 0u: { value = texel.x; }
      case 1u: { value = texel.y; }
      case 2u: { value = texel.z; }
      default: { value = texel.w; }
    }
    outputBuffer[baseIndex + ch] = value;
  }
}
