struct ProportionalDownsampleParams {
  srcWidth: f32,
  srcHeight: f32,
  srcChannels: f32,
  kernelWidth: f32,
  kernelHeight: f32,
  dstWidth: f32,
  dstHeight: f32,
  padding: f32,
};

@group(0) @binding(0) var inputTex : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> outputBuffer : array<f32>;
@group(0) @binding(2) var<uniform> params : ProportionalDownsampleParams;

fn asU32(value : f32) -> u32 {
  return u32(max(value, 0.0));
}

fn asI32(value : f32) -> i32 {
  return i32(max(value, 0.0));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let dstWidth : u32 = asU32(params.dstWidth);
  let dstHeight : u32 = asU32(params.dstHeight);
  if (gid.x >= dstWidth || gid.y >= dstHeight) {
    return;
  }

  let kernelWidth : u32 = max(asU32(params.kernelWidth), 1u);
  let kernelHeight : u32 = max(asU32(params.kernelHeight), 1u);
  let srcWidth : u32 = asU32(params.srcWidth);
  let srcHeight : u32 = asU32(params.srcHeight);
  let channels : u32 = max(asU32(params.srcChannels), 1u);

  let baseX : u32 = gid.x * kernelWidth;
  let baseY : u32 = gid.y * kernelHeight;

  var sum : vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  for (var ky : u32 = 0u; ky < kernelHeight; ky = ky + 1u) {
    let sy : u32 = min(baseY + ky, srcHeight - 1u);
    for (var kx : u32 = 0u; kx < kernelWidth; kx = kx + 1u) {
      let sx : u32 = min(baseX + kx, srcWidth - 1u);
      sum = sum + textureLoad(inputTex, vec2<i32>(i32(sx), i32(sy)), 0);
    }
  }

  let count : f32 = f32(kernelWidth * kernelHeight);
  let avg : vec4<f32> = sum / vec4<f32>(count);

  let pixelIndex : u32 = gid.y * dstWidth + gid.x;
  let baseIndex : u32 = pixelIndex * channels;
  for (var ch : u32 = 0u; ch < channels; ch = ch + 1u) {
    var value : f32;
    switch ch {
      case 0u: { value = avg.x; }
      case 1u: { value = avg.y; }
      case 2u: { value = avg.z; }
      default: { value = avg.w; }
    }
    outputBuffer[baseIndex + ch] = value;
  }
}
