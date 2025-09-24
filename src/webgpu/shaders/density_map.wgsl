struct DensityParams {
  dims0 : vec4<u32>;
  dims1 : vec4<u32>;
};

@group(0) @binding(0) var inputTexture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> binIndices : array<u32>;
@group(0) @binding(2) var<storage, read_write> histogram : array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> outputBuffer : array<f32>;
@group(0) @binding(4) var<uniform> params : DensityParams;

fn sampleChannel(texel : vec4<f32>, channel : u32) -> f32 {
  if (channel == 0u) {
    return texel.x;
  }
  if (channel == 1u) {
    return texel.y;
  }
  if (channel == 2u) {
    return texel.z;
  }
  return texel.w;
}

fn computeBin(value : f32, bins : u32) -> u32 {
  if (bins <= 1u) {
    return 0u;
  }
  let clamped : f32 = clamp(value, 0.0, 1.0);
  let scaled : f32 = clamped * f32(bins - 1u);
  var bin : u32 = u32(floor(scaled + 1e-6));
  if (bin >= bins) {
    bin = bins - 1u;
  }
  return bin;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let width : u32 = params.dims0.x;
  let height : u32 = params.dims0.y;
  if (gid.x >= width || gid.y >= height) {
    return;
  }

  let channels : u32 = max(params.dims0.z, 1u);
  let bins : u32 = max(params.dims0.w, 1u);
  let total : u32 = params.dims1.x;
  let stage : u32 = params.dims1.y;

  let pixelIndex : u32 = gid.y * width + gid.x;
  let baseIndex : u32 = pixelIndex * channels;
  if (baseIndex >= total) {
    return;
  }

  let texel : vec4<f32> = textureLoad(
    inputTexture,
    vec2<i32>(i32(gid.x), i32(gid.y)),
    0
  );

  if (stage == 0u) {
    var channel : u32 = 0u;
    loop {
      if (channel >= channels) {
        break;
      }
      let componentIndex : u32 = baseIndex + channel;
      if (componentIndex >= total) {
        break;
      }
      let value : f32 = sampleChannel(texel, channel);
      let bin : u32 = computeBin(value, bins);
      binIndices[componentIndex] = bin;
      atomicAdd(&histogram[bin], 1u);
      channel = channel + 1u;
    }
    return;
  }

  if (stage == 1u) {
    var channel : u32 = 0u;
    loop {
      if (channel >= channels) {
        break;
      }
      let componentIndex : u32 = baseIndex + channel;
      if (componentIndex >= total) {
        break;
      }
      let bin : u32 = binIndices[componentIndex];
      let count : u32 = atomicLoad(&histogram[bin]);
      outputBuffer[componentIndex] = f32(count);
      channel = channel + 1u;
    }
    return;
  }
}
