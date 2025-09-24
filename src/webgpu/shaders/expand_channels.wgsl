struct ExpandParams {
  width : u32,
  height : u32,
  channels : u32,
  _pad : u32,
};

@group(0) @binding(0) var inputTexture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> outputBuffer : array<f32>;
@group(0) @binding(2) var<uniform> params : ExpandParams;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let width : u32 = max(params.width, 1u);
  let height : u32 = max(params.height, 1u);
  if (gid.x >= width || gid.y >= height) {
    return;
  }

  let channelCount : u32 = max(params.channels, 1u);
  let coords : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
  let sample : vec4<f32> = textureLoad(inputTexture, coords, 0);
  let value : f32 = sample.x;

  let baseIndex : u32 = (gid.y * width + gid.x) * channelCount;
  for (var channel : u32 = 0u; channel < channelCount; channel = channel + 1u) {
    outputBuffer[baseIndex + channel] = value;
  }
}
