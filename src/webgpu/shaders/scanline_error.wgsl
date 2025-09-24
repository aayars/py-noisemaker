struct ScanlineErrorParams {
  dims : vec4<f32>;
  extras : vec4<f32>;
};

@group(0) @binding(0) var inputTexture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read> lineBuffer : array<f32>;
@group(0) @binding(2) var<storage, read> whiteBuffer : array<f32>;
@group(0) @binding(3) var<storage, read> errorBuffer : array<f32>;
@group(0) @binding(4) var<storage, read_write> outputBuffer : array<f32>;
@group(0) @binding(5) var<uniform> params : ScanlineErrorParams;

fn wrap_index(value : i32, width : i32) -> u32 {
  if (width <= 0) {
    return 0u;
  }
  var wrapped : i32 = value % width;
  if (wrapped < 0) {
    wrapped = wrapped + width;
  }
  return u32(wrapped);
}

fn sample_channel(texel : vec4<f32>, channel : u32) -> f32 {
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

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let width : u32 = u32(params.dims.x);
  let height : u32 = u32(params.dims.y);
  if (gid.x >= width || gid.y >= height) {
    return;
  }

  let channels : u32 = u32(params.dims.z);
  if (channels == 0u) {
    return;
  }

  let index : u32 = gid.y * width + gid.x;
  let shift_scale : f32 = params.dims.w;
  let extra_scale : f32 = params.extras.x;

  let shift_value : f32 = errorBuffer[index] * shift_scale;
  let shift : i32 = i32(floor(shift_value));
  let sample_x : i32 = i32(gid.x) - shift;
  let wrapped_x : u32 = wrap_index(sample_x, i32(width));
  let texel : vec4<f32> = textureLoad(
    inputTexture,
    vec2<i32>(i32(wrapped_x), i32(gid.y)),
    0,
  );

  let extra : f32 = lineBuffer[index] * whiteBuffer[index] * extra_scale;
  let base : u32 = index * channels;

  var channel : u32 = 0u;
  loop {
    if (channel >= channels) {
      break;
    }
    let value : f32 = sample_channel(texel, channel);
    let adjusted : f32 = min(value + extra, 1.0);
    outputBuffer[base + channel] = adjusted;
    channel = channel + 1u;
  }
}
