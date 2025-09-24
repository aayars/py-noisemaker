struct GlitchParams {
  dims : vec4<f32>;
};

@group(0) @binding(0) var inputTexture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read> noiseBuffer : array<f32>;
@group(0) @binding(2) var<storage, read_write> outputBuffer : array<f32>;
@group(0) @binding(3) var<uniform> params : GlitchParams;

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

  let width_i : i32 = i32(width);
  let x_i : i32 = i32(gid.x);
  let y_i : i32 = i32(gid.y);
  let index : u32 = gid.y * width + gid.x;
  let shift : i32 = i32(floor(noiseBuffer[index] * 4.0));
  let base_index : u32 = index * channels;

  var channel : u32 = 0u;
  loop {
    if (channel >= channels) {
      break;
    }

    var sample_x : i32 = x_i;
    if (channel == 0u) {
      sample_x = sample_x + shift;
    } else if (channel == 2u) {
      sample_x = sample_x - shift;
    }

    let wrapped_x : u32 = wrap_index(sample_x, width_i);
    let texel : vec4<f32> = textureLoad(inputTexture, vec2<i32>(i32(wrapped_x), y_i), 0);
    outputBuffer[base_index + channel] = sample_channel(texel, channel);

    channel = channel + 1u;
  }
}
