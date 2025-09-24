struct Stage4Params {
  width : f32,
  height : f32,
  channels : f32,
  padding : f32,
};

struct GlowBuffer {
  values : array<f32>,
};

struct OutputBuffer {
  values : array<f32>,
};

@group(0) @binding(0) var<storage, read> glow_buffer : GlowBuffer;
@group(0) @binding(1) var input_tex : texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> output_buffer : OutputBuffer;
@group(0) @binding(3) var<uniform> params : Stage4Params;

fn as_u32(value : f32) -> u32 {
  return u32(max(value, 0.0));
}

fn fetch_channel(texel : vec4<f32>, channel : u32) -> f32 {
  let idx = min(channel, 3u);
  switch idx {
    case 0u: { return texel.x; }
    case 1u: { return texel.y; }
    case 2u: { return texel.z; }
    default: { return texel.w; }
  }
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let width = as_u32(params.width);
  let height = as_u32(params.height);
  if (gid.x >= width || gid.y >= height) {
    return;
  }

  let channel_count = max(as_u32(params.channels), 1u);
  let coords = vec2<i32>(i32(gid.x), i32(gid.y));
  let texel = textureLoad(input_tex, coords, 0);
  let base_index = (gid.y * width + gid.x) * channel_count;

  for (var channel : u32 = 0u; channel < channel_count; channel = channel + 1u) {
    let edge_val = glow_buffer.values[base_index + channel];
    let src_val = fetch_channel(texel, channel);
    let combined = 1.0 - (1.0 - clamp(edge_val, 0.0, 1.0)) * (1.0 - clamp(src_val, 0.0, 1.0));
    output_buffer.values[base_index + channel] = clamp(combined, 0.0, 1.0);
  }
}
