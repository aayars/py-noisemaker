// Compute shader for the inner_tile effect. It extracts an inner grid of the
// source image by selecting pixels spaced by the provided frequency and writes
// the tiled result into a storage buffer for subsequent resampling.

struct InnerTileParams {
  dimsA : vec4<u32>; // width, height, channels, freqX
  dimsB : vec4<u32>; // freqY, innerW, innerH, tileWidth
  dimsC : vec4<u32>; // tileHeight, padding, padding, padding
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : InnerTileParams;

fn sample_channel(value : vec4<f32>, index : u32) -> f32 {
  switch index {
    case 0u: { return value.x; }
    case 1u: { return value.y; }
    case 2u: { return value.z; }
    default: { return value.w; }
  }
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
  let tile_width : u32 = max(params.dimsB.w, 1u);
  let tile_height : u32 = max(params.dimsC.x, 1u);
  if (global_id.x >= tile_width || global_id.y >= tile_height) {
    return;
  }

  let src_width : u32 = max(params.dimsA.x, 1u);
  let src_height : u32 = max(params.dimsA.y, 1u);
  let channel_count : u32 = max(min(params.dimsA.z, 4u), 1u);
  let freq_x : u32 = max(params.dimsA.w, 1u);
  let freq_y : u32 = max(params.dimsB.x, 1u);
  let inner_w : u32 = max(params.dimsB.y, 1u);
  let inner_h : u32 = max(params.dimsB.z, 1u);

  let inner_x : u32 = global_id.x % inner_w;
  let inner_y : u32 = global_id.y % inner_h;
  var sample_x : u32 = inner_x * freq_x;
  var sample_y : u32 = inner_y * freq_y;
  if (sample_x >= src_width) {
    sample_x = src_width - 1u;
  }
  if (sample_y >= src_height) {
    sample_y = src_height - 1u;
  }

  let coords : vec2<i32> = vec2<i32>(i32(sample_x), i32(sample_y));
  let value : vec4<f32> = textureLoad(input_texture, coords, 0);

  let base_index : u32 = (global_id.y * tile_width + global_id.x) * channel_count;
  var channel : u32 = 0u;
  loop {
    if (channel >= channel_count) {
      break;
    }
    output_buffer[base_index + channel] = sample_channel(value, channel);
    channel = channel + 1u;
  }
}
