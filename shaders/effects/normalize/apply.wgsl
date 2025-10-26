// Normalize Pass 2: Apply normalization using computed min/max values.
// This pass remaps all pixel values to the [0, 1] range.

struct NormalizeParams {
    dimensions : vec4<f32>, // width, height, channels, _pad0
    animation : vec4<f32>, // time, speed, _pad1, _pad2
};

const CHANNEL_COUNT : u32 = 4u;
const F32_MAX : f32 = 0x1.fffffep+127;
const F32_MIN : f32 = -0x1.fffffep+127;

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : NormalizeParams;
@group(0) @binding(3) var<storage, read_write> stats_buffer : array<f32>;

fn as_u32(value : f32) -> u32 {
    return u32(max(round(value), 0.0));
}

fn pixel_base_index(x : u32, y : u32, width : u32) -> u32 {
    let pixel_index : u32 = y * width + x;
    return pixel_index * CHANNEL_COUNT;
}

fn write_pixel(base_index : u32, color : vec4<f32>) {
    output_buffer[base_index + 0u] = color.x;
    output_buffer[base_index + 1u] = color.y;
    output_buffer[base_index + 2u] = color.z;
    output_buffer[base_index + 3u] = color.w;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let width : u32 = as_u32(params.dimensions.x);
    let height : u32 = as_u32(params.dimensions.y);
    
    // Early exit for out-of-bounds threads
    if (gid.x >= width || gid.y >= height) {
        return;
    }
    
    // Read final min/max (computed by pass 1, reduced by pass 2)
    let min_val : f32 = stats_buffer[0];
    let max_val : f32 = stats_buffer[1];
    
    // Check for invalid values (NaN via self-inequality)
    let min_nan : bool = !(min_val == min_val);
    let max_nan : bool = !(max_val == max_val);
    
    let coords : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
    let texel : vec4<f32> = textureLoad(input_texture, coords, 0);
    let base_index : u32 = pixel_base_index(gid.x, gid.y, width);
    
    // If NaN or min == max, just copy the source
    if (min_nan || max_nan || min_val == max_val) {
        write_pixel(base_index, texel);
        return;
    }
    
    // Apply normalization: (value - min) / (max - min)
    let inv_range : f32 = 1.0 / (max_val - min_val);
    let min_vec : vec4<f32> = vec4<f32>(min_val);
    let inv_range_vec : vec4<f32> = vec4<f32>(inv_range);
    let normalized : vec4<f32> = (texel - min_vec) * inv_range_vec;
    
    write_pixel(base_index, normalized);
}
