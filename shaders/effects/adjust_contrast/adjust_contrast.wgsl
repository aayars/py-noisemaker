// Adjusts per-pixel contrast by scaling distance from mid-gray (0.5), matching tf.image.adjust_contrast
// behaviour for normalized inputs while favouring performance over an exact global mean.
struct AdjustContrastParams {
    width : f32,
    height : f32,
    channel_count : f32,
    amount : f32,
    time : f32,
    speed : f32,
    _pad0 : f32,
    _pad1 : f32,
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : AdjustContrastParams;

fn as_u32(value : f32) -> u32 {
    return u32(max(round(value), 0.0));
}

fn clamp01_vec3(value : vec3<f32>) -> vec3<f32> {
    let min_value : vec3<f32> = vec3<f32>(0.0);
    let max_value : vec3<f32> = vec3<f32>(1.0);
    return clamp(value, min_value, max_value);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let width : u32 = as_u32(params.width);
    let height : u32 = as_u32(params.height);
    if (width == 0u || height == 0u) {
        return;
    }

    if (global_id.x >= width || global_id.y >= height) {
        return;
    }

    let coords : vec2<i32> = vec2<i32>(i32(global_id.x), i32(global_id.y));
    let texel : vec4<f32> = textureLoad(input_texture, coords, 0);
    let pixel_index : u32 = global_id.y * width + global_id.x;
    let base_index : u32 = pixel_index * 4u;

    let amount : f32 = params.amount;
    if (amount == 1.0) {
        output_buffer[base_index + 0u] = texel.x;
        output_buffer[base_index + 1u] = texel.y;
        output_buffer[base_index + 2u] = texel.z;
        output_buffer[base_index + 3u] = texel.w;
        return;
    }

    let mid_gray : vec3<f32> = vec3<f32>(0.5);
    let adjusted_rgb : vec3<f32> = clamp01_vec3((texel.xyz - mid_gray) * vec3<f32>(amount) + mid_gray);

    output_buffer[base_index + 0u] = adjusted_rgb.x;
    output_buffer[base_index + 1u] = adjusted_rgb.y;
    output_buffer[base_index + 2u] = adjusted_rgb.z;
    output_buffer[base_index + 3u] = texel.w;
}
