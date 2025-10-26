// Adjusts image brightness by adding a uniform delta and clamping to [-1, 1],
// mirroring tf.image.adjust_brightness from the Python reference.
struct AdjustBrightnessParams {
    width : f32,
    height : f32,
    channel_count : f32,
    amount : f32,
    time : f32,
    speed : f32,
    _pad0 : f32,
    _pad1 : f32,
};

const CHANNEL_COUNT : u32 = 4u;

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : AdjustBrightnessParams;

fn clamp_symmetric_vec3(value : vec3<f32>) -> vec3<f32> {
    let limits : vec3<f32> = vec3<f32>(1.0);
    return clamp(value, -limits, limits);
}

fn write_pixel(base_index : u32, rgb : vec3<f32>, alpha : f32) {
    output_buffer[base_index + 0u] = rgb.x;
    output_buffer[base_index + 1u] = rgb.y;
    output_buffer[base_index + 2u] = rgb.z;
    output_buffer[base_index + 3u] = alpha;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let width : u32 = u32(max(round(params.width), 0.0));
    let height : u32 = u32(max(round(params.height), 0.0));
    if (width == 0u || height == 0u) {
        return;
    }
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    let coords : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
    let texel : vec4<f32> = textureLoad(input_texture, coords, 0);
    let pixel_index : u32 = gid.y * width + gid.x;
    let base_index : u32 = pixel_index * CHANNEL_COUNT;

    let brightness_delta : f32 = params.amount;
    let adjusted_rgb : vec3<f32> = clamp_symmetric_vec3(texel.xyz + vec3<f32>(brightness_delta));

    write_pixel(base_index, adjusted_rgb, texel.w);
}
