// Erosion Worms - Pass 0: Initialize from previous frame
// Copies the previous trail texture into the working buffer with configurable fade.

struct ErosionWormsParams {
    size : vec4<f32>,
    controls0 : vec4<f32>,
    controls1 : vec4<f32>,
    controls2 : vec4<f32>,
};

@group(0) @binding(0) var input_texture : texture_2d<f32>; // Unused but required for layout parity
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : ErosionWormsParams;
@group(0) @binding(3) var prev_texture : texture_2d<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let dims : vec2<u32> = textureDimensions(prev_texture, 0);
    let width : u32 = dims.x;
    let height : u32 = dims.y;

    if (width == 0u || height == 0u) {
        return;
    }

    if (gid.x >= width || gid.y >= height) {
        return;
    }

    let fade : f32 = clamp(params.controls1.x, 0.0, 1.0);
    let pixel_idx : u32 = gid.y * width + gid.x;
    let base : u32 = pixel_idx * 4u;

    let prev_sample : vec4<f32> = textureLoad(prev_texture, vec2<i32>(i32(gid.x), i32(gid.y)), 0);
    let faded : vec4<f32> = prev_sample * fade;

    output_buffer[base + 0u] = faded.x;
    output_buffer[base + 1u] = faded.y;
    output_buffer[base + 2u] = faded.z;
    output_buffer[base + 3u] = faded.w;
}
