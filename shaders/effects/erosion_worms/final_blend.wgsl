// Erosion Worms - Pass 2: Final Blend
// Composites the accumulated trail buffer with the current input texture.

struct ErosionWormsParams {
    size : vec4<f32>,
    controls0 : vec4<f32>,
    controls1 : vec4<f32>,
    controls2 : vec4<f32>,
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : ErosionWormsParams;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let dims : vec2<u32> = textureDimensions(input_texture, 0);
    let width : u32 = dims.x;
    let height : u32 = dims.y;

    if (gid.x >= width || gid.y >= height) {
        return;
    }

    let pixel_idx : u32 = gid.y * width + gid.x;
    let base_index : u32 = pixel_idx * 4u;

    let trail_color : vec4<f32> = vec4<f32>(
        output_buffer[base_index + 0u],
        output_buffer[base_index + 1u],
        output_buffer[base_index + 2u],
        output_buffer[base_index + 3u]
    );

    let input_sample : vec4<f32> = textureLoad(input_texture, vec2<i32>(i32(gid.x), i32(gid.y)), 0);
    let base_intensity : f32 = clamp(params.controls2.y, 0.0, 1.0);
    let base_rgb : vec3<f32> = input_sample.xyz * base_intensity;

    let combined_rgb : vec3<f32> = clamp(base_rgb + trail_color.xyz, vec3<f32>(0.0), vec3<f32>(1.0));
    let combined_alpha : f32 = clamp(max(input_sample.w, trail_color.w), 0.0, 1.0);

    output_buffer[base_index + 0u] = combined_rgb.x;
    output_buffer[base_index + 1u] = combined_rgb.y;
    output_buffer[base_index + 2u] = combined_rgb.z;
    output_buffer[base_index + 3u] = combined_alpha;
}
