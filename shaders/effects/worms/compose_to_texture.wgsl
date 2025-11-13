// Worms effect - Compose trails with input texture for display

struct WormsParams {
    size : vec4<f32>,
    behavior_density_stride_padding : vec4<f32>,
    stride_deviation_alpha_kink : vec3<f32>,
    quantize_time_padding_intensity : vec4<f32>,
    inputIntensity_lifetime_padding : vec4<f32>,
};

@group(0) @binding(0) var<storage, read> output_buffer : array<f32>;
@group(0) @binding(1) var output_texture : texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> params : WormsParams;
@group(0) @binding(3) var input_texture : texture_2d<f32>;

fn clamp_01(value : f32) -> f32 {
    return clamp(value, 0.0, 1.0);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let width : u32 = u32(max(params.size.x, 1.0));
    let height : u32 = u32(max(params.size.y, 1.0));
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    let pixel_idx : u32 = gid.y * width + gid.x;
    let base_index : u32 = pixel_idx * 4u;

    let base_sample : vec4<f32> = textureLoad(input_texture, vec2<i32>(i32(gid.x), i32(gid.y)), 0);
    let input_intensity : f32 = clamp_01(params.inputIntensity_lifetime_padding.x);
    let base_color : vec4<f32> = vec4<f32>(base_sample.xyz * input_intensity, base_sample.w);

    let trail_color : vec4<f32> = vec4<f32>(
        output_buffer[base_index + 0u],
        output_buffer[base_index + 1u],
        output_buffer[base_index + 2u],
        output_buffer[base_index + 3u]
    );

    let combined_rgb : vec3<f32> = clamp(base_color.xyz + trail_color.xyz, vec3<f32>(0.0), vec3<f32>(1.0));
    let final_alpha : f32 = clamp(max(base_color.w, trail_color.w), 0.0, 1.0);

    textureStore(output_texture, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(combined_rgb, final_alpha));
}
