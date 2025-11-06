// Worms effect - Pass 3: Final blend
// Blend accumulated trails with input texture

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : WormsParams;

struct WormsParams {
    size : vec4<f32>,
    behavior_density_stride_padding : vec4<f32>,
    stride_deviation_alpha_kink_drunkenness : vec4<f32>,
    quantize_time_padding_intensity : vec4<f32>,
    inputIntensity_padding : vec4<f32>,
};

fn clamp_01(value : f32) -> f32 {
    return clamp(value, 0.0, 1.0);
}

fn clamp_vec3(value : vec3<f32>) -> vec3<f32> {
    return clamp(value, vec3<f32>(0.0), vec3<f32>(1.0));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let dims : vec2<u32> = textureDimensions(input_texture, 0);
    let width : u32 = dims.x;
    let height : u32 = dims.y;

    if (gid.x >= width || gid.y >= height) {
        return;
    }

    let pixel_idx : u32 = gid.y * width + gid.x;
    let base : u32 = pixel_idx * 4u;

    let base_sample : vec4<f32> = textureLoad(input_texture, vec2<i32>(i32(gid.x), i32(gid.y)), 0);
    let input_intensity : f32 = clamp(params.inputIntensity_padding.x, 0.0, 1.0);
    let base_color : vec4<f32> = vec4<f32>(base_sample.xyz * input_intensity, base_sample.w);

    let worms_color : vec4<f32> = vec4<f32>(
        output_buffer[base + 0u],
        output_buffer[base + 1u],
        output_buffer[base + 2u],
        output_buffer[base + 3u]
    );

    let combined_rgb : vec3<f32> = clamp_vec3(base_color.xyz + worms_color.xyz);
    let final_alpha : f32 = clamp(max(base_color.w, worms_color.w), 0.0, 1.0);

    output_buffer[base + 0u] = combined_rgb.x;
    output_buffer[base + 1u] = combined_rgb.y;
    output_buffer[base + 2u] = combined_rgb.z;
    output_buffer[base + 3u] = final_alpha;
}
