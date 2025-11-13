// Converts linear RGBA storage buffer data into a 2D storage texture.

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

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let width : u32 = u32(max(params.size.x, 1.0));
    let height : u32 = u32(max(params.size.y, 1.0));
    if (gid.x >= width || gid.y >= height) {
        return;
    }
    let pixel_idx : u32 = gid.y * width + gid.x;
    let base : u32 = pixel_idx * 4u;
    let color : vec4<f32> = vec4<f32>(
        output_buffer[base + 0u],
        output_buffer[base + 1u],
        output_buffer[base + 2u],
        output_buffer[base + 3u]
    );
    textureStore(output_texture, vec2<i32>(i32(gid.x), i32(gid.y)), color);
}
