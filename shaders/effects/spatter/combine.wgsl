// Blends intermediate spatter layers into a final mask.
// Inputs are grayscale textures (smear, spatter 1, spatter 2, removal).

const CHANNEL_COUNT : u32 = 4u;

struct CombineParams {
    size : vec4<f32>, // (width, height, channel_count, unused)
};

@group(0) @binding(0) var smear_texture : texture_2d<f32>;
@group(0) @binding(1) var spatter_primary_texture : texture_2d<f32>;
@group(0) @binding(2) var spatter_secondary_texture : texture_2d<f32>;
@group(0) @binding(3) var removal_texture : texture_2d<f32>;
@group(0) @binding(4) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(5) var<uniform> params : CombineParams;

fn as_u32(value : f32) -> u32 {
    if (value <= 0.0) {
        return 0u;
    }
    return u32(round(value));
}

fn clamp01(value : f32) -> f32 {
    return clamp(value, 0.0, 1.0);
}

fn sample_grayscale(tex : texture_2d<f32>, coords : vec2<i32>) -> f32 {
    return clamp01(textureLoad(tex, coords, 0).x);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let width : u32 = as_u32(params.size.x);
    let height : u32 = as_u32(params.size.y);
    if (width == 0u || height == 0u) {
        return;
    }
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    let coords : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
    let smear_value : f32 = sample_grayscale(smear_texture, coords);
    let primary_value : f32 = sample_grayscale(spatter_primary_texture, coords);
    let secondary_value : f32 = sample_grayscale(spatter_secondary_texture, coords);
    let removal_value : f32 = sample_grayscale(removal_texture, coords);

    let combined : f32 = max(smear_value, max(primary_value, secondary_value));
    let masked : f32 = max(0.0, combined - removal_value);
    let result : f32 = clamp01(masked);

    let base_index : u32 = (gid.y * width + gid.x) * CHANNEL_COUNT;
    output_buffer[base_index + 0u] = result;
    output_buffer[base_index + 1u] = result;
    output_buffer[base_index + 2u] = result;
    output_buffer[base_index + 3u] = 1.0;
}
