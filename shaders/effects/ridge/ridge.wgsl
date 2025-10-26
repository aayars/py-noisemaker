// Ridge effect.
// Implements the ridge transform from noisemaker/value.py: 1 - abs(n * 2 - 1).

struct RidgeParams {
    width : f32,
    height : f32,
    channels : f32,
    time : f32,
    speed : f32,
    _pad0 : f32,
    _pad1 : f32,
    _pad2 : f32,
};

const CHANNEL_COUNT : u32 = 4u;
const RIDGE_SCALE : f32 = 2.0;
const RIDGE_OFFSET : f32 = 1.0;

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : RidgeParams;

fn as_u32(value : f32) -> u32 {
    return u32(max(round(value), 0.0));
}

fn ridge_transform(value : vec4<f32>) -> vec4<f32> {
    let scaled : vec4<f32> = value * RIDGE_SCALE - vec4<f32>(RIDGE_OFFSET);
    return clamp(vec4<f32>(1.0) - abs(scaled), vec4<f32>(0.0), vec4<f32>(1.0));
}

fn write_pixel(base_index : u32, color : vec4<f32>) {
    output_buffer[base_index + 0u] = color.x;
    output_buffer[base_index + 1u] = color.y;
    output_buffer[base_index + 2u] = color.z;
    output_buffer[base_index + 3u] = color.w;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    // Derive dimensions from the bound input texture to avoid relying on uniforms
    let dims : vec2<u32> = textureDimensions(input_texture, 0);
    let width : u32 = dims.x;
    let height : u32 = dims.y;
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    let coords : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
    let texel : vec4<f32> = textureLoad(input_texture, coords, 0);
    let pixel_index : u32 = gid.y * width + gid.x;
    let base_index : u32 = pixel_index * CHANNEL_COUNT;
    let ridged : vec4<f32> = ridge_transform(texel);
    // Preserve alpha from source if present, otherwise default to 1
    let out_color : vec4<f32> = vec4<f32>(ridged.xyz, select(1.0, texel.w, texel.w > 0.0));
    write_pixel(base_index, out_color);
}
