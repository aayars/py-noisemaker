// Stray Hair final combine pass.
// Generates sparse, long hair-like strands using worms with high kink values.

struct StrayHairParams {
    width : f32,
    height : f32,
    channel_count : f32,
    _pad0 : f32,
    time : f32,
    speed : f32,
    seed : f32,
    _pad1 : f32,
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var worm_texture : texture_2d<f32>;
@group(0) @binding(2) var brightness_texture : texture_2d<f32>;
@group(0) @binding(3) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(4) var<uniform> params : StrayHairParams;

const CHANNEL_COUNT : u32 = 4u;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let dims : vec2<u32> = textureDimensions(input_texture, 0);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let coords : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
    let base_color : vec4<f32> = textureLoad(input_texture, coords, 0);
    let worm_mask : vec4<f32> = textureLoad(worm_texture, coords, 0);
    let brightness_sample : vec4<f32> = textureLoad(brightness_texture, coords, 0);

    let mask_rgb : vec3<f32> = clamp(worm_mask.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    let blend_factor : vec3<f32> = clamp(mask_rgb * 0.666, vec3<f32>(0.0), vec3<f32>(1.0));
    let brightness_rgb : vec3<f32> = clamp(brightness_sample.rgb * 0.333, vec3<f32>(0.0), vec3<f32>(1.0));

    let base_component : vec3<f32> = base_color.rgb * (vec3<f32>(1.0) - blend_factor);
    let hair_component : vec3<f32> = brightness_rgb * blend_factor;
    let hair_rgb : vec3<f32> = clamp(base_component + hair_component, vec3<f32>(0.0), vec3<f32>(1.0));

    let width_u : u32 = dims.x;
    let index : u32 = (gid.y * width_u + gid.x) * CHANNEL_COUNT;
    output_buffer[index + 0u] = hair_rgb.r;
    output_buffer[index + 1u] = hair_rgb.g;
    output_buffer[index + 2u] = hair_rgb.b;
    output_buffer[index + 3u] = base_color.a;
}

