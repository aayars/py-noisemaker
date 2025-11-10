// Glowing Edges final combine shader
// Reuses: posterize, convolve (Sobel), sobel (combine), bloom, normalize
// This shader performs only the last blend step:
//   edges_prep = clamp(edges * 8.0, 0.0, 1.0) * clamp(base * 1.25, 0.0, 1.0)
//   screen = 1.0 - (1.0 - edges_prep) * (1.0 - base)
//   out = mix(base, screen, alpha)

const CHANNEL_COUNT : u32 = 4u;

struct FinalParams {
    width : f32,
    height : f32,
    channel_count : f32,
    alpha : f32,
    sobel_metric : f32,
    time : f32,
    speed : f32,
};

@group(0) @binding(0) var base_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : FinalParams;
@group(0) @binding(3) var edges_texture : texture_2d<f32>;

fn write_pixel(base_index : u32, color : vec4<f32>) {
    output_buffer[base_index + 0u] = color.x;
    output_buffer[base_index + 1u] = color.y;
    output_buffer[base_index + 2u] = color.z;
    output_buffer[base_index + 3u] = color.w;
}

fn clamp01(v : vec3<f32>) -> vec3<f32> {
    return clamp(v, vec3<f32>(0.0), vec3<f32>(1.0));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let width : u32 = max(u32(params.width), 1u);
    let height : u32 = max(u32(params.height), 1u);
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    let xy : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
    let base : vec4<f32> = textureLoad(base_texture, xy, 0);
    let edges : vec4<f32> = textureLoad(edges_texture, xy, 0);

    let edges_scaled : vec3<f32> = clamp01(edges.xyz * 8.0);
    let base_scaled : vec3<f32> = clamp01(base.xyz * 1.25);
    let edges_prep : vec3<f32> = edges_scaled * base_scaled;

    let screen_rgb : vec3<f32> = vec3<f32>(1.0) - (vec3<f32>(1.0) - edges_prep) * (vec3<f32>(1.0) - base.xyz);
    let alpha : f32 = clamp(params.alpha, 0.0, 1.0);
    let mixed_rgb : vec3<f32> = mix(base.xyz, screen_rgb, alpha);

    let out_color : vec4<f32> = vec4<f32>(clamp01(mixed_rgb), base.w);
    let base_index : u32 = (gid.y * width + gid.x) * CHANNEL_COUNT;
    write_pixel(base_index, out_color);
}
