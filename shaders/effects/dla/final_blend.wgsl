// DLA - Final Blend Pass
// Blend accumulated cluster with input, preserving cluster strength

struct DlaParams {
    size_padding : vec4<f32>,
    density_time : vec4<f32>,
    speed_padding : vec4<f32>,
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : DlaParams;
@group(0) @binding(3) var prev_texture : texture_2d<f32>;
@group(0) @binding(4) var<storage, read> glider_buffer : array<f32>;

fn clamp01(v : f32) -> f32 { return clamp(v, 0.0, 1.0); }

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    // Get dimensions from params, NOT from texture (they may differ)
    let width : u32 = u32(params.size_padding.x);
    let height : u32 = u32(params.size_padding.y);
    if (width == 0u || height == 0u) { return; }
    if (gid.x >= width || gid.y >= height) { return; }

    let pos : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
    let input_color : vec4<f32> = textureLoad(input_texture, pos, 0);
    
    let idx : u32 = gid.y * width + gid.x;
    let base : u32 = idx * 4u;
    
    // Cluster state lives in output_buffer, glider overlays live in glider_buffer
    let cluster_r : f32 = clamp01(output_buffer[base + 0u]);
    let cluster_b : f32 = clamp01(output_buffer[base + 2u]);
    let cluster_a : f32 = clamp01(output_buffer[base + 3u]);

    let glider_g : f32 = clamp01(glider_buffer[base + 1u]);

    // Build simulation colors
    let cluster_color : vec3<f32> = vec3<f32>(cluster_r, 0.0, cluster_b);
    let glider_color : vec3<f32> = vec3<f32>(0.0, glider_g, 0.0);
    let sim_result : vec3<f32> = cluster_color + glider_color;

    // Blend simulation over input using alpha parameter
    let alpha_param : f32 = params.density_time.z;
    let blended : vec3<f32> = mix(input_color.xyz, input_color.xyz + sim_result, alpha_param);

    // Write blended display, but preserve PURE cluster state in R and B for feedback
    output_buffer[base + 0u] = mix(blended.x, cluster_r, cluster_r);  // Preserve magenta R
    output_buffer[base + 1u] = blended.y;
    output_buffer[base + 2u] = mix(blended.z, cluster_b, cluster_b);  // Preserve magenta B
    output_buffer[base + 3u] = max(input_color.w, max(cluster_a, glider_g));
}
