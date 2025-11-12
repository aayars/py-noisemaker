// Lowpoly combine pass. Consumes Voronoi range + color textures produced by
// shared lower-level shaders and applies faceted shading before the normalize
// stage runs in JavaScript.

const CHANNEL_COUNT : u32 = 4u;
const NORMAL_Z_SCALE : f32 = 1.6;

struct LowpolyParams {
    dims : vec4<f32>,                     // width, height, channel count, unused
    distrib_freq_time_speed : vec4<f32>,  // distrib, freq, time, speed
    dist_metric_pad : vec4<f32>,          // dist_metric, pad
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : LowpolyParams;
@group(0) @binding(3) var voronoi_color_texture : texture_2d<f32>;
@group(0) @binding(4) var voronoi_range_texture : texture_2d<f32>;

fn as_u32(value : f32) -> u32 {
    return u32(max(round(value), 0.0));
}

fn clamp_coord(value : i32, limit : i32) -> i32 {
    if (limit <= 0) {
        return 0;
    }
    if (limit == 1) {
        return 0;
    }
    return clamp(value, 0, limit - 1);
}

fn sample_range(coord : vec2<i32>) -> f32 {
    return textureLoad(voronoi_range_texture, coord, 0).x;
}

fn luminance(color : vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.299, 0.587, 0.114));
}

fn compute_shading(coords : vec2<i32>, width : i32, height : i32) -> f32 {
    if (width <= 0 || height <= 0) {
        return 1.0;
    }

    let left_coord : vec2<i32> = vec2<i32>(clamp_coord(coords.x - 1, width), coords.y);
    let right_coord : vec2<i32> = vec2<i32>(clamp_coord(coords.x + 1, width), coords.y);
    let up_coord : vec2<i32> = vec2<i32>(coords.x, clamp_coord(coords.y - 1, height));
    let down_coord : vec2<i32> = vec2<i32>(coords.x, clamp_coord(coords.y + 1, height));

    let left_val : f32 = sample_range(left_coord);
    let right_val : f32 = sample_range(right_coord);
    let up_val : f32 = sample_range(up_coord);
    let down_val : f32 = sample_range(down_coord);

    let dx : f32 = right_val - left_val;
    let dy : f32 = down_val - up_val;
    let normal : vec3<f32> = normalize(vec3<f32>(-dx, -dy, NORMAL_Z_SCALE));
    let light_dir : vec3<f32> = normalize(vec3<f32>(0.35, 0.55, 1.0));
    let lambert : f32 = clamp(dot(normal, light_dir), 0.1, 1.0);
    let rim : f32 = pow(1.0 - clamp(sample_range(coords), 0.0, 1.0), 2.0);
    return clamp(lambert * 0.85 + rim * 0.15, 0.1, 1.2);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let width_u : u32 = as_u32(params.dims.x);
    let height_u : u32 = as_u32(params.dims.y);
    if (gid.x >= width_u || gid.y >= height_u) {
        return;
    }

    let coords : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
    let base_sample : vec4<f32> = textureLoad(input_texture, coords, 0);
    let voronoi_color_sample : vec4<f32> = textureLoad(voronoi_color_texture, coords, 0);
    let voronoi_range_sample : vec4<f32> = textureLoad(voronoi_range_texture, coords, 0);

    let range_value : f32 = clamp(voronoi_range_sample.x, 0.0, 1.0);
    let distance_rgb : vec3<f32> = vec3<f32>(range_value, range_value, range_value);
    let color_rgb : vec3<f32> = voronoi_color_sample.xyz;
    let direct_mix : vec3<f32> = mix(distance_rgb, color_rgb, 0.125);

    let width_i : i32 = i32(width_u);
    let height_i : i32 = i32(height_u);
    let shade : f32 = compute_shading(coords, width_i, height_i);

    let detail_mix : f32 = clamp(luminance(color_rgb) * 0.75 + 0.25, 0.0, 1.5);
    let faceted : vec3<f32> = direct_mix * shade * detail_mix;
    let subtle_original : vec3<f32> = mix(faceted, base_sample.xyz, 0.1);

    let base_index : u32 = (gid.y * width_u + gid.x) * CHANNEL_COUNT;
    output_buffer[base_index + 0u] = subtle_original.x;
    output_buffer[base_index + 1u] = subtle_original.y;
    output_buffer[base_index + 2u] = subtle_original.z;
    output_buffer[base_index + 3u] = base_sample.w;
}
