// FXAA antialiasing pass translated from noisemaker/value.py:fxaa.
// Applies an edge-aware blur weighted by luminance differences while preserving alpha.

struct FxaaParams {
    size : vec4<f32>,       // (width, height, channels, unused)
    time_speed : vec4<f32>, // (time, speed, unused, unused)
};

const CHANNEL_COUNT : u32 = 4u;
const EPSILON : f32 = 1e-10;
const LUMA_WEIGHTS : vec3<f32> = vec3<f32>(0.299, 0.587, 0.114);

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : FxaaParams;

fn as_u32(value : f32) -> u32 {
    return u32(max(round(value), 0.0));
}

fn sanitized_channel_count(channel_value : f32) -> u32 {
    let rounded : i32 = i32(round(channel_value));
    if (rounded <= 1) {
        return 1u;
    }
    if (rounded >= 4) {
        return 4u;
    }
    return u32(rounded);
}

fn reflect_coord(coord : i32, limit : i32) -> i32 {
    if (limit <= 1) {
        return 0;
    }

    let period : i32 = 2 * limit - 2;
    var wrapped : i32 = coord % period;
    if (wrapped < 0) {
        wrapped = wrapped + period;
    }

    if (wrapped < limit) {
        return wrapped;
    }

    return period - wrapped;
}

fn load_texel(coord : vec2<i32>, size : vec2<i32>) -> vec4<f32> {
    let reflected_x : i32 = reflect_coord(coord.x, size.x);
    let reflected_y : i32 = reflect_coord(coord.y, size.y);
    return textureLoad(input_texture, vec2<i32>(reflected_x, reflected_y), 0);
}

fn luminance_from_rgb(rgb : vec3<f32>) -> f32 {
    return dot(rgb, LUMA_WEIGHTS);
}

fn weight_from_luma(center_luma : f32, neighbor_luma : f32) -> f32 {
    return exp(-abs(center_luma - neighbor_luma));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let width_u : u32 = max(as_u32(params.size.x), 1u);
    let height_u : u32 = max(as_u32(params.size.y), 1u);
    if (global_id.x >= width_u || global_id.y >= height_u) {
        return;
    }

    let pixel_index : u32 = global_id.y * width_u + global_id.x;
    let base_index : u32 = pixel_index * CHANNEL_COUNT;
    let channel_count : u32 = sanitized_channel_count(params.size.z);

    let image_size : vec2<i32> = vec2<i32>(i32(width_u), i32(height_u));
    let pixel_coord : vec2<i32> = vec2<i32>(i32(global_id.x), i32(global_id.y));

    let center_texel : vec4<f32> = load_texel(pixel_coord, image_size);
    let north_texel : vec4<f32> = load_texel(pixel_coord + vec2<i32>(0, -1), image_size);
    let south_texel : vec4<f32> = load_texel(pixel_coord + vec2<i32>(0, 1), image_size);
    let west_texel : vec4<f32> = load_texel(pixel_coord + vec2<i32>(-1, 0), image_size);
    let east_texel : vec4<f32> = load_texel(pixel_coord + vec2<i32>(1, 0), image_size);

    let center_rgb : vec3<f32> = center_texel.xyz;
    let north_rgb : vec3<f32> = north_texel.xyz;
    let south_rgb : vec3<f32> = south_texel.xyz;
    let west_rgb : vec3<f32> = west_texel.xyz;
    let east_rgb : vec3<f32> = east_texel.xyz;

    var center_luma : f32;
    var north_luma : f32;
    var south_luma : f32;
    var west_luma : f32;
    var east_luma : f32;

    if (channel_count >= 3u) {
        center_luma = luminance_from_rgb(center_rgb);
        north_luma = luminance_from_rgb(north_rgb);
        south_luma = luminance_from_rgb(south_rgb);
        west_luma = luminance_from_rgb(west_rgb);
        east_luma = luminance_from_rgb(east_rgb);
    } else {
        center_luma = center_texel.x;
        north_luma = north_texel.x;
        south_luma = south_texel.x;
        west_luma = west_texel.x;
        east_luma = east_texel.x;
    }

    let weight_center : f32 = 1.0;
    let weight_north : f32 = weight_from_luma(center_luma, north_luma);
    let weight_south : f32 = weight_from_luma(center_luma, south_luma);
    let weight_west : f32 = weight_from_luma(center_luma, west_luma);
    let weight_east : f32 = weight_from_luma(center_luma, east_luma);
    let weight_sum : f32 = weight_center + weight_north + weight_south + weight_west + weight_east + EPSILON;

    var result_texel : vec4<f32> = center_texel;
    if (channel_count <= 2u) {
        let blended_luma : f32 = (
            center_texel.x * weight_center
            + north_texel.x * weight_north
            + south_texel.x * weight_south
            + west_texel.x * weight_west
            + east_texel.x * weight_east
        ) / weight_sum;

        result_texel.x = blended_luma;
        if (channel_count == 1u) {
            result_texel.y = center_texel.y;
            result_texel.z = center_texel.z;
        }
    } else {
        let blended_rgb : vec3<f32> = (
            center_rgb * weight_center
            + north_rgb * weight_north
            + south_rgb * weight_south
            + west_rgb * weight_west
            + east_rgb * weight_east
        ) / weight_sum;

    result_texel = vec4<f32>(blended_rgb, result_texel.w);
    }

    result_texel.w = center_texel.w;

    output_buffer[base_index + 0u] = result_texel.x;
    output_buffer[base_index + 1u] = result_texel.y;
    output_buffer[base_index + 2u] = result_texel.z;
    output_buffer[base_index + 3u] = result_texel.w;
}
