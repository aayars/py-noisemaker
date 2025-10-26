// Vaseline effect: blend bloom toward the center using a Chebyshev mask.
// Mirrors noisemaker.effects.vaseline and bloom implementations.

struct VaselineParams {
    width : f32,
    height : f32,
    channel_count : f32,
    _pad0 : f32,
    alpha : f32,
    time : f32,
    speed : f32,
    _pad1 : f32,
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : VaselineParams;

const SAMPLE_POSITIONS : array<vec2<f32>, 9> = array<vec2<f32>, 9>(
    vec2<f32>(1.0 / 6.0, 1.0 / 6.0),
    vec2<f32>(0.5, 1.0 / 6.0),
    vec2<f32>(5.0 / 6.0, 1.0 / 6.0),
    vec2<f32>(1.0 / 6.0, 0.5),
    vec2<f32>(0.5, 0.5),
    vec2<f32>(5.0 / 6.0, 0.5),
    vec2<f32>(1.0 / 6.0, 5.0 / 6.0),
    vec2<f32>(0.5, 5.0 / 6.0),
    vec2<f32>(5.0 / 6.0, 5.0 / 6.0),
);

const SAMPLE_COUNT : u32 = 9u;
const BOOST : f32 = 4.0;
const BRIGHTNESS_ADJUST : f32 = 0.25;
const CONTRAST_SCALE : f32 = 1.5;
const OFFSET_SCALE : f32 = -0.05;

struct DownsampleResult {
    value : vec4<f32>,
    weight : f32,
};

fn as_u32(value : f32) -> u32 {
    return u32(max(round(value), 0.0));
}

fn clamp_vec01(value : vec4<f32>) -> vec4<f32> {
    return clamp(value, vec4<f32>(0.0), vec4<f32>(1.0));
}

fn clamp_vec_symmetric(value : vec4<f32>) -> vec4<f32> {
    return clamp(value, vec4<f32>(-1.0), vec4<f32>(1.0));
}

fn wrap_index(value : i32, size : i32) -> i32 {
    if (size <= 0) {
        return 0;
    }
    var wrapped : i32 = value % size;
    if (wrapped < 0) {
        wrapped = wrapped + size;
    }
    return wrapped;
}

fn clamp_coord(coord : vec2<i32>, image_size : vec2<i32>) -> vec2<i32> {
    return vec2<i32>(
        clamp(coord.x, 0, max(image_size.x - 1, 0)),
        clamp(coord.y, 0, max(image_size.y - 1, 0)),
    );
}

fn sample_pre_down(coord : vec2<i32>, image_size : vec2<i32>) -> vec4<f32> {
    let clamped : vec2<i32> = clamp_coord(coord, image_size);
    let texel : vec4<f32> = textureLoad(input_texture, clamped, 0);
    return clamp_vec01(texel * vec4<f32>(2.0) - vec4<f32>(1.0));
}

fn cell_origin(cell : vec2<i32>, kernel : vec2<i32>) -> vec2<i32> {
    return vec2<i32>(cell.x * kernel.x, cell.y * kernel.y);
}

fn cell_extent(origin : vec2<i32>, kernel : vec2<i32>, image_size : vec2<i32>) -> vec2<i32> {
    let max_corner : vec2<i32> = vec2<i32>(
        min(origin.x + kernel.x, image_size.x),
        min(origin.y + kernel.y, image_size.y),
    );
    return vec2<i32>(
        max(max_corner.x - origin.x, 0),
        max(max_corner.y - origin.y, 0),
    );
}

fn fetch_downsample_cell(
    cell : vec2<i32>,
    kernel : vec2<i32>,
    image_size : vec2<i32>,
) -> DownsampleResult {
    let origin : vec2<i32> = cell_origin(cell, kernel);
    let extent : vec2<i32> = cell_extent(origin, kernel, image_size);
    let safe_extent : vec2<i32> = vec2<i32>(max(extent.x, 1), max(extent.y, 1));
    let range : vec2<f32> = vec2<f32>(
        f32(max(safe_extent.x - 1, 0)),
        f32(max(safe_extent.y - 1, 0)),
    );

    var accum : vec4<f32> = vec4<f32>(0.0);
    for (var i : u32 = 0u; i < SAMPLE_COUNT; i = i + 1u) {
        let offset_norm : vec2<f32> = SAMPLE_POSITIONS[i];
        let offset_float : vec2<f32> = vec2<f32>(
            offset_norm.x * range.x,
            offset_norm.y * range.y
        );
        let local : vec2<i32> = vec2<i32>(
            clamp(i32(round(offset_float.x)), 0, safe_extent.x - 1),
            clamp(i32(round(offset_float.y)), 0, safe_extent.y - 1)
        );
        let sample_coord : vec2<i32> = origin + local;
        accum = accum + sample_pre_down(sample_coord, image_size);
    }

    let average : vec4<f32> = accum / vec4<f32>(f32(SAMPLE_COUNT));
    let weight : f32 = f32(safe_extent.x * safe_extent.y);
    return DownsampleResult(average, weight);
}

fn cubic_interpolate(
    a : vec4<f32>,
    b : vec4<f32>,
    c : vec4<f32>,
    d : vec4<f32>,
    t : f32,
) -> vec4<f32> {
    let t2 : f32 = t * t;
    let t3 : f32 = t2 * t;
    let a0 : vec4<f32> = d - c - a + b;
    let a1 : vec4<f32> = a - b - a0;
    let a2 : vec4<f32> = c - a;
    let a3 : vec4<f32> = b;
    return ((a0 * t3) + (a1 * t2)) + (a2 * t) + a3;
}

fn store_pixel(pixel_index : u32, value : vec4<f32>) {
    output_buffer[pixel_index + 0u] = value.x;
    output_buffer[pixel_index + 1u] = value.y;
    output_buffer[pixel_index + 2u] = value.z;
    output_buffer[pixel_index + 3u] = value.w;
}

fn chebyshev_mask(uv : vec2<f32>, dimensions : vec2<f32>) -> f32 {
    if (dimensions.x <= 0.0 || dimensions.y <= 0.0) {
        return 0.0;
    }

    let centered : vec2<f32> = abs(uv - vec2<f32>(0.5, 0.5));
    let px : f32 = centered.x * dimensions.x;
    let py : f32 = centered.y * dimensions.y;
    let dist : f32 = max(px, py);
    let max_dimension : f32 = max(dimensions.x, dimensions.y) * 0.5;
    if (max_dimension <= 0.0) {
        return 0.0;
    }

    return clamp(dist / max_dimension, 0.0, 1.0);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let width : u32 = as_u32(params.width);
    let height : u32 = as_u32(params.height);
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    let base_coords : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
    let base_color : vec4<f32> = clamp_vec01(textureLoad(input_texture, base_coords, 0));

    let alpha_raw : f32 = params.alpha;
    let alpha : f32 = clamp(alpha_raw, 0.0, 1.0);
    if (alpha <= 0.0) {
        let pixel_index_no_blend : u32 = (gid.y * width + gid.x) * 4u;
        store_pixel(pixel_index_no_blend, base_color);
        return;
    }

    let width_i : i32 = i32(width);
    let height_i : i32 = i32(height);
    let image_size : vec2<i32> = vec2<i32>(width_i, height_i);
    let image_size_f : vec2<f32> = vec2<f32>(max(params.width, 1.0), max(params.height, 1.0));

    let down_width : i32 = max(i32(floor(params.width / 100.0)), 1);
    let down_height : i32 = max(i32(floor(params.height / 100.0)), 1);
    let down_size_f : vec2<f32> = vec2<f32>(f32(down_width), f32(down_height));

    let kernel_width : i32 = max(width_i / down_width, 1);
    let kernel_height : i32 = max(height_i / down_height, 1);
    let kernel_size : vec2<i32> = vec2<i32>(kernel_width, kernel_height);

    let offset_x : i32 = i32(f32(width_i) * OFFSET_SCALE);
    let offset_y : i32 = i32(f32(height_i) * OFFSET_SCALE);
    let shifted : vec2<i32> = vec2<i32>(
        wrap_index(base_coords.x + offset_x, width_i),
        wrap_index(base_coords.y + offset_y, height_i),
    );

    let shifted_f : vec2<f32> = (
        vec2<f32>(f32(shifted.x), f32(shifted.y)) + vec2<f32>(0.5, 0.5)
    ) / image_size_f;
    let down_coord : vec2<f32> = shifted_f * down_size_f;

    let base_floor_x : i32 = clamp(i32(floor(down_coord.x)), 0, down_width - 1);
    let base_floor_y : i32 = clamp(i32(floor(down_coord.y)), 0, down_height - 1);
    let frac : vec2<f32> = vec2<f32>(
        clamp(down_coord.x - f32(base_floor_x), 0.0, 1.0),
        clamp(down_coord.y - f32(base_floor_y), 0.0, 1.0),
    );

    let sample_x : array<i32, 4> = array<i32, 4>(
        wrap_index(base_floor_x - 1, down_width),
        base_floor_x,
        wrap_index(base_floor_x + 1, down_width),
        wrap_index(base_floor_x + 2, down_width),
    );
    let sample_y : array<i32, 4> = array<i32, 4>(
        wrap_index(base_floor_y - 1, down_height),
        base_floor_y,
        wrap_index(base_floor_y + 1, down_height),
        wrap_index(base_floor_y + 2, down_height),
    );

    var neighbors : array<vec4<f32>, 16>;
    var neighbor_assigned : array<u32, 16>;
    for (var idx : u32 = 0u; idx < 16u; idx = idx + 1u) {
        neighbors[idx] = vec4<f32>(0.0);
        neighbor_assigned[idx] = 0u;
    }

    var bright_sum : vec3<f32> = vec3<f32>(0.0);
    var total_weight : f32 = 0.0;

    for (var y : i32 = 0; y < down_height; y = y + 1) {
        for (var x : i32 = 0; x < down_width; x = x + 1) {
            let cell_coord : vec2<i32> = vec2<i32>(x, y);
            let downsample : DownsampleResult = fetch_downsample_cell(
                cell_coord,
                kernel_size,
                image_size,
            );
            let boosted : vec4<f32> = downsample.value * vec4<f32>(BOOST);
            let brightened : vec4<f32> = clamp_vec_symmetric(
                boosted + vec4<f32>(BRIGHTNESS_ADJUST),
            );

            bright_sum = bright_sum + brightened.xyz * downsample.weight;
            total_weight = total_weight + downsample.weight;

            for (var j : u32 = 0u; j < 4u; j = j + 1u) {
                if (cell_coord.y == sample_y[j]) {
                    for (var i : u32 = 0u; i < 4u; i = i + 1u) {
                        if (cell_coord.x == sample_x[i]) {
                            let neighbor_index : u32 = j * 4u + i;
                            neighbors[neighbor_index] = boosted;
                            neighbor_assigned[neighbor_index] = 1u;
                        }
                    }
                }
            }
        }
    }

    let reference_index : u32 = 5u;
    for (var idx : u32 = 0u; idx < 16u; idx = idx + 1u) {
        if (neighbor_assigned[idx] == 0u) {
            neighbors[idx] = neighbors[reference_index];
        }
    }

    let row0 : vec4<f32> = cubic_interpolate(
        neighbors[0u],
        neighbors[1u],
        neighbors[2u],
        neighbors[3u],
        frac.x,
    );
    let row1 : vec4<f32> = cubic_interpolate(
        neighbors[4u],
        neighbors[5u],
        neighbors[6u],
        neighbors[7u],
        frac.x,
    );
    let row2 : vec4<f32> = cubic_interpolate(
        neighbors[8u],
        neighbors[9u],
        neighbors[10u],
        neighbors[11u],
        frac.x,
    );
    let row3 : vec4<f32> = cubic_interpolate(
        neighbors[12u],
        neighbors[13u],
        neighbors[14u],
        neighbors[15u],
        frac.x,
    );
    let pre_brightness : vec4<f32> = cubic_interpolate(row0, row1, row2, row3, frac.y);

    let safe_weight : f32 = select(1.0, total_weight, total_weight > 0.0);
    let global_mean : vec3<f32> = bright_sum / safe_weight;

    let pre_brightness_rgb : vec3<f32> = pre_brightness.xyz;
    let brightened_pixel : vec3<f32> = clamp(
        pre_brightness_rgb + vec3<f32>(BRIGHTNESS_ADJUST),
        vec3<f32>(-1.0),
        vec3<f32>(1.0),
    );
    let contrasted : vec3<f32> = (brightened_pixel - global_mean)
        * vec3<f32>(CONTRAST_SCALE) + global_mean;
    let blurred : vec3<f32> = clamp(
        contrasted,
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );

    let bloom_color : vec3<f32> = clamp(
        (base_color.xyz + blurred) * 0.5,
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );
    let mask_base : f32 = chebyshev_mask(
        (
            vec2<f32>(f32(base_coords.x), f32(base_coords.y)) + vec2<f32>(0.5, 0.5)
        ) / image_size_f,
        image_size_f,
    );
    let mask : f32 = pow(mask_base, 2.0);
    let center_blend : vec3<f32> = mix(base_color.xyz, bloom_color, vec3<f32>(mask));
    let final_rgb : vec3<f32> = clamp(
        mix(base_color.xyz, center_blend, vec3<f32>(alpha)),
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );
    let final_color : vec4<f32> = vec4<f32>(final_rgb, base_color.w);

    let pixel_index : u32 = (gid.y * width + gid.x) * 4u;
    store_pixel(pixel_index, final_color);
}
