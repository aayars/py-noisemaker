// Bloom effect implemented as a two-pass pipeline mirroring the Python
// reference. Pass one downsamples highlight information into a coarse grid.
// Pass two upsamples with bicubic filtering, applies brightness/contrast
// adjustments, and blends the result back into the source image.

const CHANNEL_COUNT : u32 = 4u;
const BOOST : f32 = 4.0;
const BRIGHTNESS_ADJUST : f32 = 0.25;
const CONTRAST_SCALE : f32 = 1.5;

struct BloomParams {
    size_alpha : vec4<f32>,   // width, height, channel_count, alpha
    anim_down : vec4<f32>,    // time, speed, downsample_width, downsample_height
    inv_offset : vec4<f32>,   // inv_downsample_width, inv_downsample_height, offset_x, offset_y
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : BloomParams;
@group(0) @binding(3) var<storage, read_write> downsample_buffer : array<f32>;

fn clamp_vec01(value : vec3<f32>) -> vec3<f32> {
    return clamp(value, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn clamp_vec11(value : vec3<f32>) -> vec3<f32> {
    return clamp(value, vec3<f32>(-1.0), vec3<f32>(1.0));
}

fn wrap_index(value : i32, limit : i32) -> i32 {
    if (limit <= 0) {
        return 0;
    }

    var wrapped : i32 = value % limit;
    if (wrapped < 0) {
        wrapped = wrapped + limit;
    }

    return wrapped;
}

fn ceil_div_i32(numerator : i32, denominator : i32) -> i32 {
    if (denominator <= 0) {
        return 0;
    }
    return (numerator + denominator - 1) / denominator;
}

fn read_compressed_cell(coord : vec2<i32>, down_size : vec2<i32>) -> vec4<f32> {
    let width : i32 = max(down_size.x, 1);
    let height : i32 = max(down_size.y, 1);
    let safe_x : i32 = wrap_index(coord.x, width);
    let safe_y : i32 = wrap_index(coord.y, height);
    let base_index : u32 = (u32(safe_y) * u32(width) + u32(safe_x)) * CHANNEL_COUNT;
    return vec4<f32>(
        downsample_buffer[base_index + 0u],
        downsample_buffer[base_index + 1u],
        downsample_buffer[base_index + 2u],
        downsample_buffer[base_index + 3u],
    );
}

fn cubic_interpolate_vec3(a : vec3<f32>, b : vec3<f32>, c : vec3<f32>, d : vec3<f32>, t : f32) -> vec3<f32> {
    let t2 : f32 = t * t;
    let t3 : f32 = t2 * t;
    let a0 : vec3<f32> = d - c - a + b;
    let a1 : vec3<f32> = a - b - a0;
    let a2 : vec3<f32> = c - a;
    let a3 : vec3<f32> = b;
    return ((a0 * t3) + (a1 * t2)) + (a2 * t) + a3;
}

@compute @workgroup_size(8, 8, 1)
fn downsample_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let down_width : i32 = max(i32(round(params.anim_down.z)), 1);
    let down_height : i32 = max(i32(round(params.anim_down.w)), 1);
    if (gid.x >= u32(down_width) || gid.y >= u32(down_height)) {
        return;
    }

    let width : i32 = max(i32(round(params.size_alpha.x)), 1);
    let height : i32 = max(i32(round(params.size_alpha.y)), 1);
    let kernel_width : i32 = max(ceil_div_i32(width, down_width), 1);
    let kernel_height : i32 = max(ceil_div_i32(height, down_height), 1);

    let origin_x : i32 = i32(gid.x) * kernel_width;
    let origin_y : i32 = i32(gid.y) * kernel_height;

    var accum : vec3<f32> = vec3<f32>(0.0);
    var sample_count : f32 = 0.0;

    for (var ky : i32 = 0; ky < kernel_height; ky = ky + 1) {
        let sample_y : i32 = origin_y + ky;
        if (sample_y >= height) {
            break;
        }

        for (var kx : i32 = 0; kx < kernel_width; kx = kx + 1) {
            let sample_x : i32 = origin_x + kx;
            if (sample_x >= width) {
                break;
            }

            let texel : vec3<f32> = textureLoad(input_texture, vec2<i32>(sample_x, sample_y), 0).xyz;
            let highlight : vec3<f32> = clamp(texel * vec3<f32>(2.0) - vec3<f32>(1.0), vec3<f32>(0.0), vec3<f32>(1.0));
            accum = accum + highlight;
            sample_count = sample_count + 1.0;
        }
    }

    let base_index : u32 = (gid.y * u32(down_width) + gid.x) * CHANNEL_COUNT;
    if (sample_count <= 0.0) {
        downsample_buffer[base_index + 0u] = 0.0;
        downsample_buffer[base_index + 1u] = 0.0;
        downsample_buffer[base_index + 2u] = 0.0;
        downsample_buffer[base_index + 3u] = 0.0;
        return;
    }

    let average : vec3<f32> = accum / sample_count;
    let boosted : vec3<f32> = average * vec3<f32>(BOOST);

    downsample_buffer[base_index + 0u] = boosted.x;
    downsample_buffer[base_index + 1u] = boosted.y;
    downsample_buffer[base_index + 2u] = boosted.z;
    downsample_buffer[base_index + 3u] = sample_count;
}

@compute @workgroup_size(8, 8, 1)
fn upsample_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let width : i32 = max(i32(round(params.size_alpha.x)), 1);
    let height : i32 = max(i32(round(params.size_alpha.y)), 1);
    if (gid.x >= u32(width) || gid.y >= u32(height)) {
        return;
    }

    let alpha : f32 = clamp(params.size_alpha.w, 0.0, 1.0);
    let coords : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
    let source_sample : vec4<f32> = textureLoad(input_texture, coords, 0);
    let pixel_index : u32 = (gid.y * u32(width) + gid.x) * CHANNEL_COUNT;

    if (alpha <= 0.0 || params.size_alpha.z < 3.0) {
        let clamped_source : vec3<f32> = clamp_vec01(source_sample.xyz);
        output_buffer[pixel_index + 0u] = clamped_source.x;
        output_buffer[pixel_index + 1u] = clamped_source.y;
        output_buffer[pixel_index + 2u] = clamped_source.z;
        output_buffer[pixel_index + 3u] = source_sample.w;
        return;
    }

    let down_width : i32 = max(i32(round(params.anim_down.z)), 1);
    let down_height : i32 = max(i32(round(params.anim_down.w)), 1);
    let down_size : vec2<i32> = vec2<i32>(down_width, down_height);

    let offset_x : i32 = i32(round(params.inv_offset.z));
    let offset_y : i32 = i32(round(params.inv_offset.w));
    let shifted : vec2<i32> = vec2<i32>(
        wrap_index(coords.x + offset_x, width),
        wrap_index(coords.y + offset_y, height),
    );

    let width_f : f32 = max(params.size_alpha.x, 1.0);
    let height_f : f32 = max(params.size_alpha.y, 1.0);
    let inv_down_width : f32 = max(params.inv_offset.x, 1e-6);
    let inv_down_height : f32 = max(params.inv_offset.y, 1e-6);
    let down_width_f : f32 = 1.0 / inv_down_width;
    let down_height_f : f32 = 1.0 / inv_down_height;

    let sample_pos : vec2<f32> = vec2<f32>(
        (f32(shifted.x) + 0.5) / width_f * down_width_f,
        (f32(shifted.y) + 0.5) / height_f * down_height_f,
    );

    let base_floor : vec2<i32> = vec2<i32>(
        clamp(i32(floor(sample_pos.x)), 0, down_width - 1),
        clamp(i32(floor(sample_pos.y)), 0, down_height - 1),
    );
    let frac : vec2<f32> = vec2<f32>(
        clamp(sample_pos.x - f32(base_floor.x), 0.0, 1.0),
        clamp(sample_pos.y - f32(base_floor.y), 0.0, 1.0),
    );

    var sample_x : array<i32, 4> = array<i32, 4>(
        wrap_index(base_floor.x - 1, down_width),
        base_floor.x,
        wrap_index(base_floor.x + 1, down_width),
        wrap_index(base_floor.x + 2, down_width),
    );
    var sample_y : array<i32, 4> = array<i32, 4>(
        wrap_index(base_floor.y - 1, down_height),
        base_floor.y,
        wrap_index(base_floor.y + 1, down_height),
        wrap_index(base_floor.y + 2, down_height),
    );

    var rows : array<vec3<f32>, 4>;
    for (var j : i32 = 0; j < 4; j = j + 1) {
        var samples : array<vec3<f32>, 4>;
        for (var i : i32 = 0; i < 4; i = i + 1) {
            let cell : vec4<f32> = read_compressed_cell(vec2<i32>(sample_x[i], sample_y[j]), down_size);
            samples[u32(i)] = cell.xyz;
        }
        rows[u32(j)] = cubic_interpolate_vec3(samples[0u], samples[1u], samples[2u], samples[3u], frac.x);
    }

    let boosted_sample : vec3<f32> = cubic_interpolate_vec3(rows[0u], rows[1u], rows[2u], rows[3u], frac.y);
    let brightened_pixel : vec3<f32> = clamp_vec11(boosted_sample + vec3<f32>(BRIGHTNESS_ADJUST));

    var bright_sum : vec3<f32> = vec3<f32>(0.0);
    var total_weight : f32 = 0.0;
    for (var y : i32 = 0; y < down_height; y = y + 1) {
        for (var x : i32 = 0; x < down_width; x = x + 1) {
            let cell : vec4<f32> = read_compressed_cell(vec2<i32>(x, y), down_size);
            if (cell.w <= 0.0) {
                continue;
            }
            let brightened_cell : vec3<f32> = clamp_vec11(cell.xyz + vec3<f32>(BRIGHTNESS_ADJUST));
            bright_sum = bright_sum + brightened_cell * cell.w;
            total_weight = total_weight + cell.w;
        }
    }

    var global_mean : vec3<f32> = brightened_pixel;
    if (total_weight > 0.0) {
        global_mean = bright_sum / total_weight;
    }

    let contrasted : vec3<f32> = (brightened_pixel - global_mean) * vec3<f32>(CONTRAST_SCALE) + global_mean;
    let blurred : vec3<f32> = clamp_vec01(contrasted);

    let source_clamped : vec3<f32> = clamp_vec01(source_sample.xyz);
    let mixed : vec3<f32> = clamp_vec01((source_clamped + blurred) * 0.5);
    let final_rgb : vec3<f32> = clamp_vec01(source_clamped * (1.0 - alpha) + mixed * alpha);

    output_buffer[pixel_index + 0u] = final_rgb.x;
    output_buffer[pixel_index + 1u] = final_rgb.y;
    output_buffer[pixel_index + 2u] = final_rgb.z;
    output_buffer[pixel_index + 3u] = source_sample.w;
}
