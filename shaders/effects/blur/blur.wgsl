// Blur effect implemented to match the Python reference `effects.blur`.
// Pass 1 downsamples into a coarse buffer. Pass 2 upsamples with the
// configured interpolation for the final image.

const PI : f32 = 3.14159265358979323846;

const INTERPOLATION_CONSTANT : u32 = 0u;
const INTERPOLATION_LINEAR : u32 = 1u;
const INTERPOLATION_COSINE : u32 = 2u;
const INTERPOLATION_BICUBIC : u32 = 3u;

const CHANNEL_COUNT : u32 = 4u;

struct BlurParams {
    width : f32,
    height : f32,
    channel_count : f32,
    amount : f32,
    time : f32,
    spline_order : f32,
    downsample_width : f32,
    downsample_height : f32,
    inv_downsample_width : f32,
    inv_downsample_height : f32,
    _pad0 : f32,
    _pad1 : f32,
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : BlurParams;
@group(0) @binding(3) var<storage, read_write> downsample_buffer : array<f32>;

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

fn interpolation_weight(value : f32, spline_order : u32) -> f32 {
    if (spline_order == INTERPOLATION_COSINE) {
        let clamped : f32 = clamp(value, 0.0, 1.0);
        return (1.0 - cos(clamped * PI)) * 0.5;
    }

    return clamp(value, 0.0, 1.0);
}

fn blend_cubic(a : vec4<f32>, b : vec4<f32>, c : vec4<f32>, d : vec4<f32>, factor : f32) -> vec4<f32> {
    let t : f32 = factor;
    let t2 : f32 = t * t;
    let a0 : vec4<f32> = (d - c) - (a - b);
    let a1 : vec4<f32> = (a - b) - a0;
    let a2 : vec4<f32> = c - a;
    let a3 : vec4<f32> = b;
    return ((a0 * t) * t2) + (a1 * t2) + (a2 * t) + a3;
}

fn read_downsample(coord : vec2<i32>, down_size : vec2<i32>) -> vec4<f32> {
    let safe_x : i32 = wrap_index(coord.x, down_size.x);
    let safe_y : i32 = wrap_index(coord.y, down_size.y);
    let base_index : u32 = (u32(safe_y) * u32(max(down_size.x, 1)) + u32(safe_x)) * CHANNEL_COUNT;
    return vec4<f32>(
        downsample_buffer[base_index + 0u],
        downsample_buffer[base_index + 1u],
        downsample_buffer[base_index + 2u],
        downsample_buffer[base_index + 3u],
    );
}

fn write_pixel(pixel_index : u32, value : vec4<f32>) {
    output_buffer[pixel_index + 0u] = value.x;
    output_buffer[pixel_index + 1u] = value.y;
    output_buffer[pixel_index + 2u] = value.z;
    output_buffer[pixel_index + 3u] = value.w;
}

fn sample_resampled_value(
    base_coord : vec2<i32>,
    frac : vec2<f32>,
    down_size : vec2<i32>,
    spline_order : u32,
) -> vec4<f32> {
    if (spline_order == INTERPOLATION_CONSTANT) {
        return read_downsample(base_coord, down_size);
    }

    if (spline_order == INTERPOLATION_LINEAR || spline_order == INTERPOLATION_COSINE) {
        let weight_x : f32 = interpolation_weight(frac.x, spline_order);
        let weight_y : f32 = interpolation_weight(frac.y, spline_order);

        let v00 : vec4<f32> = read_downsample(base_coord, down_size);
        let v10 : vec4<f32> = read_downsample(base_coord + vec2<i32>(1, 0), down_size);
        let v01 : vec4<f32> = read_downsample(base_coord + vec2<i32>(0, 1), down_size);
        let v11 : vec4<f32> = read_downsample(base_coord + vec2<i32>(1, 1), down_size);

        let x0 : vec4<f32> = v00 + ((v10 - v00) * weight_x);
        let x1 : vec4<f32> = v01 + ((v11 - v01) * weight_x);
        return x0 + ((x1 - x0) * weight_y);
    }

    var rows : array<vec4<f32>, 4>;
    for (var j : i32 = 0; j < 4; j = j + 1) {
        var samples : array<vec4<f32>, 4>;
        for (var i : i32 = 0; i < 4; i = i + 1) {
            samples[u32(i)] = read_downsample(
                base_coord + vec2<i32>(i - 1, j - 1),
                down_size,
            );
        }

        rows[u32(j)] = blend_cubic(samples[0u], samples[1u], samples[2u], samples[3u], frac.x);
    }

    return blend_cubic(rows[0u], rows[1u], rows[2u], rows[3u], frac.y);
}

@compute @workgroup_size(8, 8, 1)
fn downsample_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let down_width : i32 = max(i32(round(params.downsample_width)), 1);
    let down_height : i32 = max(i32(round(params.downsample_height)), 1);
    if (gid.x >= u32(down_width) || gid.y >= u32(down_height)) {
        return;
    }

    let width : i32 = max(i32(round(params.width)), 1);
    let height : i32 = max(i32(round(params.height)), 1);
    let kernel_width : i32 = max(ceil_div_i32(width, down_width), 1);
    let kernel_height : i32 = max(ceil_div_i32(height, down_height), 1);

    let origin_x : i32 = i32(gid.x) * kernel_width;
    let origin_y : i32 = i32(gid.y) * kernel_height;

    var accum : vec4<f32> = vec4<f32>(0.0);
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

            accum = accum + textureLoad(input_texture, vec2<i32>(sample_x, sample_y), 0);
            sample_count = sample_count + 1.0;
        }
    }

    if (sample_count <= 0.0) {
        return;
    }

    let scale : f32 = max(params.channel_count, 1.0);
    let average : vec4<f32> = (accum / sample_count) * vec4<f32>(scale);

    let base_index : u32 = (gid.y * u32(down_width) + gid.x) * CHANNEL_COUNT;
    downsample_buffer[base_index + 0u] = average.x;
    downsample_buffer[base_index + 1u] = average.y;
    downsample_buffer[base_index + 2u] = average.z;
    downsample_buffer[base_index + 3u] = average.w;
}

@compute @workgroup_size(8, 8, 1)
fn upsample_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let width : u32 = max(u32(round(params.width)), 1u);
    let height : u32 = max(u32(round(params.height)), 1u);
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    let down_width : f32 = max(params.downsample_width, 1.0);
    let down_height : f32 = max(params.downsample_height, 1.0);
    let spl_raw : f32 = clamp(params.spline_order, 0.0, 3.0);
    let spline_order : u32 = u32(round(spl_raw));

    let width_f : f32 = max(params.width, 1.0);
    let height_f : f32 = max(params.height, 1.0);

    let sample_pos : vec2<f32> = vec2<f32>(
        f32(gid.x) * down_width / width_f,
        f32(gid.y) * down_height / height_f,
    );

    let base_floor : vec2<f32> = floor(sample_pos);
    let base_coord : vec2<i32> = vec2<i32>(i32(base_floor.x), i32(base_floor.y));
    let frac : vec2<f32> = fract(sample_pos);

    let down_size : vec2<i32> = vec2<i32>(
        max(i32(round(params.downsample_width)), 1),
        max(i32(round(params.downsample_height)), 1),
    );

    let color : vec4<f32> = sample_resampled_value(base_coord, frac, down_size, spline_order);
    let pixel_index : u32 = (gid.y * width + gid.x) * CHANNEL_COUNT;
    write_pixel(pixel_index, color);
}
