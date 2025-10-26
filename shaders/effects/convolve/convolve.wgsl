// Convolve effect compute shader.
//
// Mirrors the Python implementation in noisemaker/value.py::convolve.
// Applies the selected convolution kernel with wrap-around sampling,
// optional normalization, and alpha blending. Work is split into three
// dispatches to keep each GPU pass bounded:
//   1. reset_stats_main  – reset the global min/max encodings
//   2. convolve_main     – apply the kernel per pixel, track min/max
//   3. main              – normalize (optional) and alpha blend

const FLOAT_MAX : f32 = 3.402823466e38;
const FLOAT_MIN : f32 = -3.402823466e38;

const CHANNEL_CAP : u32 = 4u;

const KERNEL_SIZE_3 : i32 = 3;
const KERNEL_SIZE_5 : i32 = 5;

const KERNEL_CONV2D_BLUR : i32 = 800;
const KERNEL_CONV2D_DERIV_X : i32 = 801;
const KERNEL_CONV2D_DERIV_Y : i32 = 802;
const KERNEL_CONV2D_EDGES : i32 = 803;
const KERNEL_CONV2D_EMBOSS : i32 = 804;
const KERNEL_CONV2D_INVERT : i32 = 805;
const KERNEL_CONV2D_RAND : i32 = 806;
const KERNEL_CONV2D_SHARPEN : i32 = 807;
const KERNEL_CONV2D_SOBEL_X : i32 = 808;
const KERNEL_CONV2D_SOBEL_Y : i32 = 809;
const KERNEL_CONV2D_BOX_BLUR : i32 = 810;

const KERNEL_CONV2D_BLUR_WEIGHTS : array<f32, 25> = array<f32, 25>(
    1.0, 4.0, 6.0, 4.0, 1.0,
    4.0, 16.0, 24.0, 16.0, 4.0,
    6.0, 24.0, 36.0, 24.0, 6.0,
    4.0, 16.0, 24.0, 16.0, 4.0,
    1.0, 4.0, 6.0, 4.0, 1.0
);

const KERNEL_CONV2D_BOX_BLUR_WEIGHTS : array<f32, 9> = array<f32, 9>(
    1.0, 2.0, 1.0,
    2.0, 4.0, 2.0,
    1.0, 2.0, 1.0
);

const KERNEL_CONV2D_DERIV_X_WEIGHTS : array<f32, 9> = array<f32, 9>(
    0.0, 0.0, 0.0,
    0.0, 1.0, -1.0,
    0.0, 0.0, 0.0
);

const KERNEL_CONV2D_DERIV_Y_WEIGHTS : array<f32, 9> = array<f32, 9>(
    0.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, -1.0, 0.0
);

const KERNEL_CONV2D_EDGES_WEIGHTS : array<f32, 9> = array<f32, 9>(
    1.0, 2.0, 1.0,
    2.0, -12.0, 2.0,
    1.0, 2.0, 1.0
);

const KERNEL_CONV2D_EMBOSS_WEIGHTS : array<f32, 9> = array<f32, 9>(
    0.0, 2.0, 4.0,
    -2.0, 1.0, 2.0,
    -4.0, -2.0, 0.0
);

const KERNEL_CONV2D_INVERT_WEIGHTS : array<f32, 9> = array<f32, 9>(
    0.0, 0.0, 0.0,
    0.0, -1.0, 0.0,
    0.0, 0.0, 0.0
);

const KERNEL_CONV2D_RAND_WEIGHTS : array<f32, 25> = array<f32, 25>(
    1.382026172983832, 0.7000786041836117, 0.9893689920528697, 1.620446599600729, 1.4337789950749837,
    0.011361060061794492, 0.9750442087627946, 0.42432139585115103, 0.4483905741032211, 0.7052992509691862,
    0.572021785580439, 1.2271367534814877, 0.8805188625734968, 0.5608375082464142, 0.7219316163727129,
    0.6668371636871334, 1.2470395365788032, 0.39742086811709953, 0.6565338508254507, 0.07295213034913761,
    -0.7764949079170393, 0.8268092977201803, 0.9322180994297529, 0.12891748979677903, 1.6348773119938038
);

const KERNEL_CONV2D_SHARPEN_WEIGHTS : array<f32, 9> = array<f32, 9>(
    0.0, -1.0, 0.0,
    -1.0, 5.0, -1.0,
    0.0, -1.0, 0.0
);

const KERNEL_CONV2D_SOBEL_X_WEIGHTS : array<f32, 9> = array<f32, 9>(
    1.0, 0.0, -1.0,
    2.0, 0.0, -2.0,
    1.0, 0.0, -1.0
);

const KERNEL_CONV2D_SOBEL_Y_WEIGHTS : array<f32, 9> = array<f32, 9>(
    1.0, 2.0, 1.0,
    0.0, 0.0, 0.0,
    -1.0, -2.0, -1.0
);

struct ConvolveParams {
    size : vec4<f32>,      // width, height, channel_count, kernel id
    control : vec4<f32>,   // normalize flag, alpha, time, speed (unused)
};

struct StatsBuffer {
    min_value : atomic<u32>,
    max_value : atomic<u32>,
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : ConvolveParams;
@group(0) @binding(3) var<storage, read_write> stats_buffer : StatsBuffer;

fn as_u32(value : f32) -> u32 {
    return u32(max(round(value), 0.0));
}

fn clamp_channel_count(raw_count : u32) -> u32 {
    if (raw_count == 0u) {
        return 0u;
    }
    if (raw_count > CHANNEL_CAP) {
        return CHANNEL_CAP;
    }
    return raw_count;
}

fn wrap_index(coord : i32, limit : i32) -> i32 {
    if (limit <= 0) {
        return 0;
    }
    var wrapped : i32 = coord % limit;
    if (wrapped < 0) {
        wrapped = wrapped + limit;
    }
    return wrapped;
}

fn float_to_ordered(value : f32) -> u32 {
    let bits : u32 = bitcast<u32>(value);
    if ((bits & 0x80000000u) != 0u) {
        return ~bits;
    }
    return bits | 0x80000000u;
}

fn ordered_to_float(value : u32) -> f32 {
    if ((value & 0x80000000u) != 0u) {
        return bitcast<f32>(value & 0x7fffffffu);
    }
    return bitcast<f32>(~value);
}

fn kernel_dimensions(kernel_id : i32) -> vec2<i32> {
    switch kernel_id {
        case KERNEL_CONV2D_BLUR, KERNEL_CONV2D_RAND: {
            return vec2<i32>(KERNEL_SIZE_5, KERNEL_SIZE_5);
        }
        case KERNEL_CONV2D_BOX_BLUR,
             KERNEL_CONV2D_DERIV_X,
             KERNEL_CONV2D_DERIV_Y,
             KERNEL_CONV2D_EDGES,
             KERNEL_CONV2D_EMBOSS,
             KERNEL_CONV2D_INVERT,
             KERNEL_CONV2D_SHARPEN,
             KERNEL_CONV2D_SOBEL_X,
             KERNEL_CONV2D_SOBEL_Y: {
            return vec2<i32>(KERNEL_SIZE_3, KERNEL_SIZE_3);
        }
        default: {
            return vec2<i32>(1, 1);
        }
    }
}

fn kernel_weight(kernel_id : i32, row : i32, col : i32) -> f32 {
    switch kernel_id {
        case KERNEL_CONV2D_BLUR: {
            let index : u32 = u32(row * KERNEL_SIZE_5 + col);
            return KERNEL_CONV2D_BLUR_WEIGHTS[index];
        }
        case KERNEL_CONV2D_BOX_BLUR: {
            let index : u32 = u32(row * KERNEL_SIZE_3 + col);
            return KERNEL_CONV2D_BOX_BLUR_WEIGHTS[index];
        }
        case KERNEL_CONV2D_DERIV_X: {
            let index : u32 = u32(row * KERNEL_SIZE_3 + col);
            return KERNEL_CONV2D_DERIV_X_WEIGHTS[index];
        }
        case KERNEL_CONV2D_DERIV_Y: {
            let index : u32 = u32(row * KERNEL_SIZE_3 + col);
            return KERNEL_CONV2D_DERIV_Y_WEIGHTS[index];
        }
        case KERNEL_CONV2D_EDGES: {
            let index : u32 = u32(row * KERNEL_SIZE_3 + col);
            return KERNEL_CONV2D_EDGES_WEIGHTS[index];
        }
        case KERNEL_CONV2D_EMBOSS: {
            let index : u32 = u32(row * KERNEL_SIZE_3 + col);
            return KERNEL_CONV2D_EMBOSS_WEIGHTS[index];
        }
        case KERNEL_CONV2D_INVERT: {
            let index : u32 = u32(row * KERNEL_SIZE_3 + col);
            return KERNEL_CONV2D_INVERT_WEIGHTS[index];
        }
        case KERNEL_CONV2D_RAND: {
            let index : u32 = u32(row * KERNEL_SIZE_5 + col);
            return KERNEL_CONV2D_RAND_WEIGHTS[index];
        }
        case KERNEL_CONV2D_SHARPEN: {
            let index : u32 = u32(row * KERNEL_SIZE_3 + col);
            return KERNEL_CONV2D_SHARPEN_WEIGHTS[index];
        }
        case KERNEL_CONV2D_SOBEL_X: {
            let index : u32 = u32(row * KERNEL_SIZE_3 + col);
            return KERNEL_CONV2D_SOBEL_X_WEIGHTS[index];
        }
        case KERNEL_CONV2D_SOBEL_Y: {
            let index : u32 = u32(row * KERNEL_SIZE_3 + col);
            return KERNEL_CONV2D_SOBEL_Y_WEIGHTS[index];
        }
        default: {
            return 1.0;
        }
    }
}

fn kernel_max_abs(kernel_id : i32) -> f32 {
    let dims : vec2<i32> = kernel_dimensions(kernel_id);
    let width : i32 = max(dims.x, 1);
    let height : i32 = max(dims.y, 1);
    var max_abs : f32 = 0.0;
    for (var row : i32 = 0; row < height; row = row + 1) {
        for (var col : i32 = 0; col < width; col = col + 1) {
            let w : f32 = abs(kernel_weight(kernel_id, row, col));
            max_abs = max(max_abs, w);
        }
    }
    if (max_abs == 0.0) {
        return 1.0;
    }
    return max_abs;
}

fn get_component(value : vec4<f32>, index : u32) -> f32 {
    switch index {
        case 0u: { return value.x; }
        case 1u: { return value.y; }
        case 2u: { return value.z; }
        default: { return value.w; }
    }
}

fn set_component(dst : ptr<function, vec4<f32>>, index : u32, value : f32) {
    switch index {
        case 0u: { (*dst).x = value; }
        case 1u: { (*dst).y = value; }
        case 2u: { (*dst).z = value; }
        default: { (*dst).w = value; }
    }
}

fn lerp_vec4(a : vec4<f32>, b : vec4<f32>, t : f32) -> vec4<f32> {
    return a + (b - a) * t;
}

fn write_pixel(base_index : u32, color : vec4<f32>) {
    output_buffer[base_index + 0u] = color.x;
    output_buffer[base_index + 1u] = color.y;
    output_buffer[base_index + 2u] = color.z;
    output_buffer[base_index + 3u] = color.w;
}

fn read_pixel(base_index : u32) -> vec4<f32> {
    return vec4<f32>(
        output_buffer[base_index + 0u],
        output_buffer[base_index + 1u],
        output_buffer[base_index + 2u],
        output_buffer[base_index + 3u]
    );
}

@compute @workgroup_size(1, 1, 1)
fn reset_stats_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    if (gid.x != 0u || gid.y != 0u || gid.z != 0u) {
        return;
    }

    atomicStore(&stats_buffer.min_value, float_to_ordered(FLOAT_MAX));
    atomicStore(&stats_buffer.max_value, float_to_ordered(FLOAT_MIN));
}

@compute @workgroup_size(8, 8, 1)
fn convolve_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let width : u32 = as_u32(params.size.x);
    let height : u32 = as_u32(params.size.y);
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    let channel_count : u32 = clamp_channel_count(as_u32(params.size.z));
    if (channel_count == 0u) {
        return;
    }

    let kernel_id : i32 = i32(round(params.size.w));
    let dims : vec2<i32> = kernel_dimensions(kernel_id);
    if (dims.x <= 0 || dims.y <= 0) {
        return;
    }

    let denom : f32 = kernel_max_abs(kernel_id);

    let xi : i32 = i32(gid.x);
    let yi : i32 = i32(gid.y);
    let width_i : i32 = i32(width);
    let height_i : i32 = i32(height);

    var accum : vec4<f32> = vec4<f32>(0.0);
    for (var ky : i32 = 0; ky < dims.y; ky = ky + 1) {
        for (var kx : i32 = 0; kx < dims.x; kx = kx + 1) {
            let offset_x : i32 = kx - dims.x / 2;
            let offset_y : i32 = ky - dims.y / 2;
            let sample_x : i32 = wrap_index(xi + offset_x, width_i);
            let sample_y : i32 = wrap_index(yi + offset_y, height_i);
            let sample : vec4<f32> = textureLoad(input_texture, vec2<i32>(sample_x, sample_y), 0);
            let weight : f32 = kernel_weight(kernel_id, ky, kx) / denom;
            accum = accum + sample * weight;
        }
    }

    let original : vec4<f32> = textureLoad(input_texture, vec2<i32>(xi, yi), 0);
    var processed : vec4<f32> = original;
    var pixel_min : f32 = FLOAT_MAX;
    var pixel_max : f32 = FLOAT_MIN;

    for (var c : u32 = 0u; c < channel_count; c = c + 1u) {
        let component : f32 = get_component(accum, c);
        set_component(&processed, c, component);
        pixel_min = min(pixel_min, component);
        pixel_max = max(pixel_max, component);
    }

    processed.w = original.w;

    let pixel_index : u32 = gid.y * width + gid.x;
    let base_index : u32 = pixel_index * CHANNEL_CAP;
    write_pixel(base_index, processed);

    if (params.control.x > 0.5) {
        let encoded_min : u32 = float_to_ordered(pixel_min);
        let encoded_max : u32 = float_to_ordered(pixel_max);
        atomicMin(&stats_buffer.min_value, encoded_min);
        atomicMax(&stats_buffer.max_value, encoded_max);
    }
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let width : u32 = as_u32(params.size.x);
    let height : u32 = as_u32(params.size.y);
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    let channel_count : u32 = clamp_channel_count(as_u32(params.size.z));
    if (channel_count == 0u) {
        return;
    }

    let do_normalize : bool = params.control.x > 0.5;
    let alpha : f32 = clamp(params.control.y, 0.0, 1.0);
    let kernel_id : i32 = i32(round(params.size.w));

    let min_bits : u32 = atomicLoad(&stats_buffer.min_value);
    let max_bits : u32 = atomicLoad(&stats_buffer.max_value);
    var min_value : f32 = ordered_to_float(min_bits);
    var max_value : f32 = ordered_to_float(max_bits);

    if (min_value > max_value) {
        min_value = 0.0;
        max_value = 0.0;
    }

    var inv_range : f32 = 0.0;
    if (do_normalize && max_value > min_value) {
        inv_range = 1.0 / (max_value - min_value);
    }

    let pixel_index : u32 = gid.y * width + gid.x;
    let base_index : u32 = pixel_index * CHANNEL_CAP;

    var processed : vec4<f32> = read_pixel(base_index);

    if (do_normalize && max_value > min_value) {
        for (var c : u32 = 0u; c < channel_count; c = c + 1u) {
            let value : f32 = get_component(processed, c);
            let normalized : f32 = (value - min_value) * inv_range;
            set_component(&processed, c, normalized);
        }
    }

    if (kernel_id == KERNEL_CONV2D_EDGES) {
        for (var c_edge : u32 = 0u; c_edge < channel_count; c_edge = c_edge + 1u) {
            let value_edge : f32 = get_component(processed, c_edge);
            let adjusted : f32 = abs(value_edge - 0.5) * 2.0;
            set_component(&processed, c_edge, adjusted);
        }
    }

    let coord : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
    let original : vec4<f32> = textureLoad(input_texture, coord, 0);
    var result : vec4<f32> = lerp_vec4(original, processed, alpha);
    result.w = original.w;

    write_pixel(base_index, result);
}
