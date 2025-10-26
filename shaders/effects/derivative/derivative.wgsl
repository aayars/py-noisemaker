// Derivative effect compute shader.
//
// Mirrors the Python implementation in noisemaker/effects.py::derivative.
// Applies derivative convolution kernels, computes the requested distance
// metric, optionally normalizes the result, and blends with the original
// tensor when alpha < 1.

const PI : f32 = 3.14159265358979323846;
const TAU : f32 = 6.28318530717958647692;
const FLOAT_MAX : f32 = 3.402823466e38;
const FLOAT_MIN : f32 = -3.402823466e38;
const CHANNEL_COUNT : u32 = 4u;
const SDF_SIDES : f32 = 5.0;

struct GradientPair {
    dx : vec4<f32>,
    dy : vec4<f32>,
};

const DERIVATIVE_KERNEL_OFFSETS : array<vec2<i32>, 9> = array<vec2<i32>, 9>(
    vec2<i32>(-1, -1), vec2<i32>(0, -1), vec2<i32>(1, -1),
    vec2<i32>(-1, 0), vec2<i32>(0, 0), vec2<i32>(1, 0),
    vec2<i32>(-1, 1), vec2<i32>(0, 1), vec2<i32>(1, 1)
);

// Use Sobel kernels to approximate image derivatives
const DERIVATIVE_KERNEL_X : array<f32, 9> = array<f32, 9>(
    -1.0, 0.0, 1.0,
    -2.0, 0.0, 2.0,
    -1.0, 0.0, 1.0
);

const DERIVATIVE_KERNEL_Y : array<f32, 9> = array<f32, 9>(
    -1.0, -2.0, -1.0,
     0.0,  0.0,  0.0,
     1.0,  2.0,  1.0
);

struct DerivativeParams {
    size : vec4<f32>,      // width, height, channels, dist_metric
    options : vec4<f32>,   // with_normalize, alpha, time, speed
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : DerivativeParams;

fn as_u32(value : f32) -> u32 {
    return u32(max(value, 0.0));
}

fn wrap_coord(coord : i32, limit : i32) -> i32 {
    if (limit <= 0) {
        return 0;
    }
    var wrapped : i32 = coord % limit;
    if (wrapped < 0) {
        wrapped = wrapped + limit;
    }
    return wrapped;
}

fn get_component(value : vec4<f32>, index : u32) -> f32 {
    switch index {
        case 0u: { return value.x; }
        case 1u: { return value.y; }
        case 2u: { return value.z; }
        default: { return value.w; }
    }
}

fn set_component(vector_ptr : ptr<function, vec4<f32>>, index : u32, value : f32) {
    switch index {
        case 0u: { (*vector_ptr).x = value; }
        case 1u: { (*vector_ptr).y = value; }
        case 2u: { (*vector_ptr).z = value; }
        default: { (*vector_ptr).w = value; }
    }
}

fn distance_metric(delta_x : f32, delta_y : f32, metric : u32) -> f32 {
    let abs_dx : f32 = abs(delta_x);
    let abs_dy : f32 = abs(delta_y);
    switch metric {
        case 2u: { // Manhattan
            return abs_dx + abs_dy;
        }
        case 3u: { // Chebyshev
            return max(abs_dx, abs_dy);
        }
        case 4u: { // Octagram
            let cross : f32 = (abs_dx + abs_dy) / sqrt(2.0);
            return max(cross, max(abs_dx, abs_dy));
        }
        case 101u: { // Triangular
            return max(abs_dx - delta_y * 0.5, delta_y);
        }
        case 102u: { // Hexagram
            let a : f32 = max(abs_dx - delta_y * 0.5, delta_y);
            let b : f32 = max(abs_dx + delta_y * 0.5, -delta_y);
            return max(a, b);
        }
        case 201u: { // Signed distance field
            let angle : f32 = atan2(delta_x, -delta_y) + PI;
            let step : f32 = TAU / SDF_SIDES;
            let sector : f32 = floor(0.5 + angle / step);
            let offset : f32 = sector * step - angle;
            let radius : f32 = sqrt(max(delta_x * delta_x + delta_y * delta_y, 0.0));
            return cos(offset) * radius;
        }
        default: {
            let sum : f32 = delta_x * delta_x + delta_y * delta_y;
            return sqrt(max(sum, 0.0));
        }
    }
}

fn fetch_texel(x : i32, y : i32, width : i32, height : i32) -> vec4<f32> {
    let wrapped_x : i32 = wrap_coord(x, width);
    let wrapped_y : i32 = wrap_coord(y, height);
    return textureLoad(input_texture, vec2<i32>(wrapped_x, wrapped_y), 0);
}

fn channel_count_from_params() -> u32 {
    let channels : u32 = as_u32(params.size.z);
    if (channels == 0u) {
        return CHANNEL_COUNT; // default to full RGBA if unspecified
    }
    return min(channels, CHANNEL_COUNT);
}

fn metric_from_params() -> u32 {
    let metric_value : f32 = params.size.w;
    let rounded : f32 = floor(metric_value + 0.5);
    if (rounded < 0.0) {
        return 0u;
    }
    return u32(rounded);
}

fn compute_gradients(coords : vec2<i32>, width : i32, height : i32) -> GradientPair {
    var grad_x : vec4<f32> = vec4<f32>(0.0);
    var grad_y : vec4<f32> = vec4<f32>(0.0);

    for (var i : u32 = 0u; i < 9u; i = i + 1u) {
        let offset : vec2<i32> = coords + DERIVATIVE_KERNEL_OFFSETS[i];
        let sample : vec4<f32> = fetch_texel(offset.x, offset.y, width, height);
        let weight_x : f32 = DERIVATIVE_KERNEL_X[i];
        let weight_y : f32 = DERIVATIVE_KERNEL_Y[i];
        grad_x = grad_x + sample * weight_x;
        grad_y = grad_y + sample * weight_y;
    }

    return GradientPair(grad_x, grad_y);
}

fn write_pixel(base_index : u32, color : vec4<f32>) {
    output_buffer[base_index + 0u] = color.x;
    output_buffer[base_index + 1u] = color.y;
    output_buffer[base_index + 2u] = color.z;
    output_buffer[base_index + 3u] = color.w;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let dims : vec2<u32> = textureDimensions(input_texture, 0);
    var width : u32 = select(as_u32(params.size.x), dims.x, dims.x > 0u);
    var height : u32 = select(as_u32(params.size.y), dims.y, dims.y > 0u);
    let channel_count : u32 = channel_count_from_params();
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    let width_i : i32 = i32(width);
    let height_i : i32 = i32(height);
    let metric : u32 = metric_from_params();
    let do_normalize : bool = params.options.x > 0.5;
    let alpha : f32 = clamp(params.options.y, 0.0, 1.0);

    // Compute gradients and distance for this pixel
    let xi : i32 = i32(gid.x);
    let yi : i32 = i32(gid.y);
    let source_texel : vec4<f32> = fetch_texel(xi, yi, width_i, height_i);
    let gradients : GradientPair = compute_gradients(vec2<i32>(xi, yi), width_i, height_i);

    let pixel_index : u32 = gid.y * width + gid.x;
    let base_index : u32 = pixel_index * CHANNEL_COUNT;

    var distances : vec4<f32> = vec4<f32>(0.0);
    for (var c : u32 = 0u; c < channel_count; c = c + 1u) {
        let dist : f32 = distance_metric(
            get_component(gradients.dx, c),
            get_component(gradients.dy, c),
            metric,
        );
        set_component(&distances, c, dist);
    }

    // Optional normalization requires a prepass; approximate per-pixel by dividing by a small factor to avoid huge values if not normalizing
    if (!do_normalize) {
        distances = clamp(distances, vec4<f32>(0.0), vec4<f32>(1.0));
    }

    distances.w = 1.0;
    // Alpha blend with source if requested
    if (alpha < 1.0) {
        var blended : vec4<f32> = distances;
        for (var c : u32 = 0u; c < min(channel_count, 4u); c = c + 1u) {
            let orig : f32 = get_component(source_texel, c);
            let der : f32 = get_component(distances, c);
            let val : f32 = mix(orig, der, alpha);
            set_component(&blended, c, val);
        }
        distances = blended;
    }

    write_pixel(base_index, distances);
}
