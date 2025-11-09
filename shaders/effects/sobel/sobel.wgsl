// Sobel combine shader
// Reuses low-level convolve passes (Sobel X/Y) and combines their results
// into an edge magnitude using the selected distance metric.

const CHANNEL_COUNT : u32 = 4u;

struct SobelParams {
    width : f32,
    height : f32,
    channel_count : f32,
    dist_metric : f32,
    _pad0 : f32,
    alpha : f32,
    time : f32,
    speed : f32,
};

@group(0) @binding(0) var<storage, read> sobel_x_buffer : array<f32>;
@group(0) @binding(1) var<storage, read> sobel_y_buffer : array<f32>;
@group(0) @binding(2) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(3) var<uniform> params : SobelParams;

fn as_u32(value : f32) -> u32 {
    return u32(max(round(value), 0.0));
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
        case 101u: { // Triangular (approx)
            return max(abs_dx - delta_y * 0.5, abs_dy);
        }
        case 102u: { // Hexagram (approx)
            let a : f32 = max(abs_dx - delta_y * 0.5, abs_dy);
            let b : f32 = max(abs_dx + delta_y * 0.5, abs_dy);
            return max(a, b);
        }
        case 201u: { // SDF-like
            let r : f32 = sqrt(max(delta_x * delta_x + delta_y * delta_y, 0.0));
            return r;
        }
        default: { // Euclidean
            return sqrt(max(delta_x * delta_x + delta_y * delta_y, 0.0));
        }
    }
}

fn write_pixel(base_index : u32, color : vec4<f32>) {
    output_buffer[base_index + 0u] = color.x;
    output_buffer[base_index + 1u] = color.y;
    output_buffer[base_index + 2u] = color.z;
    output_buffer[base_index + 3u] = color.w;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let width : u32 = as_u32(params.width);
    let height : u32 = as_u32(params.height);
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    let channels : u32 = max(1u, min(as_u32(params.channel_count), CHANNEL_COUNT));
    let pixel_index : u32 = gid.y * width + gid.x;
    let base : u32 = pixel_index * CHANNEL_COUNT;

    var result : vec4<f32> = vec4<f32>(0.0);
    let metric : u32 = as_u32(params.dist_metric);

    for (var c : u32 = 0u; c < channels; c = c + 1u) {
        let gx : f32 = sobel_x_buffer[base + c];
        let gy : f32 = sobel_y_buffer[base + c];
        var d : f32 = distance_metric(gx, gy, metric);

        // Rough normalization to [0,1] based on kernel bounds
        var denom : f32 = 1.0;
        switch metric {
            case 2u: { denom = 16.0; }              // Manhattan: 8 + 8
            case 3u: { denom = 8.0; }               // Chebyshev: max 8
            case 4u: { denom = 11.314f; }           // Octagram approx
            case 201u: { denom = 11.314f; }         // SDF/Euclid approx
            default: { denom = 11.314f; }           // Euclidean: 8*sqrt(2)
        }
        d = clamp(d / max(denom, 1e-6), 0.0, 1.0);
        set_component(&result, c, d);
    }

    // Preserve opaque alpha
    result.w = 1.0;
    write_pixel(base, result);
}
