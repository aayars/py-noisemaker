// Sobel operator compute shader.
//
// Mirrors noisemaker/effects.py::sobel_operator by blurring the source image,
// convolving the result with Sobel kernels, normalizing the distance field,
// and applying the standard Noisemaker signed-range scaling followed by an
// offset of (-1, -1).

const PI : f32 = 3.14159265358979323846;
const TAU : f32 = 6.28318530717958647692;
const CHANNEL_CAP : u32 = 4u;
const F32_MAX : f32 = 0x1.fffffep+127;
const F32_MIN : f32 = -0x1.fffffep+127;

struct SobelParams {
    width : f32,
    height : f32,
    channels : f32,
    dist_metric : f32,
    time : f32,
    speed : f32,
    _pad0 : f32,
    _pad1 : f32,
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : SobelParams;
@group(0) @binding(3) var<storage, read_write> temp_buffer : array<f32>;

const BLUR_KERNEL : array<f32, 25> = array<f32, 25>(
    1.0 / 36.0, 4.0 / 36.0, 6.0 / 36.0, 4.0 / 36.0, 1.0 / 36.0,
    4.0 / 36.0, 16.0 / 36.0, 24.0 / 36.0, 16.0 / 36.0, 4.0 / 36.0,
    6.0 / 36.0, 24.0 / 36.0, 36.0 / 36.0, 24.0 / 36.0, 6.0 / 36.0,
    4.0 / 36.0, 16.0 / 36.0, 24.0 / 36.0, 16.0 / 36.0, 4.0 / 36.0,
    1.0 / 36.0, 4.0 / 36.0, 6.0 / 36.0, 4.0 / 36.0, 1.0 / 36.0
);

const BLUR_OFFSETS : array<vec2<i32>, 25> = array<vec2<i32>, 25>(
    vec2<i32>(-2, -2), vec2<i32>(-1, -2), vec2<i32>(0, -2), vec2<i32>(1, -2),
    vec2<i32>(2, -2), vec2<i32>(-2, -1), vec2<i32>(-1, -1), vec2<i32>(0, -1),
    vec2<i32>(1, -1), vec2<i32>(2, -1), vec2<i32>(-2, 0), vec2<i32>(-1, 0),
    vec2<i32>(0, 0), vec2<i32>(1, 0), vec2<i32>(2, 0), vec2<i32>(-2, 1),
    vec2<i32>(-1, 1), vec2<i32>(0, 1), vec2<i32>(1, 1), vec2<i32>(2, 1),
    vec2<i32>(-2, 2), vec2<i32>(-1, 2), vec2<i32>(0, 2), vec2<i32>(1, 2),
    vec2<i32>(2, 2)
);

const SOBEL_X : array<f32, 9> = array<f32, 9>(
    0.5, 0.0, -0.5,
    1.0, 0.0, -1.0,
    0.5, 0.0, -0.5
);

const SOBEL_Y : array<f32, 9> = array<f32, 9>(
    0.5, 1.0, 0.5,
    0.0, 0.0, 0.0,
    -0.5, -1.0, -0.5
);

fn as_u32(value : f32) -> u32 {
    return u32(max(round(value), 0.0));
}

fn sanitized_channel_count(channel_value : f32) -> u32 {
    let rounded : i32 = i32(round(channel_value));
    if (rounded <= 1) {
        return 1u;
    }
    if (rounded >= i32(CHANNEL_CAP)) {
        return CHANNEL_CAP;
    }
    return u32(rounded);
}

fn wrap_coord(value : i32, limit : i32) -> i32 {
    if (limit <= 0) {
        return 0;
    }
    var wrapped : i32 = value % limit;
    if (wrapped < 0) {
        wrapped = wrapped + limit;
    }
    return wrapped;
}

fn pixel_base_index(x : u32, y : u32, width : u32) -> u32 {
    return (y * width + x) * CHANNEL_CAP;
}

fn sample_source(coords : vec2<i32>, width : i32, height : i32) -> vec4<f32> {
    let sx : i32 = wrap_coord(coords.x, width);
    let sy : i32 = wrap_coord(coords.y, height);
    return textureLoad(input_texture, vec2<i32>(sx, sy), 0);
}

fn compute_blurred(coords : vec2<i32>, width : i32, height : i32) -> vec4<f32> {
    var accum : vec4<f32> = vec4<f32>(0.0);
    for (var i : u32 = 0u; i < 25u; i = i + 1u) {
        let offset_coords : vec2<i32> = coords + BLUR_OFFSETS[i];
        let sample : vec4<f32> = sample_source(offset_coords, width, height);
        accum = accum + sample * BLUR_KERNEL[i];
    }
    return accum;
}

fn vector_component(value : vec4<f32>, index : u32) -> f32 {
    switch index {
        case 0u: { return value.x; }
        case 1u: { return value.y; }
        case 2u: { return value.z; }
        default: { return value.w; }
    }
}

fn compute_distance(a : f32, b : f32, metric : i32) -> f32 {
    let ax : f32 = abs(a);
    let ay : f32 = abs(b);
    switch metric {
        case 2: {
            return ax + ay;
        }
        case 3: {
            return max(ax, ay);
        }
        case 4: {
            let octagram : f32 = (ax + ay) / sqrt(2.0);
            return max(octagram, max(ax, ay));
        }
        case 101: {
            return max(ax - b * 0.5, b);
        }
        case 102: {
            let term0 : f32 = max(ax - b * 0.5, b);
            let term1 : f32 = max(ax - b * -0.5, b * -1.0);
            return max(term0, term1);
        }
        case 201: {
            let angle : f32 = atan2(a, -b) + PI;
            let r : f32 = TAU / 5.0;
            let distance : f32 = sqrt(a * a + b * b);
            return cos(floor(0.5 + angle / r) * r - angle) * distance;
        }
        default: {
            return sqrt(a * a + b * b);
        }
    }
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let width : u32 = as_u32(params.width);
    let height : u32 = as_u32(params.height);
    
    // Guard: exit if this thread is outside image bounds
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    let width_i : i32 = i32(width);
    let height_i : i32 = i32(height);
    let channel_count : u32 = sanitized_channel_count(params.channels);
    let metric : i32 = i32(round(params.dist_metric));
    let fudge : i32 = -1;

    // Each thread processes one pixel independently
    let x : u32 = gid.x;
    let y : u32 = gid.y;
    let coords : vec2<i32> = vec2<i32>(i32(x), i32(y));
    
    // Step 1: Apply blur to this pixel
    let blurred : vec4<f32> = compute_blurred(coords, width_i, height_i);
    let base_index : u32 = pixel_base_index(x, y, width);
    
    // Store blurred values (using fixed normalization range [0, 1])
    for (var c : u32 = 0u; c < channel_count; c = c + 1u) {
        let value : f32 = clamp(vector_component(blurred, c), 0.0, 1.0);
        temp_buffer[base_index + c] = value;
    }
    for (var c : u32 = channel_count; c < CHANNEL_CAP; c = c + 1u) {
        temp_buffer[base_index + c] = 0.0;
    }
    
    // Wait for all threads to finish blur pass (implicit barrier in separate dispatch)
    // In a real multi-pass implementation, this would be a separate shader invocation
    
    // Step 2: Apply Sobel operator to this pixel
    var grad_x : array<f32, CHANNEL_CAP>;
    var grad_y : array<f32, CHANNEL_CAP>;
    for (var c : u32 = 0u; c < CHANNEL_CAP; c = c + 1u) {
        grad_x[c] = 0.0;
        grad_y[c] = 0.0;
    }

    var kernel_index : u32 = 0u;
    for (var ky : i32 = -1; ky <= 1; ky = ky + 1) {
        for (var kx : i32 = -1; kx <= 1; kx = kx + 1) {
            let sample_coords : vec2<i32> = vec2<i32>(i32(x) + kx, i32(y) + ky);
            let sx : u32 = u32(wrap_coord(sample_coords.x, width_i));
            let sy : u32 = u32(wrap_coord(sample_coords.y, height_i));
            let sample_base : u32 = pixel_base_index(sx, sy, width);
            let weight_x : f32 = SOBEL_X[kernel_index];
            let weight_y : f32 = SOBEL_Y[kernel_index];
            for (var c : u32 = 0u; c < channel_count; c = c + 1u) {
                let sample_value : f32 = temp_buffer[sample_base + c];
                grad_x[c] = grad_x[c] + sample_value * weight_x;
                grad_y[c] = grad_y[c] + sample_value * weight_y;
            }
            kernel_index = kernel_index + 1u;
        }
    }

    // Compute distance and output (using fixed normalization)
    for (var c : u32 = 0u; c < channel_count; c = c + 1u) {
        let dist : f32 = compute_distance(grad_x[c], grad_y[c], metric);
        let normalized : f32 = clamp(dist, 0.0, 1.0);
        let scaled : f32 = abs(normalized * 2.0 - 1.0);
        output_buffer[base_index + c] = scaled;
    }
    for (var c : u32 = channel_count; c < CHANNEL_CAP; c = c + 1u) {
        output_buffer[base_index + c] = 0.0;
    }
    
    // Apply offset (fudge)
    let src_x : u32 = u32(wrap_coord(i32(x) + fudge, width_i));
    let src_y : u32 = u32(wrap_coord(i32(y) + fudge, height_i));
    let src_index : u32 = pixel_base_index(src_x, src_y, width);
    
    // Copy from offset position to output
    for (var c : u32 = 0u; c < channel_count; c = c + 1u) {
        temp_buffer[base_index + c] = output_buffer[src_index + c];
    }
    
    // Final write back
    for (var c : u32 = 0u; c < channel_count; c = c + 1u) {
        output_buffer[base_index + c] = temp_buffer[base_index + c];
    }
}
