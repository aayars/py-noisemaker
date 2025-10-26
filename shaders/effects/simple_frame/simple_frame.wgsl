// Simple frame mask derived from the Chebyshev distance singularity in the Python reference.
// Applies a binary blend between the source image and a constant brightness value.
struct SimpleFrameParams {
    size : vec4<f32>,       // width, height, channels, brightness
    time_speed : vec4<f32>, // time, speed, unused, unused
};

const CHANNEL_COUNT : u32 = 4u;
const BORDER_BLEND : f32 = 0.55;

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : SimpleFrameParams;

fn as_u32(value : f32) -> u32 {
    return u32(max(round(value), 0.0));
}

fn axis_min_max(size : u32, size_f : f32) -> vec2<f32> {
    if (size <= 1u) {
        return vec2<f32>(0.5, 0.5);
    }
    if ((size & 1u) == 0u) {
        return vec2<f32>(0.0, 0.5);
    }

    let half_floor : f32 = f32(size / 2u);
    let min_val : f32 = 0.5 / size_f;
    let max_val : f32 = (half_floor - 0.5) / size_f;
    return vec2<f32>(min_val, max_val);
}

fn axis_distance(coord : f32, center : f32, dimension : f32) -> f32 {
    if (dimension <= 0.0) {
        return 0.0;
    }

    return abs(coord - center) / dimension;
}

fn posterize_level_one(value : f32) -> f32 {
    let scaled : f32 = value * BORDER_BLEND;
    return clamp(floor(scaled + 0.5), 0.0, 1.0);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let width_u : u32 = max(as_u32(params.size.x), 1u);
    let height_u : u32 = max(as_u32(params.size.y), 1u);
    if (global_id.x >= width_u || global_id.y >= height_u) {
        return;
    }

    let width_f : f32 = max(f32(width_u), 1.0);
    let height_f : f32 = max(f32(height_u), 1.0);
    let half_width_u : u32 = width_u / 2u;
    let half_height_u : u32 = height_u / 2u;
    let center_x : f32 = width_f * 0.5;
    let center_y : f32 = height_f * 0.5;

    let fx : f32 = f32(global_id.x);
    let fy : f32 = f32(global_id.y);
    let dx : f32 = axis_distance(fx, center_x, width_f);
    let dy : f32 = axis_distance(fy, center_y, height_f);

    let axis_x : vec2<f32> = axis_min_max(width_u, width_f);
    let axis_y : vec2<f32> = axis_min_max(height_u, height_f);
    let min_dist : f32 = max(axis_x.x, axis_y.x);
    let max_dist : f32 = max(axis_x.y, axis_y.y);
    let dist : f32 = max(dx, dy);
    let delta : f32 = max_dist - min_dist;

    var normalized : f32;
    if (delta > 0.0) {
        normalized = clamp((dist - min_dist) / delta, 0.0, 1.0);
    } else {
        normalized = clamp(dist, 0.0, 1.0);
    }

    let ramp : f32 = sqrt(normalized);
    let mask : f32 = posterize_level_one(ramp);

    let pixel_index : u32 = global_id.y * width_u + global_id.x;
    let base_index : u32 = pixel_index * CHANNEL_COUNT;
    let coords : vec2<i32> = vec2<i32>(i32(global_id.x), i32(global_id.y));
    let sample : vec4<f32> = textureLoad(input_texture, coords, 0);

    let brightness : f32 = params.size.w;
    let brightness_vec : vec3<f32> = vec3<f32>(brightness);
    let blended_rgb : vec3<f32> = mix(sample.xyz, brightness_vec, vec3<f32>(mask));

    output_buffer[base_index + 0u] = blended_rgb.x;
    output_buffer[base_index + 1u] = blended_rgb.y;
    output_buffer[base_index + 2u] = blended_rgb.z;
    output_buffer[base_index + 3u] = sample.w;
}
