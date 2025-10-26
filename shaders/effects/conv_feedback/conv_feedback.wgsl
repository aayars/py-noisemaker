// Conv2d feedback loop effect mirroring the conv_feedback CPU implementation.
//
// The shader samples a proportional downsample of the source image around the
// current pixel, iteratively blurs and sharpens the gathered patch to simulate
// repeated convolution, then normalizes and blends the feedback result with the
// original texel.  Iterations, time, and speed parameters are accepted for API
// parity, though the Python reference always clamps the iteration count to 100.

const CHANNEL_COUNT : u32 = 4u;
struct ConvFeedbackParams {
    size : vec4<f32>,      // width, height, channels, _pad0
    options : vec4<f32>,   // iterations, alpha, time, speed
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
// Optional previous frame feedback; when bound by the viewer as 'prev_texture',
// the shader will treat it as the feedback source for iterative updates.
@group(0) @binding(3) var prev_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : ConvFeedbackParams;

fn as_u32(value : f32) -> u32 {
    return u32(max(round(value), 0.0));
}

fn lerp_vec4(a : vec4<f32>, b : vec4<f32>, t : f32) -> vec4<f32> {
    return a + (b - a) * t;
}

fn combine_value(value : f32) -> f32 {
    let up : f32 = max((value - 0.5) * 2.0, 0.0);
    let down : f32 = min(value * 2.0, 1.0);
    return clamp(up + (1.0 - down), 0.0, 1.0);
}

fn write_pixel(base_index : u32, color : vec4<f32>) {
    output_buffer[base_index + 0u] = color.x;
    output_buffer[base_index + 1u] = color.y;
    output_buffer[base_index + 2u] = color.z;
    output_buffer[base_index + 3u] = color.w;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let width_u : u32 = as_u32(params.size.x);
    let height_u : u32 = as_u32(params.size.y);
    if (gid.x >= width_u || gid.y >= height_u) {
        return;
    }

    let alpha : f32 = clamp(params.options.y, 0.0, 1.0);
    let iteration_raw : f32 = max(params.options.x, 0.0);

    let width : i32 = i32(width_u);
    let height : i32 = i32(height_u);
    let prev_width : i32 = max(width / 2, 1);
    let prev_height : i32 = max(height / 2, 1);
    let max_prev : vec2<i32> = vec2<i32>(prev_width - 1, prev_height - 1);

    let down_x : i32 = clamp(i32(gid.x / 2u), 0, max_prev.x);
    let down_y : i32 = clamp(i32(gid.y / 2u), 0, max_prev.y);

    let prev_center : vec4<f32> = textureLoad(prev_texture, vec2<i32>(down_x, down_y), 0);
    let prev_xp : vec4<f32> = textureLoad(prev_texture, vec2<i32>(clamp(down_x + 1, 0, max_prev.x), down_y), 0);
    let prev_xn : vec4<f32> = textureLoad(prev_texture, vec2<i32>(clamp(down_x - 1, 0, max_prev.x), down_y), 0);
    let prev_yp : vec4<f32> = textureLoad(prev_texture, vec2<i32>(down_x, clamp(down_y + 1, 0, max_prev.y)), 0);
    let prev_yn : vec4<f32> = textureLoad(prev_texture, vec2<i32>(down_x, clamp(down_y - 1, 0, max_prev.y)), 0);

    let blur_avg : vec4<f32> = (prev_center * 4.0 + prev_xp + prev_xn + prev_yp + prev_yn) / 8.0;
    let sharpened : vec4<f32> = clamp(prev_center * 1.5 - blur_avg * 0.5, vec4<f32>(0.0), vec4<f32>(1.0));

    let iteration_mix : f32 = clamp(iteration_raw / 100.0, 0.0, 1.0);
    let feedback_base : vec4<f32> = lerp_vec4(sharpened, blur_avg, iteration_mix);

    var combined : vec4<f32> = vec4<f32>(
        combine_value(feedback_base.x),
        combine_value(feedback_base.y),
        combine_value(feedback_base.z),
        combine_value(feedback_base.w),
    );

    let original : vec4<f32> = textureLoad(
        input_texture,
        vec2<i32>(i32(gid.x), i32(gid.y)),
        0,
    );
    combined = lerp_vec4(original, combined, alpha);
    combined = clamp(combined, vec4<f32>(0.0), vec4<f32>(1.0));

    let pixel_index : u32 = gid.y * width_u + gid.x;
    let base_index : u32 = pixel_index * CHANNEL_COUNT;
    write_pixel(base_index, combined);
}
