// Lens distortion effect matching `noisemaker.effects.lens_distortion`.
// The shader warps sample coordinates radially around the frame center,
// optionally zooming when the displacement is negative. Time and speed are
// accepted for API parity but do not influence the current math, mirroring the
// reference Python implementation.

struct LensDistortionParams {
    width : f32,
    height : f32,
    channels : f32,
    displacement : f32,
    time : f32,
    speed : f32,
    _pad0 : f32,
    _pad1 : f32,
};

const CHANNEL_COUNT : u32 = 4u;
const HALF_FRAME : f32 = 0.5;
const MAX_DISTANCE : f32 = sqrt(HALF_FRAME * HALF_FRAME + HALF_FRAME * HALF_FRAME);

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : LensDistortionParams;

fn as_u32(value : f32) -> u32 {
    return u32(max(round(value), 0.0));
}

fn wrap_index(value : f32, size : u32) -> i32 {
    if (size == 0u) {
        return 0;
    }

    let truncated : i32 = i32(value);
    let limit : i32 = i32(size);
    var wrapped : i32 = truncated % limit;
    if (wrapped < 0) {
        wrapped = wrapped + limit;
    }
    return wrapped;
}

fn write_pixel(base_index : u32, texel : vec4<f32>) {
    output_buffer[base_index + 0u] = texel.x;
    output_buffer[base_index + 1u] = texel.y;
    output_buffer[base_index + 2u] = texel.z;
    output_buffer[base_index + 3u] = texel.w;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let width : u32 = as_u32(params.width);
    let height : u32 = as_u32(params.height);
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    let width_f : f32 = max(params.width, 1.0);
    let height_f : f32 = max(params.height, 1.0);
    let displacement : f32 = params.displacement;
    let zoom : f32 = select(0.0, displacement * -0.25, displacement < 0.0);

    // Keep layout parity with the Python signature.
    let time : f32 = params.time;
    let speed : f32 = params.speed;
    let _unused : f32 = (time + speed) * 0.0;

    let x_index : f32 = f32(gid.x) / width_f;
    let y_index : f32 = f32(gid.y) / height_f;
    let x_dist : f32 = x_index - HALF_FRAME;
    let y_dist : f32 = y_index - HALF_FRAME;

    let distance_from_center : f32 = sqrt(x_dist * x_dist + y_dist * y_dist);
    let normalized_distance : f32 = clamp(distance_from_center / MAX_DISTANCE, 0.0, 1.0);
    let center_weight : f32 = 1.0 - normalized_distance;
    let center_weight_sq : f32 = center_weight * center_weight + _unused;

    let x_offset : f32 = (
        x_index -
        x_dist * zoom -
        x_dist * center_weight_sq * displacement
    ) * width_f;
    let y_offset : f32 = (
        y_index -
        y_dist * zoom -
        y_dist * center_weight_sq * displacement
    ) * height_f;

    let xi : i32 = wrap_index(x_offset, width);
    let yi : i32 = wrap_index(y_offset, height);
    let sample_coords : vec2<i32> = vec2<i32>(xi, yi);
    let texel : vec4<f32> = textureLoad(input_texture, sample_coords, 0);

    let pixel_index : u32 = gid.y * width + gid.x;
    let base_index : u32 = pixel_index * CHANNEL_COUNT;
    write_pixel(base_index, texel);
}
