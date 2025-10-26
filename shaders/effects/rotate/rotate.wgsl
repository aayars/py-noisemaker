// Rotate effect: matches Noisemaker's rotate() by tiling the input into a square,
// rotating in normalized space, and cropping back to the original dimensions.

const CHANNEL_COUNT : u32 = 4u;

struct RotateParams {
    dims : vec4<f32>,                 // (width, height, channels, unused)
    angle_time_speed_pad : vec4<f32>, // (angle_degrees, time, speed, _pad0)
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : RotateParams;

fn as_u32(value : f32) -> u32 {
    return u32(max(round(value), 0.0));
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

fn write_pixel(base_index : u32, color : vec4<f32>) {
    output_buffer[base_index + 0u] = color.x;
    output_buffer[base_index + 1u] = color.y;
    output_buffer[base_index + 2u] = color.z;
    output_buffer[base_index + 3u] = color.w;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let width : u32 = as_u32(params.dims.x);
    let height : u32 = as_u32(params.dims.y);
    if (global_id.x >= width || global_id.y >= height) {
        return;
    }

    let width_i : i32 = i32(width);
    let height_i : i32 = i32(height);
    if (width_i <= 0 || height_i <= 0) {
        return;
    }

    let padded_size_i : i32 = max(width_i, height_i) * 2;
    if (padded_size_i <= 0) {
        return;
    }

    let padded_size_f : f32 = f32(padded_size_i);
    let crop_offset_x : i32 = (padded_size_i - width_i) / 2;
    let crop_offset_y : i32 = (padded_size_i - height_i) / 2;
    let tile_offset_x : i32 = width_i / 2;
    let tile_offset_y : i32 = height_i / 2;

    let padded_coord : vec2<i32> = vec2<i32>(
        i32(global_id.x) + crop_offset_x,
        i32(global_id.y) + crop_offset_y
    );

    let padded_coord_f : vec2<f32> = vec2<f32>(
        f32(padded_coord.x),
        f32(padded_coord.y)
    );
    let normalized : vec2<f32> =
        padded_coord_f / padded_size_f - vec2<f32>(0.5, 0.5);

    let angle_radians : f32 = radians(params.angle_time_speed_pad.x);
    let cos_angle : f32 = cos(angle_radians);
    let sin_angle : f32 = sin(angle_radians);
    let rotation : mat2x2<f32> = mat2x2<f32>(
        cos_angle, -sin_angle,
        sin_angle, cos_angle
    );

    let rotated : vec2<f32> = rotation * normalized + vec2<f32>(0.5, 0.5);
    let rotated_scaled : vec2<f32> = rotated * padded_size_f;

    let padded_sample : vec2<i32> = vec2<i32>(
        wrap_index(i32(rotated_scaled.x), padded_size_i),
        wrap_index(i32(rotated_scaled.y), padded_size_i)
    );

    let source : vec2<i32> = vec2<i32>(
        wrap_index(padded_sample.x + tile_offset_x, width_i),
        wrap_index(padded_sample.y + tile_offset_y, height_i)
    );

    let coords : vec2<i32> = source;
    let texel : vec4<f32> = textureLoad(input_texture, coords, 0);

    let pixel_index : u32 = global_id.y * width + global_id.x;
    let base_index : u32 = pixel_index * CHANNEL_COUNT;
    write_pixel(base_index, texel);
}
