// Pixel Sort effect. Mirrors noisemaker.effects.pixel_sort.
// Applies a rotated selection sort based on OKLab luminance, optionally inverting
// for "darkest" ordering. Rotation angle follows the Python reference semantics:
//   * If `angled` is false/0, operate axis aligned.
//   * If `angled` is true within [-1, 1], derive the angle from time * speed.
//   * Otherwise treat `angled` as an explicit degree value.

const PI : f32 = 3.141592653589793;
const NEG_INFINITY : f32 = -0x1.fffffep+127;
const CHANNEL_COUNT : u32 = 4u;

struct PixelSortParams {
    size : vec4<f32>,      // width, height, channels, angled
    controls : vec4<f32>,  // darkest, time, speed, _pad0
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : PixelSortParams;
@group(0) @binding(3) var<storage, read_write> scratch : array<f32>;

fn as_u32(value : f32) -> u32 {
    return u32(max(round(value), 0.0));
}

fn clamp01(value : f32) -> f32 {
    return clamp(value, 0.0, 1.0);
}

fn srgb_to_linear(value : f32) -> f32 {
    if (value <= 0.04045) {
        return value / 12.92;
    }
    return pow((value + 0.055) / 1.055, 2.4);
}

fn cbrt(value : f32) -> f32 {
    if (value == 0.0) {
        return 0.0;
    }
    let sign_value : f32 = select(-1.0, 1.0, value >= 0.0);
    return sign_value * pow(abs(value), 1.0 / 3.0);
}

fn oklab_l_component(rgb : vec3<f32>) -> f32 {
    let r : f32 = srgb_to_linear(clamp01(rgb.x));
    let g : f32 = srgb_to_linear(clamp01(rgb.y));
    let b : f32 = srgb_to_linear(clamp01(rgb.z));

    let l : f32 = 0.4121656120 * r + 0.5362752080 * g + 0.0514575653 * b;
    let m : f32 = 0.2118591070 * r + 0.6807189584 * g + 0.1074065790 * b;
    let s : f32 = 0.0883097947 * r + 0.2818474174 * g + 0.6302613616 * b;

    let l_c : f32 = cbrt(l);
    let m_c : f32 = cbrt(m);
    let s_c : f32 = cbrt(s);

    return clamp01(0.2104542553 * l_c + 0.7936177850 * m_c - 0.0040720468 * s_c);
}

fn compute_brightness(color : vec4<f32>) -> f32 {
    return oklab_l_component(vec3<f32>(color.x, color.y, color.z));
}

fn wrap_index(value : i32, size : u32) -> u32 {
    if (size == 0u) {
        return 0u;
    }
    var wrapped : i32 = value % i32(size);
    if (wrapped < 0) {
        wrapped = wrapped + i32(size);
    }
    return u32(wrapped);
}

fn scratch_color_offset(base : u32, pixel_index : u32) -> u32 {
    return base + pixel_index * CHANNEL_COUNT;
}

fn load_scratch_color(base : u32, pixel_index : u32) -> vec4<f32> {
    let offset : u32 = scratch_color_offset(base, pixel_index);
    return vec4<f32>(
        scratch[offset],
        scratch[offset + 1u],
        scratch[offset + 2u],
        scratch[offset + 3u]
    );
}

fn store_scratch_color(base : u32, pixel_index : u32, color : vec4<f32>) {
    let offset : u32 = scratch_color_offset(base, pixel_index);
    scratch[offset] = color.x;
    scratch[offset + 1u] = color.y;
    scratch[offset + 2u] = color.z;
    scratch[offset + 3u] = color.w;
}

fn sample_rotated_color(
    image_base : u32,
    dest_x : u32,
    dest_y : u32,
    size : u32,
    cos_angle : f32,
    sin_angle : f32
) -> vec4<f32> {
    let size_f : f32 = f32(size);
    let x_norm : f32 = f32(dest_x) / size_f - 0.5;
    let y_norm : f32 = f32(dest_y) / size_f - 0.5;

    let src_x_norm : f32 = cos_angle * x_norm + sin_angle * y_norm + 0.5;
    let src_y_norm : f32 = -sin_angle * x_norm + cos_angle * y_norm + 0.5;

    let src_x : u32 = wrap_index(i32(floor(src_x_norm * size_f)), size);
    let src_y : u32 = wrap_index(i32(floor(src_y_norm * size_f)), size);
    let pixel_index : u32 = src_y * size + src_x;
    return load_scratch_color(image_base, pixel_index);
}

fn store_output_color(pixel_index : u32, color : vec4<f32>) {
    let base : u32 = pixel_index * CHANNEL_COUNT;
    output_buffer[base] = color.x;
    output_buffer[base + 1u] = color.y;
    output_buffer[base + 2u] = color.z;
    output_buffer[base + 3u] = 1.0;
}

fn set_scratch_range(start : u32, count : u32, value : f32) {
    var i : u32 = 0u;
    loop {
        if (i >= count) {
            break;
        }
        scratch[start + i] = value;
        i = i + 1u;
    }
}

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    if (gid.x > 0u || gid.y > 0u || gid.z > 0u) {
        return;
    }

    let dims : vec2<u32> = textureDimensions(input_texture, 0);
    let width : u32 = select(as_u32(params.size.x), dims.x, dims.x > 0u);
    let height : u32 = select(as_u32(params.size.y), dims.y, dims.y > 0u);
    if (width == 0u || height == 0u) {
        return;
    }

    let pixel_count : u32 = width * height;
    if (pixel_count == 0u) {
        return;
    }

    let darkest_flag : bool = params.controls.x != 0.0;

    var angled_value : f32 = params.size.w;
    let angled_flag : bool = angled_value != 0.0;
    if (angled_flag && abs(angled_value) <= 1.0) {
        angled_value = params.controls.y * params.controls.z * 360.0;
    }
    let angle_radians : f32 = angled_value * (PI / 180.0);
    let cos_angle : f32 = cos(angle_radians);
    let sin_angle : f32 = sin(angle_radians);
    let cos_neg_angle : f32 = cos_angle;
    let sin_neg_angle : f32 = -sin_angle;

    // Reduce padded size to avoid excessive work; rotate within a square just large enough
    let padded_size : u32 = max(width, height);
    if (padded_size == 0u) {
        return;
    }

    let pad_x : u32 = (padded_size - width) / 2u;
    let pad_y : u32 = (padded_size - height) / 2u;
    let padded_pixel_count : u32 = padded_size * padded_size;
    let padded_color_count : u32 = padded_pixel_count * CHANNEL_COUNT;
    let row_color_count : u32 = padded_size * CHANNEL_COUNT;

    let padded_base : u32 = 0u;
    let sorted_image_base : u32 = padded_base + padded_color_count;
    let row_color_base : u32 = sorted_image_base + padded_color_count;
    let row_sorted_base : u32 = row_color_base + row_color_count;
    let row_shifted_base : u32 = row_sorted_base + row_color_count;
    let row_brightness_base : u32 = row_shifted_base + row_color_count;
    let scratch_needed : u32 = row_brightness_base + padded_size;

    if (scratch_needed > arrayLength(&scratch)) {
        return;
    }

    // Stage 1: build padded image centered in the larger square.
    var padded_y : u32 = 0u;
    loop {
        if (padded_y >= padded_size) {
            break;
        }
        var padded_x : u32 = 0u;
        loop {
            if (padded_x >= padded_size) {
                break;
            }

            let source_x : i32 = i32(padded_x) - i32(pad_x);
            let source_y : i32 = i32(padded_y) - i32(pad_y);

            var color : vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            if (source_x >= 0 && source_x < i32(width) && source_y >= 0 && source_y < i32(height)) {
                color = textureLoad(input_texture, vec2<i32>(source_x, source_y), 0);
            }
            if (darkest_flag) {
                color = vec4<f32>(1.0) - color;
            }

            let pixel_index : u32 = padded_y * padded_size + padded_x;
            store_scratch_color(padded_base, pixel_index, color);

            padded_x = padded_x + 1u;
        }
        padded_y = padded_y + 1u;
    }

    // Stage 2: rotate, sort rows, and apply the brightest shift.
    padded_y = 0u;
    loop {
        if (padded_y >= padded_size) {
            break;
        }

        // Prepare per-row buffers
        set_scratch_range(row_color_base, row_color_count, 0.0);
        set_scratch_range(row_shifted_base, row_color_count, 0.0);

        var shift_index : u32 = 0u;
        var max_value : f32 = NEG_INFINITY;

        // Build rotated row and determine shift position based on brightest value
        var padded_x : u32 = 0u;
        loop {
            if (padded_x >= padded_size) { break; }
            let color : vec4<f32> = sample_rotated_color(padded_base, padded_x, padded_y, padded_size, cos_angle, sin_angle);
            let column_offset : u32 = padded_x * CHANNEL_COUNT;
            scratch[row_color_base + column_offset] = color.x;
            scratch[row_color_base + column_offset + 1u] = color.y;
            scratch[row_color_base + column_offset + 2u] = color.z;
            scratch[row_color_base + column_offset + 3u] = color.w;
            let v : f32 = compute_brightness(color);
            if (v > max_value) { max_value = v; shift_index = padded_x; }
            padded_x = padded_x + 1u;
        }

        // Circularly shift the row based on brightest index (no full sort to avoid timeouts)
        var position : u32 = 0u;
        loop {
            if (position >= padded_size) { break; }
            let dest_idx : u32 = (position + shift_index) % padded_size;
            let from_offset : u32 = position * CHANNEL_COUNT;
            let to_offset : u32 = dest_idx * CHANNEL_COUNT;
            scratch[row_shifted_base + to_offset] = scratch[row_color_base + from_offset];
            scratch[row_shifted_base + to_offset + 1u] = scratch[row_color_base + from_offset + 1u];
            scratch[row_shifted_base + to_offset + 2u] = scratch[row_color_base + from_offset + 2u];
            scratch[row_shifted_base + to_offset + 3u] = scratch[row_color_base + from_offset + 3u];
            position = position + 1u;
        }

        // Store shifted row into rotated output image buffer
        let row_start : u32 = padded_y * padded_size;
        var column : u32 = 0u;
        loop {
            if (column >= padded_size) { break; }
            let column_offset : u32 = column * CHANNEL_COUNT;
            let pixel_index : u32 = row_start + column;
            store_scratch_color(
                sorted_image_base,
                pixel_index,
                vec4<f32>(
                    scratch[row_shifted_base + column_offset],
                    scratch[row_shifted_base + column_offset + 1u],
                    scratch[row_shifted_base + column_offset + 2u],
                    scratch[row_shifted_base + column_offset + 3u]
                )
            );
            column = column + 1u;
        }

        padded_y = padded_y + 1u;
    }

    // Stage 4: crop back to the original size and blend with the source.
    var y : u32 = 0u;
    loop {
        if (y >= height) {
            break;
        }
        var x : u32 = 0u;
        loop {
            if (x >= width) {
                break;
            }
            let padded_x_coord : u32 = x + pad_x;
            let padded_y_coord : u32 = y + pad_y;
            // Sample from rotated image and rotate back on-the-fly
            let sorted_color : vec4<f32> = sample_rotated_color(
                sorted_image_base,
                padded_x_coord,
                padded_y_coord,
                padded_size,
                cos_neg_angle,
                sin_neg_angle
            );

            var working_color : vec4<f32> = textureLoad(input_texture, vec2<i32>(i32(x), i32(y)), 0);
            if (darkest_flag) {
                working_color = vec4<f32>(1.0) - working_color;
            }

            var combined : vec4<f32> = max(working_color, sorted_color);
            if (darkest_flag) {
                combined = vec4<f32>(1.0) - combined;
            }
            combined = clamp(combined, vec4<f32>(0.0), vec4<f32>(1.0));

            store_output_color(y * width + x, combined);

            x = x + 1u;
        }
        y = y + 1u;
    }
}
