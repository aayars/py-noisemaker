// Pixel Sort effect shader. Mirrors noisemaker.effects.pixel_sort and the
// JavaScript reference implementation. Three-stage pipeline:
//   1. prepare   - pad and rotate into a square buffer.
//   2. sort_rows - per-row counting sort with brightest alignment.
//   3. finalize  - rotate back, crop, and blend with the source image.

const PI : f32 = 3.141592653589793;
const CHANNEL_COUNT : u32 = 4u;
const NUM_BUCKETS : u32 = 256u;
const MAX_ROW_PIXELS : u32 = 4096u; // safeguard to match JS implementation limits

struct PixelSortParams {
    width : f32,
    height : f32,
    channel_count : f32,
    angled : f32,
    darkest : f32,
    time : f32,
    speed : f32,
    want_size : f32,
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : PixelSortParams;
@group(0) @binding(3) var<storage, read_write> prepared_buffer : array<f32>;
@group(0) @binding(4) var<storage, read_write> sorted_buffer : array<f32>;

fn clamp01(value : f32) -> f32 {
    return clamp(value, 0.0, 1.0);
}

fn srgb_to_linear(value : f32) -> f32 {
    if (value <= 0.04045) {
        return value / 12.92;
    }
    return pow((value + 0.055) / 1.055, 2.4);
}

fn oklab_l_component(rgb : vec3<f32>) -> f32 {
    let r : f32 = srgb_to_linear(clamp01(rgb.x));
    let g : f32 = srgb_to_linear(clamp01(rgb.y));
    let b : f32 = srgb_to_linear(clamp01(rgb.z));

    let l : f32 = 0.4121656120 * r + 0.5362752080 * g + 0.0514575653 * b;
    let m : f32 = 0.2118591070 * r + 0.6807189584 * g + 0.1074065790 * b;
    let s : f32 = 0.0883097947 * r + 0.2818474174 * g + 0.6302613616 * b;

    let l_c : f32 = pow(abs(l), 1.0 / 3.0) * sign(l);
    let m_c : f32 = pow(abs(m), 1.0 / 3.0) * sign(m);
    let s_c : f32 = pow(abs(s), 1.0 / 3.0) * sign(s);

    return clamp01(0.2104542553 * l_c + 0.7936177850 * m_c - 0.0040720468 * s_c);
}

fn compute_brightness(color : vec4<f32>) -> f32 {
    return oklab_l_component(vec3<f32>(color.x, color.y, color.z));
}

fn clamp_bucket(value : f32) -> u32 {
    let scaled : f32 = clamp(value, 0.0, 0.999999) * f32(NUM_BUCKETS - 1u);
    return u32(scaled + 0.5);
}

fn resolve_angle() -> f32 {
    var angle : f32 = params.angled;
    if (angle != 0.0 && abs(angle) <= 1.0) {
        angle = params.time * params.speed * 360.0;
    }
    return angle * PI / 180.0;
}

fn resolve_want_size() -> u32 {
    let safe_want : f32 = clamp(round(max(params.want_size, 0.0)), 0.0, f32(MAX_ROW_PIXELS));
    return u32(safe_want);
}

fn write_output_pixel(index : u32, color : vec4<f32>) {
    output_buffer[index + 0u] = color.x;
    output_buffer[index + 1u] = color.y;
    output_buffer[index + 2u] = color.z;
    output_buffer[index + 3u] = color.w;
}

// -----------------------------------------------------------------------------
// Pass 1: pad and rotate into prepared_buffer
// -----------------------------------------------------------------------------

@compute @workgroup_size(8, 8, 1)
fn prepare(@builtin(global_invocation_id) gid : vec3<u32>) {
    let want : u32 = resolve_want_size();
    if (want == 0u || gid.x >= want || gid.y >= want) {
        return;
    }

    let width : u32 = max(u32(params.width), 1u);
    let height : u32 = max(u32(params.height), 1u);

    let pad_x : i32 = (i32(want) - i32(width)) / 2;
    let pad_y : i32 = (i32(want) - i32(height)) / 2;
    let angle_rad : f32 = resolve_angle();
    let cos_a : f32 = cos(angle_rad);
    let sin_a : f32 = sin(angle_rad);

    let center : f32 = (f32(want) - 1.0) * 0.5;
    let px : f32 = f32(gid.x);
    let py : f32 = f32(gid.y);

    let dx : f32 = px - center;
    let dy : f32 = py - center;

    let src_x_f : f32 = cos_a * dx + sin_a * dy + center;
    let src_y_f : f32 = -sin_a * dx + cos_a * dy + center;

    let src_x : i32 = i32(round(src_x_f));
    let src_y : i32 = i32(round(src_y_f));

    let orig_x : i32 = src_x - pad_x;
    let orig_y : i32 = src_y - pad_y;

    var color : vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    if (orig_x >= 0 && orig_x < i32(width) && orig_y >= 0 && orig_y < i32(height)) {
        color = textureLoad(input_texture, vec2<i32>(orig_x, orig_y), 0);
    }

    if (params.darkest != 0.0) {
        color = vec4<f32>(1.0) - color;
    }

    let base : u32 = (gid.y * want + gid.x) * CHANNEL_COUNT;
    prepared_buffer[base + 0u] = color.x;
    prepared_buffer[base + 1u] = color.y;
    prepared_buffer[base + 2u] = color.z;
    prepared_buffer[base + 3u] = color.w;

    sorted_buffer[base + 0u] = color.x;
    sorted_buffer[base + 1u] = color.y;
    sorted_buffer[base + 2u] = color.z;
    sorted_buffer[base + 3u] = color.w;
}

// -----------------------------------------------------------------------------
// Pass 2: per-row counting sort with brightest alignment
// -----------------------------------------------------------------------------

@compute @workgroup_size(1, 1, 1)
fn sort_rows(@builtin(global_invocation_id) gid : vec3<u32>) {
    let want : u32 = resolve_want_size();
    if (want == 0u || gid.y >= want) {
        return;
    }

    let channel_limit : u32 = min(CHANNEL_COUNT, max(u32(params.channel_count), 1u));
    let row_index : u32 = gid.y;
    let row_start : u32 = row_index * want * CHANNEL_COUNT;

    var max_brightness : f32 = -1.0;
    var brightest_index : u32 = 0u;
    for (var x : u32 = 0u; x < want; x = x + 1u) {
        let base : u32 = row_start + x * CHANNEL_COUNT;
        let color : vec4<f32> = vec4<f32>(
            prepared_buffer[base + 0u],
            prepared_buffer[base + 1u],
            prepared_buffer[base + 2u],
            prepared_buffer[base + 3u]
        );
        let brightness : f32 = compute_brightness(color);
        if (brightness > max_brightness) {
            max_brightness = brightness;
            brightest_index = x;
        }
    }

    var histogram : array<u32, NUM_BUCKETS>;
    var positions : array<u32, NUM_BUCKETS>;

    var shift : u32 = want - brightest_index;
    if (shift == want) {
        shift = 0u;
    }

    for (var channel : u32 = 0u; channel < channel_limit; channel = channel + 1u) {
        for (var i : u32 = 0u; i < NUM_BUCKETS; i = i + 1u) {
            histogram[i] = 0u;
        }

        for (var x : u32 = 0u; x < want; x = x + 1u) {
            let idx : u32 = row_start + x * CHANNEL_COUNT + channel;
            let value : f32 = prepared_buffer[idx];
            let bucket : u32 = clamp_bucket(value);
            histogram[bucket] = histogram[bucket] + 1u;
        }

        var cumulative : u32 = 0u;
        for (var i : u32 = 0u; i < NUM_BUCKETS; i = i + 1u) {
            let count : u32 = histogram[i];
            positions[i] = cumulative;
            cumulative = cumulative + count;
        }

        for (var x : u32 = 0u; x < want; x = x + 1u) {
            let idx : u32 = row_start + x * CHANNEL_COUNT + channel;
            let value : f32 = prepared_buffer[idx];
            let bucket : u32 = clamp_bucket(value);
            let offset : u32 = positions[bucket];
            positions[bucket] = offset + 1u;

            var rotated_index : u32 = offset + shift;
            if (rotated_index >= want) {
                rotated_index = rotated_index - want;
            }

            let dest_idx : u32 = row_start + rotated_index * CHANNEL_COUNT + channel;
            sorted_buffer[dest_idx] = value;
        }
    }

    if (channel_limit < CHANNEL_COUNT) {
        let alpha_channel : u32 = CHANNEL_COUNT - 1u;
        for (var x : u32 = 0u; x < want; x = x + 1u) {
            let idx : u32 = row_start + x * CHANNEL_COUNT + alpha_channel;
            let value : f32 = prepared_buffer[idx];
            var rotated_index : u32 = x + shift;
            if (rotated_index >= want) {
                rotated_index = rotated_index - want;
            }
            let dest_idx : u32 = row_start + rotated_index * CHANNEL_COUNT + alpha_channel;
            sorted_buffer[dest_idx] = value;
        }
    }
}

// -----------------------------------------------------------------------------
// Pass 3: rotate back, crop, and blend with the source
// -----------------------------------------------------------------------------

@compute @workgroup_size(8, 8, 1)
fn finalize(@builtin(global_invocation_id) gid : vec3<u32>) {
    let width : u32 = max(u32(params.width), 1u);
    let height : u32 = max(u32(params.height), 1u);

    if (gid.x >= width || gid.y >= height) {
        return;
    }

    let want : u32 = resolve_want_size();
    let pixel_index : u32 = gid.y * width + gid.x;
    let base_index : u32 = pixel_index * CHANNEL_COUNT;

    let original_color : vec4<f32> = textureLoad(
        input_texture,
        vec2<i32>(i32(gid.x), i32(gid.y)),
        0
    );

    if (want == 0u || want > MAX_ROW_PIXELS) {
        write_output_pixel(base_index, original_color);
        return;
    }

    let pad_x : i32 = (i32(want) - i32(width)) / 2;
    let pad_y : i32 = (i32(want) - i32(height)) / 2;
    let angle_rad : f32 = resolve_angle();
    let cos_a : f32 = cos(angle_rad);
    let sin_a : f32 = sin(angle_rad);
    let center : f32 = (f32(want) - 1.0) * 0.5;

    let padded_x : f32 = f32(i32(gid.x) + pad_x);
    let padded_y : f32 = f32(i32(gid.y) + pad_y);

    let dx : f32 = padded_x - center;
    let dy : f32 = padded_y - center;

    let rot_x_f : f32 = cos_a * dx - sin_a * dy + center;
    let rot_y_f : f32 = sin_a * dx + cos_a * dy + center;

    let rot_x : i32 = i32(round(rot_x_f));
    let rot_y : i32 = i32(round(rot_y_f));

    var sorted_color : vec4<f32> = original_color;
    if (rot_x >= 0 && rot_x < i32(want) && rot_y >= 0 && rot_y < i32(want)) {
        let sorted_base : u32 = (u32(rot_y) * want + u32(rot_x)) * CHANNEL_COUNT;
        sorted_color = vec4<f32>(
            sorted_buffer[sorted_base + 0u],
            sorted_buffer[sorted_base + 1u],
            sorted_buffer[sorted_base + 2u],
            sorted_buffer[sorted_base + 3u]
        );
    }

    var working_source : vec4<f32> = original_color;
    var working_sorted : vec4<f32> = sorted_color;

    if (params.darkest != 0.0) {
        working_source = vec4<f32>(1.0) - working_source;
        working_sorted = vec4<f32>(1.0) - working_sorted;
    }

    var blended : vec4<f32> = max(working_source, working_sorted);
    blended = clamp(blended, vec4<f32>(0.0), vec4<f32>(1.0));
    blended.w = working_source.w;

    if (params.darkest != 0.0) {
        blended = vec4<f32>(1.0) - blended;
        blended.w = original_color.w;
    } else {
        blended.w = original_color.w;
    }

    write_output_pixel(base_index, blended);
}
