// Density map effect compute shader.
//
// Mirrors noisemaker/effects.py::density_map using five passes:
//   1. reset_histogram_main      – clear histogram + reset stats
//   2. reduce_minmax_main        – find global min/max across the image
//   3. histogram_main            – build the bin counts atomically
//   4. finalize_histogram_main   – derive histogram min/max counts
//   5. main                      – remap counts back to pixels

const CHANNEL_COUNT : u32 = 4u;
const FLOAT_MAX : f32 = 3.402823466e38;
const FLOAT_MIN : f32 = -3.402823466e38;
const UINT32_MAX : u32 = 0xffffffffu;

struct DensityMapParams {
    size : vec4<f32>,      // width, height, channels, unused
    controls : vec4<f32>,  // time, speed, unused0, unused1
};

struct DensityMapStats {
    min_value : atomic<u32>,
    max_value : atomic<u32>,
    min_count : atomic<u32>,
    max_count : atomic<u32>,
};

struct HistogramStorage {
    values : array<atomic<u32>>,
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<storage, read_write> histogram_buffer : HistogramStorage;
@group(0) @binding(3) var<uniform> params : DensityMapParams;
@group(0) @binding(4) var<storage, read_write> stats_buffer : DensityMapStats;

fn as_u32(value : f32) -> u32 {
    return u32(max(round(value), 0.0));
}

fn clamp_channel_count(raw_count : u32) -> u32 {
    if (raw_count == 0u) {
        return 1u;
    }
    if (raw_count > CHANNEL_COUNT) {
        return CHANNEL_COUNT;
    }
    return raw_count;
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

fn clamp01(value : f32) -> f32 {
    return clamp(value, 0.0, 1.0);
}

fn read_channel(value : vec4<f32>, channel : u32) -> f32 {
    switch channel {
        case 0u: { return value.x; }
        case 1u: { return value.y; }
        case 2u: { return value.z; }
        default: { return value.w; }
    }
}

fn compute_bin_index(normalized : f32, bin_count : u32) -> u32 {
    if (bin_count <= 1u) {
        return 0u;
    }
    let scaled : f32 = clamp01(normalized) * f32(bin_count - 1u);
    return u32(clamp(floor(scaled), 0.0, f32(bin_count - 1u)));
}

fn normalize_value(raw_value : f32, min_value : f32, max_value : f32) -> f32 {
    let delta : f32 = max_value - min_value;
    if (delta <= 0.0) {
        return clamp01(raw_value);
    }
    return clamp01((raw_value - min_value) / delta);
}

@compute @workgroup_size(64, 1, 1)
fn reset_histogram_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let width : u32 = as_u32(params.size.x);
    let height : u32 = as_u32(params.size.y);
    let bin_count : u32 = max(width, height);
    let histogram_length : u32 = arrayLength(&histogram_buffer.values);
    if (bin_count == 0u || histogram_length == 0u) {
        return;
    }

    let index : u32 = gid.x;
    if (index < min(bin_count, histogram_length)) {
        atomicStore(&histogram_buffer.values[index], 0u);
    }

    if (gid.x == 0u && gid.y == 0u && gid.z == 0u) {
        atomicStore(&stats_buffer.min_value, float_to_ordered(FLOAT_MAX));
        atomicStore(&stats_buffer.max_value, float_to_ordered(FLOAT_MIN));
        atomicStore(&stats_buffer.min_count, UINT32_MAX);
        atomicStore(&stats_buffer.max_count, 0u);
    }
}

@compute @workgroup_size(8, 8, 1)
fn reduce_minmax_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let width : u32 = as_u32(params.size.x);
    let height : u32 = as_u32(params.size.y);
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    let channel_count : u32 = clamp_channel_count(as_u32(params.size.z));
    if (channel_count == 0u) {
        return;
    }

    let texel : vec4<f32> = textureLoad(input_texture, vec2<i32>(i32(gid.x), i32(gid.y)), 0);
    var pixel_min : f32 = FLOAT_MAX;
    var pixel_max : f32 = FLOAT_MIN;

    for (var c : u32 = 0u; c < channel_count; c = c + 1u) {
        let value : f32 = read_channel(texel, c);
        pixel_min = min(pixel_min, value);
        pixel_max = max(pixel_max, value);
    }

    if (pixel_min > pixel_max) {
        return;
    }

    atomicMin(&stats_buffer.min_value, float_to_ordered(pixel_min));
    atomicMax(&stats_buffer.max_value, float_to_ordered(pixel_max));
}

@compute @workgroup_size(8, 8, 1)
fn histogram_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let width : u32 = as_u32(params.size.x);
    let height : u32 = as_u32(params.size.y);
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    let channel_count : u32 = clamp_channel_count(as_u32(params.size.z));
    if (channel_count == 0u) {
        return;
    }

    let bin_count : u32 = max(width, height);
    let histogram_length : u32 = arrayLength(&histogram_buffer.values);
    if (bin_count == 0u || histogram_length == 0u) {
        return;
    }

    let min_bits : u32 = atomicLoad(&stats_buffer.min_value);
    let max_bits : u32 = atomicLoad(&stats_buffer.max_value);
    let min_value : f32 = ordered_to_float(min_bits);
    let max_value : f32 = ordered_to_float(max_bits);

    let texel : vec4<f32> = textureLoad(input_texture, vec2<i32>(i32(gid.x), i32(gid.y)), 0);
    for (var c : u32 = 0u; c < channel_count; c = c + 1u) {
        let normalized : f32 = normalize_value(read_channel(texel, c), min_value, max_value);
        let bin_index : u32 = compute_bin_index(normalized, bin_count);
        if (bin_index < histogram_length) {
            atomicAdd(&histogram_buffer.values[bin_index], 1u);
        }
    }
}

@compute @workgroup_size(64, 1, 1)
fn finalize_histogram_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let width : u32 = as_u32(params.size.x);
    let height : u32 = as_u32(params.size.y);
    let bin_count : u32 = max(width, height);
    let histogram_length : u32 = arrayLength(&histogram_buffer.values);
    if (gid.x >= bin_count || gid.x >= histogram_length) {
        return;
    }

    let count : u32 = atomicLoad(&histogram_buffer.values[gid.x]);
    atomicMin(&stats_buffer.min_count, count);
    atomicMax(&stats_buffer.max_count, count);
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

    let bin_count : u32 = max(width, height);
    let histogram_length : u32 = arrayLength(&histogram_buffer.values);
    if (bin_count == 0u || histogram_length == 0u) {
        return;
    }

    let min_bits : u32 = atomicLoad(&stats_buffer.min_value);
    let max_bits : u32 = atomicLoad(&stats_buffer.max_value);
    let min_value : f32 = ordered_to_float(min_bits);
    let max_value : f32 = ordered_to_float(max_bits);

    var min_count_bits : u32 = atomicLoad(&stats_buffer.min_count);
    let max_count_bits : u32 = atomicLoad(&stats_buffer.max_count);
    if (min_count_bits == UINT32_MAX) {
        min_count_bits = 0u;
    }

    let texel : vec4<f32> = textureLoad(input_texture, vec2<i32>(i32(gid.x), i32(gid.y)), 0);
    let pixel_index : u32 = gid.y * width + gid.x;
    let base_index : u32 = pixel_index * CHANNEL_COUNT;

    let min_count : f32 = f32(min_count_bits);
    let max_count : f32 = f32(max_count_bits);
    let range_count : f32 = max_count - min_count;

    for (var c : u32 = 0u; c < channel_count; c = c + 1u) {
        let normalized : f32 = normalize_value(read_channel(texel, c), min_value, max_value);
        let bin_index : u32 = compute_bin_index(normalized, bin_count);
        var count : f32 = 0.0;
        if (bin_index < histogram_length) {
            count = f32(atomicLoad(&histogram_buffer.values[bin_index]));
        }

        var final_value : f32;
        if (range_count <= 0.0 || min_count == max_count) {
            final_value = count / max(max_count, 1.0);
        } else {
            final_value = clamp01((count - min_count) / range_count);
        }
        output_buffer[base_index + c] = final_value;
    }

    if (channel_count == 1u) {
        let value : f32 = output_buffer[base_index + 0u];
        output_buffer[base_index + 1u] = value;
        output_buffer[base_index + 2u] = value;
        output_buffer[base_index + 3u] = 1.0;
    } else if (channel_count == 2u) {
        let value : f32 = output_buffer[base_index + 0u];
        output_buffer[base_index + 1u] = value;
        output_buffer[base_index + 2u] = value;
        output_buffer[base_index + 3u] = 1.0;
    } else if (channel_count == 3u) {
        output_buffer[base_index + 3u] = 1.0;
    }

}
