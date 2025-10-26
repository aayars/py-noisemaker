// Multi-octave image "reverberation" effect mirroring the CPU implementation.
//
// The shader optionally ridge-transforms the input image, then accumulates
// downsampled-and-tiled layers across the requested octaves and iterations.
// Each downsampled value averages the contributing source pixels so the result
// matches value.proportional_downsample() followed by expand_tile(). Finally the
// combined image is normalized to [0, 1] just like value.normalize().

const CHANNEL_CAP : u32 = 4u;
const FLOAT_MAX : f32 = 3.402823e38;
const FLOAT_MIN : f32 = -3.402823e38;

struct ReverbParams {
    width : f32,
    height : f32,
    channels : f32,
    octaves : f32,
    iterations : f32,
    ridges : f32,
    time : f32,
    speed : f32,
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : ReverbParams;

fn as_u32(value : f32) -> u32 {
    return u32(max(round(value), 0.0));
}

fn bool_from_f32(value : f32) -> bool {
    return abs(value) >= 0.5;
}

fn channel_count_from_params(value : f32) -> u32 {
    let raw : u32 = as_u32(value);
    if (raw == 0u) {
        return 0u;
    }
    return min(raw, CHANNEL_CAP);
}

fn wrap_index(value : i32, limit : i32) -> i32 {
    if (limit <= 0) {
        return 0;
    }
    var wrapped : i32 = value % limit;
    if (wrapped < 0) {
        wrapped = wrapped + limit;
    }
    return wrapped;
}

fn clamp_coord(coord : i32, limit : i32) -> i32 {
    return clamp(coord, 0, max(limit - 1, 0));
}

fn ridge_transform(color : vec4<f32>) -> vec4<f32> {
    return vec4<f32>(1.0) - abs(color * 2.0 - vec4<f32>(1.0));
}

fn load_source_pixel(x : i32, y : i32, width : i32, height : i32) -> vec4<f32> {
    let safe_x : i32 = clamp_coord(x, width);
    let safe_y : i32 = clamp_coord(y, height);
    return textureLoad(input_texture, vec2<i32>(safe_x, safe_y), 0);
}

fn load_reference_pixel(x : i32, y : i32, width : i32, height : i32, use_ridges : bool) -> vec4<f32> {
    let sample : vec4<f32> = load_source_pixel(x, y, width, height);
    if (use_ridges) {
        return ridge_transform(sample);
    }
    return sample;
}

fn compute_kernel_size(dimension : i32, downsampled : i32) -> i32 {
    if (downsampled <= 0) {
        return 0;
    }
    let ratio : i32 = dimension / downsampled;
    return max(ratio, 1);
}

fn compute_block_start(tile_index : i32, kernel : i32, dimension : i32) -> i32 {
    if (kernel <= 0 || dimension <= 0) {
        return 0;
    }
    let max_start : i32 = max(dimension - kernel, 0);
    let unclamped : i32 = tile_index * kernel;
    return clamp(unclamped, 0, max_start);
}

fn downsampled_value(
    tile : vec2<i32>,
    width : i32,
    height : i32,
    down_width : i32,
    down_height : i32,
    use_ridges : bool,
) -> vec4<f32> {
    let kernel_w : i32 = compute_kernel_size(width, down_width);
    let kernel_h : i32 = compute_kernel_size(height, down_height);
    if (kernel_w <= 0 || kernel_h <= 0) {
        return vec4<f32>(0.0);
    }

    let start_x : i32 = compute_block_start(tile.x, kernel_w, width);
    let start_y : i32 = compute_block_start(tile.y, kernel_h, height);

    var sum : vec4<f32> = vec4<f32>(0.0);
    for (var ky : i32 = 0; ky < kernel_h; ky = ky + 1) {
        let sample_y : i32 = start_y + ky;
        for (var kx : i32 = 0; kx < kernel_w; kx = kx + 1) {
            let sample_x : i32 = start_x + kx;
            let sample : vec4<f32> = load_reference_pixel(sample_x, sample_y, width, height, use_ridges);
            sum = sum + sample;
        }
    }

    let sample_count : i32 = kernel_w * kernel_h;
    let inv_count : f32 = 1.0 / f32(max(sample_count, 1));
    return sum * inv_count;
}

fn write_texel(base_index : u32, texel : vec4<f32>) {
    output_buffer[base_index + 0u] = texel.x;
    output_buffer[base_index + 1u] = texel.y;
    output_buffer[base_index + 2u] = texel.z;
    output_buffer[base_index + 3u] = texel.w;
}

fn copy_source_to_output(width : u32, height : u32) {
    for (var y : u32 = 0u; y < height; y = y + 1u) {
        for (var x : u32 = 0u; x < width; x = x + 1u) {
            let pixel_index : u32 = y * width + x;
            let base_index : u32 = pixel_index * CHANNEL_CAP;
            let texel : vec4<f32> = textureLoad(input_texture, vec2<i32>(i32(x), i32(y)), 0);
            write_texel(base_index, texel);
        }
    }
}

fn clamp01(value : f32) -> f32 {
    return clamp(value, 0.0, 1.0);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let dims : vec2<u32> = textureDimensions(input_texture, 0);
    let width : u32 = select(as_u32(params.width), dims.x, dims.x > 0u);
    let height : u32 = select(as_u32(params.height), dims.y, dims.y > 0u);
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    let octaves : u32 = as_u32(params.octaves);
    let iterations : u32 = as_u32(params.iterations);
    let use_ridges : bool = bool_from_f32(params.ridges);

    let xi : i32 = i32(gid.x);
    let yi : i32 = i32(gid.y);

    let width_i : i32 = i32(width);
    let height_i : i32 = i32(height);

    let source_texel : vec4<f32> = load_source_pixel(xi, yi, width_i, height_i);
    var accum : vec4<f32> = load_reference_pixel(xi, yi, width_i, height_i, use_ridges);

    if (iterations > 0u && octaves > 0u) {
        for (var iter : u32 = 0u; iter < iterations; iter = iter + 1u) {
            for (var octave : u32 = 1u; octave <= octaves; octave = octave + 1u) {
                let clamped_octave : u32 = min(octave, 30u);
                let multiplier_u : u32 = 1u << clamped_octave;
                if (multiplier_u == 0u) {
                    continue;
                }
                let multiplier : i32 = max(i32(multiplier_u), 1);

                let down_width : i32 = max(width_i / multiplier, 1);
                let down_height : i32 = max(height_i / multiplier, 1);
                if (down_width <= 0 || down_height <= 0) {
                    break;
                }

                let offset_x : i32 = down_width / 2;
                let offset_y : i32 = down_height / 2;
                let tile_x : i32 = wrap_index(xi + offset_x, down_width);
                let tile_y : i32 = wrap_index(yi + offset_y, down_height);

                let averaged : vec4<f32> = downsampled_value(
                    vec2<i32>(tile_x, tile_y),
                    i32(width_i),
                    i32(height_i),
                    down_width,
                    down_height,
                    use_ridges
                );

                accum = accum + averaged / f32(multiplier);
            }
        }
    }

    let pixel_index : u32 = gid.y * width + gid.x;
    let base_index : u32 = pixel_index * CHANNEL_CAP;

    // Write clamped accum, fallback to source for any missing channels; alpha to 1.0
    output_buffer[base_index + 0u] = clamp01(accum.x);
    output_buffer[base_index + 1u] = clamp01(accum.y);
    output_buffer[base_index + 2u] = clamp01(accum.z);
    output_buffer[base_index + 3u] = 1.0;
}
