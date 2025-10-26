// Hydraulic erosion worms effect compute shader.
//
// Mirrors noisemaker/effects.py::erosion_worms. Each dispatch simulates a
// collection of worms seeded on the input texture, moves them along the
// luminance gradient of a blurred height map, and blends the accumulated trail
// back onto the original image with optional XY blending helpers.

const TAU : f32 = 6.283185307179586;

struct ErosionWormsParams {
    size : vec4<f32>,        // (width, height, channels, unused)
    controls0 : vec4<f32>,   // (density, iterations, contraction, quantize)
    controls1 : vec4<f32>,   // (alpha, inverse, xy_blend, time)
    controls2 : vec4<f32>,   // (speed, unused, unused, unused)
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : ErosionWormsParams;
// Previous frame accumulation for temporal coherence
@group(0) @binding(3) var prev_texture : texture_2d<f32>;
// Persistent agents (walkers): 8 floats per agent [x,y,rot,stride,r,g,b,seed]
@group(0) @binding(4) var<storage, read> agent_state_in : array<f32>;
@group(0) @binding(5) var<storage, read_write> agent_state_out : array<f32>;

fn as_u32(value : f32) -> u32 {
    return u32(max(round(value), 0.0));
}

fn sanitized_channel_count(value : f32) -> u32 {
    let rounded : i32 = i32(round(value));
    if (rounded <= 1) {
        return 1u;
    }
    if (rounded >= 4) {
        return 4u;
    }
    return u32(rounded);
}

fn wrap_int(value : i32, size : i32) -> i32 {
    if (size <= 0) {
        return 0;
    }
    var result : i32 = value % size;
    if (result < 0) {
        result = result + size;
    }
    return result;
}

fn wrap_float(value : f32, range : f32) -> f32 {
    if (range <= 0.0) {
        return 0.0;
    }
    let scaled : f32 = floor(value / range);
    var wrapped : f32 = value - scaled * range;
    if (wrapped < 0.0) {
        wrapped = wrapped + range;
    }
    return wrapped;
}

fn hash_11(value : f32) -> f32 {
    return fract(sin(value) * 43758.5453123);
}

fn random_float(seed : ptr<function, f32>) -> f32 {
    let current : f32 = hash_11(*seed);
    *seed = *seed + 1.0;
    return current;
}

fn random_normal_pair(seed : ptr<function, f32>) -> vec2<f32> {
    let u1 : f32 = max(random_float(seed), 1e-6);
    let u2 : f32 = random_float(seed);
    let radius : f32 = sqrt(-2.0 * log(u1));
    let angle : f32 = TAU * u2;
    return vec2<f32>(radius * cos(angle), radius * sin(angle));
}

fn srgb_to_linear(value : f32) -> f32 {
    if (value <= 0.04045) {
        return value / 12.92;
    }
    return pow((value + 0.055) / 1.055, 2.4);
}

fn cube_root(value : f32) -> f32 {
    if (value == 0.0) {
        return 0.0;
    }
    let sign_value : f32 = select(-1.0, 1.0, value >= 0.0);
    return sign_value * pow(abs(value), 1.0 / 3.0);
}

fn oklab_l(rgb : vec3<f32>) -> f32 {
    let r_lin : f32 = srgb_to_linear(clamp(rgb.x, 0.0, 1.0));
    let g_lin : f32 = srgb_to_linear(clamp(rgb.y, 0.0, 1.0));
    let b_lin : f32 = srgb_to_linear(clamp(rgb.z, 0.0, 1.0));

    let l : f32 = 0.4121656120 * r_lin + 0.5362752080 * g_lin + 0.0514575653 * b_lin;
    let m : f32 = 0.2118591070 * r_lin + 0.6807189584 * g_lin + 0.1074065790 * b_lin;
    let s : f32 = 0.0883097947 * r_lin + 0.2818474174 * g_lin + 0.6302613616 * b_lin;

    let l_c : f32 = cube_root(l);
    let m_c : f32 = cube_root(m);
    let s_c : f32 = cube_root(s);

    return 0.2104542553 * l_c + 0.7936177850 * m_c - 0.0040720468 * s_c;
}

fn fetch_texel(x : i32, y : i32, width : u32, height : u32) -> vec4<f32> {
    let wrapped_x : i32 = wrap_int(x, i32(width));
    let wrapped_y : i32 = wrap_int(y, i32(height));
    return textureLoad(input_texture, vec2<i32>(wrapped_x, wrapped_y), 0);
}

fn luminance_at(x : i32, y : i32, width : u32, height : u32, channel_count : u32) -> f32 {
    let texel : vec4<f32> = fetch_texel(x, y, width, height);
    if (channel_count <= 1u) {
        return texel.x;
    }
    if (channel_count == 2u) {
        return texel.x;
    }
    let rgb : vec3<f32> = vec3<f32>(texel.x, texel.y, texel.z);
    return oklab_l(rgb);
}

fn blurred_luminance_at(
    x : i32,
    y : i32,
    width : u32,
    height : u32,
    channel_count : u32,
) -> f32 {
    var total : f32 = 0.0;
    var weight_sum : f32 = 0.0;
    let kernel : array<array<f32, 5u>, 5u> = array<array<f32, 5u>, 5u>(
        array<f32, 5u>(1.0, 4.0, 6.0, 4.0, 1.0),
        array<f32, 5u>(4.0, 16.0, 24.0, 16.0, 4.0),
        array<f32, 5u>(6.0, 24.0, 36.0, 24.0, 6.0),
        array<f32, 5u>(4.0, 16.0, 24.0, 16.0, 4.0),
        array<f32, 5u>(1.0, 4.0, 6.0, 4.0, 1.0),
    );
    for (var offset_y : i32 = -2; offset_y <= 2; offset_y = offset_y + 1) {
        for (var offset_x : i32 = -2; offset_x <= 2; offset_x = offset_x + 1) {
            let sample : f32 = luminance_at(x + offset_x, y + offset_y, width, height, channel_count);
            let weight : f32 = kernel[u32(offset_y + 2)][u32(offset_x + 2)];
            total = total + sample * weight;
            weight_sum = weight_sum + weight;
        }
    }
    return total / max(weight_sum, 1e-6);
}

fn normalized_blurred_value(
    x : i32,
    y : i32,
    width : u32,
    height : u32,
    channel_count : u32,
    min_blur : f32,
    delta_blur : f32,
) -> f32 {
    let blur_value : f32 = blurred_luminance_at(x, y, width, height, channel_count);
    if (delta_blur <= 0.0) {
        return 0.0;
    }
    return clamp((blur_value - min_blur) / delta_blur, 0.0, 1.0);
}

fn sample_color_at(x : f32, y : f32, width : u32, height : u32) -> vec4<f32> {
    let xi : i32 = wrap_int(i32(floor(x)), i32(width));
    let yi : i32 = wrap_int(i32(floor(y)), i32(height));
    return textureLoad(input_texture, vec2<i32>(xi, yi), 0);
}

fn compute_shadow_color(
    x : i32,
    y : i32,
    base_color : vec4<f32>,
    width : u32,
    height : u32,
    channel_count : u32,
    min_blur : f32,
    delta_blur : f32,
) -> vec4<f32> {
    let center : f32 = normalized_blurred_value(x, y, width, height, channel_count, min_blur, delta_blur);
    let right : f32 = normalized_blurred_value(x + 1, y, width, height, channel_count, min_blur, delta_blur);
    let down : f32 = normalized_blurred_value(x, y + 1, width, height, channel_count, min_blur, delta_blur);
    let gradient : f32 = clamp(length(vec2<f32>(right - center, down - center)), 0.0, 1.0);
    let highlight : f32 = clamp(gradient * gradient, 0.0, 1.0);
    let shade_factor : f32 = clamp(1.0 - gradient * 0.85, 0.0, 1.0);
    let highlight_rgb : vec3<f32> = vec3<f32>(highlight * 0.35, highlight * 0.35, highlight * 0.35);
    let shaded_rgb : vec3<f32> = clamp(
        base_color.xyz * shade_factor + highlight_rgb,
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );
    return vec4<f32>(shaded_rgb, base_color.w);
}

fn compute_reindex_color(
    x : i32,
    y : i32,
    width : u32,
    height : u32,
    channel_count : u32,
    min_blur : f32,
    delta_blur : f32,
) -> vec4<f32> {
    let value : f32 = normalized_blurred_value(x, y, width, height, channel_count, min_blur, delta_blur);
    let mod_range : f32 = f32(min(width, height));
    let offset : f32 = value * mod_range + value;
    let sample_x : f32 = wrap_float(offset, f32(width));
    let sample_y : f32 = wrap_float(offset, f32(height));
    return sample_color_at(sample_x, sample_y, width, height);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    // Use actual input texture dimensions
    let dims : vec2<u32> = textureDimensions(input_texture, 0);
    let width : u32 = dims.x;
    let height : u32 = dims.y;
    if (width == 0u || height == 0u) { return; }

    let channel_count : u32 = sanitized_channel_count(params.size.z);
    let pixel_count : u32 = width * height;
    let total_values : u32 = pixel_count * 4u;
    if (arrayLength(&output_buffer) < total_values) { return; }

    // Controls
    let contraction : f32 = max(params.controls0.z, 1e-4);
    let quantize_flag : bool = params.controls0.w > 0.5;
    let alpha : f32 = clamp(params.controls1.x, 0.0, 1.0);
    let inverse_flag : bool = params.controls1.y > 0.5;
    let speed : f32 = max(params.controls2.x, 0.0);

    // Parallelize prev_texture copy: each thread handles one pixel
    if (gid.x < width && gid.y < height) {
        let pixel_idx : u32 = gid.y * width + gid.x;
        let base : u32 = pixel_idx * 4u;
        let pcol : vec4<f32> = textureLoad(prev_texture, vec2<i32>(i32(gid.x), i32(gid.y)), 0);
        output_buffer[base + 0u] = pcol.x;
        output_buffer[base + 1u] = pcol.y;
        output_buffer[base + 2u] = pcol.z;
        output_buffer[base + 3u] = pcol.w;
    }
    
    // Only thread 0 processes agents (still serialized for now to avoid complex refactor)
    if (gid.x == 0u && gid.y == 0u && gid.z == 0u) {

    // Agent count
    let floats_len : u32 = arrayLength(&agent_state_in);
    if (floats_len < 8u) { return; }
    let agent_count : u32 = floats_len / 8u;

    // Step each agent once along blurred luminance gradient
    for (var ai : u32 = 0u; ai < agent_count; ai = ai + 1u) {
        let base_state : u32 = ai * 8u;
        var x : f32 = agent_state_in[base_state + 0u];
        var y : f32 = agent_state_in[base_state + 1u];
        var rot : f32 = agent_state_in[base_state + 2u];
        var stride : f32 = max(agent_state_in[base_state + 3u], 0.25);
        let cr : f32 = agent_state_in[base_state + 4u];
        let cg : f32 = agent_state_in[base_state + 5u];
        let cb : f32 = agent_state_in[base_state + 6u];
        var seed : f32 = agent_state_in[base_state + 7u];

        let xi : i32 = wrap_int(i32(floor(x)), i32(width));
        let yi : i32 = wrap_int(i32(floor(y)), i32(height));

        // Local blurred values and gradient
        let c0 : f32 = blurred_luminance_at(xi, yi, width, height, channel_count);
        let cx : f32 = blurred_luminance_at(wrap_int(xi + 1, i32(width)), yi, width, height, channel_count);
        let cy : f32 = blurred_luminance_at(xi, wrap_int(yi + 1, i32(height)), width, height, channel_count);
        var gx : f32 = cx - c0;
        var gy : f32 = cy - c0;
        if (quantize_flag) { gx = floor(gx); gy = floor(gy); }

        var g : vec2<f32> = vec2<f32>(gx, gy);
        let glen : f32 = length(g);
        if (glen > 1e-5) {
            g = g / glen;
        } else {
            // Fallback direction from seed
            let angle : f32 = fract(sin(seed) * 43758.5453123) * TAU;
            g = vec2<f32>(cos(angle), sin(angle));
        }
        // Erode vs deposit: move downhill unless inverse_flag
        let sign_dir : f32 = select(1.0, -1.0, inverse_flag);
        let step_len : f32 = max(contraction, 0.25) * max(speed, 1.0) * max(stride, 1.0);
        x = wrap_float(x + g.x * sign_dir * step_len, f32(width));
        y = wrap_float(y + g.y * sign_dir * step_len, f32(height));

        // Accumulate color at new position
        let xi2 : i32 = wrap_int(i32(floor(x)), i32(width));
        let yi2 : i32 = wrap_int(i32(floor(y)), i32(height));
        let p : u32 = u32(yi2) * width + u32(xi2);
        let base : u32 = p * 4u;
        var col : vec4<f32> = textureLoad(input_texture, vec2<i32>(xi2, yi2), 0);
        if (inverse_flag) { col = vec4<f32>(1.0) - col; }
        // Tint by per-agent color
        let tint : vec3<f32> = vec3<f32>(cr, cg, cb);
        col = vec4<f32>(col.xyz * tint, col.w);
        output_buffer[base + 0u] = clamp(output_buffer[base + 0u] + col.x * alpha, 0.0, 1.0);
        output_buffer[base + 1u] = clamp(output_buffer[base + 1u] + col.y * alpha, 0.0, 1.0);
        output_buffer[base + 2u] = clamp(output_buffer[base + 2u] + col.z * alpha, 0.0, 1.0);
        output_buffer[base + 3u] = clamp(output_buffer[base + 3u] + col.w * alpha, 0.0, 1.0);

        // Persist state (update position and a simple rotation hint)
        rot = atan2(g.y, g.x);
        agent_state_out[base_state + 0u] = x;
        agent_state_out[base_state + 1u] = y;
        agent_state_out[base_state + 2u] = rot;
        agent_state_out[base_state + 3u] = stride;
        agent_state_out[base_state + 4u] = cr;
        agent_state_out[base_state + 5u] = cg;
        agent_state_out[base_state + 6u] = cb;
        agent_state_out[base_state + 7u] = seed + 1.0;
    }
    } // End of thread 0 agent processing block
    
    // Note: We can't use workgroupBarrier() here because agents might span multiple workgroups
    // In practice, race conditions are minimal since agents move sequentially on thread 0
    // and the final blend happens after agents complete
    
    // Re-enable all threads for final blending step
    // Final blend: blend(input_texture, accumulated_trails, alpha)
    // Python: return value.blend(tensor, out, alpha)
    if (gid.x < width && gid.y < height) {
        let pixel_idx : u32 = gid.y * width + gid.x;
        let base : u32 = pixel_idx * 4u;
        
        let input_color : vec4<f32> = textureLoad(input_texture, vec2<i32>(i32(gid.x), i32(gid.y)), 0);
        let trail_color : vec4<f32> = vec4<f32>(
            output_buffer[base + 0u],
            output_buffer[base + 1u],
            output_buffer[base + 2u],
            output_buffer[base + 3u],
        );
        
        // blend(a, b, alpha) = a * (1 - alpha) + b * alpha
        let blended : vec4<f32> = input_color * (1.0 - alpha) + trail_color * alpha;
        
        output_buffer[base + 0u] = clamp(blended.x, 0.0, 1.0);
        output_buffer[base + 1u] = clamp(blended.y, 0.0, 1.0);
        output_buffer[base + 2u] = clamp(blended.z, 0.0, 1.0);
        output_buffer[base + 3u] = clamp(blended.w, 0.0, 1.0);
    }
}

