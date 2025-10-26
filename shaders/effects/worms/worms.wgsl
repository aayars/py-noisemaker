// Worms effect compute shader.
//
// Faithfully reproduces the TensorFlow implementation found in
// noisemaker/effects.py::worms. Each dispatch simulates a collection of worms
// that follow a flow field derived from the input texture and blends the
// results back onto the original image.

const PI : f32 = 3.14159265358979323846;
const TAU : f32 = 6.28318530717958647692;
const MAX_FLOAT : f32 = 3.402823466e38;
const MIN_FLOAT : f32 = -MAX_FLOAT;

// Limit worms to control additional arrays' size; avoid large per-image arrays to keep stack usage low.
const MAX_WORMS : u32 = 128u;

const BEHAVIOR_OBEDIENT : u32 = 1u;
const BEHAVIOR_CROSSHATCH : u32 = 2u;
const BEHAVIOR_UNRULY : u32 = 3u;
const BEHAVIOR_CHAOTIC : u32 = 4u;
const BEHAVIOR_RANDOM : u32 = 5u;
const BEHAVIOR_MEANDERING : u32 = 10u;

struct WormsParams {
    size : vec4<f32>, // (width, height, channels, unused)
    behavior_density_duration_stride : vec4<f32>, // (behavior, density, duration, stride)
    stride_deviation_alpha_kink_drunkenness : vec4<f32>, // (strideDeviation, alpha, kink, drunkenness)
    quantize_time_speed_padding : vec4<f32>, // (quantizeFlag, time, speed, _pad0)
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : WormsParams;
// Optional previous framebuffer for accumulation
@group(0) @binding(3) var prev_texture : texture_2d<f32>;
// Persistent per-agent state, provided by the viewer when declared
@group(0) @binding(4) var<storage, read> agent_state_in : array<f32>;
@group(0) @binding(5) var<storage, read_write> agent_state_out : array<f32>;

fn clamp_01(value : f32) -> f32 {
    return clamp(value, 0.0, 1.0);
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
    var wrapped : i32 = value % size;
    if (wrapped < 0) {
        wrapped = wrapped + size;
    }
    return wrapped;
}

fn wrap_float(value : f32, size : f32) -> f32 {
    if (size <= 0.0) {
        return 0.0;
    }
    let scaled : f32 = floor(value / size);
    var wrapped : f32 = value - scaled * size;
    if (wrapped < 0.0) {
        wrapped = wrapped + size;
    }
    return wrapped;
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
    let r_lin : f32 = srgb_to_linear(clamp_01(rgb.x));
    let g_lin : f32 = srgb_to_linear(clamp_01(rgb.y));
    let b_lin : f32 = srgb_to_linear(clamp_01(rgb.z));

    let l : f32 = 0.4121656120 * r_lin + 0.5362752080 * g_lin + 0.0514575653 * b_lin;
    let m : f32 = 0.2118591070 * r_lin + 0.6807189584 * g_lin + 0.1074065790 * b_lin;
    let s : f32 = 0.0883097947 * r_lin + 0.2818474174 * g_lin + 0.6302613616 * b_lin;

    let l_c : f32 = cube_root(l);
    let m_c : f32 = cube_root(m);
    let s_c : f32 = cube_root(s);

    return 0.2104542553 * l_c + 0.7936177850 * m_c - 0.0040720468 * s_c;
}

fn value_map_component(texel : vec4<f32>, channel_count : u32) -> f32 {
    if (channel_count <= 1u) {
        return texel.x;
    }
    if (channel_count == 2u) {
        return texel.x;
    }
    let rgb : vec3<f32> = vec3<f32>(texel.x, texel.y, texel.z);
    return oklab_l(rgb);
}

fn normalized_sine(value : f32) -> f32 {
    return (sin(value) + 1.0) * 0.5;
}

fn periodic_value(time_value : f32, raw_value : f32) -> f32 {
    return normalized_sine((time_value - raw_value) * TAU);
}

fn rng_next(state : ptr<function, u32>) -> f32 {
    var current : u32 = *state;
    current = (current + 0x6D2B79F5u) & 0xFFFFFFFFu;
    current = ((current ^ (current >> 15u)) * (current | 1u)) & 0xFFFFFFFFu;
    current = (current ^ (current + ((current ^ (current >> 7u)) * (current | 61u)))) & 0xFFFFFFFFu;
    *state = current;
    let result : u32 = (current ^ (current >> 14u)) & 0xFFFFFFFFu;
    return f32(result) / 4294967296.0;
}

fn rng_uniform(state : ptr<function, u32>, min_value : f32, max_value : f32) -> f32 {
    let span : f32 = max_value - min_value;
    return min_value + span * rng_next(state);
}

fn rng_normal(state : ptr<function, u32>, mean : f32, stddev : f32) -> f32 {
    var u1 : f32 = rng_next(state);
    loop {
        if (u1 > 0.0) {
            break;
        }
        u1 = rng_next(state);
    }
    let u2 : f32 = rng_next(state);
    let magnitude : f32 = sqrt(-2.0 * log(u1));
    let angle : f32 = TAU * u2;
    let z0 : f32 = magnitude * cos(angle);
    return mean + stddev * z0;
}

fn fetch_texel(x : i32, y : i32, width : i32, height : i32) -> vec4<f32> {
    let wrapped_x : i32 = wrap_int(x, width);
    let wrapped_y : i32 = wrap_int(y, height);
    return textureLoad(input_texture, vec2<i32>(wrapped_x, wrapped_y), 0);
}

fn behavior_to_enum(raw : f32) -> u32 {
    let rounded : i32 = i32(round(raw));
    if (rounded == 2) {
        return BEHAVIOR_CROSSHATCH;
    }
    if (rounded == 3) {
        return BEHAVIOR_UNRULY;
    }
    if (rounded == 4) {
        return BEHAVIOR_CHAOTIC;
    }
    if (rounded == 5) {
        return BEHAVIOR_RANDOM;
    }
    if (rounded == 10) {
        return BEHAVIOR_MEANDERING;
    }
    return BEHAVIOR_OBEDIENT;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    // Single-dispatch, serial pass over the frame. We use prev_texture for accumulation,
    // and agent_state_in/out to persist walker state across frames.
    if (gid.x != 0u || gid.y != 0u || gid.z != 0u) { return; }

    let dims : vec2<u32> = textureDimensions(input_texture, 0);
    let width : u32 = dims.x;
    let height : u32 = dims.y;
    if (width == 0u || height == 0u) { return; }

    let channel_count : u32 = sanitized_channel_count(params.size.z);
    let pixel_count : u32 = width * height;
    if (pixel_count == 0u) { return; }

    let behavior_raw : f32 = params.behavior_density_duration_stride.x;
    let density : f32 = params.behavior_density_duration_stride.y;
    let stride_mean : f32 = params.behavior_density_duration_stride.w;
    let stride_deviation : f32 = params.stride_deviation_alpha_kink_drunkenness.x;
    let alpha : f32 = clamp_01(params.stride_deviation_alpha_kink_drunkenness.y);
    let kink : f32 = params.stride_deviation_alpha_kink_drunkenness.z;
    let quantize_flag : bool = params.quantize_time_speed_padding.x > 0.5;
    let speed : f32 = params.quantize_time_speed_padding.z;
    let behavior : u32 = behavior_to_enum(behavior_raw);

    let max_dim : f32 = max(f32(width), f32(height));
    var worm_count : u32 = 0u;
    if (density > 0.0 && max_dim > 0.0) {
        let desired : f32 = floor(max_dim * density);
        if (desired > 0.0) { worm_count = min(u32(desired), MAX_WORMS); }
    }

    let total_channels : u32 = 4u * pixel_count;
    // Seed output from prev frame texture for temporal accumulation
    var idx : u32 = 0u;
    for (var y0 : u32 = 0u; y0 < height; y0 = y0 + 1u) {
        for (var x0 : u32 = 0u; x0 < width; x0 = x0 + 1u) {
            let base_idx : u32 = idx * 4u;
            let prev_col : vec4<f32> = textureLoad(prev_texture, vec2<i32>(i32(x0), i32(y0)), 0);
            if (base_idx + 3u < total_channels) {
                output_buffer[base_idx + 0u] = prev_col.x;
                output_buffer[base_idx + 1u] = prev_col.y;
                output_buffer[base_idx + 2u] = prev_col.z;
                output_buffer[base_idx + 3u] = prev_col.w;
            }
            idx = idx + 1u;
        }
    }

    let width_i : i32 = i32(width);
    let height_i : i32 = i32(height);

    // Scan min/max for normalization
    var min_value : f32 = MAX_FLOAT;
    var max_value : f32 = MIN_FLOAT;
    for (var y : u32 = 0u; y < height; y = y + 1u) {
        for (var x : u32 = 0u; x < width; x = x + 1u) {
            let texel : vec4<f32> = textureLoad(input_texture, vec2<i32>(i32(x), i32(y)), 0);
            let v : f32 = value_map_component(texel, channel_count);
            min_value = min(min_value, v);
            max_value = max(max_value, v);
        }
    }
    if (!(max_value > min_value)) { min_value = 0.0; max_value = 1.0; }
    let inv_delta : f32 = 1.0 / max(max_value - min_value, 1e-6);

    if (worm_count == 0u) {
        // Nothing to draw; softly mix input to output
        var p : u32 = 0u;
        for (var yy : u32 = 0u; yy < height; yy = yy + 1u) {
            for (var xx : u32 = 0u; xx < width; xx = xx + 1u) {
                let texel : vec4<f32> = textureLoad(input_texture, vec2<i32>(i32(xx), i32(yy)), 0);
                let base : u32 = p * 4u;
                let mix_color : vec4<f32> = mix(texel, vec4<f32>(0.0), vec4<f32>(alpha));
                if (base + 3u < total_channels) {
                    output_buffer[base + 0u] = clamp_01(mix_color.x);
                    output_buffer[base + 1u] = clamp_01(mix_color.y);
                    output_buffer[base + 2u] = clamp_01(mix_color.z);
                    output_buffer[base + 3u] = clamp_01(mix_color.w);
                }
                p = p + 1u;
            }
        }
        return;
    }

    // Agent-based single step
    let exposure : f32 = 1.0;
    for (var i : u32 = 0u; i < worm_count; i = i + 1u) {
        let base_state : u32 = i * 8u;
        var wx : f32 = agent_state_in[base_state + 0u];
        var wy : f32 = agent_state_in[base_state + 1u];
        var wrot : f32 = agent_state_in[base_state + 2u];
        var wstride : f32 = agent_state_in[base_state + 3u];
        let cr : f32 = agent_state_in[base_state + 4u];
        let cg : f32 = agent_state_in[base_state + 5u];
        let cb : f32 = agent_state_in[base_state + 6u];

        let xi : i32 = wrap_int(i32(floor(wx)), width_i);
        let yi : i32 = wrap_int(i32(floor(wy)), height_i);
        let pixel_index : u32 = u32(yi) * width + u32(xi);
        let base : u32 = pixel_index * 4u;
        if (base + 3u < total_channels) {
            let sample_color : vec4<f32> = textureLoad(input_texture, vec2<i32>(xi, yi), 0) * vec4<f32>(exposure);
            let color : vec4<f32> = vec4<f32>(sample_color.x * cr, sample_color.y * cg, sample_color.z * cb, sample_color.w);
            output_buffer[base + 0u] = output_buffer[base + 0u] + color.x;
            output_buffer[base + 1u] = output_buffer[base + 1u] + color.y;
            output_buffer[base + 2u] = output_buffer[base + 2u] + color.z;
            output_buffer[base + 3u] = output_buffer[base + 3u] + color.w;
        }

        // Flow field angle from normalized channel(s)
        let texel_here : vec4<f32> = textureLoad(input_texture, vec2<i32>(xi, yi), 0);
        let v_here : f32 = value_map_component(texel_here, channel_count);
        let norm_here : f32 = clamp_01((v_here - min_value) * inv_delta);
        var angle : f32 = norm_here * TAU * kink + wrot;
        if (quantize_flag) { angle = round(angle); }
        wy = wrap_float(wy + cos(angle) * wstride, f32(height));
        wx = wrap_float(wx + sin(angle) * wstride, f32(width));

        // Persist
        agent_state_out[base_state + 0u] = wx;
        agent_state_out[base_state + 1u] = wy;
        agent_state_out[base_state + 2u] = angle;
        agent_state_out[base_state + 3u] = wstride;
        agent_state_out[base_state + 4u] = cr;
        agent_state_out[base_state + 5u] = cg;
        agent_state_out[base_state + 6u] = cb;
        agent_state_out[base_state + 7u] = 0.0;
    }
}
