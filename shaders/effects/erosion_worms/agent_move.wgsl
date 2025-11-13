// Erosion Worms - Pass 1: Agent Movement
// Each invocation advances one agent along the luminance gradient and deposits trail energy.

const TAU : f32 = 6.283185307179586;

struct ErosionWormsParams {
    size : vec4<f32>,
    controls0 : vec4<f32>,
    controls1 : vec4<f32>,
    controls2 : vec4<f32>,
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : ErosionWormsParams;
@group(0) @binding(3) var prev_texture : texture_2d<f32>;
@group(0) @binding(4) var<storage, read> agent_state_in : array<f32>;
@group(0) @binding(5) var<storage, read_write> agent_state_out : array<f32>;

fn sanitized_channel_count(value : f32) -> u32 {
    let rounded : i32 = i32(round(value));
    if (rounded <= 1) { return 1u; }
    if (rounded >= 4) { return 4u; }
    return u32(rounded);
}

fn wrap_int(value : i32, size : i32) -> i32 {
    if (size <= 0) { return 0; }
    var result : i32 = value % size;
    if (result < 0) { result = result + size; }
    return result;
}

fn wrap_float(value : f32, range : f32) -> f32 {
    if (range <= 0.0) { return 0.0; }
    let scaled : f32 = floor(value / range);
    var wrapped : f32 = value - scaled * range;
    if (wrapped < 0.0) { wrapped = wrapped + range; }
    return wrapped;
}

fn srgb_to_linear(value : f32) -> f32 {
    if (value <= 0.04045) { return value / 12.92; }
    return pow((value + 0.055) / 1.055, 2.4);
}

fn cube_root(value : f32) -> f32 {
    if (value == 0.0) { return 0.0; }
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

fn hash(seed : u32) -> f32 {
    var x : u32 = seed;
    x = x ^ (x >> 16u);
    x = x * 0x7feb352du;
    x = x ^ (x >> 15u);
    x = x * 0x846ca68bu;
    x = x ^ (x >> 16u);
    return f32(x) / f32(0xffffffffu);
}

fn hash2(seed : u32) -> vec2<f32> {
    return vec2<f32>(hash(seed), hash(seed + 1u));
}

fn fetch_texel(x : i32, y : i32, width : u32, height : u32) -> vec4<f32> {
    let wrapped_x : i32 = wrap_int(x, i32(width));
    let wrapped_y : i32 = wrap_int(y, i32(height));
    return textureLoad(input_texture, vec2<i32>(wrapped_x, wrapped_y), 0);
}

fn luminance_at(x : i32, y : i32, width : u32, height : u32, channel_count : u32) -> f32 {
    let texel : vec4<f32> = fetch_texel(x, y, width, height);
    if (channel_count <= 1u) { return texel.x; }
    if (channel_count == 2u) { return texel.x; }
    let rgb : vec3<f32> = vec3<f32>(texel.x, texel.y, texel.z);
    return oklab_l(rgb);
}

fn blurred_luminance_at(x : i32, y : i32, width : u32, height : u32, channel_count : u32) -> f32 {
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

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let dims : vec2<u32> = textureDimensions(input_texture, 0);
    let width : u32 = dims.x;
    let height : u32 = dims.y;
    if (width == 0u || height == 0u) { return; }

    let channel_count : u32 = sanitized_channel_count(params.size.z);
    let floats_len : u32 = arrayLength(&agent_state_in);
    if (floats_len < 9u) { return; }
    let agent_count : u32 = floats_len / 9u;

    let agent_id : u32 = gid.x;
    if (agent_id >= agent_count) { return; }

    let stride : f32 = max(params.controls0.y, 0.1);
    let quantize_flag : bool = params.controls0.z > 0.5;
    let intensity : f32 = clamp(params.controls1.x, 0.0, 1.0);
    let inverse_flag : bool = params.controls1.y > 0.5;
    let worm_lifetime : f32 = max(params.controls2.x, 0.0);
    let time_value : f32 = params.controls1.w;

    let base_state : u32 = agent_id * 9u;
    var x : f32 = agent_state_in[base_state + 0u];
    var y : f32 = agent_state_in[base_state + 1u];
    var x_dir : f32 = agent_state_in[base_state + 2u];
    var y_dir : f32 = agent_state_in[base_state + 3u];
    var cr : f32 = agent_state_in[base_state + 4u];
    var cg : f32 = agent_state_in[base_state + 5u];
    var cb : f32 = agent_state_in[base_state + 6u];
    let inertia : f32 = clamp(agent_state_in[base_state + 7u], 0.0, 1.0);
    var age : f32 = agent_state_in[base_state + 8u];

    let normalized_lifetime : f32 = worm_lifetime / 60.0;
    let normalized_index : f32 = select(0.0, f32(agent_id) / max(f32(agent_count), 1.0), agent_count > 0u);
    let agent_phase : f32 = fract(normalized_index);
    let needs_initial_color : bool = age < 0.0;

    let time_in_cycle : f32 = fract(time_value + agent_phase);
    let prev_time_in_cycle : f32 = fract(time_value - (1.0 / 60.0) + agent_phase);
    let respawn_check : bool = worm_lifetime > 0.0
        && normalized_lifetime > 0.0
        && time_in_cycle < normalized_lifetime
        && prev_time_in_cycle >= normalized_lifetime;

    if (needs_initial_color || respawn_check) {
        let seed : u32 = agent_id + u32(time_value * 1000.0);
        if (respawn_check) {
            let pos : vec2<f32> = hash2(seed);
            x = pos.x * f32(width);
            y = pos.y * f32(height);

            let dir_seed : u32 = seed + 12345u;
            let dir_raw : vec2<f32> = hash2(dir_seed) * 2.0 - 1.0;
            let dir_len : f32 = length(dir_raw);
            if (dir_len > 1e-5) {
                x_dir = dir_raw.x / dir_len;
                y_dir = dir_raw.y / dir_len;
            } else {
                x_dir = 1.0;
                y_dir = 0.0;
            }
        }

        let xi : i32 = wrap_int(i32(floor(x)), i32(width));
        let yi : i32 = wrap_int(i32(floor(y)), i32(height));
        let sample_color : vec4<f32> = textureLoad(input_texture, vec2<i32>(xi, yi), 0);
        cr = sample_color.x;
        cg = sample_color.y;
        cb = sample_color.z;
        age = 0.0;
    }

    let xi : i32 = wrap_int(i32(floor(x)), i32(width));
    let yi : i32 = wrap_int(i32(floor(y)), i32(height));
    let x1i : i32 = wrap_int(xi + 1, i32(width));
    let y1i : i32 = wrap_int(yi + 1, i32(height));

    let u : f32 = x - floor(x);
    let v : f32 = y - floor(y);

    let c00 : f32 = blurred_luminance_at(xi, yi, width, height, channel_count);
    let c10 : f32 = blurred_luminance_at(x1i, yi, width, height, channel_count);
    let c01 : f32 = blurred_luminance_at(xi, y1i, width, height, channel_count);
    let c11 : f32 = blurred_luminance_at(x1i, y1i, width, height, channel_count);

    var gx : f32 = mix(c01 - c00, c11 - c10, u);
    var gy : f32 = mix(c10 - c00, c11 - c01, v);

    if (quantize_flag) {
        gx = floor(gx);
        gy = floor(gy);
    }

    let glen : f32 = length(vec2<f32>(gx, gy));
    if (glen > 1e-6) {
        let scale : f32 = stride / glen;
        gx = gx * scale;
        gy = gy * scale;
    } else {
        gx = 0.0;
        gy = 0.0;
    }

    x_dir = mix(x_dir, gx, inertia);
    y_dir = mix(y_dir, gy, inertia);

    x = wrap_float(x + x_dir, f32(width));
    y = wrap_float(y + y_dir, f32(height));

    let xi2 : i32 = wrap_int(i32(floor(x)), i32(width));
    let yi2 : i32 = wrap_int(i32(floor(y)), i32(height));
    let pixel_index : u32 = u32(yi2) * width + u32(xi2);
    let base : u32 = pixel_index * 4u;

    let tint : vec3<f32> = vec3<f32>(cr, cg, cb);
    var deposit_rgb : vec3<f32> = tint;
    if (inverse_flag) {
        deposit_rgb = vec3<f32>(1.0) - deposit_rgb;
    }

    output_buffer[base + 0u] = clamp(output_buffer[base + 0u] + deposit_rgb.x * intensity, 0.0, 1.0);
    output_buffer[base + 1u] = clamp(output_buffer[base + 1u] + deposit_rgb.y * intensity, 0.0, 1.0);
    output_buffer[base + 2u] = clamp(output_buffer[base + 2u] + deposit_rgb.z * intensity, 0.0, 1.0);
    output_buffer[base + 3u] = clamp(output_buffer[base + 3u] + intensity, 0.0, 1.0);

    age = age + 1.0;

    agent_state_out[base_state + 0u] = x;
    agent_state_out[base_state + 1u] = y;
    agent_state_out[base_state + 2u] = x_dir;
    agent_state_out[base_state + 3u] = y_dir;
    agent_state_out[base_state + 4u] = cr;
    agent_state_out[base_state + 5u] = cg;
    agent_state_out[base_state + 6u] = cb;
    agent_state_out[base_state + 7u] = inertia;
    agent_state_out[base_state + 8u] = age;
}
