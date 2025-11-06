// Worms effect - Pass 2: Agent movement and trail deposition
// Each thread handles one agent

const TAU : f32 = 6.28318530717958647692;

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : WormsParams;
@group(0) @binding(3) var prev_texture : texture_2d<f32>;
@group(0) @binding(4) var<storage, read> agent_state_in : array<f32>;
@group(0) @binding(5) var<storage, read_write> agent_state_out : array<f32>;

struct WormsParams {
    size : vec4<f32>, // (width, height, channels, unused)
    behavior_density_stride_padding : vec4<f32>, // (behavior, density, stride, _)
    stride_deviation_alpha_kink_drunkenness : vec4<f32>,
    quantize_time_padding_intensity : vec4<f32>,
    inputIntensity_padding : vec4<f32>,
};

fn clamp_01(value : f32) -> f32 {
    return clamp(value, 0.0, 1.0);
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

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let agent_idx : u32 = gid.x;
    let floats_len : u32 = arrayLength(&agent_state_in);

    // Each agent is 8 floats: [x, y, rot, stride, r, g, b, seed]
    if (agent_idx * 8u >= floats_len) {
        return;
    }

    let dims : vec2<u32> = textureDimensions(input_texture, 0);
    let width : u32 = dims.x;
    let height : u32 = dims.y;
    if (width == 0u || height == 0u) {
        return;
    }

    let channel_count : u32 = 4u; // Always RGBA
    let kink : f32 = clamp(params.stride_deviation_alpha_kink_drunkenness.z, 0.0, 100.0);
    let quantize_flag : bool = params.quantize_time_padding_intensity.x > 0.5;

    // Load agent state
    // Agent format: [x, y, rot, stride, r, g, b, seed]
    // Note: x/y in agent state correspond to column/row (width/height)
    let base_state : u32 = agent_idx * 8u;
    var worms_x : f32 = agent_state_in[base_state + 0u];
    var worms_y : f32 = agent_state_in[base_state + 1u];
    var worms_rot : f32 = agent_state_in[base_state + 2u];
    let worms_stride : f32 = agent_state_in[base_state + 3u];
    let cr : f32 = agent_state_in[base_state + 4u];
    let cg : f32 = agent_state_in[base_state + 5u];
    let cb : f32 = agent_state_in[base_state + 6u];
    let wseed : f32 = agent_state_in[base_state + 7u];

    // Convert to pixel coordinates (wrapping)
    // Python: worm_positions = tf.cast(tf.stack([worms_y % height, worms_x % width], 1), tf.int32)
    // This means position is [row, col] = [y, x]
    let worms_y_wrapped : f32 = wrap_float(worms_y, f32(height));
    let worms_x_wrapped : f32 = wrap_float(worms_x, f32(width));
    let yi : i32 = i32(floor(worms_y_wrapped));
    let xi : i32 = i32(floor(worms_x_wrapped));

    let pixel_index : u32 = u32(yi) * width + u32(xi);
    let base : u32 = pixel_index * 4u;

    // Deposit trail at current position using agent's persistent spawn color
    let color : vec4<f32> = vec4<f32>(cr, cg, cb, 1.0);

    output_buffer[base + 0u] = output_buffer[base + 0u] + color.x;
    output_buffer[base + 1u] = output_buffer[base + 1u] + color.y;
    output_buffer[base + 2u] = output_buffer[base + 2u] + color.z;
    output_buffer[base + 3u] = output_buffer[base + 3u] + color.w;

    let texel_here : vec4<f32> = textureLoad(input_texture, vec2<i32>(xi, yi), 0);
    let index_value : f32 = value_map_component(texel_here, channel_count);

    let next_position : f32 = index_value * TAU * kink + worms_rot;

    var final_angle : f32 = next_position;
    if (quantize_flag) {
        final_angle = round(next_position);
    }

    let new_worms_y : f32 = worms_y + cos(final_angle) * worms_stride;
    let new_worms_x : f32 = worms_x + sin(final_angle) * worms_stride;

    agent_state_out[base_state + 0u] = new_worms_x;
    agent_state_out[base_state + 1u] = new_worms_y;
    agent_state_out[base_state + 2u] = worms_rot;
    agent_state_out[base_state + 3u] = worms_stride;
    agent_state_out[base_state + 4u] = cr;
    agent_state_out[base_state + 5u] = cg;
    agent_state_out[base_state + 6u] = cb;
    agent_state_out[base_state + 7u] = wseed + 1.0;
}
