// Cosine palette blending. Mirrors noisemaker.effects.palette by mapping OKLab
// lightness through cosine palette bands.

struct PaletteParams {
    width : f32,
    height : f32,
    channel_count : f32,
    alpha : f32,
    time : f32,
    speed : f32,
    palette_index : f32,
    _pad0 : f32,
};

struct PaletteEntry {
    amp : vec4<f32>,
    freq : vec4<f32>,
    offset : vec4<f32>,
    phase : vec4<f32>,
};

const PALETTE_COUNT : u32 = 38u;
const PALETTES : array<PaletteEntry, PALETTE_COUNT> = array<PaletteEntry, PALETTE_COUNT>(
    PaletteEntry(
        vec4<f32>(0.76, 0.88, 0.37, 0.0),
        vec4<f32>(1.0, 1.0, 1.0, 0.0),
        vec4<f32>(0.93, 0.97, 0.52, 0.0),
        vec4<f32>(0.21, 0.41, 0.56, 0.0)
    ),
    PaletteEntry(
        vec4<f32>(0.5, 0.5, 0.5, 0.0),
        vec4<f32>(1.0, 1.0, 1.0, 0.0),
        vec4<f32>(0.5, 0.5, 0.5, 0.0),
        vec4<f32>(0.0, 0.1, 0.2, 0.0)
    ),
    PaletteEntry(
        vec4<f32>(0.5, 0.5, 0.5, 0.0),
        vec4<f32>(1.0, 1.0, 1.0, 0.0),
        vec4<f32>(0.2, 0.64, 0.62, 0.0),
        vec4<f32>(0.15, 0.2, 0.3, 0.0)
    ),
    PaletteEntry(
        vec4<f32>(0.1, 0.9, 0.7, 0.0),
        vec4<f32>(1.0, 1.0, 1.0, 0.0),
        vec4<f32>(0.0, 0.3, 0.0, 0.0),
        vec4<f32>(0.6, 0.1, 0.6, 0.0)
    ),
    PaletteEntry(
        vec4<f32>(0.5, 0.5, 0.5, 0.0),
        vec4<f32>(0.5, 1.0, 0.5, 0.0),
        vec4<f32>(0.5, 0.5, 0.5, 0.0),
        vec4<f32>(0.5, 0.0, 1.0, 0.0)
    ),
    PaletteEntry(
        vec4<f32>(0.5, 0.5, 0.5, 0.0),
        vec4<f32>(1.0, 1.0, 1.0, 0.0),
        vec4<f32>(0.83, 0.6, 0.63, 0.0),
        vec4<f32>(0.3, 0.1, 0.0, 0.0)
    ),
    PaletteEntry(
        vec4<f32>(0.5, 0.5, 0.5, 0.0),
        vec4<f32>(1.0, 1.0, 1.0, 0.0),
        vec4<f32>(0.28, 0.39, 0.07, 0.0),
        vec4<f32>(0.25, 0.2, 0.1, 0.0)
    ),
    PaletteEntry(
        vec4<f32>(0.0, 0.5, 0.5, 0.0),
        vec4<f32>(0.0, 1.0, 1.0, 0.0),
        vec4<f32>(0.0, 0.5, 0.5, 0.0),
        vec4<f32>(0.0, 0.5, 0.5, 0.0)
    ),
    PaletteEntry(
        vec4<f32>(0.5, 0.5, 0.5, 0.0),
        vec4<f32>(1.0, 1.0, 1.0, 0.0),
        vec4<f32>(0.5, 0.5, 0.5, 0.0),
        vec4<f32>(0.3, 0.2, 0.2, 0.0)
    ),
    PaletteEntry(
        vec4<f32>(0.5, 0.5, 0.5, 0.0),
        vec4<f32>(1.0, 1.0, 1.0, 0.0),
        vec4<f32>(0.1, 0.4, 0.7, 0.0),
        vec4<f32>(0.1, 0.1, 0.1, 0.0)
    ),
    PaletteEntry(
        vec4<f32>(0.5, 0.5, 0.5, 0.0),
        vec4<f32>(1.0, 1.0, 1.0, 0.0),
        vec4<f32>(0.41, 0.22, 0.67, 0.0),
        vec4<f32>(0.2, 0.25, 0.2, 0.0)
    ),
    PaletteEntry(
        vec4<f32>(0.83, 0.45, 0.19, 0.0),
        vec4<f32>(1.0, 1.0, 1.0, 0.0),
        vec4<f32>(0.79, 0.45, 0.35, 0.0),
        vec4<f32>(0.28, 0.91, 0.61, 0.0)
    ),
    PaletteEntry(
        vec4<f32>(0.5, 0.5, 0.5, 0.0),
        vec4<f32>(2.0, 2.0, 2.0, 0.0),
        vec4<f32>(0.5, 0.5, 0.5, 0.0),
        vec4<f32>(1.0, 1.0, 1.0, 0.0)
    ),
    PaletteEntry(
        vec4<f32>(0.5, 0.5, 0.5, 0.0),
        vec4<f32>(1.0, 1.0, 1.0, 0.0),
        vec4<f32>(0.22, 0.48, 0.62, 0.0),
        vec4<f32>(0.1, 0.3, 0.2, 0.0)
    ),
    PaletteEntry(
        vec4<f32>(0.65, 0.4, 0.11, 0.0),
        vec4<f32>(1.0, 1.0, 1.0, 0.0),
        vec4<f32>(0.72, 0.45, 0.08, 0.0),
        vec4<f32>(0.71, 0.8, 0.84, 0.0)
    ),
    PaletteEntry(
        vec4<f32>(0.5, 0.5, 0.5, 0.0),
        vec4<f32>(1.0, 1.0, 1.0, 0.0),
        vec4<f32>(0.5, 0.5, 0.5, 0.0),
        vec4<f32>(0.27, 0.01, 0.48, 0.0)
    ),
    PaletteEntry(
        vec4<f32>(0.568, 0.774, 0.234, 0.0),
        vec4<f32>(1.0, 1.0, 1.0, 0.0),
        vec4<f32>(0.5, 0.5, 0.5, 0.0),
        vec4<f32>(0.727, 0.08, 0.104, 0.0)
    ),
    PaletteEntry(
        vec4<f32>(0.5, 0.5, 0.5, 0.0),
        vec4<f32>(1.0, 1.0, 1.0, 0.0),
        vec4<f32>(0.5, 0.5, 1.0, 0.0),
        vec4<f32>(0.0, 0.2, 0.2, 0.0)
    ),
    PaletteEntry(
        vec4<f32>(0.51, 0.39, 0.41, 0.0),
        vec4<f32>(1.0, 1.0, 1.0, 0.0),
        vec4<f32>(0.59, 0.53, 0.94, 0.0),
        vec4<f32>(0.15, 0.41, 0.46, 0.0)
    ),
    PaletteEntry(
        vec4<f32>(0.5, 0.5, 0.5, 0.0),
        vec4<f32>(1.0, 1.0, 1.0, 0.0),
        vec4<f32>(0.64, 0.12, 0.84, 0.0),
        vec4<f32>(0.1, 0.25, 0.15, 0.0)
    ),
    PaletteEntry(
        vec4<f32>(0.5, 0.5, 0.5, 0.0),
        vec4<f32>(1.0, 1.0, 1.0, 0.0),
        vec4<f32>(0.5, 0.5, 0.5, 0.0),
        vec4<f32>(0.0, 0.33, 0.67, 0.0)
    ),
    PaletteEntry(
        vec4<f32>(0.5, 0.5, 0.5, 0.0),
        vec4<f32>(1.0, 1.0, 1.0, 0.0),
        vec4<f32>(0.5, 0.5, 0.5, 0.0),
        vec4<f32>(0.25, 0.5, 0.75, 0.0)
    ),
    PaletteEntry(
        vec4<f32>(0.758, 0.628, 0.222, 0.0),
        vec4<f32>(1.0, 1.0, 1.0, 0.0),
        vec4<f32>(0.355, 0.129, 0.17, 0.0),
        vec4<f32>(0.0, 0.25, 0.5, 0.0)
    ),
    PaletteEntry(
        vec4<f32>(1.0, 0.25, 0.5, 0.0),
        vec4<f32>(1.0, 1.0, 1.0, 0.0),
        vec4<f32>(0.0, 0.0, 0.25, 0.0),
        vec4<f32>(0.5, 0.0, 0.0, 0.0)
    ),
    PaletteEntry(
        vec4<f32>(0.5, 0.5, 0.5, 0.0),
        vec4<f32>(1.0, 1.0, 1.0, 0.0),
        vec4<f32>(0.5, 0.5, 0.5, 0.0),
        vec4<f32>(0.3, 0.1, 0.1, 0.0)
    ),
    PaletteEntry(
        vec4<f32>(0.2, 0.2, 0.1, 0.0),
        vec4<f32>(1.0, 1.0, 1.0, 0.0),
        vec4<f32>(0.7, 0.2, 0.2, 0.0),
        vec4<f32>(0.5, 0.4, 0.0, 0.0)
    ),
    PaletteEntry(
        vec4<f32>(0.605, 0.175, 0.171, 0.0),
        vec4<f32>(1.0, 1.0, 1.0, 0.0),
        vec4<f32>(0.522, 0.386, 0.36, 0.0),
        vec4<f32>(0.0, 0.25, 0.5, 0.0)
    ),
    PaletteEntry(
        vec4<f32>(0.605, 0.175, 0.171, 0.0),
        vec4<f32>(2.0, 2.0, 2.0, 0.0),
        vec4<f32>(0.522, 0.386, 0.36, 0.0),
        vec4<f32>(0.0, 0.25, 0.5, 0.0)
    ),
    PaletteEntry(
        vec4<f32>(0.5, 0.5, 0.5, 0.0),
        vec4<f32>(1.0, 1.0, 1.0, 0.0),
        vec4<f32>(0.5, 0.5, 0.5, 0.0),
        vec4<f32>(0.4, 0.2, 0.0, 0.0)
    ),
    PaletteEntry(
        vec4<f32>(0.5, 0.5, 0.5, 0.0),
        vec4<f32>(1.0, 1.0, 1.0, 0.0),
        vec4<f32>(0.5, 0.5, 0.5, 0.0),
        vec4<f32>(0.0, 0.2, 0.25, 0.0)
    ),
    PaletteEntry(
        vec4<f32>(0.5, 0.5, 0.5, 0.0),
        vec4<f32>(1.0, 1.0, 1.0, 0.0),
        vec4<f32>(0.5, 0.5, 0.5, 0.0),
        vec4<f32>(0.0, 0.2, 0.4, 0.0)
    ),
    PaletteEntry(
        vec4<f32>(0.5, 0.5, 0.5, 0.0),
        vec4<f32>(2.0, 2.0, 2.0, 0.0),
        vec4<f32>(0.5, 0.5, 0.5, 0.0),
        vec4<f32>(0.0, 0.2, 0.4, 0.0)
    ),
    PaletteEntry(
        vec4<f32>(0.5, 0.5, 0.5, 0.0),
        vec4<f32>(1.0, 1.0, 1.0, 0.0),
        vec4<f32>(0.6, 0.4, 0.1, 0.0),
        vec4<f32>(0.3, 0.2, 0.1, 0.0)
    ),
    PaletteEntry(
        vec4<f32>(0.5, 0.5, 0.5, 0.0),
        vec4<f32>(1.0, 1.0, 1.0, 0.0),
        vec4<f32>(0.26, 0.57, 0.03, 0.0),
        vec4<f32>(0.0, 0.1, 0.3, 0.0)
    ),
    PaletteEntry(
        vec4<f32>(0.9, 0.76, 0.63, 0.0),
        vec4<f32>(1.0, 1.0, 1.0, 0.0),
        vec4<f32>(0.0, 0.19, 0.68, 0.0),
        vec4<f32>(0.43, 0.23, 0.32, 0.0)
    ),
    PaletteEntry(
        vec4<f32>(0.78, 0.63, 0.68, 0.0),
        vec4<f32>(1.0, 1.0, 1.0, 0.0),
        vec4<f32>(0.41, 0.03, 0.16, 0.0),
        vec4<f32>(0.81, 0.61, 0.06, 0.0)
    ),
    PaletteEntry(
        vec4<f32>(0.725, 0.7, 0.949, 0.0),
        vec4<f32>(1.0, 1.0, 1.0, 0.0),
        vec4<f32>(0.632, 0.378, 0.294, 0.0),
        vec4<f32>(0.0, 0.1, 0.2, 0.0)
    ),
    PaletteEntry(
        vec4<f32>(0.73, 0.36, 0.52, 0.0),
        vec4<f32>(1.0, 1.0, 1.0, 0.0),
        vec4<f32>(0.78, 0.68, 0.15, 0.0),
        vec4<f32>(0.74, 0.93, 0.28, 0.0)
    )
);

const CHANNEL_COUNT : u32 = 4u;

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : PaletteParams;

const PI : f32 = 3.141592653589793;
const TAU : f32 = 6.283185307179586;
const LIGHTNESS_SCALE : f32 = 0.875;
const LIGHTNESS_OFFSET : f32 = 0.0625;

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

fn srgb_to_lin(value : f32) -> f32 {
    if (value <= 0.04045) {
        return value / 12.92;
    }
    return pow((value + 0.055) / 1.055, 2.4);
}

fn oklab_l_component(rgb : vec3<f32>) -> f32 {
    let r_lin : f32 = srgb_to_lin(rgb.x);
    let g_lin : f32 = srgb_to_lin(rgb.y);
    let b_lin : f32 = srgb_to_lin(rgb.z);

    let l_val : f32 = 0.4121656120 * r_lin + 0.5362752080 * g_lin + 0.0514575653 * b_lin;
    let m_val : f32 = 0.2118591070 * r_lin + 0.6807189584 * g_lin + 0.1074065790 * b_lin;
    let s_val : f32 = 0.0883097947 * r_lin + 0.2818474174 * g_lin + 0.6302613616 * b_lin;

    let l_cbrt : f32 = pow(l_val, 1.0 / 3.0);
    let m_cbrt : f32 = pow(m_val, 1.0 / 3.0);
    let s_cbrt : f32 = pow(s_val, 1.0 / 3.0);

    return 0.2104542553 * l_cbrt + 0.7936177850 * m_cbrt - 0.0040720468 * s_cbrt;
}

fn cosine_blend_weight(blend : f32) -> f32 {
    return (1.0 - cos(blend * PI)) * 0.5;
}

fn write_pixel(base_index : u32, color : vec4<f32>) {
    output_buffer[base_index + 0u] = color.x;
    output_buffer[base_index + 1u] = color.y;
    output_buffer[base_index + 2u] = color.z;
    output_buffer[base_index + 3u] = color.w;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let dims : vec2<u32> = textureDimensions(input_texture, 0);
    let width : u32 = dims.x;
    let height : u32 = dims.y;
    
    if (global_id.x >= width || global_id.y >= height) {
        return;
    }

    let coords : vec2<i32> = vec2<i32>(i32(global_id.x), i32(global_id.y));
    let texel : vec4<f32> = textureLoad(input_texture, coords, 0);

    let palette_index : u32 = as_u32(params.palette_index);
    let channel_count : u32 = sanitized_channel_count(params.channel_count);
    
    let pixel_index : u32 = global_id.y * width + global_id.x;
    let base_index : u32 = pixel_index * CHANNEL_COUNT;

    // If no palette selected or not RGB, pass through
    if (palette_index == 0u || channel_count < 3u) {
        write_pixel(base_index, texel);
        return;
    }

    // Clamp palette index to valid range (1-38 maps to array index 0-37)
    let max_index : u32 = PALETTE_COUNT - 1u;
    let clamped_index : u32 = min(palette_index - 1u, max_index);
    let palette : PaletteEntry = PALETTES[clamped_index];

    let base_rgb : vec3<f32> = clamp(texel.xyz, vec3<f32>(0.0), vec3<f32>(1.0));
    let lightness : f32 = oklab_l_component(base_rgb);

    let freq_vec : vec3<f32> = palette.freq.xyz;
    let amp_vec : vec3<f32> = palette.amp.xyz;
    let offset_vec : vec3<f32> = palette.offset.xyz;
    let phase_vec : vec3<f32> = palette.phase.xyz + vec3<f32>(params.time);

    let cosine_arg : vec3<f32> = freq_vec * (lightness * LIGHTNESS_SCALE + LIGHTNESS_OFFSET) + phase_vec;
    let cosine_vals : vec3<f32> = cos(TAU * cosine_arg);
    let palette_rgb : vec3<f32> = offset_vec + amp_vec * cosine_vals;

    let weight : f32 = cosine_blend_weight(params.alpha);
    let blended : vec3<f32> = base_rgb * (1.0 - weight) + palette_rgb * weight;
    
    write_pixel(base_index, vec4<f32>(blended, texel.w));
}
