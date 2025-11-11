// Texture effect: generate animated ridged noise, derive a shadow from the
// noise gradient, then blend that shade back into the source pixels. This
// implementation keeps the required 3-stage algorithm (noise → shadow →
// blend) while avoiding the heavyweight single-invocation work that previously
// froze the GPU. Everything runs in a tiled 8x8 compute pass with compact math.

const INV_UINT32_MAX : f32 = 1.0 / 4294967295.0;
const OCTAVE_COUNT : u32 = 3u;
const SHADE_GAIN : f32 = 4.4;

struct TextureParams {
    width : f32,
    height : f32,
    channel_count : f32,
    _pad0 : f32,
    time : f32,
    speed : f32,
    _pad1 : f32,
    _pad2 : f32,
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : TextureParams;

fn as_u32(value : f32) -> u32 {
    return u32(max(round(value), 0.0));
}

fn clamp01(value : f32) -> f32 {
    return clamp(value, 0.0, 1.0);
}

fn fade(t : f32) -> f32 {
    return t * t * (3.0 - 2.0 * t);
}

fn freq_for_shape(base_freq : f32, dims : vec2<f32>) -> vec2<f32> {
    let width : f32 = max(dims.x, 1.0);
    let height : f32 = max(dims.y, 1.0);
    if (abs(width - height) < 0.5) {
        return vec2<f32>(base_freq, base_freq);
    }
    if (width > height) {
        return vec2<f32>(base_freq, base_freq * width / height);
    }
    return vec2<f32>(base_freq * height / width, base_freq);
}

fn fast_hash(p : vec3<i32>, salt : u32) -> f32 {
    var h : u32 = salt ^ 0x9e3779b9u;
    h ^= bitcast<u32>(p.x) * 0x27d4eb2du;
    h = (h ^ (h >> 15u)) * 0x85ebca6bu;
    h ^= bitcast<u32>(p.y) * 0xc2b2ae35u;
    h = (h ^ (h >> 13u)) * 0x27d4eb2du;
    h ^= bitcast<u32>(p.z) * 0x165667b1u;
    h = h ^ (h >> 16u);
    return f32(h) * INV_UINT32_MAX;
}

fn value_noise(uv : vec2<f32>, freq : vec2<f32>, motion : f32, salt : u32) -> f32 {
    let scaled_uv : vec2<f32> = uv * max(freq, vec2<f32>(1.0, 1.0));
    let cell_floor : vec2<f32> = floor(scaled_uv);
    let frac : vec2<f32> = fract(scaled_uv);
    let base_cell : vec2<i32> = vec2<i32>(i32(cell_floor.x), i32(cell_floor.y));

    let z_floor : f32 = floor(motion);
    let z_frac : f32 = fract(motion);
    let z0 : i32 = i32(z_floor);
    let z1 : i32 = z0 + 1;

    let c000 : f32 = fast_hash(vec3<i32>(base_cell.x + 0, base_cell.y + 0, z0), salt);
    let c100 : f32 = fast_hash(vec3<i32>(base_cell.x + 1, base_cell.y + 0, z0), salt);
    let c010 : f32 = fast_hash(vec3<i32>(base_cell.x + 0, base_cell.y + 1, z0), salt);
    let c110 : f32 = fast_hash(vec3<i32>(base_cell.x + 1, base_cell.y + 1, z0), salt);
    let c001 : f32 = fast_hash(vec3<i32>(base_cell.x + 0, base_cell.y + 0, z1), salt);
    let c101 : f32 = fast_hash(vec3<i32>(base_cell.x + 1, base_cell.y + 0, z1), salt);
    let c011 : f32 = fast_hash(vec3<i32>(base_cell.x + 0, base_cell.y + 1, z1), salt);
    let c111 : f32 = fast_hash(vec3<i32>(base_cell.x + 1, base_cell.y + 1, z1), salt);

    let tx : f32 = fade(frac.x);
    let ty : f32 = fade(frac.y);
    let tz : f32 = fade(z_frac);

    let x00 : f32 = mix(c000, c100, tx);
    let x10 : f32 = mix(c010, c110, tx);
    let x01 : f32 = mix(c001, c101, tx);
    let x11 : f32 = mix(c011, c111, tx);

    let y0 : f32 = mix(x00, x10, ty);
    let y1 : f32 = mix(x01, x11, ty);

    return mix(y0, y1, tz);
}

fn multi_octave_noise(uv : vec2<f32>, base_freq : vec2<f32>, motion : f32) -> f32 {
    var freq : vec2<f32> = max(base_freq, vec2<f32>(1.0, 1.0));
    var amplitude : f32 = 0.5;
    var accum : f32 = 0.0;
    var total : f32 = 0.0;

    for (var octave : u32 = 0u; octave < OCTAVE_COUNT; octave = octave + 1u) {
        let salt : u32 = 0x9e3779b9u * (octave + 1u);
        let sample : f32 = value_noise(uv, freq, motion + f32(octave) * 0.37, salt);
        let ridged : f32 = 1.0 - abs(sample * 2.0 - 1.0);
        accum = accum + ridged * amplitude;
        total = total + amplitude;
        freq = freq * 2.0;
        amplitude = amplitude * 0.55;
    }

    if (total <= 0.0) {
        return clamp01(accum);
    }
    return clamp01(accum / total);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let width : u32 = as_u32(params.width);
    let height : u32 = as_u32(params.height);
    if (width == 0u || height == 0u) {
        return;
    }
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    let coords : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
    let base_color : vec4<f32> = textureLoad(input_texture, coords, 0);
    let pixel_index : u32 = gid.y * width + gid.x;
    let base_index : u32 = pixel_index * 4u;

    let dims : vec2<f32> = vec2<f32>(max(params.width, 1.0), max(params.height, 1.0));
    let uv : vec2<f32> = (vec2<f32>(f32(coords.x), f32(coords.y)) + 0.5) / dims;
    let pixel_step : vec2<f32> = vec2<f32>(1.0 / dims.x, 1.0 / dims.y);

    let base_freq : vec2<f32> = freq_for_shape(24.0, dims);
    let motion : f32 = params.time * params.speed;

    let noise_center : f32 = multi_octave_noise(uv, base_freq, motion);
    let noise_right : f32 = multi_octave_noise(uv + vec2<f32>(pixel_step.x, 0.0), base_freq, motion);
    let noise_left : f32 = multi_octave_noise(uv - vec2<f32>(pixel_step.x, 0.0), base_freq, motion);
    let noise_up : f32 = multi_octave_noise(uv + vec2<f32>(0.0, pixel_step.y), base_freq, motion);
    let noise_down : f32 = multi_octave_noise(uv - vec2<f32>(0.0, pixel_step.y), base_freq, motion);

    let gx : f32 = noise_right - noise_left;
    let gy : f32 = noise_down - noise_up;
    let gradient : f32 = sqrt(gx * gx + gy * gy);
    let shade_base : f32 = clamp01(gradient * SHADE_GAIN * 0.25);

    let highlight_mix : f32 = clamp01((shade_base * shade_base) * 1.25);
    let base_factor : f32 = 0.9 + noise_center * 0.35;
    let factor : f32 = clamp(base_factor + highlight_mix * 0.35, 0.85, 1.6);

    let scaled_rgb : vec3<f32> = clamp(base_color.xyz * factor, vec3<f32>(0.0), vec3<f32>(1.0));

    output_buffer[base_index + 0u] = scaled_rgb.x;
    output_buffer[base_index + 1u] = scaled_rgb.y;
    output_buffer[base_index + 2u] = scaled_rgb.z;
    output_buffer[base_index + 3u] = base_color.w;
}
