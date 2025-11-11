// Bad VHS tracking effect replicating noisemaker.effects.vhs.
// Optimized to compute noise values efficiently by minimizing redundant calculations.

const TAU : f32 = 6.28318530717958647692;
const CHANNEL_COUNT : u32 = 4u;

struct VHSParams {
    size : vec4<f32>,    // (width, height, channels, unused)
    motion : vec4<f32>,  // (time, speed, unused, unused)
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : VHSParams;

// Simple hash function for pseudo-random values
fn hash(p : vec3<f32>) -> f32 {
    var p3 : vec3<f32> = fract(p * 0.1031);
    p3 = p3 + dot(p3, vec3<f32>(p3.y, p3.z, p3.x) + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// Fast value noise using simple hash-based interpolation
fn value_noise(p : vec3<f32>) -> f32 {
    let i : vec3<f32> = floor(p);
    let f : vec3<f32> = fract(p);
    
    // Smooth interpolation
    let u : vec3<f32> = f * f * (3.0 - 2.0 * f);
    
    // 8 corners of the cube
    let c000 : f32 = hash(i + vec3<f32>(0.0, 0.0, 0.0));
    let c100 : f32 = hash(i + vec3<f32>(1.0, 0.0, 0.0));
    let c010 : f32 = hash(i + vec3<f32>(0.0, 1.0, 0.0));
    let c110 : f32 = hash(i + vec3<f32>(1.0, 1.0, 0.0));
    let c001 : f32 = hash(i + vec3<f32>(0.0, 0.0, 1.0));
    let c101 : f32 = hash(i + vec3<f32>(1.0, 0.0, 1.0));
    let c011 : f32 = hash(i + vec3<f32>(0.0, 1.0, 1.0));
    let c111 : f32 = hash(i + vec3<f32>(1.0, 1.0, 1.0));
    
    // Trilinear interpolation
    return mix(
        mix(mix(c000, c100, u.x), mix(c010, c110, u.x), u.y),
        mix(mix(c001, c101, u.x), mix(c011, c111, u.x), u.y),
        u.z
    );
}

fn periodic_value(time : f32, value : f32) -> f32 {
    return sin((time - value) * TAU) * 0.5 + 0.5;
}

// Compute value noise with time modulation
fn compute_value_noise(coord : vec2<f32>, freq : vec2<f32>, time : f32, speed : f32,
                       base_offset : vec3<f32>, time_offset : vec3<f32>) -> f32 {
    let p : vec3<f32> = vec3<f32>(
        coord.x * freq.x + base_offset.x,
        coord.y * freq.y + base_offset.y,
        cos(time * TAU) * speed + base_offset.z
    );
    
    var value : f32 = value_noise(p);
    
    if (speed != 0.0 && time != 0.0) {
        let time_p : vec3<f32> = vec3<f32>(
            coord.x * freq.x + time_offset.x,
            coord.y * freq.y + time_offset.y,
            time_offset.z
        );
        let time_value : f32 = value_noise(time_p);
        let scaled_time : f32 = periodic_value(time, time_value) * speed;
        value = periodic_value(scaled_time, value);
    }
    
    return clamp(value, 0.0, 1.0);
}

// Compute gradient noise for VHS effect (varies vertically, constant horizontally)
fn compute_grad_value(y_norm : f32, freq_y : f32, time : f32, speed : f32) -> f32 {
    // Only sample along Y axis (X is fixed at 0.0) for horizontal consistency
    let coord : vec2<f32> = vec2<f32>(0.0, y_norm);
    let freq : vec2<f32> = vec2<f32>(1.0, freq_y);
    
    let base : f32 = compute_value_noise(
        coord,
        freq,
        time,
        speed,
        vec3<f32>(17.0, 29.0, 47.0),
        vec3<f32>(71.0, 113.0, 191.0)
    );
    
    var g : f32 = max(base - 0.5, 0.0);
    g = min(g * 2.0, 1.0);
    return g;
}

// Compute scan noise for VHS effect
fn compute_scan_noise(coord : vec2<f32>, freq : vec2<f32>, time : f32, speed : f32) -> f32 {
    return compute_value_noise(
        coord,
        freq,
        time,
        speed,
        vec3<f32>(37.0, 59.0, 83.0),
        vec3<f32>(131.0, 173.0, 211.0)
    );
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let width : u32 = u32(max(round(params.size.x), 0.0));
    let height : u32 = u32(max(round(params.size.y), 0.0));
    
    if (width == 0u || height == 0u || gid.x >= width || gid.y >= height) {
        return;
    }

    let width_f : f32 = f32(width);
    let height_f : f32 = f32(height);
    let time : f32 = params.motion.x;
    let speed : f32 = params.motion.y;

    // Normalized coordinate for current pixel
    let y_norm : f32 = (f32(gid.y) + 0.5) / height_f;
    let x_norm : f32 = (f32(gid.x) + 0.5) / width_f;
    let dest_coord : vec2<f32> = vec2<f32>(x_norm, y_norm);

    // Compute gradient noise (varies vertically only, constant per row)
    let grad_freq_y : f32 = 5.0;
    let grad_dest : f32 = compute_grad_value(y_norm, grad_freq_y, time, speed);

    // Compute scan noise frequency
    let scan_base_freq : f32 = floor(height_f * 0.5) + 1.0;
    let scan_freq : vec2<f32> = select(
        vec2<f32>(scan_base_freq * (height_f / width_f), scan_base_freq),
        vec2<f32>(scan_base_freq, scan_base_freq * (width_f / height_f)),
        height_f < width_f
    );
    
    // Compute scan noise at destination
    let scan_dest : f32 = compute_scan_noise(dest_coord, scan_freq, time, speed * 100.0);

    // Calculate horizontal shift
    let shift_amount : i32 = i32(floor(scan_dest * width_f * grad_dest * grad_dest));
    let src_x_wrapped : i32 = (i32(gid.x) - shift_amount) % i32(width);
    let src_x : i32 = select(src_x_wrapped, src_x_wrapped + i32(width), src_x_wrapped < 0);
    let src_coord : vec2<i32> = vec2<i32>(src_x, i32(gid.y));

    // Sample source pixel
    let src_texel : vec4<f32> = textureLoad(input_texture, src_coord, 0);
    
    // Compute gradient at source location for blending
    let src_x_norm : f32 = (f32(src_x) + 0.5) / width_f;
    let src_coord_norm : vec2<f32> = vec2<f32>(src_x_norm, y_norm);
    let scan_source : f32 = compute_scan_noise(src_coord_norm, scan_freq, time, speed * 100.0);
    let grad_source : f32 = compute_grad_value(y_norm, grad_freq_y, time, speed);

    // Blend source with scan noise based on gradient
    let noise_color : vec4<f32> = vec4<f32>(scan_source, scan_source, scan_source, scan_source);
    let blended : vec4<f32> = mix(src_texel, noise_color, grad_source);

    // Write output
    let base_index : u32 = (gid.y * width + gid.x) * CHANNEL_COUNT;
    for (var channel : u32 = 0u; channel < CHANNEL_COUNT; channel = channel + 1u) {
        output_buffer[base_index + channel] = blended[channel];
    }
}
