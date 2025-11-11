// CRT effect - faithful port of Python implementation
// Applies scanlines, lens warp, chromatic aberration, hue shift, saturation boost, and vignette

const PI : f32 = 3.14159265358979323846;
const TAU : f32 = 6.28318530717958647692;
const INV_THREE : f32 = 0.3333333333333333;

struct CRTParams {
    size : vec4<f32>,    // (width, height, channels, unused)
    motion : vec4<f32>,  // (time, speed, unused, unused)
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : CRTParams;

fn as_u32(value : f32) -> u32 {
    return u32(max(value, 0.0));
}

fn clamp01(value : f32) -> f32 {
    return clamp(value, 0.0, 1.0);
}

fn random_scalar(seed : f32) -> f32 {
    return fract(sin(seed) * 43758.5453123);
}

fn simplex_random(time : f32, speed : f32) -> f32 {
    let angle : f32 = time * TAU;
    let z : f32 = cos(angle) * speed;
    let w : f32 = sin(angle) * speed;
    return fract(sin(z * 157.0 + w * 113.0) * 43758.5453);
}

fn freq_for_shape(base_freq : f32, width : f32, height : f32) -> vec2<f32> {
    let freq : f32 = max(base_freq, 1.0);
    let width_safe : f32 = max(width, 1.0);
    let height_safe : f32 = max(height, 1.0);

    if (abs(width_safe - height_safe) < 1e-5) {
        return vec2<f32>(freq, freq);
    }

    if (height_safe < width_safe) {
        let scaled : f32 = floor(freq * width_safe / height_safe);
        return vec2<f32>(freq, max(scaled, 1.0));
    }

    let scaled : f32 = floor(freq * height_safe / width_safe);
    return vec2<f32>(max(scaled, 1.0), freq);
}

fn normalized_sine(value : f32) -> f32 {
    return sin(value) * 0.5 + 0.5;
}

fn periodic_value(time : f32, value : f32) -> f32 {
    return normalized_sine((time - value) * TAU);
}

fn mod289_vec3(x : vec3<f32>) -> vec3<f32> {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

fn mod289_vec4(x : vec4<f32>) -> vec4<f32> {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

fn permute(x : vec4<f32>) -> vec4<f32> {
    return mod289_vec4(((x * 34.0) + 1.0) * x);
}

fn taylor_inv_sqrt(r : vec4<f32>) -> vec4<f32> {
    return 1.79284291400159 - 0.85373472095314 * r;
}

fn simplex_noise(v : vec3<f32>) -> f32 {
    let C : vec2<f32> = vec2<f32>(1.0 / 6.0, 1.0 / 3.0);
    let D : vec4<f32> = vec4<f32>(0.0, 0.5, 1.0, 2.0);

    let i0 : vec3<f32> = floor(v + dot(v, vec3<f32>(C.y)));
    let x0 : vec3<f32> = v - i0 + dot(i0, vec3<f32>(C.x));

    let step1 : vec3<f32> = step(vec3<f32>(x0.y, x0.z, x0.x), x0);
    let l : vec3<f32> = vec3<f32>(1.0) - step1;
    let i1 : vec3<f32> = min(step1, vec3<f32>(l.z, l.x, l.y));
    let i2 : vec3<f32> = max(step1, vec3<f32>(l.z, l.x, l.y));

    let x1 : vec3<f32> = x0 - i1 + vec3<f32>(C.x);
    let x2 : vec3<f32> = x0 - i2 + vec3<f32>(C.y);
    let x3 : vec3<f32> = x0 - vec3<f32>(D.y);

    let i = mod289_vec3(i0);
    let p = permute(
        permute(
            permute(i.z + vec4<f32>(0.0, i1.z, i2.z, 1.0))
            + i.y + vec4<f32>(0.0, i1.y, i2.y, 1.0)
        )
        + i.x + vec4<f32>(0.0, i1.x, i2.x, 1.0)
    );

    let n_ : f32 = 0.14285714285714285;
    let ns : vec3<f32> = n_ * vec3<f32>(D.w, D.y, D.z) - vec3<f32>(D.x, D.z, D.x);

    let j : vec4<f32> = p - 49.0 * floor(p * ns.z * ns.z);
    let x_ : vec4<f32> = floor(j * ns.z);
    let y_ : vec4<f32> = floor(j - 7.0 * x_);

    let x = x_ * ns.x + ns.y;
    let y = y_ * ns.x + ns.y;
    let h = 1.0 - abs(x) - abs(y);

    let b0 : vec4<f32> = vec4<f32>(x.x, x.y, y.x, y.y);
    let b1 : vec4<f32> = vec4<f32>(x.z, x.w, y.z, y.w);

    let s0 : vec4<f32> = floor(b0) * 2.0 + 1.0;
    let s1 : vec4<f32> = floor(b1) * 2.0 + 1.0;
    let sh : vec4<f32> = -step(h, vec4<f32>(0.0));

    let a0 : vec4<f32> = vec4<f32>(b0.x, b0.z, b0.y, b0.w)
        + vec4<f32>(s0.x, s0.z, s0.y, s0.w) * vec4<f32>(sh.x, sh.x, sh.y, sh.y);
    let a1 : vec4<f32> = vec4<f32>(b1.x, b1.z, b1.y, b1.w)
        + vec4<f32>(s1.x, s1.z, s1.y, s1.w) * vec4<f32>(sh.z, sh.z, sh.w, sh.w);

    let g0 : vec3<f32> = vec3<f32>(a0.x, a0.y, h.x);
    let g1 : vec3<f32> = vec3<f32>(a0.z, a0.w, h.y);
    let g2 : vec3<f32> = vec3<f32>(a1.x, a1.y, h.z);
    let g3 : vec3<f32> = vec3<f32>(a1.z, a1.w, h.w);

    let norm : vec4<f32> = taylor_inv_sqrt(vec4<f32>(
        dot(g0, g0),
        dot(g1, g1),
        dot(g2, g2),
        dot(g3, g3)
    ));

    let g0n : vec3<f32> = g0 * norm.x;
    let g1n : vec3<f32> = g1 * norm.y;
    let g2n : vec3<f32> = g2 * norm.z;
    let g3n : vec3<f32> = g3 * norm.w;

    let m0 : f32 = max(0.6 - dot(x0, x0), 0.0);
    let m1 : f32 = max(0.6 - dot(x1, x1), 0.0);
    let m2 : f32 = max(0.6 - dot(x2, x2), 0.0);
    let m3 : f32 = max(0.6 - dot(x3, x3), 0.0);

    let m0sq : f32 = m0 * m0;
    let m1sq : f32 = m1 * m1;
    let m2sq : f32 = m2 * m2;
    let m3sq : f32 = m3 * m3;

    return 42.0 * (
        m0sq * m0sq * dot(g0n, x0)
        + m1sq * m1sq * dot(g1n, x1)
        + m2sq * m2sq * dot(g2n, x2)
        + m3sq * m3sq * dot(g3n, x3)
    );
}

fn wrap_float(value : f32, limit : f32) -> f32 {
    if (limit <= 0.0) {
        return 0.0;
    }
    var result : f32 = value - floor(value / limit) * limit;
    if (result < 0.0) {
        result = result + limit;
    }
    return result;
}

fn singularity_mask(uv : vec2<f32>, width : f32, height : f32) -> f32 {
    if (width <= 0.0 || height <= 0.0) {
        return 0.0;
    }

    let delta : vec2<f32> = abs(uv - vec2<f32>(0.5, 0.5));
    let aspect : f32 = width / height;
    let scaled : vec2<f32> = vec2<f32>(delta.x * aspect, delta.y);
    let max_radius : f32 = length(vec2<f32>(aspect * 0.5, 0.5));
    if (max_radius <= 0.0) {
        return 0.0;
    }

    let normalized : f32 = clamp(length(scaled) / max_radius, 0.0, 1.0);
    let masked : f32 = sqrt(normalized);
    return pow(masked, 5.0);
}

fn animated_simplex_value(uv : vec2<f32>, time : f32, speed : f32) -> f32 {
    let angle : f32 = time * TAU;
    let z_base : f32 = cos(angle) * speed;
    let base_seed : vec3<f32> = vec3<f32>(17.0, 29.0, 47.0);
    let base_noise : f32 = simplex_noise(vec3<f32>(
        uv.x + base_seed.x,
        uv.y + base_seed.y,
        z_base + base_seed.z
    ));
    var value : f32 = clamp(base_noise * 0.5 + 0.5, 0.0, 1.0);

    if (speed != 0.0 && time != 0.0) {
        let time_seed : vec3<f32> = vec3<f32>(
            base_seed.x + 54.0,
            base_seed.y + 82.0,
            base_seed.z + 124.0
        );
        let time_noise : f32 = simplex_noise(vec3<f32>(
            uv.x + time_seed.x,
            uv.y + time_seed.y,
            time_seed.z
        ));
        let time_value : f32 = clamp(time_noise * 0.5 + 0.5, 0.0, 1.0);
        let scaled_time : f32 = periodic_value(time, time_value) * speed;
        value = clamp01(periodic_value(scaled_time, value));
    }

    return clamp01(value);
}

fn compute_lens_offsets(
    sample_pos : vec2<f32>,
    width : f32,
    height : f32,
    freq : vec2<f32>,
    time : f32,
    speed : f32,
    displacement : f32
) -> vec2<f32> {
    let width_safe : f32 = max(width, 1.0);
    let height_safe : f32 = max(height, 1.0);
    let freq_x : f32 = max(freq.y, 1.0);
    let freq_y : f32 = max(freq.x, 1.0);

    let wrapped_pos : vec2<f32> = vec2<f32>(
        wrap_float(sample_pos.x, width_safe),
        wrap_float(sample_pos.y, height_safe)
    );
    let uv : vec2<f32> = vec2<f32>(
        (wrapped_pos.x / width_safe) * freq_x,
        (wrapped_pos.y / height_safe) * freq_y
    );

    let noise_value : f32 = animated_simplex_value(uv, time, speed);

    let uv_centered : vec2<f32> = (wrapped_pos + vec2<f32>(0.5, 0.5)) / vec2<f32>(width_safe, height_safe);
    let mask : f32 = singularity_mask(uv_centered, width_safe, height_safe);
    let distortion : f32 = (noise_value * 2.0 - 1.0) * mask;
    let angle : f32 = distortion * TAU;

    let offsets : vec2<f32> = vec2<f32>(cos(angle), sin(angle))
        * displacement * vec2<f32>(width_safe, height_safe);
    return offsets;
}

// Value noise implementation
fn fade(value : f32) -> f32 {
    return value * value * (3.0 - 2.0 * value);
}

fn fade_vec3(v : vec3<f32>) -> vec3<f32> {
    return vec3<f32>(fade(v.x), fade(v.y), fade(v.z));
}

fn lerp(a : f32, b : f32, t : f32) -> f32 {
    return a + (b - a) * t;
}

fn hash3(coord : vec3<i32>, seed : f32) -> f32 {
    let base : vec3<f32> = vec3<f32>(coord);
    let dot_value : f32 = dot(base, vec3<f32>(12.9898, 78.233, 37.719)) + seed * 0.001;
    return fract(sin(dot_value) * 43758.5453);
}

fn value_noise_3d(coord : vec3<f32>, seed : f32) -> f32 {
    let cell : vec3<i32> = vec3<i32>(floor(coord));
    let local : vec3<f32> = fract(coord);
    let smooth_t : vec3<f32> = fade_vec3(local);

    let c000 : f32 = hash3(cell, seed);
    let c100 : f32 = hash3(cell + vec3<i32>(1, 0, 0), seed);
    let c010 : f32 = hash3(cell + vec3<i32>(0, 1, 0), seed);
    let c110 : f32 = hash3(cell + vec3<i32>(1, 1, 0), seed);
    let c001 : f32 = hash3(cell + vec3<i32>(0, 0, 1), seed);
    let c101 : f32 = hash3(cell + vec3<i32>(1, 0, 1), seed);
    let c011 : f32 = hash3(cell + vec3<i32>(0, 1, 1), seed);
    let c111 : f32 = hash3(cell + vec3<i32>(1, 1, 1), seed);

    let x00 : f32 = lerp(c000, c100, smooth_t.x);
    let x10 : f32 = lerp(c010, c110, smooth_t.x);
    let x01 : f32 = lerp(c001, c101, smooth_t.x);
    let x11 : f32 = lerp(c011, c111, smooth_t.x);
    let y0 : f32 = lerp(x00, x10, smooth_t.y);
    let y1 : f32 = lerp(x01, x11, smooth_t.y);
    return lerp(y0, y1, smooth_t.z);
}

// Singularity (radial distance from center)
fn compute_singularity(x : f32, y : f32, width : f32, height : f32) -> f32 {
    let center_x : f32 = width * 0.5;
    let center_y : f32 = height * 0.5;
    let dx : f32 = (x - center_x) / width;
    let dy : f32 = (y - center_y) / height;
    return length(vec2<f32>(dx, dy));
}

// Helper functions for color space conversion and adjustments
fn wrap_unit(value : f32) -> f32 {
    var wrapped : f32 = value - floor(value);
    if (wrapped < 0.0) {
        wrapped = wrapped + 1.0;
    }
    return wrapped;
}

fn blend_linear(a : f32, b : f32, t : f32) -> f32 {
    return mix(a, b, clamp(t, 0.0, 1.0));
}

fn blend_cosine(a : f32, b : f32, value : f32) -> f32 {
    let clamped : f32 = clamp(value, 0.0, 1.0);
    let weight : f32 = (1.0 - cos(clamped * PI)) * 0.5;
    return mix(a, b, weight);
}

fn clamp_index(value : f32, max_index : f32) -> u32 {
    if (max_index <= 0.0) {
        return 0u;
    }
    let clamped : f32 = clamp(value, 0.0, max_index);
    return u32(clamped);
}

fn rgb_to_hsv(rgb : vec3<f32>) -> vec3<f32> {
    let c_max : f32 = max(max(rgb.x, rgb.y), rgb.z);
    let c_min : f32 = min(min(rgb.x, rgb.y), rgb.z);
    let delta : f32 = c_max - c_min;

    var hue : f32 = 0.0;
    if (delta > 0.0) {
        if (c_max == rgb.x) {
            var segment : f32 = (rgb.y - rgb.z) / delta;
            if (segment < 0.0) {
                segment = segment + 6.0;
            }
            hue = segment;
        } else if (c_max == rgb.y) {
            hue = ((rgb.z - rgb.x) / delta) + 2.0;
        } else {
            hue = ((rgb.x - rgb.y) / delta) + 4.0;
        }
        hue = wrap_unit(hue / 6.0);
    }

    let saturation : f32 = select(0.0, delta / c_max, c_max != 0.0);
    return vec3<f32>(hue, saturation, c_max);
}

fn hsv_to_rgb(hsv : vec3<f32>) -> vec3<f32> {
    let h : f32 = hsv.x;
    let s : f32 = hsv.y;
    let v : f32 = hsv.z;

    let dh : f32 = h * 6.0;
    let r_comp : f32 = clamp01(abs(dh - 3.0) - 1.0);
    let g_comp : f32 = clamp01(-abs(dh - 2.0) + 2.0);
    let b_comp : f32 = clamp01(-abs(dh - 4.0) + 2.0);

    let one_minus_s : f32 = 1.0 - s;
    let sr : f32 = s * r_comp;
    let sg : f32 = s * g_comp;
    let sb : f32 = s * b_comp;

    let r : f32 = clamp01((one_minus_s + sr) * v);
    let g : f32 = clamp01((one_minus_s + sg) * v);
    let b : f32 = clamp01((one_minus_s + sb) * v);

    return vec3<f32>(r, g, b);
}

fn adjust_hue(color : vec3<f32>, amount : f32) -> vec3<f32> {
    var hsv : vec3<f32> = rgb_to_hsv(color);
    hsv.x = wrap_unit(hsv.x + amount);
    hsv.y = clamp01(hsv.y);
    hsv.z = clamp01(hsv.z);
    return clamp(vec3<f32>(hsv_to_rgb(hsv)), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn adjust_saturation(color : vec3<f32>, amount : f32) -> vec3<f32> {
    var hsv : vec3<f32> = rgb_to_hsv(color);
    hsv.y = clamp01(hsv.y * amount);
    hsv.z = clamp01(hsv.z);
    return clamp(vec3<f32>(hsv_to_rgb(hsv)), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn apply_vignette(value : f32, brightness : f32, mask : f32, alpha : f32) -> f32 {
    let edge_mix : f32 = mix(value, brightness, mask);
    return mix(value, edge_mix, clamp(alpha, 0.0, 1.0));
}

// Generate base scanline values (2x1 noise pattern)
fn get_scanline_base_values(time : f32, speed : f32) -> vec2<f32> {
    let time_scaled : f32 = time * speed * 0.1;
    let noise0 : f32 = value_noise_3d(vec3<f32>(0.0, 0.0, time_scaled), 19.37);
    let noise1 : f32 = value_noise_3d(vec3<f32>(1.0, 0.0, time_scaled), 19.37);
    return vec2<f32>(noise0, noise1);
}

// Get interpolated scanline value for a given y coordinate
// The scanline pattern is based on Y position to create horizontal lines
fn get_scanline_value_interpolated(y : f32, height : f32, base_values : vec2<f32>) -> f32 {
    // Goal: ~500 bars for 1000px image = ~2px per bar, increased by 25% = 2.5px per bar
    // Each bar alternates between the 2 base values
    let pixels_per_bar : f32 = 2.5;
    let y_scaled : f32 = y / pixels_per_bar;
    let scanline_index : i32 = i32(floor(y_scaled)) % 2;
    
    return select(base_values.y, base_values.x, scanline_index == 0);
}

// Sample scanline with bilinear interpolation at fractional coordinates
fn sample_scanline_bilinear(sample_x : f32, sample_y : f32, width : f32, height : f32, base_values : vec2<f32>) -> f32 {
    // Wrap coordinates
    var wrapped_x : f32 = sample_x - floor(sample_x / width) * width;
    var wrapped_y : f32 = sample_y - floor(sample_y / height) * height;
    
    if (wrapped_x < 0.0) { wrapped_x = wrapped_x + width; }
    if (wrapped_y < 0.0) { wrapped_y = wrapped_y + height; }
    
    wrapped_x = clamp(wrapped_x, 0.0, width - 1.0);
    wrapped_y = clamp(wrapped_y, 0.0, height - 1.0);
    
    // Bilinear interpolation
    let x0 : f32 = floor(wrapped_x);
    let y0 : f32 = floor(wrapped_y);
    let x1 : f32 = min(x0 + 1.0, width - 1.0);
    let y1 : f32 = min(y0 + 1.0, height - 1.0);
    
    let x_fract : f32 = clamp(wrapped_x - x0, 0.0, 1.0);
    let y_fract : f32 = clamp(wrapped_y - y0, 0.0, 1.0);
    
    // Get scanline values at the 4 corners
    let val_x0_y0 : f32 = get_scanline_value_interpolated(y0, height, base_values);
    let val_x1_y0 : f32 = get_scanline_value_interpolated(y0, height, base_values);
    let val_x0_y1 : f32 = get_scanline_value_interpolated(y1, height, base_values);
    let val_x1_y1 : f32 = get_scanline_value_interpolated(y1, height, base_values);
    
    // Bilinear blend
    let val_y0 : f32 = mix(val_x0_y0, val_x1_y0, x_fract);
    let val_y1 : f32 = mix(val_x0_y1, val_x1_y1, x_fract);
    
    return mix(val_y0, val_y1, y_fract);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let width : u32 = as_u32(params.size.x);
    let height : u32 = as_u32(params.size.y);
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    let width_f : f32 = max(params.size.x, 1.0);
    let height_f : f32 = max(params.size.y, 1.0);
    let time : f32 = params.motion.x;
    let speed : f32 = params.motion.y;
    let x : f32 = f32(gid.x);
    let y : f32 = f32(gid.y);

    let displacement : f32 = 0.0625;
    let freq : vec2<f32> = freq_for_shape(2.0, width_f, height_f);
    let base_offsets : vec2<f32> = compute_lens_offsets(
        vec2<f32>(x, y),
        width_f,
        height_f,
        freq,
        time,
        speed,
        displacement
    );

    // Step 2: Sample the procedural scanline texture at the WARPED coordinates.
    // This correctly applies the lens warp to the scanlines.
    let scanline_base : vec2<f32> = get_scanline_base_values(time, speed);
    let scan_value : f32 = sample_scanline_bilinear(x + base_offsets.x, y + base_offsets.y, width_f, height_f, scanline_base);

    // Step 3: Sample the input texture at the ORIGINAL, un-warped coordinates.
    let base_color : vec3<f32> = textureLoad(input_texture, vec2<i32>(i32(x), i32(y)), 0).xyz;

    // Step 4: Blend the original input color with the warped scanlines.
    var color : vec3<f32> = mix(
        base_color,
        (base_color + scan_value) * scan_value,
        0.5
    );
    color = clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
    
    // Step 5: Chromatic aberration, hue shift, saturation, and vignette
    if (params.size.z >= 2.5) { // channels == 3
        let seed_base : f32 = 17.0;
        let displacement_base : f32 = 0.0125 + random_scalar(seed_base + 0.37) * 0.00625;
        let simplex_value : f32 = random_scalar(seed_base + 0.73);
        let displacement_pixels : f32 = displacement_base * width_f * simplex_value;
        
        let singularity : f32 = compute_singularity(x, y, width_f, height_f);
        let aber_mask : f32 = pow(singularity, 3.0);
        let gradient : f32 = clamp(x / (width_f - 1.0), 0.0, 1.0);
        
        let hue_shift : f32 = random_scalar(seed_base + 1.91) * 0.25 - 0.125;
        
        // Red channel sample point (aberration shift)
        var red_x : f32 = min(x + displacement_pixels, width_f - 1.0);
        red_x = blend_linear(red_x, x, gradient);
        let red_sample_x : f32 = blend_cosine(x, red_x, aber_mask);
        
        let red_base_col : vec3<f32> = textureLoad(input_texture, vec2<i32>(i32(red_sample_x), i32(y)), 0).xyz;
        let red_offsets : vec2<f32> = compute_lens_offsets(
            vec2<f32>(red_sample_x, y),
            width_f,
            height_f,
            freq,
            time,
            speed,
            displacement
        );
        let red_scan_val : f32 = sample_scanline_bilinear(red_sample_x + red_offsets.x, y + red_offsets.y, width_f, height_f, scanline_base);
        let red_blended : vec3<f32> = mix(red_base_col, (red_base_col + red_scan_val) * red_scan_val, 0.5);

        // Green channel is the original computed color for this pixel
        let green_blended : vec3<f32> = color;

        // Blue channel sample point (aberration shift)
        var blue_x : f32 = max(x - displacement_pixels, 0.0);
        blue_x = blend_linear(x, blue_x, gradient);
        let blue_sample_x : f32 = blend_cosine(x, blue_x, aber_mask);

        let blue_base_col : vec3<f32> = textureLoad(input_texture, vec2<i32>(i32(blue_sample_x), i32(y)), 0).xyz;
        let blue_offsets : vec2<f32> = compute_lens_offsets(
            vec2<f32>(blue_sample_x, y),
            width_f,
            height_f,
            freq,
            time,
            speed,
            displacement
        );
        let blue_scan_val : f32 = sample_scanline_bilinear(blue_sample_x + blue_offsets.x, y + blue_offsets.y, width_f, height_f, scanline_base);
        let blue_blended : vec3<f32> = mix(blue_base_col, (blue_base_col + blue_scan_val) * blue_scan_val, 0.5);

        // Combine, applying hue shift to each component before assembling
        color = vec3<f32>(
            adjust_hue(red_blended, hue_shift).r,
            adjust_hue(green_blended, hue_shift).g,
            adjust_hue(blue_blended, hue_shift).b
        );
        
        // Restore original hue
        color = adjust_hue(color, -hue_shift);
        
        // Step 6: Saturation boost
        color = adjust_saturation(color, 1.125);
        
        // Step 7: Vignette
        let vignette_alpha : f32 = random_scalar(seed_base + 3.17) * 0.175;
        let vignette_mask : f32 = singularity;
        color.x = apply_vignette(color.x, 0.0, vignette_mask, vignette_alpha);
        color.y = apply_vignette(color.y, 0.0, vignette_mask, vignette_alpha);
        color.z = apply_vignette(color.z, 0.0, vignette_mask, vignette_alpha);
    }
    
    // Step 8: Normalize (contrast adjustment around mean)
    let local_mean : f32 = (color.x + color.y + color.z) * INV_THREE;
    color = clamp((color - local_mean) * 1.25 + local_mean, vec3<f32>(0.0), vec3<f32>(1.0));
    
    // Write output
    let pixel_index : u32 = gid.y * width + gid.x;
    let base_index : u32 = pixel_index * 4u;
    output_buffer[base_index + 0u] = color.x;
    output_buffer[base_index + 1u] = color.y;
    output_buffer[base_index + 2u] = color.z;
    output_buffer[base_index + 3u] = 1.0;
}
