// Voronoi diagram effect converted from Noisemaker's Python reference implementation.
// Supports range, region, and flow diagram variants with optional refract blending.

const PI : f32 = 3.14159265358979323846;
const TAU : f32 = 6.28318530717958647692;

const MAX_POINTS : u32 = 256u;
const EPSILON : f32 = 1e-6;

const PD_RANDOM : i32 = 1000000;
const PD_SQUARE : i32 = 1000001;
const PD_WAFFLE : i32 = 1000002;
const PD_CHESS : i32 = 1000003;
const PD_H_HEX : i32 = 1000010;
const PD_V_HEX : i32 = 1000011;
const PD_SPIRAL : i32 = 1000050;
const PD_CIRCULAR : i32 = 1000100;
const PD_CONCENTRIC : i32 = 1000101;
const PD_ROTATING : i32 = 1000102;

var<workgroup> shared_points : array<vec2<f32>, MAX_POINTS>;
var<workgroup> shared_point_colors : array<vec4<f32>, MAX_POINTS>;
var<workgroup> shared_point_count : u32;

struct VoronoiParams {
    dims : vec4<f32>,                    // width, height, channels, diagram_type
    nth_metric_sdf_alpha : vec4<f32>,    // nth, dist_metric, sdf_sides, alpha
    refract_inverse_xy : vec4<f32>,      // with_refract, inverse, xy_mode, xy_count (unused)
    ridge_refract_time_speed : vec4<f32>, // ridges_hint, refract_y_from_offset, time, speed
    freq_gen_distrib_drift : vec4<f32>,  // point_freq, point_generations, point_distrib, point_drift
    corners_downsample_pad : vec4<f32>,  // point_corners, downsample, pad0, pad1
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : VoronoiParams;

fn as_u32(value : f32) -> u32 {
    return u32(max(round(value), 0.0));
}

fn bool_from(value : f32) -> bool {
    return value > 0.5;
}

fn clamp01(value : f32) -> f32 {
    return clamp(value, 0.0, 1.0);
}

fn clamp_color(color : vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        clamp01(color.x),
        clamp01(color.y),
        clamp01(color.z),
        clamp01(color.w)
    );
}

fn wrap_float(value : f32, limit : f32) -> f32 {
    if (limit <= 0.0) {
        return 0.0;
    }
    var wrapped : f32 = value - floor(value / limit) * limit;
    if (wrapped < 0.0) {
        wrapped = wrapped + limit;
    }
    return wrapped;
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

fn append_point(
    points : ptr<function, array<vec2<f32>, MAX_POINTS>>,
    point_count : ptr<function, u32>,
    point : vec2<f32>,
) {
    if ((*point_count) >= MAX_POINTS) {
        return;
    }
    let index : u32 = (*point_count);
    (*points)[index] = point;
    (*point_count) = index + 1u;
}

fn hash31(p : vec3<f32>) -> f32 {
    let h : f32 = dot(p, vec3<f32>(127.1, 311.7, 74.7));
    return fract(sin(h) * 43758.5453);
}

fn random_scalar(seed : vec3<f32>) -> f32 {
    return hash31(seed);
}

fn random_vec2(seed : vec3<f32>) -> vec2<f32> {
    let x : f32 = hash31(seed);
    let y : f32 = hash31(seed + vec3<f32>(19.19, 73.73, 37.37));
    return vec2<f32>(x, y);
}

fn width_value() -> f32 {
    let dims : vec2<u32> = textureDimensions(input_texture, 0);
    let w_tex : f32 = f32(dims.x);
    return max(select(params.dims.x, w_tex, w_tex > 0.0), 1.0);
}

fn height_value() -> f32 {
    let dims : vec2<u32> = textureDimensions(input_texture, 0);
    let h_tex : f32 = f32(dims.y);
    return max(select(params.dims.y, h_tex, h_tex > 0.0), 1.0);
}

fn diagram_type() -> i32 {
    return i32(round(params.dims.w));
}

fn nth_value() -> i32 {
    return i32(round(params.nth_metric_sdf_alpha.x));
}

fn distance_metric_id() -> i32 {
    return i32(round(params.nth_metric_sdf_alpha.y));
}

fn sdf_sides() -> f32 {
    return max(params.nth_metric_sdf_alpha.z, 3.0);
}

fn alpha_value() -> f32 {
    return clamp01(params.nth_metric_sdf_alpha.w);
}

fn with_refract_amount() -> f32 {
    return params.refract_inverse_xy.x;
}

fn inverse_flag() -> bool {
    return bool_from(params.refract_inverse_xy.y);
}

fn ridges_hint_flag() -> bool {
    return bool_from(params.ridge_refract_time_speed.x);
}

fn refract_y_from_offset_flag() -> bool {
    return bool_from(params.ridge_refract_time_speed.y);
}

fn current_time() -> f32 {
    return params.ridge_refract_time_speed.z;
}

fn current_speed() -> f32 {
    return params.ridge_refract_time_speed.w;
}

fn point_frequency() -> i32 {
    return max(i32(round(params.freq_gen_distrib_drift.x)), 1);
}

fn point_generations() -> i32 {
    return max(i32(round(params.freq_gen_distrib_drift.y)), 1);
}

fn point_distribution() -> i32 {
    return i32(round(params.freq_gen_distrib_drift.z));
}

fn point_drift() -> f32 {
    return params.freq_gen_distrib_drift.w;
}

fn point_corners_flag() -> bool {
    return bool_from(params.corners_downsample_pad.x);
}

fn downsample_flag() -> bool {
    return bool_from(params.corners_downsample_pad.y);
}

fn lowpoly_pack_enabled() -> bool {
    return params.corners_downsample_pad.z > 0.5;
}

fn is_grid_distribution(distrib : i32) -> bool {
    return distrib >= PD_SQUARE && distrib < PD_SPIRAL;
}

fn is_circular_distribution(distrib : i32) -> bool {
    return distrib >= PD_CIRCULAR;
}

fn distance_metric(dx : f32, dy : f32, metric : i32, sides : f32) -> f32 {
    switch (metric) {
        case 2: {
            return abs(dx) + abs(dy);
        }
        case 3: {
            return max(abs(dx), abs(dy));
        }
        case 4: {
            let sum : f32 = (abs(dx) + abs(dy)) * 0.7071067811865476;
            return max(sum, max(abs(dx), abs(dy)));
        }
        case 101: {
            return max(abs(dx) - dy * 0.5, dy);
        }
        case 102: {
            let a : f32 = max(abs(dx) - dy * 0.5, dy);
            let b : f32 = max(abs(dx) + dy * 0.5, -dy);
            return max(a, b);
        }
        case 201: {
            let angle : f32 = atan2(dx, -dy) + PI;
            let r : f32 = TAU / max(sides, 3.0);
            let k : f32 = floor(0.5 + angle / r) * r - angle;
            let base : f32 = sqrt(dx * dx + dy * dy);
            return cos(k) * base;
        }
        default: {
            let sum : f32 = dx * dx + dy * dy;
            return sqrt(max(sum, 0.0));
        }
    }
}

fn blend_cosine(a : f32, b : f32, t : f32) -> f32 {
    let smooth_t : f32 = (1.0 - cos(t * PI)) * 0.5;
    return a * (1.0 - smooth_t) + b * smooth_t;
}

fn generate_random_points(
    points : ptr<function, array<vec2<f32>, MAX_POINTS>>,
    point_count : ptr<function, u32>,
    freq : i32,
    width : f32,
    height : f32,
    drift : f32,
    time : f32,
    speed : f32,
    seed : vec3<f32>,
) {
    if (freq <= 0) {
        return;
    }
    let total : i32 = freq * freq;
    let center : vec2<f32> = vec2<f32>(width * 0.5, height * 0.5);
    let range : vec2<f32> = vec2<f32>(width * 0.5, height * 0.5);
    
    // Generate two sets of points for smooth animation via cosine blending
    // Use floor(time*speed) to get stable point sets that transition smoothly
    let time_floor : f32 = floor(time * speed);
    let time_fract : f32 = fract(time * speed);
    
    for (var i : i32 = 0; i < total; i = i + 1) {
        if ((*point_count) >= MAX_POINTS) {
            return;
        }
        
        // Generate first point set (current)
        let jitter0 : vec2<f32> = random_vec2(seed + vec3<f32>(f32(i), time_floor, 0.0));
        var px0 : f32 = center.x + (jitter0.x * range.x * 2.0 - range.x);
        var py0 : f32 = center.y + (jitter0.y * range.y * 2.0 - range.y);
        
        // Generate second point set (next)
        let jitter1 : vec2<f32> = random_vec2(seed + vec3<f32>(f32(i), time_floor + 1.0, 0.0));
        var px1 : f32 = center.x + (jitter1.x * range.x * 2.0 - range.x);
        var py1 : f32 = center.y + (jitter1.y * range.y * 2.0 - range.y);
        
        // Blend between the two point sets
        var px : f32 = blend_cosine(px0, px1, time_fract);
        var py : f32 = blend_cosine(py0, py1, time_fract);
        
        if (abs(drift) > EPSILON) {
            let drift_vec0 : vec2<f32> = random_vec2(seed + vec3<f32>(f32(i) + 53.0, time_floor, 0.0)) * 2.0 - vec2<f32>(1.0, 1.0);
            let drift_vec1 : vec2<f32> = random_vec2(seed + vec3<f32>(f32(i) + 53.0, time_floor + 1.0, 0.0)) * 2.0 - vec2<f32>(1.0, 1.0);
            let drift_vec : vec2<f32> = vec2<f32>(
                blend_cosine(drift_vec0.x, drift_vec1.x, time_fract),
                blend_cosine(drift_vec0.y, drift_vec1.y, time_fract)
            );
            let scale : f32 = drift / max(f32(freq), 1.0);
            px = px + drift_vec.x * width * scale;
            py = py + drift_vec.y * height * scale;
        }
        append_point(points, point_count, vec2<f32>(wrap_float(px, width), wrap_float(py, height)));
    }
}

fn generate_grid_points(
    points : ptr<function, array<vec2<f32>, MAX_POINTS>>,
    point_count : ptr<function, u32>,
    freq : i32,
    width : f32,
    height : f32,
    distrib : i32,
    corners : bool,
    drift : f32,
    time : f32,
    speed : f32,
    seed : vec3<f32>,
) {
    if (freq <= 0) {
        return;
    }
    let center : vec2<f32> = vec2<f32>(width * 0.5, height * 0.5);
    let range : vec2<f32> = vec2<f32>(width * 0.5, height * 0.5);
    let drift_amount : f32 = 0.5 / max(f32(freq), 1.0);
    var base_drift : f32 = drift_amount;
    if ((freq & 1) == 0) {
    base_drift = select(0.0, drift_amount, corners);
    } else {
    base_drift = select(drift_amount, 0.0, corners);
    }
    for (var a : i32 = 0; a < freq; a = a + 1) {
        for (var b : i32 = 0; b < freq; b = b + 1) {
            if (distrib == PD_WAFFLE && (b & 1) == 0 && (a & 1) == 0) {
                continue;
            }
            if (distrib == PD_CHESS && ((a & 1) == (b & 1))) {
                continue;
            }
            var x_drift : f32 = 0.0;
            if (distrib == PD_H_HEX && (b & 1) == 1) {
                x_drift = drift_amount;
            }
            var y_drift : f32 = 0.0;
            if (distrib == PD_V_HEX && (a & 1) == 0) {
                y_drift = drift_amount;
            }
            let nx : f32 = (f32(a) / max(f32(freq), 1.0)) + base_drift + x_drift;
            let ny : f32 = (f32(b) / max(f32(freq), 1.0)) + base_drift + y_drift;
            var px : f32 = center.x + nx * range.x * 2.0;
            var py : f32 = center.y + ny * range.y * 2.0;
            if ((*point_count) >= MAX_POINTS) {
                return;
            }
            if (abs(drift) > EPSILON) {
                // Use floor/fract for stable animation
                let time_floor : f32 = floor(time * speed);
                let time_fract : f32 = fract(time * speed);
                
                let drift_vec0 : vec2<f32> = random_vec2(seed + vec3<f32>(f32(a * freq + b), time_floor, 0.0)) * 2.0 - vec2<f32>(1.0, 1.0);
                let drift_vec1 : vec2<f32> = random_vec2(seed + vec3<f32>(f32(a * freq + b), time_floor + 1.0, 0.0)) * 2.0 - vec2<f32>(1.0, 1.0);
                
                let drift_vec : vec2<f32> = vec2<f32>(
                    blend_cosine(drift_vec0.x, drift_vec1.x, time_fract),
                    blend_cosine(drift_vec0.y, drift_vec1.y, time_fract)
                );
                
                let scale : f32 = drift / max(f32(freq), 1.0);
                px = px + drift_vec.x * width * scale;
                py = py + drift_vec.y * height * scale;
            }
            append_point(points, point_count, vec2<f32>(wrap_float(px, width), wrap_float(py, height)));
        }
    }
}

fn generate_spiral_points(
    points : ptr<function, array<vec2<f32>, MAX_POINTS>>,
    point_count : ptr<function, u32>,
    freq : i32,
    width : f32,
    height : f32,
    drift : f32,
    time : f32,
    speed : f32,
    seed : vec3<f32>,
) {
    if (freq <= 0) {
        return;
    }
    let count : i32 = freq * freq;
    let center : vec2<f32> = vec2<f32>(width * 0.5, height * 0.5);
    let range : vec2<f32> = vec2<f32>(width * 0.5, height * 0.5);
    
    let time_floor : f32 = floor(time * speed);
    let time_fract : f32 = fract(time * speed);
    let kink0 : f32 = 0.5 + random_scalar(seed + vec3<f32>(time_floor, speed, 41.0)) * 0.5;
    let kink1 : f32 = 0.5 + random_scalar(seed + vec3<f32>(time_floor + 1.0, speed, 41.0)) * 0.5;
    let kink : f32 = blend_cosine(kink0, kink1, time_fract);
    
    for (var i : i32 = 0; i < count; i = i + 1) {
        if ((*point_count) >= MAX_POINTS) {
            return;
        }
        let fract_val : f32 = f32(i) / max(f32(count), 1.0);
        let angle : f32 = fract_val * TAU * kink;
        var px : f32 = center.x + sin(angle) * fract_val * range.x;
        var py : f32 = center.y + cos(angle) * fract_val * range.y;
        if (abs(drift) > EPSILON) {
            let drift_vec0 : vec2<f32> = random_vec2(seed + vec3<f32>(f32(i) + 11.0, time_floor, 0.0)) * 2.0 - vec2<f32>(1.0, 1.0);
            let drift_vec1 : vec2<f32> = random_vec2(seed + vec3<f32>(f32(i) + 11.0, time_floor + 1.0, 0.0)) * 2.0 - vec2<f32>(1.0, 1.0);
            let drift_vec : vec2<f32> = vec2<f32>(
                blend_cosine(drift_vec0.x, drift_vec1.x, time_fract),
                blend_cosine(drift_vec0.y, drift_vec1.y, time_fract)
            );
            let scale : f32 = drift / max(f32(freq), 1.0);
            px = px + drift_vec.x * width * scale;
            py = py + drift_vec.y * height * scale;
        }
        append_point(points, point_count, vec2<f32>(wrap_float(px, width), wrap_float(py, height)));
    }
}

fn generate_circular_points(
    points : ptr<function, array<vec2<f32>, MAX_POINTS>>,
    point_count : ptr<function, u32>,
    freq : i32,
    width : f32,
    height : f32,
    distrib : i32,
    drift : f32,
    time : f32,
    speed : f32,
    seed : vec3<f32>,
) {
    if (freq <= 0) {
        return;
    }
    let center : vec2<f32> = vec2<f32>(width * 0.5, height * 0.5);
    let range : vec2<f32> = vec2<f32>(width * 0.5, height * 0.5);
    let ring_count : i32 = freq;
    let dot_count : i32 = freq;
    let rotation : f32 = TAU / max(f32(dot_count), 1.0);
    
    let time_floor : f32 = floor(time * speed);
    let time_fract : f32 = fract(time * speed);
    let kink0 : f32 = 0.5 + random_scalar(seed + vec3<f32>(time_floor, speed, 79.0)) * 0.5;
    let kink1 : f32 = 0.5 + random_scalar(seed + vec3<f32>(time_floor + 1.0, speed, 79.0)) * 0.5;
    let kink : f32 = blend_cosine(kink0, kink1, time_fract);
    
    if ((*point_count) < MAX_POINTS) {
        append_point(points, point_count, center);
    }
    for (var i : i32 = 1; i <= ring_count; i = i + 1) {
        let dist_fract : f32 = f32(i) / max(f32(ring_count), 1.0);
        for (var j : i32 = 1; j <= dot_count; j = j + 1) {
            if ((*point_count) >= MAX_POINTS) {
                return;
            }
            var radians : f32 = f32(j) * rotation;
            if (distrib == PD_CIRCULAR) {
                radians = radians + rotation * 0.5 * f32(i);
            }
            if (distrib == PD_ROTATING) {
                radians = radians + rotation * dist_fract * kink;
            }
            var px : f32 = center.x + sin(radians) * dist_fract * range.x;
            var py : f32 = center.y + cos(radians) * dist_fract * range.y;
            if (abs(drift) > EPSILON) {
                let drift_vec0 : vec2<f32> = random_vec2(seed + vec3<f32>(f32(i * dot_count + j), time_floor, 0.0)) * 2.0 - vec2<f32>(1.0, 1.0);
                let drift_vec1 : vec2<f32> = random_vec2(seed + vec3<f32>(f32(i * dot_count + j), time_floor + 1.0, 0.0)) * 2.0 - vec2<f32>(1.0, 1.0);
                let drift_vec : vec2<f32> = vec2<f32>(
                    blend_cosine(drift_vec0.x, drift_vec1.x, time_fract),
                    blend_cosine(drift_vec0.y, drift_vec1.y, time_fract)
                );
                let scale : f32 = drift / max(f32(freq), 1.0);
                px = px + drift_vec.x * width * scale;
                py = py + drift_vec.y * height * scale;
            }
            append_point(points, point_count, vec2<f32>(wrap_float(px, width), wrap_float(py, height)));
        }
    }
}

fn build_point_cloud(
    points : ptr<function, array<vec2<f32>, MAX_POINTS>>,
    point_count : ptr<function, u32>,
    width : f32,
    height : f32,
    freq : i32,
    generations : i32,
    distrib : i32,
    corners : bool,
    drift : f32,
    time : f32,
    speed : f32,
) {
    if (freq <= 0 || generations <= 0) {
        return;
    }
    let distrib_id : f32 = params.freq_gen_distrib_drift.z;
    let drift_value : f32 = params.freq_gen_distrib_drift.w;
    for (var gen : i32 = 0; gen < generations; gen = gen + 1) {
        // Use a stable base seed so point layouts evolve smoothly over time.
        // Avoid feeding raw time into the seed; animation comes from the per-point blending logic.
        let seed : vec3<f32> = vec3<f32>(
            f32(gen) * 37.0 + distrib_id * 0.01,
            drift_value,
            distrib_id * 0.001 + f32(gen)
        );
        if (distrib == PD_RANDOM) {
            generate_random_points(points, point_count, freq, width, height, drift, time, speed, seed);
        } else if (is_grid_distribution(distrib)) {
            generate_grid_points(points, point_count, freq, width, height, distrib, corners, drift, time, speed, seed);
        } else if (distrib == PD_SPIRAL) {
            generate_spiral_points(points, point_count, freq, width, height, drift, time, speed, seed);
        } else if (is_circular_distribution(distrib)) {
            generate_circular_points(points, point_count, freq, width, height, distrib, drift, time, speed, seed);
        } else {
            generate_random_points(points, point_count, freq, width, height, drift, time, speed, seed);
        }
        if ((*point_count) >= MAX_POINTS) {
            return;
        }
    }
}

fn select_nth_index(
    sorted_indices : ptr<function, array<u32, MAX_POINTS>>,
    count : u32,
    nth_param : i32,
) -> u32 {
    if (count == 0u) {
        return 0u;
    }
    if (nth_param >= 0) {
        let clamped : i32 = clamp(nth_param, 0, i32(count) - 1);
        return (*sorted_indices)[u32(clamped)];
    }
    let pos : i32 = clamp(-nth_param, 1, i32(count));
    let resolved : i32 = i32(count) - pos;
    return (*sorted_indices)[u32(max(resolved, 0))];
}

fn select_nth_distance(
    sorted_distances : ptr<function, array<f32, MAX_POINTS>>,
    count : u32,
    nth_param : i32,
) -> f32 {
    if (count == 0u) {
        return 0.0;
    }
    if (nth_param >= 0) {
        let clamped : i32 = clamp(nth_param, 0, i32(count) - 1);
        return (*sorted_distances)[u32(clamped)];
    }
    let pos : i32 = clamp(-nth_param, 1, i32(count));
    let resolved : i32 = i32(count) - pos;
    let idx : u32 = u32(max(resolved, 0));
    return (*sorted_distances)[idx];
}

fn is_flow_diagram(diagram : i32) -> bool {
    return diagram == 41 || diagram == 42;
}

fn needs_color_regions(diagram : i32) -> bool {
    return diagram == 22 || diagram == 31 || diagram == 42;
}

fn needs_range_slice(diagram : i32) -> bool {
    return diagram == 11 || diagram == 12 || diagram == 31 || diagram == 41;
}

fn luminance_from(color : vec4<f32>) -> f32 {
    return dot(color.xyz, vec3<f32>(0.299, 0.587, 0.114));
}

fn refract_color(
    pixel_coord : vec2<f32>,
    width : u32,
    height : u32,
    reference : vec4<f32>,
    displacement : f32,
    y_from_offset : bool,
) -> vec4<f32> {
    if (displacement == 0.0) {
        let xi : i32 = clamp(i32(pixel_coord.x), 0, i32(width) - 1);
        let yi : i32 = clamp(i32(pixel_coord.y), 0, i32(height) - 1);
        return textureLoad(input_texture, vec2<i32>(xi, yi), 0);
    }
    let width_f : f32 = max(f32(width), 1.0);
    let height_f : f32 = max(f32(height), 1.0);
    let ref_color : vec4<f32> = clamp_color(reference);
    let angle : f32 = luminance_from(ref_color) * TAU;
    let direction : vec2<f32> = vec2<f32>(cos(angle), sin(angle));
    let offset : vec2<f32> = direction * displacement;
    var sample_x : f32 = pixel_coord.x + offset.x * width_f;
    var sample_y : f32 = pixel_coord.y + offset.y * height_f;
    if (y_from_offset) {
        sample_x = sample_x + width_f * 0.5;
        sample_y = sample_y + height_f * 0.5;
    }
    let wrapped_x : i32 = wrap_index(i32(floor(sample_x + 0.5)), i32(width));
    let wrapped_y : i32 = wrap_index(i32(floor(sample_y + 0.5)), i32(height));
    return textureLoad(input_texture, vec2<i32>(wrapped_x, wrapped_y), 0);
}

@compute @workgroup_size(8, 8, 1)
fn main(
    @builtin(global_invocation_id) gid : vec3<u32>,
    @builtin(local_invocation_id) lid : vec3<u32>,
) {
    let dims : vec2<u32> = textureDimensions(input_texture, 0);
    let width : u32 = select(as_u32(params.dims.x), dims.x, dims.x > 0u);
    let height : u32 = select(as_u32(params.dims.y), dims.y, dims.y > 0u);
    let in_bounds : bool = gid.x < width && gid.y < height;

    let width_f : f32 = width_value();
    let height_f : f32 = height_value();
    let downsample_enabled : bool = downsample_flag();
    let sample_scale : f32 = select(1.0, 0.5, downsample_enabled);
    let sample_width : f32 = max(width_f * sample_scale, 1.0);
    let sample_height : f32 = max(height_f * sample_scale, 1.0);

    let metric : i32 = distance_metric_id();
    let diagram : i32 = diagram_type();
    let sides : f32 = sdf_sides();
    let nth_param : i32 = nth_value();
    let inverse_diagram : bool = inverse_flag();
    let ridges : bool = ridges_hint_flag();
    let refract_y_from_offset : bool = refract_y_from_offset_flag();
    let needs_colors : bool = needs_color_regions(diagram) || diagram == 12 || diagram == 42;
    let lowpoly_enabled : bool = lowpoly_pack_enabled();

    let freq : i32 = point_frequency();
    let generations : i32 = point_generations();
    let distrib : i32 = point_distribution();
    let drift : f32 = point_drift();
    let corners : bool = point_corners_flag();
    let time : f32 = current_time();
    let speed : f32 = current_speed();

    if (lid.x == 0u && lid.y == 0u && lid.z == 0u) {
        shared_point_count = 0u;
        var local_points : array<vec2<f32>, MAX_POINTS>;
        var local_count : u32 = 0u;
        build_point_cloud(&local_points, &local_count, sample_width, sample_height, freq, generations, distrib, corners, drift, time, speed);
        shared_point_count = local_count;

        let inv_scale : f32 = select(1.0, 1.0 / sample_scale, downsample_enabled);
        if (needs_colors) {
            for (var i : u32 = 0u; i < local_count; i = i + 1u) {
                let point : vec2<f32> = local_points[i];
                shared_points[i] = point;
                let sample_point : vec2<f32> = point * inv_scale;
                let sx : i32 = clamp(i32(round(sample_point.x)), 0, i32(width) - 1);
                let sy : i32 = clamp(i32(round(sample_point.y)), 0, i32(height) - 1);
                var color : vec4<f32> = clamp_color(textureLoad(input_texture, vec2<i32>(sx, sy), 0));
                if (ridges && (diagram == 22 || diagram == 31 || diagram == 42)) {
                    color = abs(color * 2.0 - vec4<f32>(1.0, 1.0, 1.0, 1.0));
                }
                shared_point_colors[i] = color;
            }
        } else {
            for (var i : u32 = 0u; i < local_count; i = i + 1u) {
                shared_points[i] = local_points[i];
                shared_point_colors[i] = vec4<f32>(0.0, 0.0, 0.0, 1.0);
            }
        }
    }

    workgroupBarrier();

    if (!in_bounds) {
        return;
    }

    let point_count : u32 = shared_point_count;
    if (point_count == 0u) {
        let base_index : u32 = (gid.y * width + gid.x) * 4u;
        let color : vec4<f32> = textureLoad(input_texture, vec2<i32>(i32(gid.x), i32(gid.y)), 0);
        output_buffer[base_index + 0u] = color.x;
        output_buffer[base_index + 1u] = color.y;
        output_buffer[base_index + 2u] = color.z;
        output_buffer[base_index + 3u] = 1.0;
        return;
    }

    var sorted_distances : array<f32, MAX_POINTS>;
    var sorted_indices : array<u32, MAX_POINTS>;
    var sorted_count : u32 = 0u;
    var nearest_distance : f32 = 1e30;
    var nearest_index : u32 = 0u;
    var second_distance : f32 = 1e30;
    var second_index : u32 = 0u;
    var min_local_distance_accum : f32 = 1e30;
    var max_local_distance_accum : f32 = -1e30;

    let pixel_coord_sample : vec2<f32> = vec2<f32>(f32(gid.x), f32(gid.y)) * sample_scale;
    let pixel_coord_original : vec2<f32> = vec2<f32>(f32(gid.x), f32(gid.y));
    let half_w : f32 = sample_width * 0.5;
    let half_h : f32 = sample_height * 0.5;
    let is_triangular_metric : bool = metric == 101 || metric == 102 || metric == 201;
    let is_sdf_metric : bool = metric == 201;

    var flow_sum : f32 = 0.0;
    var color_flow_sum : vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);

    let inverse_sign : f32 = select(1.0, -1.0, inverse_diagram);
    let lowpoly_fast_path : bool = lowpoly_enabled && diagram == 22 && nth_param == 0;

    for (var i : u32 = 0u; i < point_count; i = i + 1u) {
        let point : vec2<f32> = shared_points[i];
        var point_color : vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        if (needs_colors) {
            point_color = shared_point_colors[i];
        }

        var dx : f32;
        var dy : f32;
        if (is_triangular_metric) {
            dx = (pixel_coord_sample.x - point.x) / sample_width;
            dy = ((pixel_coord_sample.y - point.y) * inverse_sign) / sample_height;
        } else {
            let x0 : f32 = pixel_coord_sample.x - point.x - half_w;
            let x1 : f32 = pixel_coord_sample.x - point.x + half_w;
            let y0 : f32 = pixel_coord_sample.y - point.y - half_h;
            let y1 : f32 = pixel_coord_sample.y - point.y + half_h;
            dx = min(abs(x0), abs(x1)) / sample_width;
            dy = min(abs(y0), abs(y1)) / sample_height;
        }
        var dist : f32 = distance_metric(dx, dy, metric, sides);
        if (is_sdf_metric) {
            if (abs(dist) < EPSILON) {
                dist = select(-EPSILON, EPSILON, dist >= 0.0);
            }
        } else {
            dist = max(dist, EPSILON);
        }
        if (lowpoly_fast_path) {
            min_local_distance_accum = min(min_local_distance_accum, dist);
            max_local_distance_accum = max(max_local_distance_accum, dist);
            if (dist < nearest_distance) {
                second_distance = nearest_distance;
                second_index = nearest_index;
                nearest_distance = dist;
                nearest_index = i;
            } else if (dist < second_distance) {
                second_distance = dist;
                second_index = i;
            }
        } else {
            // Maintain full sorted order for general diagrams.
            if (sorted_count == 0u) {
                sorted_distances[0u] = dist;
                sorted_indices[0u] = i;
                sorted_count = 1u;
            } else if (sorted_count < point_count && sorted_count < MAX_POINTS) {
                var j : u32 = sorted_count;
                loop {
                    if (j == 0u || dist >= sorted_distances[j - 1u]) {
                        break;
                    }
                    sorted_distances[j] = sorted_distances[j - 1u];
                    sorted_indices[j] = sorted_indices[j - 1u];
                    j = j - 1u;
                }
                sorted_distances[j] = dist;
                sorted_indices[j] = i;
                sorted_count = sorted_count + 1u;
            } else {
                if (sorted_count == 0u) {
                    sorted_count = 1u;
                    sorted_distances[0u] = dist;
                    sorted_indices[0u] = i;
                } else if (dist < sorted_distances[sorted_count - 1u]) {
                    var j2 : u32 = sorted_count - 1u;
                    loop {
                        if (j2 == 0u || dist >= sorted_distances[j2 - 1u]) {
                            break;
                        }
                        sorted_distances[j2] = sorted_distances[j2 - 1u];
                        sorted_indices[j2] = sorted_indices[j2 - 1u];
                        j2 = j2 - 1u;
                    }
                    sorted_distances[j2] = dist;
                    sorted_indices[j2] = i;
                }
            }
        }

        if (is_flow_diagram(diagram)) {
            let log_val : f32 = clamp(log(dist), -10.0, 10.0);
            flow_sum = flow_sum + log_val;
            if (diagram == 42) {
                color_flow_sum = color_flow_sum + point_color * log_val;
            }
        }
    }

    if (lowpoly_fast_path) {
        sorted_count = 1u;
        sorted_distances[0u] = nearest_distance;
        sorted_indices[0u] = nearest_index;
        if (second_distance < 1e29) {
            sorted_count = 2u;
            sorted_distances[1u] = second_distance;
            sorted_indices[1u] = second_index;
        }
    }

    if (!lowpoly_fast_path && sorted_count == 0u) {
        let base_index : u32 = (gid.y * width + gid.x) * 4u;
        let color : vec4<f32> = textureLoad(input_texture, vec2<i32>(i32(gid.x), i32(gid.y)), 0);
        output_buffer[base_index + 0u] = color.x;
        output_buffer[base_index + 1u] = color.y;
        output_buffer[base_index + 2u] = color.z;
        output_buffer[base_index + 3u] = 1.0;
        return;
    }

    let selected_index : u32 = select_nth_index(&sorted_indices, sorted_count, nth_param);
    let selected_distance : f32 = select_nth_distance(&sorted_distances, sorted_count, nth_param);

    var min_local_distance : f32 = sorted_distances[0u];
    var max_local_distance : f32 = sorted_distances[sorted_count - 1u];
    if (lowpoly_fast_path) {
        min_local_distance = min_local_distance_accum;
        max_local_distance = max_local_distance_accum;
    }

    var normalized_value : f32;
    if (is_sdf_metric) {
        let denominator : f32 = max(max_local_distance - min_local_distance, EPSILON);
        normalized_value = clamp((selected_distance - min_local_distance) / denominator, 0.0, 1.0);
    } else {
        let denominator : f32 = max(max_local_distance, EPSILON);
        normalized_value = clamp(selected_distance / denominator, 0.0, 1.0);
    }

    var range_value : f32 = sqrt(normalized_value);

    var lowpoly_range_value : f32 = range_value;
    if (lowpoly_enabled) {
        var alt_distance : f32 = select_nth_distance(&sorted_distances, sorted_count, 1);
        if (lowpoly_fast_path && second_distance < 1e29) {
            alt_distance = second_distance;
        }
        var alt_normalized : f32;
        if (is_sdf_metric) {
            let denominator : f32 = max(max_local_distance - min_local_distance, EPSILON);
            alt_normalized = clamp((alt_distance - min_local_distance) / denominator, 0.0, 1.0);
        } else {
            let denominator : f32 = max(max_local_distance, EPSILON);
            alt_normalized = clamp(alt_distance / denominator, 0.0, 1.0);
        }
        lowpoly_range_value = sqrt(alt_normalized);
    }

    if (needs_range_slice(diagram) && inverse_diagram) {
        range_value = 1.0 - range_value;
    }

    let input_color : vec4<f32> = clamp_color(textureLoad(input_texture, vec2<i32>(i32(gid.x), i32(gid.y)), 0));
    var range_color : vec4<f32> = vec4<f32>(range_value, range_value, range_value, 1.0);

    if (diagram == 12) {
        let tinted : vec4<f32> = input_color * range_value;
        range_color = tinted * (1.0 - range_value) + vec4<f32>(range_value, range_value, range_value, 1.0) * range_value;
    }

    var effect_color : vec4<f32> = input_color;

    if (diagram == 0) {
        effect_color = input_color;
    } else if (diagram == 11) {
        effect_color = range_color;
    } else if (diagram == 12) {
        let tinted : vec4<f32> = input_color * range_value;
        effect_color = tinted * (1.0 - range_value) + vec4<f32>(range_value, range_value, range_value, 1.0) * range_value;
    } else if (diagram == 21) {
        var region_index_value : f32 = 0.0;
        if (sorted_count > 1u) {
            region_index_value = f32(selected_index) / f32(sorted_count - 1u);
        }
        effect_color = vec4<f32>(region_index_value, region_index_value, region_index_value, 1.0);
    } else if (diagram == 22) {
        effect_color = shared_point_colors[selected_index];
        if (lowpoly_enabled) {
            effect_color.w = clamp(lowpoly_range_value, 0.0, 1.0);
        }
    } else if (diagram == 31) {
        let region_color : vec4<f32> = shared_point_colors[selected_index];
        let blend_amount : f32 = clamp(range_value * range_value, 0.0, 1.0);
        effect_color = range_color * (1.0 - blend_amount) + region_color * blend_amount;
    } else if (diagram == 41) {
        let avg : f32 = flow_sum / f32(sorted_count);
        var flow_value : f32 = (avg + 1.75) / 1.45;
        if (inverse_diagram) {
            flow_value = 1.0 - flow_value;
        }
        flow_value = clamp(flow_value, 0.0, 1.0);
        effect_color = vec4<f32>(flow_value, flow_value, flow_value, 1.0);
    } else if (diagram == 42) {
        let avg_log : f32 = flow_sum / f32(sorted_count);
        let scalar : f32 = clamp((avg_log + 1.75) / 1.45, 0.0, 1.0);
        var color_flow : vec4<f32> = vec4<f32>(scalar, scalar, scalar, 1.0);
        if (length(color_flow_sum.xyz) > 0.0) {
            color_flow = clamp_color(color_flow_sum / f32(sorted_count));
        }
        if (inverse_diagram) {
            color_flow = vec4<f32>(1.0, 1.0, 1.0, 1.0) - color_flow;
        }
        effect_color = color_flow;
    }

    var final_color : vec4<f32> = effect_color;
    let refract_amount : f32 = max(with_refract_amount(), 0.0);
    if (refract_amount > 0.0) {
        final_color = refract_color(pixel_coord_original, width, height, effect_color, refract_amount, refract_y_from_offset);
    }

    let alpha : f32 = alpha_value();
    final_color = input_color * (1.0 - alpha) + final_color * alpha;
    if (lowpoly_enabled) {
        final_color.w = clamp(lowpoly_range_value, 0.0, 1.0);
    } else {
        final_color.w = 1.0;
    }

    let base_index : u32 = (gid.y * width + gid.x) * 4u;
    output_buffer[base_index + 0u] = final_color.x;
    output_buffer[base_index + 1u] = final_color.y;
    output_buffer[base_index + 2u] = final_color.z;
    output_buffer[base_index + 3u] = final_color.w;
}

