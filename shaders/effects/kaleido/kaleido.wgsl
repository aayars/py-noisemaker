// Kaleido - mirrors the input texture into kaleidoscope wedges driven by a Voronoi radius
// field. Mirrors the behavior of `noisemaker.effects.kaleido`.

struct KaleidoParams {
    width : f32,
    height : f32,
    channels : f32,
    sides : f32,
    sdf_sides : f32,
    xy_mode : f32,
    xy_x : f32,
    xy_y : f32,
    blend_edges : f32,
    time : f32,
    speed : f32,
    point_freq : f32,
    point_generations : f32,
    point_distrib : f32,
    point_drift : f32,
    point_corners : f32,
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : KaleidoParams;

const PI : f32 = 3.14159265358979323846;
const TAU : f32 = 6.28318530717958647692;

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

const MAX_POINTS : u32 = 1024u;

fn as_u32(value : f32) -> u32 {
    return u32(max(value, 0.0));
}

fn clamp_01(value : f32) -> f32 {
    return clamp(value, 0.0, 1.0);
}

fn positive_mod(value : f32, modulus : f32) -> f32 {
    if (modulus == 0.0) {
        return 0.0;
    }
    var result : f32 = value - floor(value / modulus) * modulus;
    if (result < 0.0) {
        result = result + modulus;
    }
    return result;
}

fn wrap_index(value : i32, limit : i32) -> u32 {
    if (limit <= 0) {
        return 0u;
    }
    var wrapped : i32 = value % limit;
    if (wrapped < 0) {
        wrapped = wrapped + limit;
    }
    return u32(wrapped);
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

fn random_scalar(seed : vec3<f32>) -> f32 {
    let dot_value : f32 = dot(seed, vec3<f32>(12.9898, 78.233, 37.719));
    return fract(sin(dot_value) * 43758.5453123);
}

fn random_vec2(seed : vec3<f32>) -> vec2<f32> {
    let x : f32 = random_scalar(seed);
    let y : f32 = random_scalar(seed + vec3<f32>(17.0, 59.4, 83.1));
    return vec2<f32>(x, y);
}

fn blend_cosine(a : f32, b : f32, g : f32) -> f32 {
    let weight : f32 = (1.0 - cos(g * PI)) * 0.5;
    return mix(a, b, weight);
}

fn is_grid_distribution(distrib : i32) -> bool {
    return distrib >= PD_SQUARE && distrib < PD_SPIRAL;
}

fn is_circular_distribution(distrib : i32) -> bool {
    return distrib >= PD_CIRCULAR;
}

fn distance_metric(dx : f32, dy : f32, metric : i32, sdf_sides : f32) -> f32 {
    switch (metric) {
        case 201: {
            let angle : f32 = atan2(dx, -dy) + PI;
            let radians : f32 = TAU / max(sdf_sides, 3.0);
            return cos(floor(0.5 + angle / radians) * radians - angle) * sqrt(dx * dx + dy * dy);
        }
        default: {
            return sqrt(dx * dx + dy * dy);
        }
    }
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
    for (var i : i32 = 0; i < total; i = i + 1) {
        if (*point_count >= MAX_POINTS) {
            return;
        }
        let jitter : vec2<f32> = random_vec2(seed + vec3<f32>(f32(i), time, speed));
        let jittered : vec2<f32> = jitter * range * 2.0 - range;
        var point : vec2<f32> = center + jittered;
        if (abs(drift) > 1e-6) {
            let drift_vec : vec2<f32> = random_vec2(seed + vec3<f32>(f32(i) + 53.0, speed, time)) * 2.0 - vec2<f32>(1.0, 1.0);
            let scale : f32 = drift / max(f32(freq), 1.0);
            point = point + vec2<f32>(drift_vec.x * width * scale, drift_vec.y * height * scale);
        }
        (*points)[*point_count] = vec2<f32>(wrap_float(point.x, width), wrap_float(point.y, height));
        *point_count = *point_count + 1u;
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
            let nx : f32 = (f32(a) / f32(freq)) + base_drift + x_drift;
            let ny : f32 = (f32(b) / f32(freq)) + base_drift + y_drift;
            var point : vec2<f32> = vec2<f32>(nx * range.x * 2.0, ny * range.y * 2.0);
            if (abs(drift) > 1e-6) {
                let drift_vec : vec2<f32> = (
                    random_vec2(seed + vec3<f32>(f32(a * freq + b), time, speed)) * 2.0
                ) - vec2<f32>(1.0, 1.0);
                let scale : f32 = drift / max(f32(freq), 1.0);
                point = point + vec2<f32>(drift_vec.x * width * scale, drift_vec.y * height * scale);
            }
            if (*point_count >= MAX_POINTS) {
                return;
            }
            (*points)[*point_count] = vec2<f32>(wrap_float(point.x, width), wrap_float(point.y, height));
            *point_count = *point_count + 1u;
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
    let kink : f32 = 0.5 + random_scalar(seed + vec3<f32>(time, speed, 41.0)) * 0.5;
    for (var i : i32 = 0; i < count; i = i + 1) {
        if (*point_count >= MAX_POINTS) {
            return;
        }
        let fract_value : f32 = f32(i) / max(f32(count), 1.0);
        let angle : f32 = fract_value * TAU * kink;
        var point : vec2<f32> = vec2<f32>(
            center.x + sin(angle) * fract_value * range.x,
            center.y + cos(angle) * fract_value * range.y,
        );
        if (abs(drift) > 1e-6) {
            let drift_vec : vec2<f32> = (
                random_vec2(seed + vec3<f32>(f32(i) + 11.0, speed, time)) * 2.0
            ) - vec2<f32>(1.0, 1.0);
            let scale : f32 = drift / max(f32(freq), 1.0);
            point = point + vec2<f32>(drift_vec.x * width * scale, drift_vec.y * height * scale);
        }
        (*points)[*point_count] = vec2<f32>(wrap_float(point.x, width), wrap_float(point.y, height));
        *point_count = *point_count + 1u;
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
    let kink : f32 = 0.5 + random_scalar(seed + vec3<f32>(time, speed, 79.0)) * 0.5;
    if (*point_count < MAX_POINTS) {
        (*points)[*point_count] = center;
        *point_count = *point_count + 1u;
    }
    for (var i : i32 = 1; i <= ring_count; i = i + 1) {
        let dist_fract : f32 = f32(i) / max(f32(ring_count), 1.0);
        for (var j : i32 = 1; j <= dot_count; j = j + 1) {
            if (*point_count >= MAX_POINTS) {
                return;
            }
            var radians : f32 = f32(j) * rotation;
            if (distrib == PD_CIRCULAR) {
                radians = radians + rotation * 0.5 * f32(i);
            }
            if (distrib == PD_ROTATING) {
                radians = radians + rotation * dist_fract * kink;
            }
            var point : vec2<f32> = vec2<f32>(
                center.x + sin(radians) * dist_fract * range.x,
                center.y + cos(radians) * dist_fract * range.y,
            );
            if (abs(drift) > 1e-6) {
                let drift_vec : vec2<f32> = (
                    random_vec2(seed + vec3<f32>(f32(i * dot_count + j), speed, time)) * 2.0
                ) - vec2<f32>(1.0, 1.0);
                let scale : f32 = drift / max(f32(freq), 1.0);
                point = point + vec2<f32>(drift_vec.x * width * scale, drift_vec.y * height * scale);
            }
            (*points)[*point_count] = vec2<f32>(wrap_float(point.x, width), wrap_float(point.y, height));
            *point_count = *point_count + 1u;
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
    var effective_distrib : i32 = distrib;
    if (freq == 1) {
        effective_distrib = PD_SQUARE;
    }
    for (var gen : i32 = 0; gen < generations; gen = gen + 1) {
        let seed : vec3<f32> = vec3<f32>(f32(gen), time, speed);
        if (effective_distrib == PD_RANDOM) {
            generate_random_points(points, point_count, freq, width, height, drift, time, speed, seed);
        } else if (is_grid_distribution(effective_distrib)) {
            generate_grid_points(points, point_count, freq, width, height, effective_distrib, corners, drift, time, speed, seed);
        } else if (effective_distrib == PD_SPIRAL) {
            generate_spiral_points(points, point_count, freq, width, height, drift, time, speed, seed);
        } else if (is_circular_distribution(effective_distrib)) {
            generate_circular_points(points, point_count, freq, width, height, effective_distrib, drift, time, speed, seed);
        } else {
            generate_random_points(points, point_count, freq, width, height, drift, time, speed, seed);
        }
    }
}

fn blend_point_ring(
    points : ptr<function, array<vec2<f32>, MAX_POINTS>>,
    point_count : u32,
    time : f32,
) {
    if (point_count <= 1u) {
        return;
    }
    var blended : array<vec2<f32>, MAX_POINTS>;
    for (var i : u32 = 0u; i < point_count; i = i + 1u) {
        let next : u32 = (i + 1u) % point_count;
        let current_point : vec2<f32> = (*points)[i];
        let next_point : vec2<f32> = (*points)[next];
        blended[i] = vec2<f32>(
            blend_cosine(current_point.x, next_point.x, time),
            blend_cosine(current_point.y, next_point.y, time),
        );
    }
    for (var i : u32 = 0u; i < point_count; i = i + 1u) {
        (*points)[i] = blended[i];
    }
}

fn compute_voronoi_radius(
    coord : vec2<f32>,
    width : f32,
    height : f32,
    metric : i32,
    sdf_sides : f32,
    points : ptr<function, array<vec2<f32>, MAX_POINTS>>,
    point_count : u32,
) -> f32 {
    if (point_count == 0u) {
        let safe_size : vec2<f32> = vec2<f32>(max(width, 1.0), max(height, 1.0));
        return length(coord / safe_size);
    }
    let half_width : f32 = width * 0.5;
    let half_height : f32 = height * 0.5;
    var best : f32 = 1e9;
    let use_tri_metric : bool = metric == 201;
    for (var i : u32 = 0u; i < point_count; i = i + 1u) {
        let point : vec2<f32> = (*points)[i];
        var dx : f32;
        var dy : f32;
        if (use_tri_metric) {
            dx = (coord.x - point.x) / max(width, 1.0);
            dy = (coord.y - point.y) / max(height, 1.0);
        } else {
            let x0 : f32 = coord.x - point.x - half_width;
            let x1 : f32 = coord.x - point.x + half_width;
            let y0 : f32 = coord.y - point.y - half_height;
            let y1 : f32 = coord.y - point.y + half_height;
            dx = min(abs(x0), abs(x1)) / max(width, 1.0);
            dy = min(abs(y0), abs(y1)) / max(height, 1.0);
        }
        let dist : f32 = distance_metric(dx, dy, metric, sdf_sides);
        if (dist < best) {
            best = dist;
        }
    }
    return clamp(best, 0.0, 1e6);
}

fn compute_edge_fade(x : u32, y : u32, width : u32, height : u32) -> f32 {
    if (width <= 1u || height <= 1u) {
        return 0.0;
    }
    let nx : f32 = f32(x) / f32(width - 1u);
    let ny : f32 = f32(y) / f32(height - 1u);
    let dx : f32 = abs(nx - 0.5) * 2.0;
    let dy : f32 = abs(ny - 0.5) * 2.0;
    let chebyshev : f32 = clamp(max(dx, dy), 0.0, 1.0);
    return pow(chebyshev, 5.0);
}

fn store_channels(base : u32, texel : vec4<f32>) {
    output_buffer[base + 0u] = texel.x;
    output_buffer[base + 1u] = texel.y;
    output_buffer[base + 2u] = texel.z;
    output_buffer[base + 3u] = texel.w;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let width : u32 = max(as_u32(params.width), 1u);
    let height : u32 = max(as_u32(params.height), 1u);
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    let width_f : f32 = max(params.width, 1.0);
    let height_f : f32 = max(params.height, 1.0);
    let sides : f32 = max(params.sides, 1.0);
    let raw_sdf_sides : f32 = params.sdf_sides;
    let use_sdf : bool = raw_sdf_sides >= 3.0;
    let sdf_sides : f32 = max(raw_sdf_sides, 3.0);
    let has_xy : bool = params.xy_mode > 0.5;
    let xy_coord : vec2<f32> = vec2<f32>(params.xy_x, params.xy_y);
    let blend_edges : bool = params.blend_edges > 0.5;
    let time : f32 = params.time;
    let speed : f32 = params.speed;
    let freq : i32 = max(i32(round(params.point_freq)), 1);
    let generations : i32 = max(i32(round(params.point_generations)), 1);
    let distrib : i32 = i32(round(params.point_distrib));
    let drift : f32 = params.point_drift;
    let corners : bool = params.point_corners > 0.5;

    var points : array<vec2<f32>, MAX_POINTS>;
    var point_count : u32 = 0u;
    if (has_xy) {
        let px : f32 = clamp(xy_coord.x, 0.0, max(width_f - 1.0, 0.0));
        let py : f32 = clamp(xy_coord.y, 0.0, max(height_f - 1.0, 0.0));
        points[0] = vec2<f32>(px, py);
        point_count = 1u;
    } else {
        build_point_cloud(
            &points,
            &point_count,
            width_f,
            height_f,
            freq,
            generations,
            distrib,
            corners,
            drift,
            time,
            speed,
        );
        if (freq > 1) {
            blend_point_ring(&points, point_count, time);
        }
    }

    let pixel_coord : vec2<f32> = vec2<f32>(f32(gid.x), f32(gid.y));
    var metric_id : i32 = 1;
    if (use_sdf) {
        metric_id = 201;
    }
    let radius : f32 = compute_voronoi_radius(pixel_coord, width_f, height_f, metric_id, sdf_sides, &points, point_count);

    var normalized_x : f32 = 0.0;
    if (width > 1u) {
        normalized_x = f32(gid.x) / f32(width - 1u) - 0.5;
    }
    var normalized_y : f32 = 0.0;
    if (height > 1u) {
        normalized_y = f32(gid.y) / f32(height - 1u) - 0.5;
    }

    let angle_step : f32 = TAU / sides;
    var angle : f32 = atan2(normalized_y, normalized_x) + PI * 0.5;
    angle = positive_mod(angle, angle_step);
    angle = abs(angle - angle_step * 0.5);

    var sample_x : f32 = radius * width_f * sin(angle);
    var sample_y : f32 = radius * height_f * cos(angle);

    if (blend_edges) {
        let fade : f32 = clamp_01(compute_edge_fade(gid.x, gid.y, width, height));
        sample_x = mix(sample_x, f32(gid.x), fade);
        sample_y = mix(sample_y, f32(gid.y), fade);
    }

    let wrapped_x : u32 = wrap_index(i32(round(sample_x)), i32(width));
    let wrapped_y : u32 = wrap_index(i32(round(sample_y)), i32(height));

    let texel : vec4<f32> = textureLoad(
        input_texture,
        vec2<i32>(i32(wrapped_x), i32(wrapped_y)),
        0,
    );
    let base_index : u32 = (gid.y * width + gid.x) * 4u;
    store_channels(base_index, texel);
}
