// Low-poly art style effect inspired by Noisemaker's Python implementation.
// Recreates the combination of Voronoi distance and color sampling blended
// together and normalized for a faceted appearance.

const PI : f32 = 3.14159265358979323846;
const TAU : f32 = 6.28318530717958647692;

const CHANNEL_COUNT : u32 = 4u;
const LARGE_DISTANCE : f32 = 1e30;

const DISTRIB_RANDOM : i32 = 1000000;
const DISTRIB_SQUARE : i32 = 1000001;
const DISTRIB_WAFFLE : i32 = 1000002;
const DISTRIB_CHESS : i32 = 1000003;
const DISTRIB_H_HEX : i32 = 1000010;
const DISTRIB_V_HEX : i32 = 1000011;
const DISTRIB_SPIRAL : i32 = 1000050;
const DISTRIB_CIRCULAR : i32 = 1000100;
const DISTRIB_CONCENTRIC : i32 = 1000101;
const DISTRIB_ROTATING : i32 = 1000102;

const METRIC_MANHATTAN : i32 = 2;
const METRIC_CHEBYSHEV : i32 = 3;
const METRIC_OCTAGRAM : i32 = 4;
const METRIC_TRIANGULAR : i32 = 101;
const METRIC_HEXAGRAM : i32 = 102;
const METRIC_SDF : i32 = 201;

struct LowpolyParams {
    size : vec4<f32>,                     // width, height, channels, unused
    distrib_freq_time_speed : vec4<f32>,  // distrib, freq, time, speed
    metric_pad : vec4<f32>,               // dist_metric, pad0, pad1, pad2
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : LowpolyParams;

fn as_u32(value : f32) -> u32 {
    return u32(max(round(value), 0.0));
}

fn hash21(p : vec2<f32>) -> f32 {
    let h : f32 = dot(p, vec2<f32>(127.1, 311.7));
    return fract(sin(h) * 43758.5453123);
}

fn hash23(p : vec2<f32>) -> vec3<f32> {
    return vec3<f32>(
        hash21(p + vec2<f32>(0.0, 17.0)),
        hash21(p + vec2<f32>(11.0, 5.0)),
        hash21(p + vec2<f32>(29.0, 47.0))
    );
}

fn wrap_coord(coord : i32, limit : i32) -> i32 {
    if (limit <= 0) {
        return 0;
    }
    var wrapped : i32 = coord % limit;
    if (wrapped < 0) {
        wrapped = wrapped + limit;
    }
    return wrapped;
}

fn wrap_cell(value : i32, limit : i32) -> i32 {
    if (limit <= 0) {
        return 0;
    }
    var wrapped : i32 = value % limit;
    if (wrapped < 0) {
        wrapped = wrapped + limit;
    }
    return wrapped;
}

fn tile_shift(value : i32, limit : i32) -> f32 {
    if (limit <= 0) {
        return 0.0;
    }
    return floor(f32(value) / f32(limit));
}

fn animated_jitter(seed : vec2<f32>, amount : f32, time_value : f32, speed_value : f32) -> vec2<f32> {
    let phase : f32 = time_value * speed_value;
    let base : vec3<f32> = hash23(seed);
    let next : vec3<f32> = hash23(
        seed + vec2<f32>(phase, phase * 1.61803398875)
    );
    let blend : f32 = fract(phase);
    let jitter : vec2<f32> = mix(base.xy, next.xy, blend) - vec2<f32>(0.5, 0.5);
    return jitter * amount;
}

fn base_point_in_cell(
    cell : vec2<i32>,
    freq_int : i32,
    distribution : i32,
    time_value : f32,
    speed_value : f32
) -> vec2<f32> {
    let freq_f : f32 = f32(freq_int);
    let cell_f : vec2<f32> = vec2<f32>(f32(cell.x), f32(cell.y));
    let cell_index : i32 = cell.y * freq_int + cell.x;
    let total_points : i32 = freq_int * freq_int;
    let seed : vec2<f32> = cell_f + vec2<f32>(f32(distribution) * 0.001, f32(distribution) * -0.0013);

    var point : vec2<f32> = cell_f + vec2<f32>(0.5, 0.5);

    if (distribution == DISTRIB_RANDOM) {
        let jitter : vec3<f32> = hash23(
            seed + vec2<f32>(time_value * speed_value * 0.37, time_value * speed_value * -0.23)
        );
        point = cell_f + jitter.xy;
    } else if (distribution == DISTRIB_SQUARE) {
        point = cell_f + vec2<f32>(0.5, 0.5);
    } else if (distribution == DISTRIB_CHESS) {
        let parity : i32 = (cell.x + cell.y) & 1;
        let offset : vec2<f32> = select(
            vec2<f32>(0.25, 0.75),
            vec2<f32>(0.75, 0.25),
            parity == 0
        );
        point = cell_f + offset;
    } else if (distribution == DISTRIB_WAFFLE) {
        let even_pair : bool = ((cell.x & 1) == 0) && ((cell.y & 1) == 0);
        let offset : vec2<f32> = select(
            vec2<f32>(0.5, 0.5),
            vec2<f32>(0.25, 0.75),
            even_pair
        );
        point = cell_f + offset;
    } else if (distribution == DISTRIB_H_HEX) {
        let parity : i32 = cell.y & 1;
        let offset_x : f32 = select(0.0, 0.5, parity != 0);
        point = cell_f + vec2<f32>(0.5 + offset_x, 0.5);
    } else if (distribution == DISTRIB_V_HEX) {
        let parity : i32 = cell.x & 1;
        let offset_y : f32 = select(0.0, 0.5, parity == 0);
        point = cell_f + vec2<f32>(0.5, 0.5 + offset_y);
    } else if (
        distribution == DISTRIB_SPIRAL || distribution == DISTRIB_CIRCULAR ||
        distribution == DISTRIB_CONCENTRIC || distribution == DISTRIB_ROTATING
    ) {
        let total_f : f32 = f32(max(total_points, 1));
        let index_f : f32 = f32(cell_index);
        let center : vec2<f32> = vec2<f32>(freq_f * 0.5, freq_f * 0.5);
        var radius : f32 = (index_f / total_f) * freq_f * 0.5;
        var angle : f32 = index_f / max(total_f - 1.0, 1.0) * TAU;
        if (distribution == DISTRIB_SPIRAL) {
            let kink : f32 = mix(0.5, 1.0, hash21(seed));
            angle = angle * kink;
        }
        if (
            distribution == DISTRIB_CIRCULAR || distribution == DISTRIB_CONCENTRIC ||
            distribution == DISTRIB_ROTATING
        ) {
            let rings : f32 = max(freq_f, 1.0);
            let ring_index : f32 = f32(cell.y);
            radius = (ring_index + 1.0) / rings * freq_f * 0.5;
            let steps : f32 = max(freq_f, 1.0);
            angle = (
                f32(cell.x) + hash21(seed + vec2<f32>(0.37, 0.93))
            ) / steps * TAU;
            if (distribution == DISTRIB_ROTATING) {
                angle = angle + time_value * speed_value * 0.5;
            }
        }
        point = center + vec2<f32>(sin(angle), cos(angle)) * radius;
    }

    let drift : vec2<f32> = animated_jitter(seed, 0.5, time_value, speed_value);
    point = point + drift;

    return point;
}

fn point_for_cell(
    cell : vec2<i32>,
    freq_int : i32,
    distribution : i32,
    time_value : f32,
    speed_value : f32
) -> vec2<f32> {
    let freq_f : f32 = f32(freq_int);
    let wrapped : vec2<i32> = vec2<i32>(wrap_cell(cell.x, freq_int), wrap_cell(cell.y, freq_int));
    let base : vec2<f32> = base_point_in_cell(wrapped, freq_int, distribution, time_value, speed_value);
    let shift : vec2<f32> = vec2<f32>(tile_shift(cell.x, freq_int), tile_shift(cell.y, freq_int));
    return (base / freq_f) + shift;
}

fn wrap_distance(delta : vec2<f32>) -> vec2<f32> {
    let wrapped : vec2<f32> = delta - round(delta);
    return wrapped;
}

fn distance_metric(offset : vec2<f32>, metric : i32) -> f32 {
    let dx : f32 = offset.x;
    let dy : f32 = offset.y;
    let abs_dx : f32 = abs(dx);
    let abs_dy : f32 = abs(dy);
    switch (metric) {
        case METRIC_MANHATTAN: {
            return abs_dx + abs_dy;
        }
        case METRIC_CHEBYSHEV: {
            return max(abs_dx, abs_dy);
        }
        case METRIC_OCTAGRAM: {
            let sum : f32 = (abs_dx + abs_dy) * 0.7071067811865476;
            return max(sum, max(abs_dx, abs_dy));
        }
        case METRIC_TRIANGULAR: {
            return max(abs_dx - dy * 0.5, dy);
        }
        case METRIC_HEXAGRAM: {
            let a : f32 = max(abs_dx - dy * 0.5, dy);
            let b : f32 = max(abs_dx + dy * 0.5, -dy);
            return max(a, b);
        }
        case METRIC_SDF: {
            let angle : f32 = atan2(dx, -dy) + PI;
            let sides : f32 = 6.0;
            let segment : f32 = TAU / max(sides, 3.0);
            let k : f32 = floor(0.5 + angle / segment) * segment - angle;
            let base : f32 = length(offset);
            return cos(k) * base;
        }
        default: {
            return length(offset);
        }
    }
}

fn sample_texture(coords : vec2<f32>) -> vec3<f32> {
    let width : i32 = i32(max(params.size.x, 1.0));
    let height : i32 = i32(max(params.size.y, 1.0));
    let px : i32 = wrap_coord(i32(floor(coords.x)), width);
    let py : i32 = wrap_coord(i32(floor(coords.y)), height);
    return textureLoad(input_texture, vec2<i32>(px, py), 0).xyz;
}

fn write_output(index : u32, color : vec4<f32>) {
    output_buffer[index + 0u] = color.x;
    output_buffer[index + 1u] = color.y;
    output_buffer[index + 2u] = color.z;
    output_buffer[index + 3u] = color.w;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let width : u32 = as_u32(params.size.x);
    let height : u32 = as_u32(params.size.y);
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    let coords : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
    let base_color : vec4<f32> = textureLoad(input_texture, coords, 0);
    let uv : vec2<f32> = (vec2<f32>(f32(coords.x), f32(coords.y)) + 0.5) /
        vec2<f32>(params.size.x, params.size.y);

    let distribution : i32 = i32(round(params.distrib_freq_time_speed.x));
    let freq_int : i32 = max(i32(round(params.distrib_freq_time_speed.y)), 1);
    let freq_f : f32 = f32(freq_int);
    let time_value : f32 = params.distrib_freq_time_speed.z;
    let speed_value : f32 = params.distrib_freq_time_speed.w;
    let metric : i32 = i32(round(params.metric_pad.x));

    let scaled : vec2<f32> = uv * freq_f;
    let cell : vec2<i32> = vec2<i32>(i32(floor(scaled.x)), i32(floor(scaled.y)));

    var best_distance : f32 = LARGE_DISTANCE;
    var second_distance : f32 = LARGE_DISTANCE;
    var best_point : vec2<f32> = vec2<f32>(0.0, 0.0);
    var best_color : vec3<f32> = base_color.xyz;

    for (var offset_y : i32 = -1; offset_y <= 1; offset_y = offset_y + 1) {
        for (var offset_x : i32 = -1; offset_x <= 1; offset_x = offset_x + 1) {
            let neighbor : vec2<i32> = cell + vec2<i32>(offset_x, offset_y);
            let point : vec2<f32> = point_for_cell(neighbor, freq_int, distribution, time_value, speed_value);

            let wrapped_point : vec2<f32> = fract(point);
            var adjusted_point : vec2<f32> = wrapped_point;
            if (adjusted_point.x < 0.0) {
                adjusted_point.x = adjusted_point.x + 1.0;
            }
            if (adjusted_point.y < 0.0) {
                adjusted_point.y = adjusted_point.y + 1.0;
            }

            let offset_uv : vec2<f32> = wrap_distance(point - uv);
            let offset_pixels : vec2<f32> = vec2<f32>(
                offset_uv.x * params.size.x,
                offset_uv.y * params.size.y
            );
            let distance_value : f32 = distance_metric(offset_pixels, metric);

            if (distance_value < best_distance) {
                second_distance = best_distance;
                best_distance = distance_value;
                best_point = adjusted_point;
                let sample_coords : vec2<f32> = vec2<f32>(
                    best_point.x * params.size.x,
                    best_point.y * params.size.y
                );
                best_color = sample_texture(sample_coords);
            } else if (distance_value < second_distance) {
                second_distance = distance_value;
            }
        }
    }

    let diag : f32 = max(length(vec2<f32>(params.size.x, params.size.y)), 1.0);
    let normalized_distance : f32 = clamp(1.0 - best_distance / diag, 0.0, 1.0);
    let secondary_gap : f32 = clamp((second_distance - best_distance) / diag, 0.0, 1.0);

    let distance_color : vec3<f32> = vec3<f32>(
        normalized_distance,
        normalized_distance,
        normalized_distance
    );
    let blended : vec3<f32> = mix(distance_color, best_color, 0.5);
    let shading : f32 = 0.7 + secondary_gap * 0.6;
    let final_rgb : vec3<f32> = clamp(
        blended * shading,
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );

    let base_index : u32 = (gid.y * width + gid.x) * CHANNEL_COUNT;
    write_output(base_index, vec4<f32>(final_rgb, base_color.w));
}
