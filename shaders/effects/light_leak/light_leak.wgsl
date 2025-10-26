// Light leak effect emulating Noisemaker's Python implementation.
// Pipeline reference:
//   point_cloud -> voronoi(color regions) -> wormhole(kink=1, stride=0.25)
//   -> bloom(alpha=1) -> lighten -> center_mask(power=4)
//   -> blend with original alpha -> vaseline(alpha).

const TAU : f32 = 6.28318530717958647692;

struct LightLeakParams {
    // size packs the render target dimensions and blend alpha in .w.
    size : vec4<f32>,
    // alpha_time_speed mirrors the Python signature (alpha, time, speed).
    alpha_time_speed : vec4<f32>,
};

const POINT_COUNT : u32 = 6u;
const LAYOUT_COUNT : u32 = 4u;
const LEAK_BLOOM_COUNT : u32 = 4u;
const BLOOM_SAMPLE_COUNT : u32 = 8u;
const BLOOM_CENTER_WEIGHT : f32 = 4.0;

struct PointData {
    positions : array<vec2<f32>, POINT_COUNT>,
    colors : array<vec3<f32>, POINT_COUNT>,
};

const LAYOUTS : array<vec2<u32>, LAYOUT_COUNT> = array<vec2<u32>, LAYOUT_COUNT>(
    vec2<u32>(3u, 2u),
    vec2<u32>(2u, 3u),
    vec2<u32>(1u, 6u),
    vec2<u32>(6u, 1u),
);

const LEAK_BLOOM_OFFSETS : array<vec2<i32>, LEAK_BLOOM_COUNT> = array<vec2<i32>, LEAK_BLOOM_COUNT>(
    vec2<i32>(1, 0),
    vec2<i32>(-1, 0),
    vec2<i32>(0, 1),
    vec2<i32>(0, -1),
);

const BLOOM_KERNEL_OFFSETS : array<vec2<i32>, BLOOM_SAMPLE_COUNT> = array<vec2<i32>, BLOOM_SAMPLE_COUNT>(
    vec2<i32>(1, 0),
    vec2<i32>(-1, 0),
    vec2<i32>(0, 1),
    vec2<i32>(0, -1),
    vec2<i32>(1, 1),
    vec2<i32>(-1, 1),
    vec2<i32>(1, -1),
    vec2<i32>(-1, -1),
);

const BLOOM_KERNEL_WEIGHTS : array<f32, BLOOM_SAMPLE_COUNT> = array<f32, BLOOM_SAMPLE_COUNT>(
    2.0,
    2.0,
    2.0,
    2.0,
    1.0,
    1.0,
    1.0,
    1.0,
);

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : LightLeakParams;

fn as_u32(value : f32) -> u32 {
    return u32(max(round(value), 0.0));
}

fn clamp01(value : f32) -> f32 {
    return clamp(value, 0.0, 1.0);
}

fn clamp_vec3(value : vec3<f32>) -> vec3<f32> {
    return clamp(value, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn wrap_coord(value : i32, limit : i32) -> i32 {
    if (limit <= 0) {
        return 0;
    }

    var wrapped : i32 = value % limit;
    if (wrapped < 0) {
        wrapped = wrapped + limit;
    }

    return wrapped;
}

fn hash31(p : vec3<f32>) -> f32 {
    let h : f32 = dot(p, vec3<f32>(127.1, 311.7, 74.7));
    return fract(sin(h) * 43758.5453123);
}

fn random_vec2(seed : vec3<f32>) -> vec2<f32> {
    let h1 : f32 = hash31(seed);
    let h2 : f32 = hash31(seed + vec3<f32>(17.13, 29.97, 42.75));
    return vec2<f32>(h1, h2);
}

fn select_layout(time_value : f32, speed_value : f32) -> vec2<u32> {
    let phase : f32 = floor((time_value * speed_value) * 0.25 + 0.5);
    let index_value : f32 = hash31(vec3<f32>(phase, time_value * 0.5, speed_value + 11.0));
    let layout_index : u32 = min(u32(floor(index_value * f32(LAYOUT_COUNT))), LAYOUT_COUNT - 1u);
    return LAYOUTS[layout_index];
}

fn sample_base_color(uv : vec2<f32>, width : i32, height : i32) -> vec3<f32> {
    let px : i32 = wrap_coord(i32(floor(uv.x * f32(width))), width);
    let py : i32 = wrap_coord(i32(floor(uv.y * f32(height))), height);
    return textureLoad(input_texture, vec2<i32>(px, py), 0).xyz;
}

fn luminance(color : vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.299, 0.587, 0.114));
}

fn chebyshev_mask(uv : vec2<f32>, dimensions : vec2<f32>) -> f32 {
    if (dimensions.x <= 0.0 || dimensions.y <= 0.0) {
        return 0.0;
    }

    let centered : vec2<f32> = abs(uv - vec2<f32>(0.5, 0.5));
    let px : f32 = centered.x * dimensions.x;
    let py : f32 = centered.y * dimensions.y;
    let dist : f32 = max(px, py);
    let max_dimension : f32 = max(dimensions.x, dimensions.y) * 0.5;
    if (max_dimension <= 0.0) {
        return 0.0;
    }

    return clamp(dist / max_dimension, 0.0, 1.0);
}

fn generate_point_data(
    data : ptr<function, PointData>,
    grid_layout : vec2<u32>,
    width : f32,
    height : f32,
    time_value : f32,
    speed_value : f32,
) {
    let layout_x : f32 = max(f32(grid_layout.x), 1.0);
    let layout_y : f32 = max(f32(grid_layout.y), 1.0);
    let cell_size : vec2<f32> = vec2<f32>(1.0 / layout_x, 1.0 / layout_y);
    let drift_strength : f32 = 0.05;
    let width_i : i32 = max(i32(width), 1);
    let height_i : i32 = max(i32(height), 1);

    var index : u32 = 0u;
    loop {
        if (index >= POINT_COUNT) {
            break;
        }

    let cell_x : f32 = f32(index % grid_layout.x);
    let cell_y : f32 = f32(index / grid_layout.x);
        let base_center : vec2<f32> = (vec2<f32>(cell_x, cell_y) + vec2<f32>(0.5, 0.5)) * cell_size;

        let drift_phase : f32 = time_value * speed_value;
        let oscillation : vec2<f32> = vec2<f32>(
            sin(drift_phase * 0.7 + f32(index) * 1.618),
            cos(drift_phase * 0.5 + f32(index) * 2.236),
        ) * drift_strength;

        let jitter : vec2<f32> = (random_vec2(vec3<f32>(f32(index), drift_phase, speed_value)) - vec2<f32>(0.5, 0.5))
            * (drift_strength * 0.5);

        let position : vec2<f32> = fract(base_center + oscillation + jitter);
        (*data).positions[index] = position;
        (*data).colors[index] = sample_base_color(position, width_i, height_i);

        index = index + 1u;
    }
}

fn nearest_color(uv : vec2<f32>, data : ptr<function, PointData>) -> vec3<f32> {
    var best_index : u32 = 0u;
    var best_distance : f32 = 1e9;

    var i : u32 = 0u;
    loop {
        if (i >= POINT_COUNT) {
            break;
        }

        let point : vec2<f32> = (*data).positions[i];
        let delta : vec2<f32> = abs(uv - point);
        let wrap_delta : vec2<f32> = min(delta, vec2<f32>(1.0, 1.0) - delta);
        let dist : f32 = dot(wrap_delta, wrap_delta);
        if (dist < best_distance) {
            best_distance = dist;
            best_index = i;
        }

        i = i + 1u;
    }

    return (*data).colors[best_index];
}

fn leak_bloom_color(uv : vec2<f32>, data : ptr<function, PointData>, inv_size : vec2<f32>) -> vec3<f32> {
    let base_color : vec3<f32> = nearest_color(uv, data);
    var accum : vec3<f32> = base_color * 4.0;
    var weight : f32 = 4.0;

    var i : u32 = 0u;
    loop {
        if (i >= LEAK_BLOOM_COUNT) {
            break;
        }

        let offset : vec2<i32> = LEAK_BLOOM_OFFSETS[i];
        let sample_uv : vec2<f32> = fract(uv + vec2<f32>(f32(offset.x), f32(offset.y)) * inv_size * 3.5);
        let sample_color : vec3<f32> = nearest_color(sample_uv, data);
        accum = accum + sample_color * 2.0;
        weight = weight + 2.0;

        i = i + 1u;
    }

    return clamp_vec3(accum / weight);
}

fn compute_leak_stage(
    uv : vec2<f32>,
    data : ptr<function, PointData>,
    width : f32,
    height : f32,
    time_value : f32,
    speed_value : f32,
) -> vec3<f32> {
    let width_i : i32 = max(i32(width), 1);
    let height_i : i32 = max(i32(height), 1);

    let base_sample : vec3<f32> = textureLoad(
        input_texture,
        vec2<i32>(
            wrap_coord(i32(floor(uv.x * width)), width_i),
            wrap_coord(i32(floor(uv.y * height)), height_i),
        ),
        0,
    ).xyz;

    let base_leak : vec3<f32> = nearest_color(uv, data);
    let luma : f32 = luminance(base_leak);
    let swirl_phase : f32 = time_value * speed_value * 0.5;
    let angle : f32 = luma * TAU + swirl_phase;
    let stride : f32 = 0.25;
    let warp_vector : vec2<f32> = vec2<f32>(cos(angle), sin(angle)) * stride;
    let drift_offset : vec2<f32> = vec2<f32>(time_value * 0.05, time_value * 0.033) * speed_value;
    let warped_uv : vec2<f32> = fract(uv + warp_vector + drift_offset);
    let wormhole_sample : vec3<f32> = nearest_color(warped_uv, data);
    let wormhole_color : vec3<f32> = mix(base_leak, clamp_vec3(sqrt(clamp_vec3(wormhole_sample))), 0.65);

    let inv_size : vec2<f32> = vec2<f32>(
        1.0 / max(width, 1.0),
        1.0 / max(height, 1.0),
    );
    let bloom_color : vec3<f32> = leak_bloom_color(warped_uv, data, inv_size);
    let leak_color : vec3<f32> = clamp_vec3(mix(wormhole_color, bloom_color, 0.55));

    let lighten_color : vec3<f32> = vec3<f32>(1.0) - (vec3<f32>(1.0) - base_sample) * (vec3<f32>(1.0) - leak_color);
    let mask : f32 = pow(chebyshev_mask(uv, vec2<f32>(width, height)), 4.0);
    let center_blend : vec3<f32> = mix(base_sample, lighten_color, mask);
    return clamp_vec3(center_blend);
}

fn compute_vaseline_bloom(
    uv : vec2<f32>,
    data : ptr<function, PointData>,
    width : f32,
    height : f32,
    time_value : f32,
    speed_value : f32,
    blend_alpha : f32,
    center_base : vec3<f32>,
    center_leak : vec3<f32>,
) -> vec3<f32> {
    let width_i : i32 = max(i32(width), 1);
    let height_i : i32 = max(i32(height), 1);
    let inv_size : vec2<f32> = vec2<f32>(
        1.0 / max(width, 1.0),
        1.0 / max(height, 1.0),
    );

    let center_blended : vec3<f32> = mix(center_base, center_leak, blend_alpha);
    var accum : vec3<f32> = center_blended * BLOOM_CENTER_WEIGHT;
    var weight_sum : f32 = BLOOM_CENTER_WEIGHT;

    var i : u32 = 0u;
    loop {
        if (i >= BLOOM_SAMPLE_COUNT) {
            break;
        }

        let offset : vec2<i32> = BLOOM_KERNEL_OFFSETS[i];
        let weight : f32 = BLOOM_KERNEL_WEIGHTS[i];
        let offset_uv : vec2<f32> = fract(uv + vec2<f32>(f32(offset.x), f32(offset.y)) * inv_size * 2.0);
        let sample_base : vec3<f32> = sample_base_color(offset_uv, width_i, height_i);
        let sample_leak : vec3<f32> = compute_leak_stage(offset_uv, data, width, height, time_value, speed_value);
        let sample_blended : vec3<f32> = mix(sample_base, sample_leak, blend_alpha);
        accum = accum + sample_blended * weight;
        weight_sum = weight_sum + weight;

        i = i + 1u;
    }

    return clamp_vec3(accum / weight_sum);
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
    let width_i : i32 = max(i32(width_f), 1);
    let height_i : i32 = max(i32(height_f), 1);

    let alpha : f32 = clamp01(params.alpha_time_speed.x);
    let time_value : f32 = params.alpha_time_speed.y;
    let speed_value : f32 = params.alpha_time_speed.z;

    let grid_layout : vec2<u32> = select_layout(time_value, speed_value);
    var point_data : PointData;
    generate_point_data(&point_data, grid_layout, width_f, height_f, time_value, speed_value);

    let coords : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
    let base_sample : vec4<f32> = textureLoad(input_texture, coords, 0);

    let uv : vec2<f32> = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5, 0.5)) / vec2<f32>(width_f, height_f);
    let leak_stage : vec3<f32> = compute_leak_stage(uv, &point_data, width_f, height_f, time_value, speed_value);
    let blended : vec3<f32> = mix(base_sample.xyz, leak_stage, alpha);

    let vaseline_bloom : vec3<f32> = compute_vaseline_bloom(
        uv,
        &point_data,
        width_f,
        height_f,
        time_value,
        speed_value,
        alpha,
        base_sample.xyz,
        leak_stage,
    );

    let mask : f32 = pow(chebyshev_mask(uv, vec2<f32>(width_f, height_f)), 2.0);
    let vaseline_color : vec3<f32> = mix(blended, vaseline_bloom, mask);
    let final_color : vec3<f32> = clamp_vec3(mix(blended, vaseline_color, alpha));

    let pixel_index : u32 = (gid.y * width + gid.x) * 4u;
    output_buffer[pixel_index + 0u] = final_color.x;
    output_buffer[pixel_index + 1u] = final_color.y;
    output_buffer[pixel_index + 2u] = final_color.z;
    output_buffer[pixel_index + 3u] = base_sample.w;
}
