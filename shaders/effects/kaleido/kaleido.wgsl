// Kaleido - reuses Voronoi distance field to mirror the source texture into wedge slices.
// Low-level Voronoi generation is handled by a separate pass; this shader only remaps samples.

const PI : f32 = 3.14159265358979323846;
const TAU : f32 = 6.28318530717958647692;

struct KaleidoParams {
    dims : vec4<f32>,   // width, height, channel_count, sides
    misc : vec4<f32>,   // sdf_sides, blend_edges, time, speed (time/speed unused but kept for parity)
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : KaleidoParams;
@group(0) @binding(3) var radius_texture : texture_2d<f32>;

fn as_u32(value : f32) -> u32 {
    return u32(max(round(value), 0.0));
}

fn clamp01(value : f32) -> f32 {
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

fn store_color(base : u32, color : vec4<f32>) {
    output_buffer[base + 0u] = color.x;
    output_buffer[base + 1u] = color.y;
    output_buffer[base + 2u] = color.z;
    output_buffer[base + 3u] = color.w;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let width_u : u32 = max(as_u32(params.dims.x), 1u);
    let height_u : u32 = max(as_u32(params.dims.y), 1u);
    if (gid.x >= width_u || gid.y >= height_u) {
        return;
    }

    let width_f : f32 = max(params.dims.x, 1.0);
    let height_f : f32 = max(params.dims.y, 1.0);
    let sides : f32 = max(params.dims.w, 1.0);
    let blend_edges_flag : bool = params.misc.y > 0.5;

    var normalized_x : f32 = 0.0;
    if (width_u > 1u) {
        normalized_x = f32(gid.x) / f32(width_u - 1u) - 0.5;
    }
    var normalized_y : f32 = 0.0;
    if (height_u > 1u) {
        normalized_y = f32(gid.y) / f32(height_u - 1u) - 0.5;
    }

    let coord_i : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
    var radius_sample : f32 = clamp01(textureLoad(radius_texture, coord_i, 0).x);

    let angle_step : f32 = TAU / max(sides, 1.0);
    var angle : f32 = atan2(normalized_y, normalized_x) + PI * 0.5;
    angle = positive_mod(angle, angle_step);
    angle = abs(angle - angle_step * 0.5);

    var sample_x : f32 = radius_sample * width_f * sin(angle);
    var sample_y : f32 = radius_sample * height_f * cos(angle);

    if (blend_edges_flag) {
        let fade : f32 = clamp01(compute_edge_fade(gid.x, gid.y, width_u, height_u));
        sample_x = mix(sample_x, f32(gid.x), fade);
        sample_y = mix(sample_y, f32(gid.y), fade);
    }

    let wrapped_x : i32 = wrap_index(i32(round(sample_x)), i32(width_u));
    let wrapped_y : i32 = wrap_index(i32(round(sample_y)), i32(height_u));
    let color : vec4<f32> = textureLoad(input_texture, vec2<i32>(wrapped_x, wrapped_y), 0);

    let base_index : u32 = (gid.y * width_u + gid.x) * 4u;
    store_color(base_index, color);
}
