// Vortex tiling effect.
// Port of noisemaker.effects.vortex. Builds a displacement field from a
// singularity map, modulates it with a Chebyshev fade mask, and refracts the
// source texture using a simplex-driven displacement amount.

const TAU : f32 = 6.28318530717958647692;
const CHANNEL_COUNT : u32 = 4u;

struct VortexParams {
    size_displacement : vec4<f32>,  // width, height, channels, displacement
    time_speed : vec4<f32>,         // time, speed, _pad0, _pad1
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : VortexParams;

fn to_u32(value : f32) -> u32 {
    return u32(max(value, 0.0));
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

fn wrap_float(value : f32, limit : f32) -> f32 {
    if (limit == 0.0) {
        return 0.0;
    }

    let cycles : f32 = floor(value / limit);
    var result : f32 = value - cycles * limit;
    if (result < 0.0) {
        result = result + limit;
    }

    return result;
}

fn store_texel(base_index : u32, texel : vec4<f32>) {
    output_buffer[base_index + 0u] = texel.x;
    output_buffer[base_index + 1u] = texel.y;
    output_buffer[base_index + 2u] = texel.z;
    output_buffer[base_index + 3u] = texel.w;
}

fn mod289_vec3(x : vec3<f32>) -> vec3<f32> {
    return x - floor(x / 289.0) * 289.0;
}

fn mod289_vec4(x : vec4<f32>) -> vec4<f32> {
    return x - floor(x / 289.0) * 289.0;
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
        ) + i.x + vec4<f32>(0.0, i1.x, i2.x, 1.0)
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

    let norm : vec4<f32> = taylor_inv_sqrt(
        vec4<f32>(dot(g0, g0), dot(g1, g1), dot(g2, g2), dot(g3, g3))
    );
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

fn simplex_random(time : f32, speed : f32) -> f32 {
    let angle : f32 = time * TAU;
    let z : f32 = cos(angle) * speed;
    let w : f32 = sin(angle) * speed;
    let noise_value : f32 = simplex_noise(vec3<f32>(z + 17.0, w + 29.0, 11.0));
    return clamp(noise_value * 0.5 + 0.5, 0.0, 1.0);
}

fn displacement_value(coord : vec2<i32>, dims : vec2<f32>) -> f32 {
    if (dims.x <= 0.0 || dims.y <= 0.0) {
        return 0.0;
    }

    let half_dims : vec2<f32> = dims * 0.5;
    let pos : vec2<f32> = vec2<f32>(f32(coord.x), f32(coord.y)) + vec2<f32>(0.5, 0.5);
    let centered : vec2<f32> = pos - half_dims;
    let distance : f32 = length(centered);
    let max_distance : f32 = length(half_dims);
    if (max_distance <= 0.0) {
        return 0.0;
    }

    return clamp(distance / max_distance, 0.0, 1.0);
}

fn fade_value(coord : vec2<i32>, dims : vec2<f32>) -> f32 {
    if (dims.x <= 0.0 || dims.y <= 0.0) {
        return 0.0;
    }

    let half_dims : vec2<f32> = dims * 0.5;
    let pos : vec2<f32> = vec2<f32>(f32(coord.x), f32(coord.y)) + vec2<f32>(0.5, 0.5);
    let offset : vec2<f32> = abs(pos - half_dims);
    let max_component : f32 = max(offset.x / half_dims.x, offset.y / half_dims.y);
    return 1.0 - clamp(max_component, 0.0, 1.0);
}

fn gradient_at(coord : vec2<i32>, dims : vec2<f32>) -> vec2<f32> {
    let width_i : i32 = i32(dims.x);
    let height_i : i32 = i32(dims.y);
    if (width_i <= 0 || height_i <= 0) {
        return vec2<f32>(0.0, 0.0);
    }

    let center : f32 = displacement_value(coord, dims);

    let right_coord : vec2<i32> = vec2<i32>(wrap_coord(coord.x + 1, width_i), coord.y);
    let down_coord : vec2<i32> = vec2<i32>(coord.x, wrap_coord(coord.y + 1, height_i));

    let right : f32 = displacement_value(right_coord, dims);
    let down : f32 = displacement_value(down_coord, dims);

    return vec2<f32>(center - right, center - down);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let width : u32 = max(to_u32(params.size_displacement.x), 1u);
    let height : u32 = max(to_u32(params.size_displacement.y), 1u);
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    let pixel_index : u32 = gid.y * width + gid.x;
    let base_index : u32 = pixel_index * CHANNEL_COUNT;

    let width_f : f32 = max(params.size_displacement.x, 1.0);
    let height_f : f32 = max(params.size_displacement.y, 1.0);
    let dims : vec2<f32> = vec2<f32>(width_f, height_f);
    let coord_i : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));

    let fade : f32 = fade_value(coord_i, dims);
    let gradient : vec2<f32> = gradient_at(coord_i, dims) * fade;

    let random_factor : f32 = simplex_random(params.time_speed.x, params.time_speed.y);
    let warp_amount : f32 = random_factor * 100.0 * params.size_displacement.w;

    let scale_x : f32 = warp_amount * width_f * 2.0;
    let scale_y : f32 = warp_amount * height_f * 2.0;

    let sample_x : f32 = f32(gid.x) + gradient.x * scale_x;
    let sample_y : f32 = f32(gid.y) + gradient.y * scale_y;

    let wrapped_x : f32 = wrap_float(sample_x, width_f);
    let wrapped_y : f32 = wrap_float(sample_y, height_f);

    var x0 : i32 = i32(floor(wrapped_x));
    var y0 : i32 = i32(floor(wrapped_y));

    let width_i : i32 = i32(width);
    let height_i : i32 = i32(height);

    if (x0 < 0) {
        x0 = 0;
    } else if (x0 >= width_i) {
        x0 = width_i - 1;
    }

    if (y0 < 0) {
        y0 = 0;
    } else if (y0 >= height_i) {
        y0 = height_i - 1;
    }

    let x1 : i32 = wrap_coord(x0 + 1, width_i);
    let y1 : i32 = wrap_coord(y0 + 1, height_i);

    let fx : f32 = clamp(wrapped_x - f32(x0), 0.0, 1.0);
    let fy : f32 = clamp(wrapped_y - f32(y0), 0.0, 1.0);

    let tex00 : vec4<f32> = textureLoad(input_texture, vec2<i32>(x0, y0), 0);
    let tex10 : vec4<f32> = textureLoad(input_texture, vec2<i32>(x1, y0), 0);
    let tex01 : vec4<f32> = textureLoad(input_texture, vec2<i32>(x0, y1), 0);
    let tex11 : vec4<f32> = textureLoad(input_texture, vec2<i32>(x1, y1), 0);

    let mix_x0 : vec4<f32> = mix(tex00, tex10, vec4<f32>(fx));
    let mix_x1 : vec4<f32> = mix(tex01, tex11, vec4<f32>(fx));
    let result : vec4<f32> = mix(mix_x0, mix_x1, vec4<f32>(fy));

    store_texel(base_index, result);
}
