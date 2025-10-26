// Glitch effect compute shader.
//
// Faithfully mirrors the TensorFlow implementation found in
// noisemaker/effects.py::glitch. It generates a value-noise field
// and shifts RGB channels horizontally based on that field.

const CHANNEL_COUNT : u32 = 4u;

struct GlitchParams {
    size : vec4<f32>,
    anim : vec4<f32>,
}

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : GlitchParams;

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
    let c : vec2<f32> = vec2<f32>(1.0 / 6.0, 1.0 / 3.0);
    let i0 : vec3<f32> = floor(v + dot(v, vec3<f32>(c.y)));
    let x0 : vec3<f32> = v - i0 + dot(i0, vec3<f32>(c.x));

    let step1 : vec3<f32> = step(vec3<f32>(x0.y, x0.z, x0.x), x0);
    let l : vec3<f32> = vec3<f32>(1.0) - step1;
    let i1 : vec3<f32> = min(step1, vec3<f32>(l.z, l.x, l.y));
    let i2 : vec3<f32> = max(step1, vec3<f32>(l.z, l.x, l.y));

    let x1 : vec3<f32> = x0 - i1 + vec3<f32>(c.x);
    let x2 : vec3<f32> = x0 - i2 + vec3<f32>(c.y);
    let x3 : vec3<f32> = x0 - vec3<f32>(0.5);

    let i_wrapped : vec3<f32> = mod289_vec3(i0);
    let p : vec4<f32> = permute(
        permute(
            permute(vec4<f32>(i_wrapped.z) + vec4<f32>(0.0, i1.z, i2.z, 1.0))
            + vec4<f32>(i_wrapped.y) + vec4<f32>(0.0, i1.y, i2.y, 1.0)
        ) + vec4<f32>(i_wrapped.x) + vec4<f32>(0.0, i1.x, i2.x, 1.0)
    );

    let n_s : f32 = 0.142857142857;
    let j : vec4<f32> = p - 49.0 * floor(p * n_s);

    let x_vec : vec4<f32> = floor(j * n_s);
    let y_vec : vec4<f32> = floor(j - 7.0 * x_vec);

    let x_offset : vec4<f32> = x_vec * n_s + vec4<f32>(0.5) * n_s;
    let y_offset : vec4<f32> = y_vec * n_s + vec4<f32>(0.5) * n_s;

    let gx : vec4<f32> = x_offset - vec4<f32>(0.5);
    let gy : vec4<f32> = y_offset - vec4<f32>(0.5);
    let gz : vec4<f32> = vec4<f32>(0.5) - abs(gx) - abs(gy);
    let sz : vec4<f32> = step(gz, vec4<f32>(0.0));
    let gx_adj = gx - sz * (step(vec4<f32>(0.0), gx) - 0.5);
    let gy_adj = gy - sz * (step(vec4<f32>(0.0), gy) - 0.5);

    let g0 : vec3<f32> = vec3<f32>(gx_adj.x, gy_adj.x, gz.x);
    let g1 : vec3<f32> = vec3<f32>(gx_adj.y, gy_adj.y, gz.y);
    let g2 : vec3<f32> = vec3<f32>(gx_adj.z, gy_adj.z, gz.z);
    let g3 : vec3<f32> = vec3<f32>(gx_adj.w, gy_adj.w, gz.w);

    let norm : vec4<f32> = taylor_inv_sqrt(vec4<f32>(
        dot(g0, g0),
        dot(g1, g1),
        dot(g2, g2),
        dot(g3, g3)
    ));

    let g0_n : vec3<f32> = g0 * norm.x;
    let g1_n : vec3<f32> = g1 * norm.y;
    let g2_n : vec3<f32> = g2 * norm.z;
    let g3_n : vec3<f32> = g3 * norm.w;

    let m : vec4<f32> = max(vec4<f32>(0.6) - vec4<f32>(
        dot(x0, x0),
        dot(x1, x1),
        dot(x2, x2),
        dot(x3, x3)
    ), vec4<f32>(0.0));
    let m2 : vec4<f32> = m * m;
    let m4 : vec4<f32> = m2 * m2;

    let pdotx : vec4<f32> = vec4<f32>(
        dot(g0_n, x0),
        dot(g1_n, x1),
        dot(g2_n, x2),
        dot(g3_n, x3)
    );

    return 42.0 * dot(m4, pdotx);
}

fn value_noise(coord : vec2<f32>, freq : f32, time : f32, speed : f32) -> f32 {
    let width = max(params.size.x, 1.0);
    let height = max(params.size.y, 1.0);
    let uv = vec2<f32>(coord.x / width, coord.y / height);
    let scaled_uv = uv * freq;
    let z = time * speed;
    let noise = simplex_noise(vec3<f32>(scaled_uv.x, scaled_uv.y, z));
    return clamp(noise * 0.5 + 0.5, 0.0, 1.0);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let width : u32 = u32(max(params.size.x, 1.0));
    let height : u32 = u32(max(params.size.y, 1.0));
    
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    let base_index : u32 = (gid.y * width + gid.x) * CHANNEL_COUNT;
    let coord = vec2<f32>(f32(gid.x), f32(gid.y));
    let time_value = params.anim.x;
    let speed_value = params.anim.y * 50.0;
    let noise = value_noise(coord, 4.0, time_value, speed_value);
    let shift = i32(floor(noise * 4.0));
    
    for (var channel : u32 = 0u; channel < CHANNEL_COUNT; channel++) {
        var sample_x = i32(gid.x);
        
        if (channel == 0u) {
            sample_x = sample_x + shift;
        } else if (channel == 2u) {
            sample_x = sample_x - shift;
        }
        
        let width_i = i32(width);
        var wrapped_x = sample_x % width_i;
        if (wrapped_x < 0) {
            wrapped_x = wrapped_x + width_i;
        }
        
        let texel = textureLoad(input_texture, vec2<i32>(wrapped_x, i32(gid.y)), 0);
        output_buffer[base_index + channel] = texel[channel];
    }
}
