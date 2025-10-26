// DLA - Init Seeds Pass
// Initializes the simulation with random seed points.

struct DlaParams {
    size_padding : vec4<f32>,
    density_time : vec4<f32>,
    speed_padding : vec4<f32>,
};

@group(0) @binding(0) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(1) var<uniform> params : DlaParams;

const UINT32_TO_FLOAT : f32 = 1.0 / 4294967296.0;

fn fract(v : f32) -> f32 { return v - floor(v); }

fn pcg3d(v_in : vec3<u32>) -> vec3<u32> {
    var v : vec3<u32> = v_in * 1664525u + 1013904223u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v = v ^ (v >> vec3<u32>(16u));
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    return v;
}

fn random_from_cell(cell : vec2<u32>, seed : u32) -> f32 {
    let packed : vec3<u32> = vec3<u32>(cell.x, cell.y, seed);
    let noise : vec3<u32> = pcg3d(packed);
    return f32(noise.x) * UINT32_TO_FLOAT;
}

fn next_seed(seed : f32) -> f32 {
    return fract(seed * 1.3247179572447458 + 0.123456789);
}

fn rand01(seed : ptr<function, f32>) -> f32 {
    let r : f32 = *seed;
    *seed = next_seed(*seed);
    return r;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let width : u32 = u32(params.size_padding.x);
    let height : u32 = u32(params.size_padding.y);
    
    if (gid.x >= width || gid.y >= height) {
        return;
    }
    
    let p : u32 = gid.y * width + gid.x;
    let base : u32 = p * 4u;

    // Use PCG hash like multires does
    let time_seed : u32 = u32(params.density_time.w * 1000.0);
    var seed : f32 = random_from_cell(gid.xy, time_seed);
    
    let seed_density : f32 = params.density_time.x;

    if (rand01(&seed) < seed_density) {
        // Create a MAGENTA seed point
        output_buffer[base + 0u] = 1.0;
        output_buffer[base + 1u] = 0.0;
        output_buffer[base + 2u] = 1.0;
        output_buffer[base + 3u] = 1.0;
    } else {
        // Clear the pixel
        output_buffer[base + 0u] = 0.0;
        output_buffer[base + 1u] = 0.0;
        output_buffer[base + 2u] = 0.0;
        output_buffer[base + 3u] = 0.0;
    }
}
