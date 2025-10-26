// DLA - Agent Walk Pass
// Each thread processes one walker in parallel

const EXPANDED_RANGE : i32 = 8;
const EXPANDED_WIDTH : i32 = EXPANDED_RANGE * 2 + 1;

const NEAR_DIRECTION_COUNT : u32 = 8u;
const NEAR_DIRECTIONS : array<vec2<i32>, 8> = array<vec2<i32>, 8>(
    vec2<i32>(-1, -1), vec2<i32>(-1, 0), vec2<i32>(-1, 1),
    vec2<i32>(0, -1),  vec2<i32>(0, 1),
    vec2<i32>(1, -1),  vec2<i32>(1, 0),  vec2<i32>(1, 1)
);

const BLUR_KERNEL_SIZE : u32 = 5u;
const BLUR_KERNEL_RADIUS : i32 = 2;
const BLUR_KERNEL : array<f32, 25> = array<f32, 25>(
    1.0 / 36.0, 4.0 / 36.0, 6.0 / 36.0, 4.0 / 36.0, 1.0 / 36.0,
    4.0 / 36.0, 16.0 / 36.0, 24.0 / 36.0, 16.0 / 36.0, 4.0 / 36.0,
    6.0 / 36.0, 24.0 / 36.0, 36.0 / 36.0, 24.0 / 36.0, 6.0 / 36.0,
    4.0 / 36.0, 16.0 / 36.0, 24.0 / 36.0, 16.0 / 36.0, 4.0 / 36.0,
    1.0 / 36.0, 4.0 / 36.0, 6.0 / 36.0, 4.0 / 36.0, 1.0 / 36.0,
);

struct DlaParams {
    size_padding : vec4<f32>,
    density_time : vec4<f32>,
    speed_padding : vec4<f32>,
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : DlaParams;
@group(0) @binding(3) var prev_texture : texture_2d<f32>;
@group(0) @binding(4) var<storage, read> agent_state_in : array<f32>;
@group(0) @binding(5) var<storage, read_write> agent_state_out : array<f32>;
@group(0) @binding(6) var<storage, read_write> glider_buffer : array<f32>;

fn wrap_int(v : i32, s : i32) -> i32 {
    if (s <= 0) { return 0; }
    var w : i32 = v % s;
    if (w < 0) { w = w + s; }
    return w;
}

fn luminance(rgb : vec3<f32>) -> f32 {
    return (rgb.x + rgb.y + rgb.z) * (1.0 / 3.0);
}

fn fract(v : f32) -> f32 { return v - floor(v); }

fn hash_seed(seed : f32, agent_id : u32) -> f32 {
    // Better mixing using agent ID to decorrelate seeds
    let mixed : f32 = fract(seed * 43758.5453123 + f32(agent_id) * 0.1031);
    return fract(mixed * 73856.093);
}

fn next_seed(seed : f32) -> f32 {
    return fract(seed * 1.3247179572447458 + 0.123456789);
}

fn rand01(seed : ptr<function, f32>) -> f32 {
    let r : f32 = *seed;
    *seed = next_seed(*seed);
    return r;
}

fn clamp01(v : f32) -> f32 { return clamp(v, 0.0, 1.0); }

fn uniform_index(seed : ptr<function, f32>, count : u32) -> u32 {
    if (count == 0u) { return 0u; }
    let scaled : f32 = rand01(seed) * f32(count);
    let idx : u32 = u32(floor(scaled));
    return clamp(idx, 0u, count - 1u);
}

fn random_sign(seed : ptr<function, f32>) -> i32 {
    return select(-1, 1, rand01(seed) < 0.5);
}

fn random_near_offset(seed : ptr<function, f32>) -> vec2<i32> {
    let idx : u32 = uniform_index(seed, NEAR_DIRECTION_COUNT);
    return NEAR_DIRECTIONS[idx];
}

fn random_far_offset(seed : ptr<function, f32>) -> vec2<i32> {
    let rx : f32 = rand01(seed) * f32(EXPANDED_WIDTH);
    let ry : f32 = rand01(seed) * f32(EXPANDED_WIDTH);
    var dx : i32 = i32(floor(rx)) - EXPANDED_RANGE;
    var dy : i32 = i32(floor(ry)) - EXPANDED_RANGE;

    if (dx == 0 && dy == 0) {
        let axis_choice : bool = rand01(seed) < 0.5;
        if (axis_choice) {
            dx = random_sign(seed);
        } else {
            dy = random_sign(seed);
        }
    } else if (abs(dx) == abs(dy)) {
        if (rand01(seed) < 0.5) {
            dx = 0;
        } else {
            dy = 0;
        }
    }

    return vec2<i32>(dx, dy);
}

fn kernel_value(index : u32) -> f32 {
    if (index >= 25u) { return 0.0; }
    return BLUR_KERNEL[index];
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    // Get dimensions from params, NOT from texture (they may differ)
    let width : u32 = u32(params.size_padding.x);
    let height : u32 = u32(params.size_padding.y);
    if (width == 0u || height == 0u) { return; }

    let floats_len : u32 = arrayLength(&agent_state_in);
    if (floats_len < 8u) { return; }
    let agent_count : u32 = floats_len / 8u;
    
    let agent_id : u32 = gid.x;
    if (agent_id >= agent_count) { return; }

    let base_state : u32 = agent_id * 8u;
    var x : f32 = agent_state_in[base_state + 0u];
    var y : f32 = agent_state_in[base_state + 1u];
    var seed : f32 = agent_state_in[base_state + 7u];
    
    let current_xi : i32 = wrap_int(i32(floor(x)), i32(width));
    let current_yi : i32 = wrap_int(i32(floor(y)), i32(height));

    // Check if in expanded_neighborhoods (within 8 pixels of cluster)
    var in_expanded : bool = false;
    for (var ey : i32 = -EXPANDED_RANGE; ey <= EXPANDED_RANGE; ey = ey + 1) {
        for (var ex : i32 = -EXPANDED_RANGE; ex <= EXPANDED_RANGE; ex = ex + 1) {
            let sx : i32 = wrap_int(current_xi + ex, i32(width));
            let sy : i32 = wrap_int(current_yi + ey, i32(height));
            let idx : u32 = u32(sy) * width + u32(sx);
            let base : u32 = idx * 4u;
            let r : f32 = output_buffer[base + 0u];
            let g : f32 = output_buffer[base + 1u];
            let b : f32 = output_buffer[base + 2u];
            // Check if this pixel is magenta (part of cluster)
            if (r > 0.5 && g < 0.5 && b > 0.5) {
                in_expanded = true;
                break;
            }
        }
        if (in_expanded) { break; }
    }

    // ALWAYS move exactly 1 pixel (8-directional + stay)
    let offset : vec2<i32> = random_near_offset(&seed);

    var xi : i32 = wrap_int(current_xi + offset.x, i32(width));
    var yi : i32 = wrap_int(current_yi + offset.y, i32(height));

    // Stick if near existing cluster in output_buffer (current frame's cluster state)
    var should_stick : bool = false;
    for (var oy : i32 = -1; oy <= 1; oy = oy + 1) {
        for (var ox : i32 = -1; ox <= 1; ox = ox + 1) {
            let sx : i32 = wrap_int(xi + ox, i32(width));
            let sy : i32 = wrap_int(yi + oy, i32(height));
            let idx : u32 = u32(sy) * width + u32(sx);
            let base : u32 = idx * 4u;
            let r : f32 = output_buffer[base + 0u];
            let g : f32 = output_buffer[base + 1u];
            let b : f32 = output_buffer[base + 2u];
            // Check if this pixel is magenta (part of cluster)
            if (r > 0.5 && g < 0.5 && b > 0.5) { should_stick = true; break; }
        }
        if (should_stick) { break; }
    }

    if (should_stick) {
        // Deposit MAGENTA at stick location (no kernel, just center pixel)
        let p : u32 = u32(yi) * width + u32(xi);
        let base : u32 = p * 4u;
        output_buffer[base + 0u] = 1.0;  // R
        output_buffer[base + 1u] = 0.0;  // G
        output_buffer[base + 2u] = 1.0;  // B
        output_buffer[base + 3u] = 1.0;  // A
        
        // Respawn at completely random location
        // Add extra mixing to decorrelate x and y
        let r1 : f32 = rand01(&seed);
        seed = fract(seed * 7.3891);  // Extra mixing
        let r2 : f32 = rand01(&seed);
        seed = fract(seed * 5.7721);  // Extra mixing
        let rx : u32 = u32(floor(r1 * f32(width)));
        let ry : u32 = u32(floor(r2 * f32(height)));
        x = f32(rx);
        y = f32(ry);
    } else {
        // Move to new position and record glider overlay color
        x = f32(xi);
        y = f32(yi);

        // Ensure coordinates are positive before indexing
        let ux : u32 = u32(clamp(xi, 0, i32(width) - 1));
        let uy : u32 = u32(clamp(yi, 0, i32(height) - 1));
        let p : u32 = uy * width + ux;
        let base : u32 = p * 4u;
        glider_buffer[base + 0u] = 0.0;
        glider_buffer[base + 1u] = 1.0;
        glider_buffer[base + 2u] = 0.0;
        glider_buffer[base + 3u] = 1.0;
    }

    // Persist agent state
    agent_state_out[base_state + 0u] = x;
    agent_state_out[base_state + 1u] = y;
    agent_state_out[base_state + 2u] = 0.0;
    agent_state_out[base_state + 3u] = 1.0;
    agent_state_out[base_state + 4u] = 0.0;
    agent_state_out[base_state + 5u] = 1.0;
    agent_state_out[base_state + 6u] = 0.0;
    agent_state_out[base_state + 7u] = seed;
}
