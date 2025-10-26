// Diffusion-limited aggregation effect.
// Mirrors the behavior of noisemaker.effects.dla with WebGPU-friendly data structures.

const SMALL_OFFSETS : array<i32, 3> = array<i32, 3>(-1, 0, 1);
const EXPANDED_RANGE : i32 = 8;
const EXPANDED_WIDTH : i32 = EXPANDED_RANGE * 2 + 1;

// Keep MAX_OUTPUT_VALUES very small to reduce stack pressure on Metal.
const MAX_IMAGE_WIDTH : u32 = 32u;
const MAX_IMAGE_HEIGHT : u32 = 32u;
const MAX_CHANNELS : u32 = 4u;
// Keep MAX_CELLS small to reduce large temporary arrays
const MAX_HALF_WIDTH : u32 = 32u;
const MAX_HALF_HEIGHT : u32 = 32u;
const MAX_CELLS : u32 = MAX_HALF_WIDTH * MAX_HALF_HEIGHT; // 1024
const MAX_WALKERS : u32 = MAX_CELLS;
const MAX_OUTPUT_VALUES : u32 = MAX_IMAGE_WIDTH * MAX_IMAGE_HEIGHT * MAX_CHANNELS; // 127*127*4 = 64516

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
    size_padding : vec4<f32>,      // (width, height, channels, padding)
    density_time : vec4<f32>,      // (seed_density, density, alpha, time)
    speed_padding : vec4<f32>,     // (speed, unused, unused, unused)
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : DlaParams;
// Previous frame accumulation (temporal coherence)
@group(0) @binding(3) var prev_texture : texture_2d<f32>;
// Persistent agents (walkers)
@group(0) @binding(4) var<storage, read> agent_state_in : array<f32>;
@group(0) @binding(5) var<storage, read_write> agent_state_out : array<f32>;

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

// Very cheap deterministic pseudo RNG on [0,1). Stores/returns a seed in [0,1).
fn next_seed(seed : f32) -> f32 {
    // Weyl sequence step (irrational increment)
    return fract(seed * 1.3247179572447458 + 0.123456789);
}

fn rand01(seed : ptr<function, f32>) -> f32 {
    let r : f32 = *seed;
    *seed = next_seed(*seed);
    return r;
}

fn clamp_dimension(value : f32, limit : u32) -> u32 {
    let clamped : f32 = max(value, 0.0);
    let floored : u32 = u32(floor(clamped));
    if (floored > limit) {
        return limit;
    }
    return floored;
}

fn sanitize_dimension(value : f32, maximum : u32, fallback : u32) -> u32 {
    let dimension : u32 = clamp_dimension(value, maximum);
    if (dimension == 0u) {
        return fallback;
    }
    return dimension;
}

fn sanitize_channel_count(value : f32) -> u32 {
    let clamped : i32 = i32(round(value));
    if (clamped < 1) {
        return 1u;
    }
    if (clamped > i32(MAX_CHANNELS)) {
        return MAX_CHANNELS;
    }
    return u32(clamped);
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

fn wrap_index(value : i32, limit : u32) -> u32 {
    if (limit == 0u) {
        return 0u;
    }
    let wrapped : i32 = wrap_coord(value, i32(limit));
    return u32(wrapped);
}

fn mix_seed(a : u32, b : u32, c : u32, time_seed : u32, speed_seed : u32) -> u32 {
    var state : u32 = ((a * 747796405u) ^ (b * 2891336453u) ^ (c * 277803737u) ^ time_seed ^ speed_seed);
    state ^= state >> 17;
    state *= 0xED5AD4BBu;
    state ^= state >> 11;
    state *= 0xAC4C1B51u;
    state ^= state >> 15;
    state *= 0x31848BAFu;
    state ^= state >> 14;
    return state;
}

fn random_value(a : u32, b : u32, c : u32, time_seed : u32, speed_seed : u32) -> f32 {
    let bits : u32 = mix_seed(a, b, c, time_seed, speed_seed);
    return f32(bits) * (1.0 / 4294967295.0);
}

fn select_small_offset(value : f32) -> i32 {
    let scaled : f32 = clamp(value, 0.0, 0.9999999) * 3.0;
    let index : u32 = u32(floor(scaled));
    return SMALL_OFFSETS[index];
}

fn select_expanded_offset(value : f32) -> i32 {
    let scaled : f32 = clamp(value, 0.0, 0.9999999) * f32(EXPANDED_WIDTH);
    let index : i32 = i32(floor(scaled)) - EXPANDED_RANGE;
    return index;
}

fn kernel_value(index : u32) -> f32 {
    return BLUR_KERNEL[index];
}

fn clamp01(value : f32) -> f32 {
    return clamp(value, 0.0, 1.0);
}

fn initialize_seed(
    seed_index : u32,
    half_width : u32,
    half_height : u32,
    time_seed : u32,
    speed_seed : u32,
) -> vec2<u32> {
    let rand_y : f32 = random_value(seed_index, 0u, 1u, time_seed, speed_seed);
    let rand_x : f32 = random_value(seed_index, 1u, 0u, time_seed, speed_seed);
    var node_y : u32 = u32(floor(rand_y * f32(half_height)));
    var node_x : u32 = u32(floor(rand_x * f32(half_width)));
    if (node_y >= half_height) {
        node_y = half_height - 1u;
    }
    if (node_x >= half_width) {
        node_x = half_width - 1u;
    }
    return vec2<u32>(node_x, node_y);
}

fn append_walker(
    walkers : ptr<function, array<vec2<i32>, MAX_WALKERS>>,
    walker_active : ptr<function, array<u32, MAX_WALKERS>>,
    walker_length : ptr<function, u32>,
    position : vec2<i32>,
) {
    if (*walker_length >= MAX_WALKERS) {
        return;
    }
    (*walkers)[*walker_length] = position;
    (*walker_active)[*walker_length] = 1u;
    *walker_length = *walker_length + 1u;
}

fn append_cluster_node(
    cluster_list : ptr<function, array<u32, MAX_CELLS>>,
    cluster_length : ptr<function, u32>,
    cell_index : u32,
) {
    if (*cluster_length >= MAX_CELLS) {
        return;
    }
    (*cluster_list)[*cluster_length] = cell_index;
    *cluster_length = *cluster_length + 1u;
}

fn mark_neighborhood(
    dst : ptr<function, array<u32, MAX_CELLS>>,
    half_width : u32,
    half_height : u32,
    node_x : u32,
    node_y : u32,
    range_radius : i32,
    should_wrap : bool,
) {
    let min_offset : i32 = -range_radius;
    let max_offset : i32 = range_radius;
    for (var dy : i32 = min_offset; dy <= max_offset; dy = dy + 1) {
        var sample_y : i32 = i32(node_y) + dy;
        if (should_wrap) {
            sample_y = wrap_coord(sample_y, i32(half_height));
        }
        if (!should_wrap && (sample_y < 0 || sample_y >= i32(half_height))) {
            continue;
        }
        let wrapped_y : u32 = u32(sample_y);
        for (var dx : i32 = min_offset; dx <= max_offset; dx = dx + 1) {
            var sample_x : i32 = i32(node_x) + dx;
            if (should_wrap) {
                sample_x = wrap_coord(sample_x, i32(half_width));
            }
            if (!should_wrap && (sample_x < 0 || sample_x >= i32(half_width))) {
                continue;
            }
            let wrapped_x : u32 = u32(sample_x);
            let idx : u32 = wrapped_y * half_width + wrapped_x;
            (*dst)[idx] = 1u;
        }
    }
}

fn normalized_value(value : f32, minimum : f32, inv_range : f32) -> f32 {
    return clamp01((value - minimum) * inv_range);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    // Use actual input texture dimensions; avoid large local arrays
    let dims : vec2<u32> = textureDimensions(input_texture, 0);
    let width : u32 = dims.x;
    let height : u32 = dims.y;
    if (width == 0u || height == 0u) { return; }

    let channels : u32 = 4u; // RGBA pipeline
    let pixel_count : u32 = width * height;
    let total_values : u32 = pixel_count * channels;
    if (arrayLength(&output_buffer) < total_values) { return; }

    let alpha : f32 = clamp01(params.density_time.z);
    let seed_density : f32 = max(params.density_time.x, 0.0);

    // Parallelize prev_texture copy: each thread handles one pixel
    if (gid.x < width && gid.y < height) {
        let pixel_idx : u32 = gid.y * width + gid.x;
        let base : u32 = pixel_idx * 4u;
        let pcol : vec4<f32> = textureLoad(prev_texture, vec2<i32>(i32(gid.x), i32(gid.y)), 0);
        output_buffer[base + 0u] = pcol.x;
        output_buffer[base + 1u] = pcol.y;
        output_buffer[base + 2u] = pcol.z;
        output_buffer[base + 3u] = pcol.w;
    }
    
    // Only thread 0 handles seed initialization and agent processing
    if (gid.x == 0u && gid.y == 0u) {

    // Quick occupancy check (any non-black in prev_texture?)
    var occupied : u32 = 0u;
    for (var y1 : u32 = 0u; y1 < height; y1 = y1 + 1u) {
        for (var x1 : u32 = 0u; x1 < width; x1 = x1 + 1u) {
            let c : vec4<f32> = textureLoad(prev_texture, vec2<i32>(i32(x1), i32(y1)), 0);
            if (luminance(vec3<f32>(c.x, c.y, c.z)) > 0.01) {
                occupied = 1u; break;
            }
        }
        if (occupied == 1u) { break; }
    }

    // If empty, initialize a few seeds
    if (occupied == 0u) {
        let min_dim : f32 = f32(min(width, height));
        let seeds : u32 = max(1u, u32(floor(min_dim * seed_density * 0.25)));
        var s : f32 = 0.37; // deterministic seed
        for (var i : u32 = 0u; i < seeds; i = i + 1u) {
            let rx : u32 = u32(floor(rand01(&s) * f32(width)));
            let ry : u32 = u32(floor(rand01(&s) * f32(height)));
            let cx : i32 = i32(rx);
            let cy : i32 = i32(ry);
            // Stamp a small 5x5 kernel
            for (var ky : u32 = 0u; ky < BLUR_KERNEL_SIZE; ky = ky + 1u) {
                for (var kx : u32 = 0u; kx < BLUR_KERNEL_SIZE; kx = kx + 1u) {
                    let oy : i32 = i32(ky) - BLUR_KERNEL_RADIUS;
                    let ox : i32 = i32(kx) - BLUR_KERNEL_RADIUS;
                    let yy : u32 = u32(wrap_int(cy + oy, i32(height)));
                    let xx : u32 = u32(wrap_int(cx + ox, i32(width)));
                    let p : u32 = yy * width + xx;
                    let base : u32 = p * 4u;
                    let w : f32 = kernel_value(ky * BLUR_KERNEL_SIZE + kx) * alpha;
                    output_buffer[base + 0u] = clamp01(output_buffer[base + 0u] + w);
                    output_buffer[base + 1u] = clamp01(output_buffer[base + 1u] + w);
                    output_buffer[base + 2u] = clamp01(output_buffer[base + 2u] + w);
                    output_buffer[base + 3u] = clamp01(output_buffer[base + 3u] + w);
                }
            }
        }
    }

    // Agents: 8 floats per agent
    let floats_len : u32 = arrayLength(&agent_state_in);
    if (floats_len < 8u) { return; }
    let agent_count : u32 = floats_len / 8u;

    // Step each agent once per frame (one iteration of Python loop)
    for (var ai : u32 = 0u; ai < agent_count; ai = ai + 1u) {
        let base_state : u32 = ai * 8u;
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
                let c : vec4<f32> = textureLoad(prev_texture, vec2<i32>(sx, sy), 0);
                if (luminance(vec3<f32>(c.x, c.y, c.z)) > 0.05) {
                    in_expanded = true;
                    break;
                }
            }
            if (in_expanded) { break; }
        }

        // Movement step size depends on proximity to cluster
        var dx : i32 = 0;
        var dy : i32 = 0;
        
        if (in_expanded) {
            // Near cluster: small random walk from 3x3 grid
            // Python: offsets[random] for both x and y independently
            let rx : f32 = rand01(&seed) * 3.0;
            let ry : f32 = rand01(&seed) * 3.0;
            dx = SMALL_OFFSETS[u32(floor(rx))];
            dy = SMALL_OFFSETS[u32(floor(ry))];
        } else {
            // Far from cluster: large random walk from 17x17 grid
            // Python: expanded_offsets[random] for both x and y independently
            let rx : f32 = rand01(&seed) * f32(EXPANDED_WIDTH);
            let ry : f32 = rand01(&seed) * f32(EXPANDED_WIDTH);
            dx = i32(floor(rx)) - EXPANDED_RANGE;
            dy = i32(floor(ry)) - EXPANDED_RANGE;
        }

        var xi : i32 = wrap_int(current_xi + dx, i32(width));
        var yi : i32 = wrap_int(current_yi + dy, i32(height));

        // Stick if near existing cluster in prev_texture
        var should_stick : bool = false;
        for (var oy : i32 = -1; oy <= 1; oy = oy + 1) {
            for (var ox : i32 = -1; ox <= 1; ox = ox + 1) {
                let sx : i32 = wrap_int(xi + ox, i32(width));
                let sy : i32 = wrap_int(yi + oy, i32(height));
                let c : vec4<f32> = textureLoad(prev_texture, vec2<i32>(sx, sy), 0);
                if (luminance(vec3<f32>(c.x, c.y, c.z)) > 0.05) { should_stick = true; break; }
            }
            if (should_stick) { break; }
        }

        if (should_stick) {
            // Stamp kernel and respawn
            for (var ky : u32 = 0u; ky < BLUR_KERNEL_SIZE; ky = ky + 1u) {
                for (var kx : u32 = 0u; kx < BLUR_KERNEL_SIZE; kx = kx + 1u) {
                    let oy2 : i32 = i32(ky) - BLUR_KERNEL_RADIUS;
                    let ox2 : i32 = i32(kx) - BLUR_KERNEL_RADIUS;
                    let yy : u32 = u32(wrap_int(yi + oy2, i32(height)));
                    let xx : u32 = u32(wrap_int(xi + ox2, i32(width)));
                    let p : u32 = yy * width + xx;
                    let base : u32 = p * 4u;
                    let w : f32 = kernel_value(ky * BLUR_KERNEL_SIZE + kx) * alpha;
                    output_buffer[base + 0u] = clamp01(output_buffer[base + 0u] + w);
                    output_buffer[base + 1u] = clamp01(output_buffer[base + 1u] + w);
                    output_buffer[base + 2u] = clamp01(output_buffer[base + 2u] + w);
                    output_buffer[base + 3u] = clamp01(output_buffer[base + 3u] + w);
                }
            }
            // Respawn at random location
            let rx : u32 = u32(floor(rand01(&seed) * f32(width)));
            let ry : u32 = u32(floor(rand01(&seed) * f32(height)));
            x = f32(rx);
            y = f32(ry);
        } else {
            // Continue walking
            x = f32(xi);
            y = f32(yi);
        }

        // Persist agent state
        agent_state_out[base_state + 0u] = x;
        agent_state_out[base_state + 1u] = y;
        agent_state_out[base_state + 2u] = 0.0;
        agent_state_out[base_state + 3u] = 1.0;
        agent_state_out[base_state + 4u] = 1.0;
        agent_state_out[base_state + 5u] = 1.0;
        agent_state_out[base_state + 6u] = 1.0;
        agent_state_out[base_state + 7u] = seed;
    }
    } // End of thread 0 seed/agent processing block
    
    // Final blend (all threads participate)
    // Python: return value.blend(tensor, out * tensor, alpha)
    if (gid.x < width && gid.y < height) {
        let pixel_idx : u32 = gid.y * width + gid.x;
        let base : u32 = pixel_idx * 4u;
        
        let input_color : vec4<f32> = textureLoad(input_texture, vec2<i32>(i32(gid.x), i32(gid.y)), 0);
        let trail_color : vec4<f32> = vec4<f32>(
            output_buffer[base + 0u],
            output_buffer[base + 1u],
            output_buffer[base + 2u],
            output_buffer[base + 3u],
        );
        
        // DLA uses: blend(tensor, out * tensor, alpha)
        let multiplied : vec4<f32> = trail_color * input_color;
        let blended : vec4<f32> = input_color * (1.0 - alpha) + multiplied * alpha;
        
        output_buffer[base + 0u] = clamp(blended.x, 0.0, 1.0);
        output_buffer[base + 1u] = clamp(blended.y, 0.0, 1.0);
        output_buffer[base + 2u] = clamp(blended.z, 0.0, 1.0);
        output_buffer[base + 3u] = clamp(blended.w, 0.0, 1.0);
    }
}

