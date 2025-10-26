// Stray Hair Effect - Single-frame strand generation
//
// Python reference:
//   mask = values(4, value_shape, time=time, speed=speed)
//   mask = worms(mask, behavior=unruly, density=0.0025-0.00375, duration=8-16, 
//                kink=5-50, stride=0.5, alpha=1.0)
//   brightness = values(32, value_shape, time=time, speed=speed)
//   return blend(tensor, brightness * 0.333, mask * 0.666)

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : StrayHairParams;

struct StrayHairParams {
    width : f32,
    height : f32,
    channel_count : f32,
    _pad0 : f32,
    time : f32,
    speed : f32,
    seed : f32,
    _pad1 : f32,
};

const TAU : f32 = 6.28318530718;
const CHANNEL_COUNT : u32 = 4u;

// RNG from hash
fn hash(p : u32) -> u32 {
    var h : u32 = p;
    h = h ^ (h >> 16u);
    h = h * 0x7feb352du;
    h = h ^ (h >> 15u);
    h = h * 0x846ca68bu;
    h = h ^ (h >> 16u);
    return h;
}

fn random_from_seed(seed : u32) -> f32 {
    return f32(hash(seed)) / 4294967296.0;
}

// Value noise (from multires/texture shaders)
fn wrap_unit(value : f32, frequency : f32) -> f32 {
    let wrapped : f32 = value % frequency;
    return select(wrapped, wrapped + frequency, wrapped < 0.0);
}

fn wrap_cell_2d(cell : vec2<i32>, frequency : f32) -> vec2<i32> {
    let freq_i : i32 = i32(frequency);
    var wrapped_x : i32 = cell.x % freq_i;
    var wrapped_y : i32 = cell.y % freq_i;
    
    if (wrapped_x < 0) { wrapped_x += freq_i; }
    if (wrapped_y < 0) { wrapped_y += freq_i; }
    
    return vec2<i32>(wrapped_x, wrapped_y);
}

fn random_from_cell_3d_wrapped(cell : vec3<i32>, frequency : f32) -> f32 {
    let wrapped_xy : vec2<i32> = wrap_cell_2d(cell.xy, frequency);
    let freq_i : i32 = i32(frequency);
    var wrapped_z : i32 = cell.z % freq_i;
    if (wrapped_z < 0) { wrapped_z += freq_i; }
    
    let wrapped : vec3<i32> = vec3<i32>(wrapped_xy, wrapped_z);
    let p : u32 = u32(wrapped.x) + u32(wrapped.y) * 73856093u + u32(wrapped.z) * 19349663u;
    return random_from_seed(p);
}

fn sample_value_noise(pos : vec2<f32>, z_time : f32, frequency : f32) -> f32 {
    let scaled : vec2<f32> = pos * frequency;
    let cell_base : vec2<i32> = vec2<i32>(i32(floor(scaled.x)), i32(floor(scaled.y)));
    let local : vec2<f32> = fract(scaled);
    let fade_xy : vec2<f32> = local * local * (3.0 - 2.0 * local);
    
    let z_cell_base : i32 = i32(floor(z_time));
    let z_local : f32 = fract(z_time);
    let fade_z : f32 = z_local * z_local * (3.0 - 2.0 * z_local);
    
    var sum : f32 = 0.0;
    for (var dz : i32 = 0; dz < 2; dz++) {
        let z_cell : i32 = z_cell_base + dz;
        var layer_sum : f32 = 0.0;
        
        for (var dy : i32 = 0; dy < 2; dy++) {
            for (var dx : i32 = 0; dx < 2; dx++) {
                let cell : vec2<i32> = cell_base + vec2<i32>(dx, dy);
                let cell_3d : vec3<i32> = vec3<i32>(cell, z_cell);
                let corner_value : f32 = random_from_cell_3d_wrapped(cell_3d, frequency);
                
                let weight_x : f32 = select(1.0 - fade_xy.x, fade_xy.x, dx == 1);
                let weight_y : f32 = select(1.0 - fade_xy.y, fade_xy.y, dy == 1);
                layer_sum += corner_value * weight_x * weight_y;
            }
        }
        
    let weight_z : f32 = select(1.0 - fade_z, fade_z, dz == 1);
        sum += layer_sum * weight_z;
    }
    
    return sum;
}

// Agent simulation - each invocation simulates one complete worm path
// Returns: contribution to the mask at this pixel from all simulated agents
fn simulate_agents_at_pixel(pixel_pos : vec2<u32>, time_val : f32, width : f32, height : f32) -> f32 {
    // Worm parameters - REDUCED for GPU performance
    let density : f32 = 0.0005;  // Was 0.003, now 6x less for performance
    let duration : f32 = 8.0;    // Was 12.0, shorter strands
    let kink : f32 = 27.5;       // 5-50
    let base_stride : f32 = 0.5;
    let stride_deviation : f32 = 0.25;
    
    let agent_count : u32 = max(1u, u32(max(width, height) * density));
    let iterations : u32 = max(1u, u32(sqrt(min(width, height)) * duration));
    
    var mask_contribution : f32 = 0.0;
    
    // Simulate each agent and check if it deposits at this pixel
    for (var agent_idx : u32 = 0u; agent_idx < agent_count; agent_idx++) {
        // Initialize agent with deterministic randomness based on seed and agent index (NOT time)
        let seed_base : u32 = u32(params.seed * 1000.0) + agent_idx * 12345u;
        
        // Random starting position
        var agent_x : f32 = random_from_seed(seed_base * 2u) * (width - 1.0);
        var agent_y : f32 = random_from_seed(seed_base * 2u + 1u) * (height - 1.0);
        
        // Unruly behavior: obedient + small random deviation
        let base_rot : f32 = random_from_seed(seed_base * 2u + 2u) * TAU;
        var agent_rot : f32 = base_rot + (random_from_seed(seed_base * 2u + 3u) - 0.5) * 0.25;
        
        // Stride with deviation, normalized by image size
        let stride_var : f32 = base_stride * (1.0 + (random_from_seed(seed_base * 2u + 4u) - 0.5) * 2.0 * stride_deviation);
        let agent_stride : f32 = max(0.1, stride_var * (max(width, height) / 1024.0));
        
        // Bright color
        let agent_color : vec3<f32> = vec3<f32>(
            0.5 + random_from_seed(seed_base * 2u + 5u) * 0.5,
            0.5 + random_from_seed(seed_base * 2u + 6u) * 0.5,
            0.5 + random_from_seed(seed_base * 2u + 7u) * 0.5,
        );
        
        // Simulate agent movement through iterations
        for (var i : u32 = 0u; i < iterations; i++) {
            // Sample flow field at agent position (seed affects flow field)
            let norm_pos : vec2<f32> = vec2<f32>(agent_x / width, agent_y / height);
            let flow_angle : f32 = sample_value_noise(norm_pos, time_val + params.seed, 4.0) * TAU;
            
            // Update rotation based on flow field (kink factor)
            agent_rot = flow_angle * kink;
            
            // Move agent
            agent_x += cos(agent_rot) * agent_stride;
            agent_y += sin(agent_rot) * agent_stride;
            
            // Wrap around
            agent_x = agent_x % width;
            agent_y = agent_y % height;
            if (agent_x < 0.0) { agent_x += width; }
            if (agent_y < 0.0) { agent_y += height; }
            
            // Check if agent is at this exact pixel (1px wide strands)
            let agent_px : u32 = u32(clamp(agent_x, 0.0, width - 1.0));
            let agent_py : u32 = u32(clamp(agent_y, 0.0, height - 1.0));
            
            if (agent_px == pixel_pos.x && agent_py == pixel_pos.y) {
                // Linear gradient exposure: 0 -> 1 -> 0
                let t : f32 = f32(i) / f32(iterations - 1u);
                let exposure : f32 = 1.0 - abs(1.0 - t * 2.0);
                
                // Add contribution (average of RGB for grayscale mask)
                let brightness : f32 = (agent_color.r + agent_color.g + agent_color.b) / 3.0;
                mask_contribution += brightness * exposure;
            }
        }
    }
    
    return clamp(mask_contribution, 0.0, 1.0);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let width_u : u32 = u32(params.width);
    let height_u : u32 = u32(params.height);
    
    if (gid.x >= width_u || gid.y >= height_u) {
        return;
    }
    
    let width_f : f32 = params.width;
    let height_f : f32 = params.height;
    let time_val : f32 = params.time * params.speed;
    
    // Read input texture
    let coords : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
    let input_color : vec4<f32> = textureLoad(input_texture, coords, 0);
    
    // Normalized position [0, 1]
    let norm_pos : vec2<f32> = vec2<f32>(f32(gid.x) / width_f, f32(gid.y) / height_f);
    
    // Generate strand mask using agent simulation
    let strand_mask : f32 = simulate_agents_at_pixel(gid.xy, time_val, width_f, height_f);
    
    // Sample brightness noise at frequency 32
    let brightness : f32 = sample_value_noise(norm_pos, time_val + params.seed, 32.0);
    
    // Python: blend(tensor, brightness * 0.333, mask * 0.666)
    // This means: output = input * (1 - weight) + target * weight
    // where target = brightness and weight = mask * 0.666
    let blend_weight : f32 = strand_mask * 0.666;
    let brightness_target : vec3<f32> = vec3<f32>(brightness * 0.333);
    let result_rgb : vec3<f32> = mix(input_color.rgb, brightness_target, blend_weight);
    
    // Write to output buffer (preserve alpha)
    let pixel_idx : u32 = gid.y * width_u + gid.x;
    let base : u32 = pixel_idx * CHANNEL_COUNT;
    
    output_buffer[base + 0u] = clamp(result_rgb.r, 0.0, 1.0);
    output_buffer[base + 1u] = clamp(result_rgb.g, 0.0, 1.0);
    output_buffer[base + 2u] = clamp(result_rgb.b, 0.0, 1.0);
    output_buffer[base + 3u] = input_color.a;
}
