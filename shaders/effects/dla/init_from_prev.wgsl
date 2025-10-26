// DLA - Init Pass
// Copy previous frame texture to output buffer, preserving the cluster

struct DlaParams {
    size_padding : vec4<f32>,
    density_time : vec4<f32>,
    speed_padding : vec4<f32>,
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : DlaParams;
@group(0) @binding(3) var prev_texture : texture_2d<f32>;
@group(0) @binding(4) var<storage, read_write> glider_buffer : array<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    // Get dimensions from params, NOT from texture (they may differ)
    let width : u32 = u32(params.size_padding.x);
    let height : u32 = u32(params.size_padding.y);
    
    if (gid.x >= width || gid.y >= height) {
        return;
    }
    
    let x : u32 = gid.x;
    let y : u32 = gid.y;
    let p : u32 = y * width + x;
    let base : u32 = p * 4u;
    
    // Copy ONLY cluster pixels (magenta) from prev_texture to output_buffer
    // Filter out everything else (gliders, input texture, etc.)
    let c : vec4<f32> = textureLoad(prev_texture, vec2<u32>(x, y), 0);
    
    // Check for pure magenta: R high, G low, B high
    let is_magenta : bool = (c.r > 0.5) && (c.g < 0.5) && (c.b > 0.5);
    
    if (is_magenta) {
        output_buffer[base + 0u] = 1.0;
        output_buffer[base + 1u] = 0.0;
        output_buffer[base + 2u] = 1.0;
        output_buffer[base + 3u] = 1.0;
    } else {
        output_buffer[base + 0u] = 0.0;
        output_buffer[base + 1u] = 0.0;
        output_buffer[base + 2u] = 0.0;
        output_buffer[base + 3u] = 0.0;
    }

    // Clear glider overlay buffer each frame
    glider_buffer[base + 0u] = 0.0;
    glider_buffer[base + 1u] = 0.0;
    glider_buffer[base + 2u] = 0.0;
    glider_buffer[base + 3u] = 0.0;
}
