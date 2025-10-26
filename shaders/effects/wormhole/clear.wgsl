// Wormhole - Pass 0: Clear output buffer (pixel-parallel)

const CHANNEL_COUNT : u32 = 4u;

struct WormholeParams {
    size : vec4<f32>,
    flow : vec4<f32>,
    motion : vec4<f32>,
    _pad : vec4<f32>,
};

@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : WormholeParams;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let width_u : u32 = u32(params.size.x);
    let height_u : u32 = u32(params.size.y);
    
    if (gid.x >= width_u || gid.y >= height_u) {
        return;
    }
    
    let pixel_idx : u32 = gid.y * width_u + gid.x;
    let base : u32 = pixel_idx * CHANNEL_COUNT;
    
    output_buffer[base + 0u] = 0.0;
    output_buffer[base + 1u] = 0.0;
    output_buffer[base + 2u] = 0.0;
    output_buffer[base + 3u] = 0.0;
}
