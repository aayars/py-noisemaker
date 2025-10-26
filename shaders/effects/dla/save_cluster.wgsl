// DLA - Save Cluster Pass
// After final_blend, extract and save ONLY the cluster state for next frame
// This undoes the blending so output_buffer contains only magenta cluster

struct DlaParams {
    size_padding : vec4<f32>,
    density_time : vec4<f32>,
    speed_padding : vec4<f32>,
};

@group(0) @binding(0) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(1) var<uniform> params : DlaParams;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let width : u32 = u32(params.size_padding.x);
    let height : u32 = u32(params.size_padding.y);
    
    if (gid.x >= width || gid.y >= height) {
        return;
    }
    
    let idx : u32 = gid.y * width + gid.x;
    let base : u32 = idx * 4u;
    
    // Read current pixel (which has cluster + gliders + input blended)
    let r : f32 = output_buffer[base + 0u];
    let g : f32 = output_buffer[base + 1u];
    let b : f32 = output_buffer[base + 2u];
    
    // Extract ONLY the magenta cluster component
    // If pixel is magenta-ish, keep it; otherwise clear it
    if (r > 0.5 && g < 0.5 && b > 0.5) {
        // This is a cluster pixel, keep it pure magenta
        output_buffer[base + 0u] = 1.0;
        output_buffer[base + 1u] = 0.0;
        output_buffer[base + 2u] = 1.0;
        output_buffer[base + 3u] = 1.0;
    } else {
        // Not a cluster pixel, clear it
        output_buffer[base + 0u] = 0.0;
        output_buffer[base + 1u] = 0.0;
        output_buffer[base + 2u] = 0.0;
        output_buffer[base + 3u] = 0.0;
    }
}
