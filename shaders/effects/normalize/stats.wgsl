// Normalize Pass 1: Calculate min/max statistics across all pixels.
// This pass computes the global minimum and maximum values needed for normalization.

struct NormalizeParams {
    dimensions : vec4<f32>, // width, height, channels, _pad0
    animation : vec4<f32>, // time, speed, _pad1, _pad2
};

const CHANNEL_COUNT : u32 = 4u;
const F32_MAX : f32 = 0x1.fffffep+127;
const F32_MIN : f32 = -0x1.fffffep+127;

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : NormalizeParams;
@group(0) @binding(3) var<storage, read_write> stats_buffer : array<f32>;

fn as_u32(value : f32) -> u32 {
    return u32(max(round(value), 0.0));
}

// Workgroup shared memory for parallel reduction
var<workgroup> workgroup_min : array<f32, 64>;
var<workgroup> workgroup_max : array<f32, 64>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>,
        @builtin(workgroup_id) wid : vec3<u32>) {
    let width : u32 = as_u32(params.dimensions.x);
    let height : u32 = as_u32(params.dimensions.y);
    
    let local_index : u32 = lid.y * 8u + lid.x;
    
    // Initialize local min/max
    var local_min : f32 = F32_MAX;
    var local_max : f32 = F32_MIN;
    
    // Each thread processes its assigned pixel
    if (gid.x < width && gid.y < height) {
        let coords : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
        let texel : vec4<f32> = textureLoad(input_texture, coords, 0);
        
        // Process all channels
        local_min = min(local_min, texel.x);
        local_max = max(local_max, texel.x);
        local_min = min(local_min, texel.y);
        local_max = max(local_max, texel.y);
        local_min = min(local_min, texel.z);
        local_max = max(local_max, texel.z);
        local_min = min(local_min, texel.w);
        local_max = max(local_max, texel.w);
    }
    
    // Store in workgroup shared memory
    workgroup_min[local_index] = local_min;
    workgroup_max[local_index] = local_max;
    
    workgroupBarrier();
    
    // Parallel reduction within workgroup (only first thread)
    if (local_index == 0u) {
        var wg_min : f32 = F32_MAX;
        var wg_max : f32 = F32_MIN;
        
        for (var i : u32 = 0u; i < 64u; i = i + 1u) {
            wg_min = min(wg_min, workgroup_min[i]);
            wg_max = max(wg_max, workgroup_max[i]);
        }
        
        // Atomic operations would be ideal here, but we'll use a simple approach:
        // Each workgroup writes to a unique slot and we'll reduce in pass 2
        let workgroup_index : u32 = wid.y * ((width + 7u) / 8u) + wid.x;
        
        // stats_buffer layout: [min0, max0, min1, max1, ...]
        // Reserve first two slots for final results
        let offset : u32 = 2u + workgroup_index * 2u;
        if (offset + 1u < arrayLength(&stats_buffer)) {
            stats_buffer[offset] = wg_min;
            stats_buffer[offset + 1u] = wg_max;
        }
    }
}
