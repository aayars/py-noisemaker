// Reindex Pass 2 (Reduce): Reduce workgroup statistics to final global min/max

struct ReindexParams {
    width_height_channels_displacement : vec4<f32>,
    time_speed_padding : vec4<f32>,
};

const F32_MAX : f32 = 0x1.fffffep+127;
const F32_MIN : f32 = -0x1.fffffep+127;

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : ReindexParams;
@group(0) @binding(3) var<storage, read_write> stats_buffer : array<f32>;

fn as_u32(value : f32) -> u32 {
    return u32(max(round(value), 0.0));
}

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    // Only thread (0,0,0) does the reduction
    if (gid.x != 0u || gid.y != 0u || gid.z != 0u) {
        return;
    }

    let width : u32 = as_u32(params.width_height_channels_displacement.x);
    let height : u32 = as_u32(params.width_height_channels_displacement.y);
    let num_workgroups : u32 = ((width + 7u) / 8u) * ((height + 7u) / 8u);
    
    var global_min : f32 = F32_MAX;
    var global_max : f32 = F32_MIN;
    
    // Reduce all workgroup results
    for (var i : u32 = 0u; i < num_workgroups; i = i + 1u) {
        let offset : u32 = 2u + i * 2u;
        if (offset + 1u < arrayLength(&stats_buffer)) {
            global_min = min(global_min, stats_buffer[offset]);
            global_max = max(global_max, stats_buffer[offset + 1u]);
        }
    }
    
    // Store final results in first two slots
    stats_buffer[0] = global_min;
    stats_buffer[1] = global_max;
}
