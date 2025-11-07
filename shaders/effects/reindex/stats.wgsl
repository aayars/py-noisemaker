// Reindex Pass 1 (Stats): Calculate min/max of value_map components
// Each workgroup computes local min/max which will be reduced in pass 2

struct ReindexParams {
    width_height_channels_displacement : vec4<f32>,
    time_speed_padding : vec4<f32>,
};

const CHANNEL_COUNT : u32 = 4u;
const F32_MAX : f32 = 0x1.fffffep+127;
const F32_MIN : f32 = -0x1.fffffep+127;

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : ReindexParams;
@group(0) @binding(3) var<storage, read_write> stats_buffer : array<f32>;

fn as_u32(value : f32) -> u32 {
    return u32(max(round(value), 0.0));
}

fn clamp01(value : f32) -> f32 {
    return clamp(value, 0.0, 1.0);
}

fn srgb_to_linear(value : f32) -> f32 {
    if (value <= 0.04045) {
        return value / 12.92;
    }
    return pow((value + 0.055) / 1.055, 2.4);
}

fn cube_root(value : f32) -> f32 {
    if (value == 0.0) {
        return 0.0;
    }
    let sign_value : f32 = select(-1.0, 1.0, value >= 0.0);
    return sign_value * pow(abs(value), 1.0 / 3.0);
}

fn oklab_l_component(rgb : vec3<f32>) -> f32 {
    let r_lin : f32 = srgb_to_linear(clamp01(rgb.x));
    let g_lin : f32 = srgb_to_linear(clamp01(rgb.y));
    let b_lin : f32 = srgb_to_linear(clamp01(rgb.z));

    let l : f32 = 0.4121656120 * r_lin + 0.5362752080 * g_lin + 0.0514575653 * b_lin;
    let m : f32 = 0.2118591070 * r_lin + 0.6807189584 * g_lin + 0.1074065790 * b_lin;
    let s : f32 = 0.0883097947 * r_lin + 0.2818474174 * g_lin + 0.6302613616 * b_lin;

    let l_c : f32 = cube_root(l);
    let m_c : f32 = cube_root(m);
    let s_c : f32 = cube_root(s);

    let lightness : f32 = 0.2104542553 * l_c + 0.7936177850 * m_c - 0.0040720468 * s_c;
    return clamp01(lightness);
}

fn value_map_component(texel : vec4<f32>) -> f32 {
    let rgb : vec3<f32> = vec3<f32>(texel.x, texel.y, texel.z);
    return oklab_l_component(rgb);
}

// Workgroup shared memory for parallel reduction
var<workgroup> workgroup_min : array<f32, 64>;
var<workgroup> workgroup_max : array<f32, 64>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>,
        @builtin(workgroup_id) wid : vec3<u32>) {
    let width : u32 = as_u32(params.width_height_channels_displacement.x);
    let height : u32 = as_u32(params.width_height_channels_displacement.y);
    
    let local_index : u32 = lid.y * 8u + lid.x;
    
    // Initialize local min/max
    var local_min : f32 = F32_MAX;
    var local_max : f32 = F32_MIN;
    
    // Each thread processes its assigned pixel
    if (gid.x < width && gid.y < height) {
        let coords : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
        let texel : vec4<f32> = textureLoad(input_texture, coords, 0);
        let reference_value : f32 = value_map_component(texel);
        
        local_min = min(local_min, reference_value);
        local_max = max(local_max, reference_value);
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
        
        // Each workgroup writes to a unique slot
        let workgroup_index : u32 = wid.y * ((width + 7u) / 8u) + wid.x;
        
        // stats_buffer layout: [min, max, min0, max0, min1, max1, ...]
        // Reserve first two slots for final results
        let offset : u32 = 2u + workgroup_index * 2u;
        if (offset + 1u < arrayLength(&stats_buffer)) {
            stats_buffer[offset] = wg_min;
            stats_buffer[offset + 1u] = wg_max;
        }
    }
}
