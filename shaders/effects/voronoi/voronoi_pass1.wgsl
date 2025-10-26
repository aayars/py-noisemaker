// Voronoi Pass 1: Compute distances and find min/max
// This pass computes voronoi distances for each pixel and outputs them,
// while also accumulating min/max values for normalization

const PI : f32 = 3.14159265358979323846;
const TAU : f32 = 6.28318530717958647692;

const MAX_POINTS : u32 = 256u;
const EPSILON : f32 = 1e-6;

const PD_RANDOM : i32 = 1000000;
const PD_SQUARE : i32 = 1000001;
const PD_WAFFLE : i32 = 1000002;
const PD_CHESS : i32 = 1000003;
const PD_H_HEX : i32 = 1000010;
const PD_V_HEX : i32 = 1000011;
const PD_SPIRAL : i32 = 1000050;
const PD_CIRCULAR : i32 = 1000100;
const PD_CONCENTRIC : i32 = 1000101;
const PD_ROTATING : i32 = 1000102;

struct VoronoiParams {
    dims : vec4<f32>,                    // width, height, channels, diagram_type
    nth_metric_sdf_alpha : vec4<f32>,    // nth, dist_metric, sdf_sides, alpha
    refract_inverse_xy : vec4<f32>,      // with_refract, inverse, xy_mode, xy_count (unused)
    ridge_refract_time_speed : vec4<f32>, // ridges_hint, refract_y_from_offset, time, speed
    freq_gen_distrib_drift : vec4<f32>,  // point_freq, point_generations, point_distrib, point_drift
    corners_downsample_pad : vec4<f32>,  // point_corners, downsample, pad0, pad1
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> distance_buffer : array<f32>;  // Stores selected distances
@group(0) @binding(2) var<uniform> params : VoronoiParams;
@group(0) @binding(3) var<storage, read_write> minmax_buffer : array<atomic<u32>, 2>;  // [min_as_u32, max_as_u32]

// Include all the helper functions from the main shader (point generation, distance metrics, etc.)
// ... [Copy all helper functions here]

@compute @workgroup_size(8, 8, 1)
fn pass1_compute_distances(@builtin(global_invocation_id) gid : vec3<u32>) {
    let dims : vec2<u32> = textureDimensions(input_texture, 0);
    let width : u32 = select(as_u32(params.dims.x), dims.x, dims.x > 0u);
    let height : u32 = select(as_u32(params.dims.y), dims.y, dims.y > 0u);
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    // Generate point cloud and compute distances (same as main shader)
    // ... [Point cloud generation and distance calculation code]
    
    // Select the Nth distance
    let selected_distance : f32 = select_nth_distance(&sorted_distances, sorted_count, nth_param);
    
    // Write distance to buffer
    let pixel_index : u32 = gid.y * width + gid.x;
    distance_buffer[pixel_index] = selected_distance;
    
    // Atomically update min/max using bitcast trick
    let dist_bits : u32 = bitcast<u32>(selected_distance);
    atomicMin(&minmax_buffer[0], dist_bits);
    atomicMax(&minmax_buffer[1], dist_bits);
}
