// Conv2d feedback loop: temporal feedback where each frame applies one blur+sharpen pass.
// Each frame reads previous output, applies blur then sharpen, outputs result.
// After ~100 frames the effect converges.

const CHANNEL_COUNT : u32 = 4u;

// 5x5 Gaussian blur kernel (matches Python ValueMask.conv2d_blur)
const BLUR_KERNEL : array<array<f32, 5>, 5> = array<array<f32, 5>, 5>(
    array<f32, 5>(1.0, 4.0, 6.0, 4.0, 1.0),
    array<f32, 5>(4.0, 16.0, 24.0, 16.0, 4.0),
    array<f32, 5>(6.0, 24.0, 36.0, 24.0, 6.0),
    array<f32, 5>(4.0, 16.0, 24.0, 16.0, 4.0),
    array<f32, 5>(1.0, 4.0, 6.0, 4.0, 1.0),
);
const BLUR_KERNEL_SUM : f32 = 256.0;

// 3x3 sharpen kernel (matches Python ValueMask.conv2d_sharpen)  
const SHARPEN_KERNEL : array<array<f32, 3>, 3> = array<array<f32, 3>, 3>(
    array<f32, 3>(0.0, -1.0, 0.0),
    array<f32, 3>(-1.0, 5.0, -1.0),
    array<f32, 3>(0.0, -1.0, 0.0),
);

struct ConvFeedbackParams {
    size : vec4<f32>,      // width, height, channels, _pad0
    options : vec4<f32>,   // _unused, alpha, time, speed
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(3) var prev_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : ConvFeedbackParams;

fn as_u32(value : f32) -> u32 {
    return u32(max(round(value), 0.0));
}

fn write_pixel(base_index : u32, color : vec4<f32>) {
    output_buffer[base_index + 0u] = color.x;
    output_buffer[base_index + 1u] = color.y;
    output_buffer[base_index + 2u] = color.z;
    output_buffer[base_index + 3u] = color.w;
}

// Apply 5x5 blur kernel
fn apply_blur(pos : vec2<i32>, width : i32, height : i32, source : texture_2d<f32>) -> vec3<f32> {
    var sum : vec3<f32> = vec3<f32>(0.0);
    
    for (var ky : i32 = -2; ky <= 2; ky = ky + 1) {
        for (var kx : i32 = -2; kx <= 2; kx = kx + 1) {
            let sample_x : i32 = clamp(pos.x + kx, 0, width - 1);
            let sample_y : i32 = clamp(pos.y + ky, 0, height - 1);
            let sample : vec4<f32> = textureLoad(source, vec2<i32>(sample_x, sample_y), 0);
            let weight : f32 = BLUR_KERNEL[ky + 2][kx + 2];
            sum = sum + sample.rgb * weight;
        }
    }
    
    return sum / BLUR_KERNEL_SUM;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let width_u : u32 = as_u32(params.size.x);
    let height_u : u32 = as_u32(params.size.y);
    if (gid.x >= width_u || gid.y >= height_u) {
        return;
    }

    let width : i32 = i32(width_u);
    let height : i32 = i32(height_u);
    let pos : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
    
    // Step 1: Apply blur to previous frame
    let blurred : vec3<f32> = apply_blur(pos, width, height, prev_texture);
    
    // Step 2: Apply sharpen to the blurred result
    // Sharpen needs to sample from the blurred texture, but we don't have it yet
    // We need to do a 2-pass approach: pass 1 = blur, pass 2 = sharpen
    // OR we need to apply sharpen mathematically to the blurred value
    
    // For now, let's apply sharpen by sampling from prev_texture and using blur inline
    // This isn't ideal but works for a single-pass shader
    var sharpen_sum : vec3<f32> = vec3<f32>(0.0);
    
    for (var ky : i32 = -1; ky <= 1; ky = ky + 1) {
        for (var kx : i32 = -1; kx <= 1; kx = kx + 1) {
            let sample_pos : vec2<i32> = vec2<i32>(
                clamp(pos.x + kx, 0, width - 1),
                clamp(pos.y + ky, 0, height - 1)
            );
            
            // Get blurred value at this neighbor position
            let neighbor_blurred : vec3<f32> = apply_blur(sample_pos, width, height, prev_texture);
            
            let weight : f32 = SHARPEN_KERNEL[ky + 1][kx + 1];
            sharpen_sum = sharpen_sum + neighbor_blurred * weight;
        }
    }
    
    let sharpened : vec3<f32> = clamp(sharpen_sum, vec3<f32>(0.0), vec3<f32>(1.0));
    
    // Preserve original alpha from previous frame
    let prev_pixel : vec4<f32> = textureLoad(prev_texture, pos, 0);
    let result : vec4<f32> = vec4<f32>(sharpened, prev_pixel.a);
    
    let pixel_index : u32 = gid.y * width_u + gid.x;
    let base_index : u32 = pixel_index * CHANNEL_COUNT;
    write_pixel(base_index, result);
}
