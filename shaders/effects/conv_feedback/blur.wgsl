// Conv2d feedback - Blur pass
// Applies 5x5 Gaussian blur to the current input texture, outputs to intermediate buffer

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

struct ConvFeedbackParams {
    size : vec4<f32>,      // width, height, channels, bias
    options : vec4<f32>,   // alpha, time, speed, _pad0
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> blurred_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : ConvFeedbackParams;

fn as_u32(value : f32) -> u32 {
    return u32(max(round(value), 0.0));
}

fn write_pixel(base_index : u32, color : vec4<f32>) {
    blurred_buffer[base_index + 0u] = color.x;
    blurred_buffer[base_index + 1u] = color.y;
    blurred_buffer[base_index + 2u] = color.z;
    blurred_buffer[base_index + 3u] = color.w;
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
    
    // Apply 5x5 blur
    var sum : vec3<f32> = vec3<f32>(0.0);
    
    for (var ky : i32 = -2; ky <= 2; ky = ky + 1) {
        for (var kx : i32 = -2; kx <= 2; kx = kx + 1) {
            let sample_x : i32 = clamp(pos.x + kx, 0, width - 1);
            let sample_y : i32 = clamp(pos.y + ky, 0, height - 1);
            let sample : vec4<f32> = textureLoad(input_texture, vec2<i32>(sample_x, sample_y), 0);
            let weight : f32 = BLUR_KERNEL[ky + 2][kx + 2];
            sum = sum + sample.rgb * weight;
        }
    }
    
    let blurred : vec3<f32> = sum / BLUR_KERNEL_SUM;
    
    // Preserve alpha from input frame
    let source_pixel : vec4<f32> = textureLoad(input_texture, pos, 0);
    let result : vec4<f32> = vec4<f32>(blurred, source_pixel.a);
    
    let pixel_index : u32 = gid.y * width_u + gid.x;
    let base_index : u32 = pixel_index * CHANNEL_COUNT;
    write_pixel(base_index, result);
}
