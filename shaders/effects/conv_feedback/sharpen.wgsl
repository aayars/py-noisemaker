// Conv2d feedback - Sharpen pass
// Applies 3x3 sharpen kernel to blurred buffer, outputs to final output

const CHANNEL_COUNT : u32 = 4u;

// 3x3 sharpen kernel (matches Python ValueMask.conv2d_sharpen)
const SHARPEN_KERNEL : array<array<f32, 3>, 3> = array<array<f32, 3>, 3>(
    array<f32, 3>(0.0, -1.0, 0.0),
    array<f32, 3>(-1.0, 5.0, -1.0),
    array<f32, 3>(0.0, -1.0, 0.0),
);

struct ConvFeedbackParams {
    size : vec4<f32>,      // width, height, channels, bias
    options : vec4<f32>,   // alpha, time, speed, _pad0
};

@group(0) @binding(0) var<storage, read> blurred_buffer : array<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : ConvFeedbackParams;

fn as_u32(value : f32) -> u32 {
    return u32(max(round(value), 0.0));
}

fn read_pixel(base_index : u32) -> vec4<f32> {
    return vec4<f32>(
        blurred_buffer[base_index + 0u],
        blurred_buffer[base_index + 1u],
        blurred_buffer[base_index + 2u],
        blurred_buffer[base_index + 3u],
    );
}

fn write_pixel(base_index : u32, color : vec4<f32>) {
    output_buffer[base_index + 0u] = color.x;
    output_buffer[base_index + 1u] = color.y;
    output_buffer[base_index + 2u] = color.z;
    output_buffer[base_index + 3u] = color.w;
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
    
    // Apply 3x3 sharpen to blurred image
    var sum : vec3<f32> = vec3<f32>(0.0);
    
    for (var ky : i32 = -1; ky <= 1; ky = ky + 1) {
        for (var kx : i32 = -1; kx <= 1; kx = kx + 1) {
            let sample_x : i32 = clamp(pos.x + kx, 0, width - 1);
            let sample_y : i32 = clamp(pos.y + ky, 0, height - 1);
            let sample_index : u32 = u32(sample_y) * width_u + u32(sample_x);
            let sample_base : u32 = sample_index * CHANNEL_COUNT;
            let sample : vec4<f32> = read_pixel(sample_base);
            let weight : f32 = SHARPEN_KERNEL[ky + 1][kx + 1];
            sum = sum + sample.rgb * weight;
        }
    }
    
    let sharpened : vec3<f32> = clamp(sum, vec3<f32>(0.0), vec3<f32>(1.0));
    
    // Preserve alpha
    let pixel_index : u32 = gid.y * width_u + gid.x;
    let base_index : u32 = pixel_index * CHANNEL_COUNT;
    let current_pixel : vec4<f32> = read_pixel(base_index);
    let result : vec4<f32> = vec4<f32>(sharpened, current_pixel.a);
    
    write_pixel(base_index, result);
}
