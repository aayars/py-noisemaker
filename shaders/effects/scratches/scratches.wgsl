// Scratches final combine pass.
// Reuses the worms low-level pipelines to paint scratch trails into `worm_texture`,
// then layers that onto the input image. The Python reference builds 4 scratch layers
// with randomized worm parameters, but since we run worms once per frame, the shader
// just uses the single worm pass result directly.

struct ScratchesParams {
    width : f32,
    height : f32,
    channel_count : f32,
    _pad0 : f32,
    time : f32,
    speed : f32,
    seed : f32,
    _pad1 : f32,
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var worm_texture : texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(3) var<uniform> params : ScratchesParams;
@group(0) @binding(4) var noise_texture : texture_2d<f32>;

const CHANNEL_COUNT : u32 = 4u;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let dims : vec2<u32> = textureDimensions(input_texture, 0);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let coords : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
    let base_color : vec4<f32> = textureLoad(input_texture, coords, 0);
    let worm_mask : vec4<f32> = textureLoad(worm_texture, coords, 0);
    let noise : vec4<f32> = textureLoad(noise_texture, coords, 0);

    // Python: mask -= value.values(...) * 2.0; mask = max(mask, 0.0)
    // Worms create confluent trails, then we subtract noise to punch holes (create scratches)
    let worm_luminance : f32 = (worm_mask.r + worm_mask.g + worm_mask.b) / 3.0;
    let noise_luminance : f32 = (noise.r + noise.g + noise.b) / 3.0;
    let scratched : f32 = max(worm_luminance - noise_luminance * 2.0, 0.0);
    
    // Amplify and composite (Python uses * 8.0)
    let scratch_value : f32 = scratched * 8.0;
    let scratch_rgb : vec3<f32> = max(base_color.rgb, vec3<f32>(scratch_value));

    let width_u : u32 = dims.x;
    let index : u32 = (gid.y * width_u + gid.x) * CHANNEL_COUNT;
    output_buffer[index + 0u] = scratch_rgb.r;
    output_buffer[index + 1u] = scratch_rgb.g;
    output_buffer[index + 2u] = scratch_rgb.b;
    output_buffer[index + 3u] = base_color.a;
}
