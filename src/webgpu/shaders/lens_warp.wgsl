struct LensWarpParams {
  size : vec2<f32>,
};

@group(0) @binding(0) var maskTexture : texture_2d<f32>;
@group(0) @binding(1) var noiseTexture : texture_2d<f32>;
@group(0) @binding(2) var refXTexture : texture_storage_2d<r32float, write>;
@group(0) @binding(3) var refYTexture : texture_storage_2d<r32float, write>;
@group(0) @binding(4) var<uniform> params : LensWarpParams;

const TAU : f32 = 6.283185307179586;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
  let width = u32(params.size.x);
  let height = u32(params.size.y);
  if (global_id.x >= width || global_id.y >= height) {
    return;
  }
  let coord = vec2<i32>(i32(global_id.x), i32(global_id.y));
  let maskVal = textureLoad(maskTexture, coord, 0).x;
  let noiseVal = textureLoad(noiseTexture, coord, 0).x;
  let maskPow = pow(maskVal, 5.0);
  let base = (noiseVal * 2.0 - 1.0) * maskPow;
  let angle = base * TAU;
  let cosVal = clamp(cos(angle) * 0.5 + 0.5, 0.0, 1.0);
  let sinVal = clamp(sin(angle) * 0.5 + 0.5, 0.0, 1.0);
  textureStore(refXTexture, coord, vec4<f32>(cosVal, 0.0, 0.0, 0.0));
  textureStore(refYTexture, coord, vec4<f32>(sinVal, 0.0, 0.0, 0.0));
}
