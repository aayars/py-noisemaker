// Scratch mask generation stage. Mirrors the CPU scratches implementation by
// subtracting the secondary noise field from the primary mask, amplifying the
// contrast between the worm trails before clamping the result to the [0, 1]
// range.

struct ScratchesMaskParams {
  size : vec2<f32>;
  _padding : vec2<f32>;
};

@group(0) @binding(0) var maskTexture : texture_2d<f32>;
@group(0) @binding(1) var subtractTexture : texture_2d<f32>;
@group(0) @binding(2) var outputTexture : texture_storage_2d<r32float, write>;
@group(0) @binding(3) var<uniform> params : ScratchesMaskParams;

fn asDimension(value : f32) -> u32 {
  return u32(max(round(value), 0.0));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let width = asDimension(params.size.x);
  let height = asDimension(params.size.y);
  if (gid.x >= width || gid.y >= height) {
    return;
  }

  let coords = vec2<i32>(i32(gid.x), i32(gid.y));
  let primary = textureLoad(maskTexture, coords, 0).x;
  let subtract = textureLoad(subtractTexture, coords, 0).x;
  let diff = max(primary - subtract * 2.0, 0.0);

  textureStore(outputTexture, coords, vec4<f32>(diff, 0.0, 0.0, 0.0));
}
