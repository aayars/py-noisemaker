// Generates the radial mask used by the vaseline effect. The mask mirrors the
// centre-weighted blend performed by the CPU implementation by computing a
// Chebyshev distance field in normalized UV space using pixel centres. The
// result is a single-channel texture in the range [0, 1] that weights the blend
// between the original tensor and the blurred copy produced by the blur stage.

struct MaskParams {
  size : vec4<f32>; // width, height, unused, unused
};

@group(0) @binding(0) var maskTexture : texture_storage_2d<r32float, write>;
@group(0) @binding(1) var<uniform> params : MaskParams;

fn clamp01(value : f32) -> f32 {
  return clamp(value, 0.0, 1.0);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let width : u32 = u32(max(params.size.x, 0.0));
  let height : u32 = u32(max(params.size.y, 0.0));
  if (gid.x >= width || gid.y >= height) {
    return;
  }

  let widthF : f32 = max(params.size.x, 1.0);
  let heightF : f32 = max(params.size.y, 1.0);
  let u : f32 = (f32(gid.x) + 0.5) / widthF;
  let v : f32 = (f32(gid.y) + 0.5) / heightF;
  let dx : f32 = abs(u - 0.5);
  let dy : f32 = abs(v - 0.5);
  let wrappedX : f32 = min(dx, 0.5 - dx) * 2.0;
  let wrappedY : f32 = min(dy, 0.5 - dy) * 2.0;
  let dist : f32 = clamp01(max(wrappedX, wrappedY));

  textureStore(
    maskTexture,
    vec2<i32>(i32(gid.x), i32(gid.y)),
    vec4<f32>(dist, 0.0, 0.0, 0.0),
  );
}
