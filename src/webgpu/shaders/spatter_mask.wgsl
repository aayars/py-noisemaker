// Generates the base mask for the spatter effect. The shader mirrors the CPU
// implementation by combining three FBM-style noise fields (smear, sp1, sp2)
// and subtracting a removal field before clamping the result to the [0, 1]
// range. The CPU path then normalizes the mask, so matching the arithmetic here
// ensures parity between the two implementations.

struct SpatterMaskParams {
  size : vec2<f32>;
  _padding : vec2<f32>;
};

@group(0) @binding(0) var smearTexture : texture_2d<f32>;
@group(0) @binding(1) var sp1Texture : texture_2d<f32>;
@group(0) @binding(2) var sp2Texture : texture_2d<f32>;
@group(0) @binding(3) var removerTexture : texture_2d<f32>;
@group(0) @binding(4) var outputTexture : texture_storage_2d<r32float, write>;
@group(0) @binding(5) var<uniform> params : SpatterMaskParams;

fn toU32(value : f32) -> u32 {
  return u32(max(value, 0.0));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let width : u32 = toU32(params.size.x + 0.5);
  let height : u32 = toU32(params.size.y + 0.5);
  if (gid.x >= width || gid.y >= height) {
    return;
  }

  let coords : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
  let smearVal : f32 = textureLoad(smearTexture, coords, 0).x;
  let sp1Val : f32 = textureLoad(sp1Texture, coords, 0).x;
  let sp2Val : f32 = textureLoad(sp2Texture, coords, 0).x;
  let removerVal : f32 = textureLoad(removerTexture, coords, 0).x;

  var combined : f32 = max(smearVal, sp1Val);
  combined = max(combined, sp2Val);
  let result : f32 = max(combined - removerVal, 0.0);

  textureStore(outputTexture, coords, vec4<f32>(result, 0.0, 0.0, 0.0));
}
