struct SpatterMaskParams {
  width: f32,
  height: f32,
};
@group(0) @binding(0) var smear: texture_2d<f32>;
@group(0) @binding(1) var sp1: texture_2d<f32>;
@group(0) @binding(2) var sp2: texture_2d<f32>;
@group(0) @binding(3) var remover: texture_2d<f32>;
@group(0) @binding(4) var outTex: texture_storage_2d<r32float, write>;
@group(0) @binding(5) var<uniform> params: SpatterMaskParams;

@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  let w = u32(params.width);
  let h = u32(params.height);
  if (x >= w || y >= h) { return; }
  let coord = vec2<i32>(i32(x), i32(y));
  let s = textureLoad(smear, coord, 0).x;
  let p1 = textureLoad(sp1, coord, 0).x;
  let p2 = textureLoad(sp2, coord, 0).x;
  let r = textureLoad(remover, coord, 0).x;
  var v1 = clamp(((p1 - 1.0) - 0.5) * 4.0 + 0.5, 0.0, 1.0);
  var v2 = clamp(((p2 - 1.25) - 0.5) * 4.0 + 0.5, 0.0, 1.0);
  var sm = max(s, v1);
  sm = max(sm, v2);
  sm = max(0.0, sm - r);
  textureStore(outTex, coord, vec4<f32>(sm, 0.0, 0.0, 1.0));
}
