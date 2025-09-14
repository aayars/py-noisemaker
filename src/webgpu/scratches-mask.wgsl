struct ScratchesMaskParams {
  width: f32,
  height: f32,
};
@group(0) @binding(0) var mask: texture_2d<f32>;
@group(0) @binding(1) var sub: texture_2d<f32>;
@group(0) @binding(2) var outTex: texture_storage_2d<r32float, write>;
@group(0) @binding(3) var<uniform> params: ScratchesMaskParams;

@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  let w = u32(params.width);
  let h = u32(params.height);
  if (x >= w || y >= h) { return; }
  let coord = vec2<i32>(i32(x), i32(y));
  let m = textureLoad(mask, coord, 0).x;
  let s = textureLoad(sub, coord, 0).x;
  var v = max(m - s * 2.0, 0.0);
  textureStore(outTex, coord, vec4<f32>(v, 0.0, 0.0, 1.0));
}
