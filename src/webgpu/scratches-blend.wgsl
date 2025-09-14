struct ScratchesBlendParams {
  width: f32,
  height: f32,
  channels: f32,
};
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var mask: texture_2d<f32>;
@group(0) @binding(2) var outTex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(3) var<uniform> params: ScratchesBlendParams;

@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  let w = u32(params.width);
  let h = u32(params.height);
  if (x >= w || y >= h) { return; }
  let coord = vec2<i32>(i32(x), i32(y));
  let base = textureLoad(src, coord, 0);
  let m = textureLoad(mask, coord, 0).x;
  let val = min(m * 8.0, 1.0);
  var outVal = base;
  if (params.channels > 0.0) { outVal.x = max(base.x, val); }
  if (params.channels > 1.0) { outVal.y = max(base.y, val); }
  if (params.channels > 2.0) { outVal.z = max(base.z, val); }
  if (params.channels > 3.0) { outVal.w = max(base.w, val); }
  textureStore(outTex, coord, outVal);
}
