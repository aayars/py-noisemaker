struct VaselineMaskParams {
  width: f32,
  height: f32,
};
@group(0) @binding(0) var outTex: texture_storage_2d<r32float, write>;
@group(0) @binding(1) var<uniform> params: VaselineMaskParams;

@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  let w = u32(params.width);
  let h = u32(params.height);
  if (x >= w || y >= h) { return; }
  let cx = params.width * 0.5;
  let cy = params.height * 0.5;
  let dx = abs(f32(x) + 0.5 - cx) / cx;
  let dy = abs(f32(y) + 0.5 - cy) / cy;
  let d = max(dx, dy);
  let v = clamp(d * d, 0.0, 1.0);
  textureStore(outTex, vec2<i32>(i32(x), i32(y)), vec4<f32>(v, 0.0, 0.0, 1.0));
}
