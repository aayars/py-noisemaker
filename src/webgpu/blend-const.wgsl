struct BlendParams {
  width: f32,
  height: f32,
  channels: f32,
  t: f32,
  pad0: f32,
  pad1: f32,
  pad2: f32,
  pad3: f32,
};
@group(0) @binding(0) var texA: texture_2d<f32>;
@group(0) @binding(1) var texB: texture_2d<f32>;
@group(0) @binding(2) var outTex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(3) var<uniform> params: BlendParams;

@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  let w = u32(params.width);
  let h = u32(params.height);
  if (x >= w || y >= h) { return; }
  let coord = vec2<i32>(i32(x), i32(y));
  let a = textureLoad(texA, coord, 0);
  let b = textureLoad(texB, coord, 0);
  let val = mix(a, b, vec4<f32>(params.t));
  var outVal = a;
  if (params.channels > 0.0) { outVal.x = val.x; }
  if (params.channels > 1.0) { outVal.y = val.y; }
  if (params.channels > 2.0) { outVal.z = val.z; }
  if (params.channels > 3.0) { outVal.w = val.w; }
  textureStore(outTex, coord, outVal);
}
