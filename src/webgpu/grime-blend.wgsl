struct GrimeBlendParams {
  width: f32,
  height: f32,
  channels: f32,
};
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var mask: texture_2d<f32>;
@group(0) @binding(2) var noise: texture_2d<f32>;
@group(0) @binding(3) var specks: texture_2d<f32>;
@group(0) @binding(4) var outTex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(5) var<uniform> params: GrimeBlendParams;

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
  let n = textureLoad(noise, coord, 0).x;
  let s = textureLoad(specks, coord, 0).x;
  let gate = m * m * 0.075;
  var dusty = mix(base, vec4<f32>(0.25, 0.25, 0.25, 0.25), gate);
  dusty = mix(dusty, vec4<f32>(n, n, n, n), 0.075);
  dusty = dusty * s;
  let maskVal = m * 0.75;
  let outVal = mix(base, dusty, maskVal);
  var res = base;
  if (params.channels > 0.0) { res.x = outVal.x; }
  if (params.channels > 1.0) { res.y = outVal.y; }
  if (params.channels > 2.0) { res.z = outVal.z; }
  if (params.channels > 3.0) { res.w = outVal.w; }
  textureStore(outTex, coord, res);
}
