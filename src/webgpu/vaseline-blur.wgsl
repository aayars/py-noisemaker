struct VaselineBlurParams {
  width: f32,
  height: f32,
  channels: f32,
};
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var outTex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> params: VaselineBlurParams;

fn wrapCoord(v: i32, size: i32) -> i32 {
  var m = v % size;
  if (m < 0) { m = m + size; }
  return m;
}

@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = i32(gid.x);
  let y = i32(gid.y);
  let w = i32(params.width);
  let h = i32(params.height);
  if (x >= w || y >= h) { return; }
  let xOff = i32(params.width * -0.05);
  let yOff = i32(params.height * -0.05);
  var sum: vec4<f32> = vec4<f32>(0.0);
  let radius = 2;
  let kernelSize = 25.0;
  for (var j: i32 = -radius; j <= radius; j = j + 1) {
    for (var i: i32 = -radius; i <= radius; i = i + 1) {
      let xx = wrapCoord(x + i + xOff, w);
      let yy = wrapCoord(y + j + yOff, h);
      var v = textureLoad(tex, vec2<i32>(xx, yy), 0);
      v = clamp(v * 2.0 - 1.0, vec4<f32>(0.0), vec4<f32>(1.0));
      sum = sum + v;
    }
  }
  var blurred = (sum / kernelSize) * 4.0;
  blurred = blurred + vec4<f32>(0.25);
  blurred = (blurred - vec4<f32>(0.5)) * 1.5 + vec4<f32>(0.5);
  let orig = textureLoad(tex, vec2<i32>(x, y), 0);
  var out = clamp((orig + blurred) * 0.5, vec4<f32>(0.0), vec4<f32>(1.0));
  textureStore(outTex, vec2<i32>(x, y), vec4<f32>(out.xyz, 1.0));
}
