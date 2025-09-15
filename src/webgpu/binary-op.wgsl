struct BinaryParams {
  width: u32,
  height: u32,
  channels: u32,
  op: u32,
};

@group(0) @binding(0) var texA: texture_2d<f32>;
@group(0) @binding(1) var texB: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: BinaryParams;

fn get_component(v: vec4<f32>, idx: u32) -> f32 {
  switch idx {
    case 0u: { return v.x; }
    case 1u: { return v.y; }
    case 2u: { return v.z; }
    default: { return v.w; }
  }
}

fn apply_binary(a: f32, b: f32, op: u32) -> f32 {
  switch op {
    case 0u: {
      let invA = 1.0 - a;
      let invB = 1.0 - b;
      return min(invA, invB);
    }
    default: {
      return a;
    }
  }
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  let w = params.width;
  let h = params.height;
  if (x >= w || y >= h) {
    return;
  }
  let base = (y * w + x) * params.channels;
  let aTex = textureLoad(texA, vec2<i32>(i32(x), i32(y)), 0);
  let bTex = textureLoad(texB, vec2<i32>(i32(x), i32(y)), 0);
  for (var i: u32 = 0u; i < params.channels; i = i + 1u) {
    let aVal = get_component(aTex, i);
    let bVal = get_component(bTex, i);
    out[base + i] = apply_binary(aVal, bVal, params.op);
  }
}
