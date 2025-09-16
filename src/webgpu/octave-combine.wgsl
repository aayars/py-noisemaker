struct OctaveParams {
  width: f32,
  height: f32,
  channels: f32,
  mode: f32,
  weight: f32,
  pad0: f32,
  pad1: f32,
  pad2: f32,
};

@group(0) @binding(0) var baseTex: texture_2d<f32>;
@group(0) @binding(1) var layerTex: texture_2d<f32>;
@group(0) @binding(2) var outTex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(3) var<uniform> params: OctaveParams;

fn get_component(v: vec4<f32>, idx: u32) -> f32 {
  switch idx {
    case 0u: { return v.x; }
    case 1u: { return v.y; }
    case 2u: { return v.z; }
    default: { return v.w; }
  }
}

fn combine_channel(
  channel: u32,
  aTex: vec4<f32>,
  bTex: vec4<f32>,
  channels: u32,
  mode: u32,
  weight: f32,
) -> f32 {
  let aVal = get_component(aTex, channel);
  let bVal = get_component(bTex, channel);
  switch mode {
    case 0u: { // falloff: a + b * weight
      return aVal + bVal * weight;
    }
    case 1u: { // reduce_max
      return max(aVal, bVal);
    }
    case 2u: { // alpha blending using last channel as weight
      let alphaIdx = max(channels, 1u) - 1u;
      let alpha = get_component(bTex, alphaIdx);
      return aVal * (1.0 - alpha) + bVal * alpha;
    }
    default: {
      return bVal;
    }
  }
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  let w = u32(params.width);
  let h = u32(params.height);
  if (x >= w || y >= h) {
    return;
  }
  let coord = vec2<i32>(i32(x), i32(y));
  let aTex = textureLoad(baseTex, coord, 0);
  let bTex = textureLoad(layerTex, coord, 0);
  var outVal = aTex;
  let channels = u32(params.channels + 0.5);
  let mode = u32(params.mode + 0.5);
  let weight = params.weight;
  if (channels > 0u) {
    outVal.x = combine_channel(0u, aTex, bTex, channels, mode, weight);
  }
  if (channels > 1u) {
    outVal.y = combine_channel(1u, aTex, bTex, channels, mode, weight);
  }
  if (channels > 2u) {
    outVal.z = combine_channel(2u, aTex, bTex, channels, mode, weight);
  }
  if (channels > 3u) {
    outVal.w = combine_channel(3u, aTex, bTex, channels, mode, weight);
  }
  textureStore(outTex, coord, outVal);
}
