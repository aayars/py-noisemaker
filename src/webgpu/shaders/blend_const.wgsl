struct BlendConstParams {
  width : f32;
  height : f32;
  channels : f32;
  alpha : f32;
  pad0 : f32;
  pad1 : f32;
  pad2 : f32;
  pad3 : f32;
};

@group(0) @binding(0) var inputA : texture_2d<f32>;
@group(0) @binding(1) var inputB : texture_2d<f32>;
@group(0) @binding(2) var outputTexture : texture_storage_2d<rgba32float, write>;
@group(0) @binding(3) var<uniform> params : BlendConstParams;

fn readChannel(sample : vec4<f32>, channel : u32) -> f32 {
  switch channel {
    case 0u: {
      return sample.x;
    }
    case 1u: {
      return sample.y;
    }
    case 2u: {
      return sample.z;
    }
    default: {
      return sample.w;
    }
  }
}

fn writeChannel(base : vec4<f32>, channel : u32, value : f32) -> vec4<f32> {
  switch channel {
    case 0u: {
      return vec4<f32>(value, base.y, base.z, base.w);
    }
    case 1u: {
      return vec4<f32>(base.x, value, base.z, base.w);
    }
    case 2u: {
      return vec4<f32>(base.x, base.y, value, base.w);
    }
    default: {
      return vec4<f32>(base.x, base.y, base.z, value);
    }
  }
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let width : u32 = max(u32(params.width), 1u);
  let height : u32 = max(u32(params.height), 1u);
  if (gid.x >= width || gid.y >= height) {
    return;
  }

  let channelCount : u32 = max(u32(params.channels), 1u);
  let coords : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
  let sampleA : vec4<f32> = textureLoad(inputA, coords, 0);
  let sampleB : vec4<f32> = textureLoad(inputB, coords, 0);
  let alpha : f32 = clamp(params.alpha, 0.0, 1.0);

  var outSample : vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  var channel : u32 = 0u;
  loop {
    if (channel >= channelCount) {
      break;
    }
    let aVal : f32 = readChannel(sampleA, channel);
    let bVal : f32 = readChannel(sampleB, channel);
    let blended : f32 = mix(aVal, bVal, alpha);
    outSample = writeChannel(outSample, channel, blended);
    channel = channel + 1u;
  }

  textureStore(outputTexture, coords, outSample);
}
