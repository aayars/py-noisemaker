struct BinaryParams {
  width : u32,
  height : u32,
  channels : u32,
  op : u32,
};

@group(0) @binding(0) var inputATexture : texture_2d<f32>;
@group(0) @binding(1) var inputBTexture : texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> outputBuffer : array<f32>;
@group(0) @binding(3) var<uniform> params : BinaryParams;

fn combine(a : f32, b : f32, op : u32) -> f32 {
  switch op {
    case 0u: {
      return min(a, 1.0 - b);
    }
    default: {
      return a;
    }
  }
}

fn readChannel(sample : vec4<f32>, channel : u32) -> f32 {
  switch channel {
    case 0u: { return sample.x; }
    case 1u: { return sample.y; }
    case 2u: { return sample.z; }
    default: { return sample.w; }
  }
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let width : u32 = max(params.width, 1u);
  let height : u32 = max(params.height, 1u);
  if (gid.x >= width || gid.y >= height) {
    return;
  }

  let channelCount : u32 = max(params.channels, 1u);
  let coords : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
  let sampleA : vec4<f32> = textureLoad(inputATexture, coords, 0);
  let sampleB : vec4<f32> = textureLoad(inputBTexture, coords, 0);

  let baseIndex : u32 = (gid.y * width + gid.x) * channelCount;
  let op : u32 = params.op;

  var channel : u32 = 0u;
  loop {
    if (channel >= channelCount) {
      break;
    }
    let aVal : f32 = readChannel(sampleA, channel);
    let bVal : f32 = readChannel(sampleB, channel);
    outputBuffer[baseIndex + channel] = combine(aVal, bVal, op);
    channel = channel + 1u;
  }
}
