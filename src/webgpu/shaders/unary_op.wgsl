struct UnaryParams {
  width : u32,
  height : u32,
  channels : u32,
  op : u32,
};

@group(0) @binding(0) var inputTexture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> outputBuffer : array<f32>;
@group(0) @binding(2) var<uniform> params : UnaryParams;

fn applyOp(value : f32, op : u32) -> f32 {
  switch op {
    case 0u: { // invert
      return 1.0 - value;
    }
    case 1u: { // square
      return value * value;
    }
    case 2u: { // clamp01
      return clamp(value, 0.0, 1.0);
    }
    default: {
      return value;
    }
  }
}

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

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let width : u32 = max(params.width, 1u);
  let height : u32 = max(params.height, 1u);
  if (gid.x >= width || gid.y >= height) {
    return;
  }

  let channelCount : u32 = max(params.channels, 1u);
  let coords : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
  let sample : vec4<f32> = textureLoad(inputTexture, coords, 0);

  let baseIndex : u32 = (gid.y * width + gid.x) * channelCount;
  let op : u32 = params.op;

  var channel : u32 = 0u;
  loop {
    if (channel >= channelCount) {
      break;
    }
    let value : f32 = readChannel(sample, channel);
    outputBuffer[baseIndex + channel] = applyOp(value, op);
    channel = channel + 1u;
  }
}
