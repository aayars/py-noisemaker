struct ScreenParams {
  sizeChannels : vec4<u32>;
};

@group(0) @binding(0) var sourceTex : texture_2d<f32>;
@group(0) @binding(1) var leakTex : texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> outputBuffer : array<f32>;
@group(0) @binding(3) var<uniform> params : ScreenParams;

fn clamp01(value : vec4<f32>) -> vec4<f32> {
  return clamp(value, vec4<f32>(0.0), vec4<f32>(1.0));
}

fn writePixel(index : u32, channels : u32, value : vec4<f32>) {
  if (channels == 0u) {
    return;
  }
  outputBuffer[index] = value.x;
  if (channels > 1u) {
    outputBuffer[index + 1u] = value.y;
  }
  if (channels > 2u) {
    outputBuffer[index + 2u] = value.z;
  }
  if (channels > 3u) {
    outputBuffer[index + 3u] = value.w;
  }
  if (channels > 4u) {
    var c : u32 = 4u;
    loop {
      if (c >= channels) {
        break;
      }
      outputBuffer[index + c] = value.w;
      c = c + 1u;
    }
  }
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let width : u32 = params.sizeChannels.x;
  let height : u32 = params.sizeChannels.y;
  if (gid.x >= width || gid.y >= height) {
    return;
  }

  let channels : u32 = max(params.sizeChannels.z, 1u);
  let pixelIndex : u32 = (gid.y * width + gid.x) * channels;
  let coords : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
  let baseColor : vec4<f32> = clamp01(textureLoad(sourceTex, coords, 0));
  let leakColor : vec4<f32> = clamp01(textureLoad(leakTex, coords, 0));
  let screenColor : vec4<f32> = clamp01(baseColor + leakColor - baseColor * leakColor);
  writePixel(pixelIndex, channels, screenColor);
}
