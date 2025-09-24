struct NormalizeParams {
  width: f32,
  height: f32,
  channels: f32,
  stage: f32,
  _pad0: f32,
  _pad1: f32,
  _pad2: f32,
  _pad3: f32,
};

@group(0) @binding(0) var inputTex : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> outputBuffer : array<f32>;
@group(0) @binding(2) var<storage, read_write> reduceBuffer : array<f32>;
@group(0) @binding(3) var<uniform> params : NormalizeParams;

fn asU32(value : f32) -> u32 {
  return u32(max(value, 0.0));
}

fn writeStats(minVal : f32, maxVal : f32) {
  if (arrayLength(&reduceBuffer) < 2u) {
    return;
  }
  reduceBuffer[0] = minVal;
  reduceBuffer[1] = maxVal;
}

fn processTexelForStats(texel : vec4<f32>, channelCount : u32, minVal : ptr<function, f32>, maxVal : ptr<function, f32>) {
  if (channelCount == 0u) {
    return;
  }
  *minVal = min(*minVal, texel.x);
  *maxVal = max(*maxVal, texel.x);

  if (channelCount > 1u) {
    *minVal = min(*minVal, texel.y);
    *maxVal = max(*maxVal, texel.y);
  }
  if (channelCount > 2u) {
    *minVal = min(*minVal, texel.z);
    *maxVal = max(*maxVal, texel.z);
  }
  if (channelCount > 3u) {
    *minVal = min(*minVal, texel.w);
    *maxVal = max(*maxVal, texel.w);
  }
  if (channelCount > 4u) {
    var idx : u32 = 4u;
    let fallback : f32 = texel.w;
    loop {
      if (idx >= channelCount) {
        break;
      }
      *minVal = min(*minVal, fallback);
      *maxVal = max(*maxVal, fallback);
      idx = idx + 1u;
    }
  }
}

fn normalizeChannel(value : f32, minVal : f32, invRange : f32) -> f32 {
  return (value - minVal) * invRange;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let stage : u32 = asU32(params.stage);

  if (stage == 0u) {
    if (gid.x != 0u || gid.y != 0u || gid.z != 0u) {
      return;
    }

    let width : u32 = asU32(params.width);
    let height : u32 = asU32(params.height);
    let channels : u32 = asU32(params.channels);

    if (width == 0u || height == 0u || channels == 0u) {
      writeStats(0.0, 0.0);
      return;
    }

    var minVal : f32 = 3.40282347e38;
    var maxVal : f32 = -3.40282347e38;

    for (var y : u32 = 0u; y < height; y = y + 1u) {
      for (var x : u32 = 0u; x < width; x = x + 1u) {
        let texel : vec4<f32> = textureLoad(inputTex, vec2<i32>(i32(x), i32(y)), 0);
        processTexelForStats(texel, channels, &minVal, &maxVal);
      }
    }

    writeStats(minVal, maxVal);
    return;
  }

  if (stage != 1u) {
    return;
  }

  let width : u32 = asU32(params.width);
  let height : u32 = asU32(params.height);
  if (gid.x >= width || gid.y >= height) {
    return;
  }

  let channels : u32 = asU32(params.channels);
  if (channels == 0u) {
    return;
  }

  if (arrayLength(&reduceBuffer) < 2u) {
    return;
  }

  let minVal : f32 = reduceBuffer[0];
  let maxVal : f32 = reduceBuffer[1];
  let range : f32 = maxVal - minVal;
  var invRange : f32 = 1.0;
  if (range != 0.0) {
    invRange = 1.0 / range;
  }

  let coords : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
  let texel : vec4<f32> = textureLoad(inputTex, coords, 0);
  let pixelIndex : u32 = gid.y * width + gid.x;
  let baseIndex : u32 = pixelIndex * channels;

  outputBuffer[baseIndex] = normalizeChannel(texel.x, minVal, invRange);
  if (channels > 1u) {
    outputBuffer[baseIndex + 1u] = normalizeChannel(texel.y, minVal, invRange);
  }
  if (channels > 2u) {
    outputBuffer[baseIndex + 2u] = normalizeChannel(texel.z, minVal, invRange);
  }
  if (channels > 3u) {
    outputBuffer[baseIndex + 3u] = normalizeChannel(texel.w, minVal, invRange);
  }
  if (channels > 4u) {
    let fallback : f32 = normalizeChannel(texel.w, minVal, invRange);
    var idx : u32 = 4u;
    loop {
      if (idx >= channels) {
        break;
      }
      outputBuffer[baseIndex + idx] = fallback;
      idx = idx + 1u;
    }
  }
}
