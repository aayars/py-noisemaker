struct BloomParams {
  sizeAlpha : vec4<f32>;
  offsetsAdjust : vec4<f32>;
};

@group(0) @binding(0) var inputTex : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> outputBuffer : array<f32>;
@group(0) @binding(2) var<uniform> params : BloomParams;

fn clampVec01(value : vec4<f32>) -> vec4<f32> {
  return clamp(value, vec4<f32>(0.0), vec4<f32>(1.0));
}

fn asU32(value : f32) -> u32 {
  return u32(max(value, 0.0));
}

fn wrapIndex(value : i32, size : i32) -> i32 {
  if (size <= 0) {
    return 0;
  }
  var modVal : i32 = value % size;
  if (modVal < 0) {
    modVal = modVal + size;
  }
  return modVal;
}

fn loadBright(x : i32, y : i32, width : i32, height : i32) -> vec4<f32> {
  let wrappedX : i32 = wrapIndex(x, width);
  let wrappedY : i32 = wrapIndex(y, height);
  let color : vec4<f32> = textureLoad(inputTex, vec2<i32>(wrappedX, wrappedY), 0);
  return clampVec01(color * 2.0 - vec4<f32>(1.0));
}

fn applyBrightnessContrast(color : vec4<f32>) -> vec4<f32> {
  let brightness : f32 = params.offsetsAdjust.z;
  let contrast : f32 = params.offsetsAdjust.w;
  let brightened : vec4<f32> = clampVec01(color + vec4<f32>(brightness));
  let mean : vec4<f32> = vec4<f32>(0.5);
  let contrasted : vec4<f32> = (brightened - mean) * vec4<f32>(contrast) + mean;
  return clampVec01(contrasted);
}

fn storePixel(
  pixelIndex : u32,
  channelCount : u32,
  value : vec4<f32>,
) {
  if (channelCount == 0u) {
    return;
  }
  outputBuffer[pixelIndex] = value.x;
  if (channelCount > 1u) {
    outputBuffer[pixelIndex + 1u] = value.y;
  }
  if (channelCount > 2u) {
    outputBuffer[pixelIndex + 2u] = value.z;
  }
  if (channelCount > 3u) {
    outputBuffer[pixelIndex + 3u] = value.w;
  }
  if (channelCount > 4u) {
    var ch : u32 = 4u;
    let fallback : f32 = value.w;
    loop {
      if (ch >= channelCount) {
        break;
      }
      outputBuffer[pixelIndex + ch] = fallback;
      ch = ch + 1u;
    }
  }
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let width : u32 = asU32(params.sizeAlpha.x);
  let height : u32 = asU32(params.sizeAlpha.y);
  if (gid.x >= width || gid.y >= height) {
    return;
  }

  let channelCount : u32 = max(asU32(params.sizeAlpha.z), 1u);
  let alpha : f32 = clamp(params.sizeAlpha.w, 0.0, 1.0);
  let pixelIndex : u32 = (gid.y * width + gid.x) * channelCount;
  let coords : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
  let baseColor : vec4<f32> = clampVec01(textureLoad(inputTex, coords, 0));

  if (alpha <= 0.0) {
    storePixel(pixelIndex, channelCount, baseColor);
    return;
  }

  let widthI : i32 = i32(width);
  let heightI : i32 = i32(height);
  let offsetX : i32 = i32(params.offsetsAdjust.x);
  let offsetY : i32 = i32(params.offsetsAdjust.y);
  let centerX : i32 = i32(gid.x) + offsetX;
  let centerY : i32 = i32(gid.y) + offsetY;

  let offsets : array<vec2<i32>, 9> = array<vec2<i32>, 9>(
    vec2<i32>(-1, -1),
    vec2<i32>(0, -1),
    vec2<i32>(1, -1),
    vec2<i32>(-1, 0),
    vec2<i32>(0, 0),
    vec2<i32>(1, 0),
    vec2<i32>(-1, 1),
    vec2<i32>(0, 1),
    vec2<i32>(1, 1),
  );
  let weights : array<f32, 9> = array<f32, 9>(
    1.0,
    2.0,
    1.0,
    2.0,
    4.0,
    2.0,
    1.0,
    2.0,
    1.0,
  );

  var totalWeight : f32 = 0.0;
  var blurred : vec4<f32> = vec4<f32>(0.0);
  var i : i32 = 0;
  loop {
    if (i >= 9) {
      break;
    }
    let offset : vec2<i32> = offsets[i];
    let sampleX : i32 = centerX + offset.x;
    let sampleY : i32 = centerY + offset.y;
    let sample : vec4<f32> = loadBright(sampleX, sampleY, widthI, heightI);
    let weight : f32 = weights[i];
    blurred = blurred + sample * weight;
    totalWeight = totalWeight + weight;
    i = i + 1;
  }

  if (totalWeight > 0.0) {
    blurred = blurred / totalWeight;
  }
  blurred = clampVec01(blurred * 4.0);
  blurred = applyBrightnessContrast(blurred);

  let mixed : vec4<f32> = clampVec01((baseColor + blurred) * 0.5);
  let finalColor : vec4<f32> = clampVec01(baseColor * (1.0 - alpha) + mixed * alpha);
  storePixel(pixelIndex, channelCount, finalColor);
}
