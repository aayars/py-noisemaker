// Vaseline blur pass. Mirrors the behaviour of the CPU bloom helper used by
// the vaseline effect by generating a blurred copy of the source tensor. The
// shader operates directly on textures so the result can be consumed by the
// subsequent center-mask blend step without an intermediate buffer copy.

struct BlurParams {
  size : vec4<f32>; // width, height, channels, unused
};

@group(0) @binding(0) var inputTexture : texture_2d<f32>;
@group(0) @binding(1) var outputTexture : texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> params : BlurParams;

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
  let color : vec4<f32> = textureLoad(inputTexture, vec2<i32>(wrappedX, wrappedY), 0);
  return clampVec01(color * 2.0 - vec4<f32>(1.0));
}

fn applyBrightnessContrast(color : vec4<f32>) -> vec4<f32> {
  let brightness : f32 = 0.25;
  let contrast : f32 = 1.5;
  let brightened : vec4<f32> = clampVec01(color + vec4<f32>(brightness));
  let mean : vec4<f32> = vec4<f32>(0.5);
  let contrasted : vec4<f32> = (brightened - mean) * vec4<f32>(contrast) + mean;
  return clampVec01(contrasted);
}

fn mixChannels(baseColor : vec4<f32>, blurred : vec4<f32>, channelCount : u32) -> vec4<f32> {
  let mixed : vec4<f32> = clampVec01((baseColor + blurred) * 0.5);
  var outColor : vec4<f32> = baseColor;
  if (channelCount > 0u) {
    outColor.x = mixed.x;
  }
  if (channelCount > 1u) {
    outColor.y = mixed.y;
  }
  if (channelCount > 2u) {
    outColor.z = mixed.z;
  }
  if (channelCount > 3u) {
    outColor.w = mixed.w;
  }
  return outColor;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let width : u32 = asU32(params.size.x);
  let height : u32 = asU32(params.size.y);
  if (gid.x >= width || gid.y >= height) {
    return;
  }

  let channelCount : u32 = max(asU32(params.size.z), 1u);
  let coords : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
  let baseColor : vec4<f32> = clampVec01(textureLoad(inputTexture, coords, 0));

  let widthI : i32 = i32(width);
  let heightI : i32 = i32(height);
  let offsetX : i32 = coords.x - widthI / 20;  // approximate -0.05 * width
  let offsetY : i32 = coords.y - heightI / 20; // approximate -0.05 * height

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

  var blurred : vec4<f32> = vec4<f32>(0.0);
  var totalWeight : f32 = 0.0;
  for (var i : i32 = 0; i < 9; i = i + 1) {
    let offset : vec2<i32> = offsets[u32(i)];
    let sampleX : i32 = offsetX + offset.x;
    let sampleY : i32 = offsetY + offset.y;
    let sample : vec4<f32> = loadBright(sampleX, sampleY, widthI, heightI);
    let weight : f32 = weights[u32(i)];
    blurred = blurred + sample * weight;
    totalWeight = totalWeight + weight;
  }

  if (totalWeight > 0.0) {
    blurred = blurred / totalWeight;
  }
  blurred = clampVec01(blurred * 4.0);
  blurred = applyBrightnessContrast(blurred);

  let outColor : vec4<f32> = mixChannels(baseColor, blurred, channelCount);
  textureStore(outputTexture, coords, outColor);
}
