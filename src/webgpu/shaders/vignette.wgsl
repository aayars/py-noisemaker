struct VignetteParams {
  width: f32,
  height: f32,
  channels: f32,
  brightness: f32,
  alpha: f32,
};

@group(0) @binding(0) var inputTex : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> outputBuffer : array<f32>;
@group(0) @binding(2) var<uniform> params : VignetteParams;

fn asU32(value : f32) -> u32 {
  return u32(max(value, 0.0));
}

fn applyVignette(value : f32, brightness : f32, mask : f32, alpha : f32) -> f32 {
  let edgeMix : f32 = mix(value, brightness, mask);
  return mix(value, edgeMix, alpha);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let width : u32 = asU32(params.width);
  let height : u32 = asU32(params.height);
  if (gid.x >= width || gid.y >= height) {
    return;
  }

  let channelCount : u32 = asU32(params.channels);
  if (channelCount == 0u) {
    return;
  }

  let widthF : f32 = params.width;
  let heightF : f32 = params.height;

  if (widthF <= 0.0 || heightF <= 0.0) {
    return;
  }

  let coords : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
  let texel : vec4<f32> = textureLoad(inputTex, coords, 0);

  let pixelIndex : u32 = gid.y * width + gid.x;
  let baseIndex : u32 = pixelIndex * channelCount;

  let px : f32 = f32(gid.x) + 0.5;
  let py : f32 = f32(gid.y) + 0.5;
  let dx : f32 = (px - widthF * 0.5) / widthF;
  let dy : f32 = (py - heightF * 0.5) / heightF;
  let maxDx : f32 = abs((widthF * 0.5 - 0.5) / widthF);
  let maxDy : f32 = abs((heightF * 0.5 - 0.5) / heightF);
  let maxDist : f32 = sqrt(maxDx * maxDx + maxDy * maxDy);
  let dist : f32 = sqrt(dx * dx + dy * dy);
  var mask : f32 = 0.0;
  if (maxDist > 0.0) {
    mask = clamp(dist / maxDist, 0.0, 1.0);
  }

  let brightness : f32 = params.brightness;
  let alpha : f32 = params.alpha;

  if (channelCount > 0u) {
    outputBuffer[baseIndex] = applyVignette(texel.x, brightness, mask, alpha);
  }
  if (channelCount > 1u) {
    outputBuffer[baseIndex + 1u] = applyVignette(texel.y, brightness, mask, alpha);
  }
  if (channelCount > 2u) {
    outputBuffer[baseIndex + 2u] = applyVignette(texel.z, brightness, mask, alpha);
  }
  if (channelCount > 3u) {
    outputBuffer[baseIndex + 3u] = applyVignette(texel.w, brightness, mask, alpha);
  }
  if (channelCount > 4u) {
    let fallback : f32 = applyVignette(texel.w, brightness, mask, alpha);
    var ch : u32 = 4u;
    loop {
      if (ch >= channelCount) {
        break;
      }
      outputBuffer[baseIndex + ch] = fallback;
      ch = ch + 1u;
    }
  }
}
