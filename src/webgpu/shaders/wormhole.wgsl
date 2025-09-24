const TAU : f32 = 6.28318530717958647692;

struct WormholeParams {
  width : f32,
  height : f32,
  channels : f32,
  stride : f32,
  kink : f32,
  xOff : f32,
  yOff : f32,
  pad0 : f32,
};

@group(0) @binding(0) var inputTex : texture_2d<f32>;
@group(0) @binding(1) var<storage, read> luminance : array<f32>;
@group(0) @binding(2) var<storage, read_write> outputBuffer : array<f32>;
@group(0) @binding(3) var<uniform> params : WormholeParams;

fn asU32(value : f32) -> u32 {
  return u32(max(value, 0.0));
}

fn wrapCoord(coord : i32, limit : i32) -> u32 {
  if (limit <= 0) {
    return 0u;
  }
  var wrapped : i32 = coord % limit;
  if (wrapped < 0) {
    wrapped = wrapped + limit;
  }
  return u32(wrapped);
}

fn writeChannel(index : u32, value : f32) {
  outputBuffer[index] = value;
}

fn addChannel(index : u32, value : f32) {
  outputBuffer[index] = outputBuffer[index] + value;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (gid.x != 0u || gid.y != 0u || gid.z != 0u) {
    return;
  }

  let width : u32 = asU32(params.width);
  let height : u32 = asU32(params.height);
  if (width == 0u || height == 0u) {
    return;
  }
  let channels : u32 = max(asU32(params.channels), 1u);

  let total : u32 = width * height * channels;
  var clearIndex : u32 = 0u;
  loop {
    if (clearIndex >= total) {
      break;
    }
    writeChannel(clearIndex, 0.0);
    clearIndex = clearIndex + 1u;
  }

  let widthI : i32 = i32(width);
  let heightI : i32 = i32(height);
  let xOffset : i32 = i32(floor(params.xOff));
  let yOffset : i32 = i32(floor(params.yOff));

  var y : u32 = 0u;
  loop {
    if (y >= height) {
      break;
    }
    var x : u32 = 0u;
    loop {
      if (x >= width) {
        break;
      }
      let pixelIndex : u32 = y * width + x;
      let lum : f32 = luminance[pixelIndex];
      let angle : f32 = lum * TAU * params.kink;
      let xo : f32 = (cos(angle) + 1.0) * params.stride;
      let yo : f32 = (sin(angle) + 1.0) * params.stride;

      let destX : u32 = wrapCoord(i32(floor(f32(x) + xo)) + xOffset, widthI);
      let destY : u32 = wrapCoord(i32(floor(f32(y) + yo)) + yOffset, heightI);
      let destPixel : u32 = destY * width + destX;
      let baseIndex : u32 = destPixel * channels;

      let texel : vec4<f32> = textureLoad(
        inputTex,
        vec2<i32>(i32(x), i32(y)),
        0,
      );
      let scale : f32 = lum * lum;
      let scaled : vec4<f32> = texel * vec4<f32>(scale);

      if (channels > 0u) {
        addChannel(baseIndex, scaled.x);
      }
      if (channels > 1u) {
        addChannel(baseIndex + 1u, scaled.y);
      }
      if (channels > 2u) {
        addChannel(baseIndex + 2u, scaled.z);
      }
      if (channels > 3u) {
        addChannel(baseIndex + 3u, scaled.w);
      }
      if (channels > 4u) {
        var extra : u32 = 4u;
        loop {
          if (extra >= channels) {
            break;
          }
          addChannel(baseIndex + extra, scaled.w);
          extra = extra + 1u;
        }
      }

      x = x + 1u;
    }
    y = y + 1u;
  }
}
