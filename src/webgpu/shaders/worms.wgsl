struct WormsParams {
  dims : vec4<f32>;
  config : vec4<f32>;
};

@group(0) @binding(0) var<storage, read> positionBuffer : array<f32>;
@group(0) @binding(1) var<storage, read> strideBuffer : array<f32>;
@group(0) @binding(2) var<storage, read> rotationBuffer : array<f32>;
@group(0) @binding(3) var<storage, read> colorBuffer : array<f32>;
@group(0) @binding(4) var<storage, read> indexBuffer : array<f32>;
@group(0) @binding(5) var<storage, read> drunkBuffer : array<f32>;
@group(0) @binding(6) var<storage, read_write> outputBuffer : array<atomic<u32>>;
@group(0) @binding(7) var<uniform> params : WormsParams;

fn width() -> u32 {
  return max(u32(params.dims.x + 0.5), 1u);
}

fn height() -> u32 {
  return max(u32(params.dims.y + 0.5), 1u);
}

fn channelCount() -> u32 {
  return max(u32(params.dims.z + 0.5), 1u);
}

fn wormCount() -> u32 {
  return bitcast<u32>(params.dims.w);
}

fn iterationCount() -> u32 {
  return bitcast<u32>(params.config.x);
}

fn quantize() -> bool {
  return params.config.y > 0.5;
}

fn drunkenness() -> f32 {
  return params.config.w;
}

fn wrapIndex(value : i32, size : i32) -> u32 {
  if (size <= 0) {
    return 0u;
  }
  var result : i32 = value % size;
  if (result < 0) {
    result = result + size;
  }
  return u32(result);
}

fn wrapCoord(value : f32, size : f32) -> f32 {
  if (size <= 0.0) {
    return 0.0;
  }
  let ratio : f32 = floor(value / size);
  var wrapped : f32 = value - ratio * size;
  if (wrapped < 0.0) {
    wrapped = wrapped + size;
  }
  return wrapped;
}

fn atomicAddF32(index : u32, value : f32) {
  loop {
    let oldBits : u32 = atomicLoad(&outputBuffer[index]);
    let oldValue : f32 = bitcast<f32>(oldBits);
    let newValue : f32 = oldValue + value;
    let newBits : u32 = bitcast<u32>(newValue);
    let result = atomicCompareExchangeWeak(&outputBuffer[index], oldBits, newBits);
    if (result.exchanged) {
      break;
    }
  }
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let wormIndex : u32 = gid.x;
  let count : u32 = wormCount();
  if (wormIndex >= count) {
    return;
  }

  let w : u32 = width();
  let h : u32 = height();
  let channels : u32 = channelCount();
  if (w == 0u || h == 0u || channels == 0u) {
    return;
  }

  let wf : f32 = f32(w);
  let hf : f32 = f32(h);
  let iterations : u32 = iterationCount();
  if (iterations == 0u) {
    return;
  }
  let denom : u32 = if (iterations > 1u) { iterations - 1u } else { 1u };
  let drunkFactor : f32 = drunkenness();
  let quantizeFlag : bool = quantize();

  var x : f32 = positionBuffer[wormIndex * 2u];
  var y : f32 = positionBuffer[wormIndex * 2u + 1u];
  let stride : f32 = strideBuffer[wormIndex];
  var rotation : f32 = rotationBuffer[wormIndex];
  let colorBase : u32 = wormIndex * channels;

  var iter : u32 = 0u;
  loop {
    if (iter >= iterations) {
      break;
    }

    if (drunkFactor > 0.0) {
      let drunkIdx : u32 = iter * count + wormIndex;
      if (drunkIdx < arrayLength(&drunkBuffer)) {
        rotation = rotation + drunkBuffer[drunkIdx] * drunkFactor;
      }
    }

    let xi : u32 = wrapIndex(i32(floor(x)), i32(w));
    let yi : u32 = wrapIndex(i32(floor(y)), i32(h));
    let pixelIndex : u32 = yi * w + xi;

    let exposure : f32 = if (iterations > 1u) {
      1.0 - abs(1.0 - (f32(iter) / f32(denom)) * 2.0)
    } else {
      1.0
    };

    let outBase : u32 = pixelIndex * channels;
    for (var ch : u32 = 0u; ch < channels; ch = ch + 1u) {
      let colorVal : f32 = colorBuffer[colorBase + ch] * exposure;
      atomicAddF32(outBase + ch, colorVal);
    }

    var angle : f32 = indexBuffer[pixelIndex] + rotation;
    if (quantizeFlag) {
      angle = round(angle);
    }

    y = wrapCoord(y + cos(angle) * stride, hf);
    x = wrapCoord(x + sin(angle) * stride, wf);

    iter = iter + 1u;
  }
}
