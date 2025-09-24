struct ErosionWormsParams {
  sizeCount : vec4<f32>;
  config : vec4<f32>;
};

@group(0) @binding(0) var<storage, read> wormsBuffer : array<f32>;
@group(0) @binding(1) var<storage, read> valuesBuffer : array<f32>;
@group(0) @binding(2) var<storage, read> startColorBuffer : array<f32>;
@group(0) @binding(3) var<storage, read_write> outputBuffer : array<atomic<u32>>;
@group(0) @binding(4) var<uniform> params : ErosionWormsParams;

fn width() -> u32 {
  return max(u32(params.sizeCount.x + 0.5), 1u);
}

fn height() -> u32 {
  return max(u32(params.sizeCount.y + 0.5), 1u);
}

fn wormCount() -> u32 {
  return bitcast<u32>(params.sizeCount.z);
}

fn iterationCount() -> u32 {
  return bitcast<u32>(params.sizeCount.w);
}

fn contraction() -> f32 {
  return params.config.x;
}

fn quantize() -> bool {
  return params.config.y > 0.5;
}

fn channelCount() -> u32 {
  let raw : u32 = bitcast<u32>(params.config.z);
  return max(raw, 1u);
}

fn wrapIndex(value : i32, size : i32) -> u32 {
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

fn lerp(a : f32, b : f32, t : f32) -> f32 {
  return mix(a, b, t);
}

fn gradientValues(baseIndex : u32, x1Index : u32, y1Index : u32, x1y1Index : u32, fracX : f32, fracY : f32) -> vec2<f32> {
  let baseVal : f32 = valuesBuffer[baseIndex];
  let x1Val : f32 = valuesBuffer[x1Index];
  let y1Val : f32 = valuesBuffer[y1Index];
  let x1y1Val : f32 = valuesBuffer[x1y1Index];
  let gradX : f32 = lerp(y1Val - baseVal, x1y1Val - x1Val, fracX);
  let gradY : f32 = lerp(x1Val - baseVal, x1y1Val - y1Val, fracY);
  return vec2<f32>(gradX, gradY);
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx : u32 = gid.x;
  let count : u32 = wormCount();
  if (idx >= count) {
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
  let denom : u32 = if (iterations > 1u) { iterations - 1u } else { 1u };
  let valuesLength : u32 = arrayLength(&valuesBuffer);

  let wormBase : u32 = idx * 5u;
  var x : f32 = wormsBuffer[wormBase];
  var y : f32 = wormsBuffer[wormBase + 1u];
  var dirX : f32 = wormsBuffer[wormBase + 2u];
  var dirY : f32 = wormsBuffer[wormBase + 3u];
  let inertia : f32 = wormsBuffer[wormBase + 4u];

  let dirLen : f32 = sqrt(dirX * dirX + dirY * dirY);
  if (dirLen > 0.0) {
    dirX = dirX / dirLen;
    dirY = dirY / dirLen;
  }

  let startBase : u32 = idx * channels;
  let contractionVal : f32 = contraction();
  let quantizeFlag : bool = quantize();

  var iter : u32 = 0u;
  loop {
    if (iter >= iterations) {
      break;
    }

    let baseXi : u32 = wrapIndex(i32(floor(x)), i32(w));
    let baseYi : u32 = wrapIndex(i32(floor(y)), i32(h));
    let pixelIndex : u32 = baseYi * w + baseXi;
    let exposureT : f32 = if (iterations > 1u) {
      1.0 - abs(1.0 - (f32(iter) / f32(denom)) * 2.0)
    } else {
      1.0
    };

    let outBase : u32 = pixelIndex * channels;
    for (var ch : u32 = 0u; ch < channels; ch = ch + 1u) {
      let colorVal : f32 = startColorBuffer[startBase + ch] * exposureT;
      atomicAddF32(outBase + ch, colorVal);
    }

    if (valuesLength == 0u) {
      iter = iter + 1u;
      continue;
    }

    let x1 : u32 = (baseXi + 1u) % w;
    let y1 : u32 = (baseYi + 1u) % h;
    let baseIndex : u32 = pixelIndex;
    let x1Index : u32 = baseYi * w + x1;
    let y1Index : u32 = y1 * w + baseXi;
    let x1y1Index : u32 = y1 * w + x1;

    let fracX : f32 = fract(x);
    let fracY : f32 = fract(y);
    let grads : vec2<f32> = gradientValues(
      baseIndex,
      x1Index,
      y1Index,
      x1y1Index,
      fracX,
      fracY,
    );

    var gx : f32 = grads.x;
    var gy : f32 = grads.y;

    if (quantizeFlag) {
      gx = floor(gx);
      gy = floor(gy);
    }

    var lenVal : f32 = length(vec2<f32>(gx, gy)) * contractionVal;
    if (!isFinite(lenVal) || lenVal == 0.0) {
      lenVal = 1.0;
    }

    let targetX : f32 = gx / lenVal;
    let targetY : f32 = gy / lenVal;

    dirX = lerp(dirX, targetX, inertia);
    dirY = lerp(dirY, targetY, inertia);

    x = wrapCoord(x + dirX, wf);
    y = wrapCoord(y + dirY, hf);

    iter = iter + 1u;
  }
}
