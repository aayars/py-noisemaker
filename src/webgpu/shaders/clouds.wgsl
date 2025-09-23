struct CloudsParams {
  size : vec4<f32>;
  pre : vec4<f32>;
  freqTime : vec4<f32>;
  offsetsWarp : vec4<f32>;
  seeds : vec4<u32>;
  scales : vec4<f32>;
};

@group(0) @binding(0) var inputTex : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> outBuffer : array<f32>;
@group(0) @binding(2) var<uniform> params : CloudsParams;

const BLUR_KERNEL : array<f32, 25> = array<f32, 25>(
  1.0, 4.0, 6.0, 4.0, 1.0,
  4.0, 16.0, 24.0, 16.0, 4.0,
  6.0, 24.0, 36.0, 24.0, 6.0,
  4.0, 16.0, 24.0, 16.0, 4.0,
  1.0, 4.0, 6.0, 4.0, 1.0,
);

fn wrap_component(value : f32, size : f32) -> f32 {
  if (size <= 0.0) {
    return 0.0;
  }
  return value - floor(value / size) * size;
}

fn wrap_coord(coord : vec2<f32>, dims : vec2<f32>) -> vec2<f32> {
  return vec2<f32>(wrap_component(coord.x, dims.x), wrap_component(coord.y, dims.y));
}

fn normalize_coord(coord : vec2<f32>, dims : vec2<f32>) -> vec2<f32> {
  let nx = select(0.0, coord.x / dims.x, dims.x > 0.0);
  let ny = select(0.0, coord.y / dims.y, dims.y > 0.0);
  return vec2<f32>(nx, ny);
}

fn pcg3d(v_in : vec3<u32>) -> vec3<u32> {
  var v : vec3<u32> = v_in * 1664525u + 1013904223u;
  v.x += v.y * v.z;
  v.y += v.z * v.x;
  v.z += v.x * v.y;
  v = v ^ (v >> vec3<u32>(16u));
  v.x += v.y * v.z;
  v.y += v.z * v.x;
  v.z += v.x * v.y;
  return v;
}

fn random_from_cell_3d(cell : vec3<i32>, seed : u32) -> f32 {
  let hashed : vec3<u32> = vec3<u32>(
    bitcast<u32>(cell.x) ^ seed,
    bitcast<u32>(cell.y) ^ (seed * 0x9e3779b9u + 0x7f4a7c15u),
    bitcast<u32>(cell.z) ^ (seed * 0x632be59bu + 0x5bf03635u),
  );
  let noise : vec3<u32> = pcg3d(hashed);
  return f32(noise.x) / f32(0xffffffffu);
}

fn value_noise(
  normCoord : vec2<f32>,
  freq : vec2<f32>,
  seed : u32,
  octaveIndex : u32,
  phase : f32,
) -> f32 {
  let fx = max(freq.x, 1.0);
  let fy = max(freq.y, 1.0);
  let scaled = vec2<f32>(normCoord.x * fx, normCoord.y * fy);
  let phaseScale = phase * (1.0 + f32(octaveIndex));
  let offset = vec2<f32>(phaseScale, phaseScale * 1.7320508);
  let sample = scaled + offset;
  let cell = vec2<i32>(floor(sample));
  let frac = fract(sample);
  let baseZ = i32(octaveIndex);
  let p00 = random_from_cell_3d(vec3<i32>(cell.x, cell.y, baseZ), seed);
  let p10 = random_from_cell_3d(vec3<i32>(cell.x + 1, cell.y, baseZ), seed);
  let p01 = random_from_cell_3d(vec3<i32>(cell.x, cell.y + 1, baseZ), seed);
  let p11 = random_from_cell_3d(vec3<i32>(cell.x + 1, cell.y + 1, baseZ), seed);
  let blendX0 = mix(p00, p10, frac.x);
  let blendX1 = mix(p01, p11, frac.x);
  return mix(blendX0, blendX1, frac.y);
}

fn control_value_at(coord : vec2<f32>, dims : vec2<f32>) -> f32 {
  let wrapped = wrap_coord(coord, dims);
  let norm = normalize_coord(wrapped, dims);
  let warpFreq = vec2<f32>(max(params.offsetsWarp.z, 1.0), max(params.offsetsWarp.w, 1.0));
  let phase = params.scales.w;
  let flowX = value_noise(norm, warpFreq, params.seeds.z, 0u, phase);
  let flowY = value_noise(norm + vec2<f32>(0.31, 0.73), warpFreq, params.seeds.w, 0u, phase);
  let offset = (vec2<f32>(flowX, flowY) * 2.0 - vec2<f32>(1.0, 1.0)) * params.scales.z;
  let warped = wrap_coord(wrapped + offset, dims);
  let warpedNorm = normalize_coord(warped, dims);

  let baseFreqY = max(params.freqTime.x, 1.0);
  let baseFreqX = max(params.freqTime.y, 1.0);
  let maxOctaves = u32(params.size.w + 0.5);
  let preH = max(dims.y, 1.0);
  let preW = max(dims.x, 1.0);

  var weight = 0.5;
  var weightSum = 0.0;
  var accum = 0.0;

  for (var octave : u32 = 0u; octave < maxOctaves; octave = octave + 1u) {
    let mult = f32(1u << (octave + 1u)) * 0.5;
    var freqY = floor(baseFreqY * mult);
    var freqX = floor(baseFreqX * mult);
    if (freqY <= 0.0) {
      freqY = 1.0;
    }
    if (freqX <= 0.0) {
      freqX = 1.0;
    }
    if (freqY > preH && freqX > preW) {
      break;
    }
    let freqVec = vec2<f32>(freqX, freqY);
    let sample = value_noise(warpedNorm, freqVec, params.seeds.x, octave, phase);
    let ridged = 1.0 - abs(sample * 2.0 - 1.0);
    accum = accum + ridged * weight;
    weightSum = weightSum + weight;
    weight = weight * 0.5;
  }

  if (weightSum > 0.0) {
    accum = accum / weightSum;
  }
  return clamp(accum, 0.0, 1.0);
}

fn combined_value_at(coord : vec2<f32>, dims : vec2<f32>) -> f32 {
  let control = control_value_at(coord, dims);
  return max(0.0, 1.0 - control * 2.0);
}

fn shade_value_at(coord : vec2<f32>, dims : vec2<f32>) -> f32 {
  let offset = vec2<f32>(params.offsetsWarp.x, params.offsetsWarp.y);
  let base = wrap_coord(coord + offset, dims);
  var accum = 0.0;
  var kernelIndex : u32 = 0u;
  for (var dy : i32 = -2; dy <= 2; dy = dy + 1) {
    for (var dx : i32 = -2; dx <= 2; dx = dx + 1) {
      let sampleCoord = wrap_coord(base + vec2<f32>(f32(dx), f32(dy)), dims);
      let combined = combined_value_at(sampleCoord, dims);
      let scaled = clamp(combined * params.scales.x, 0.0, 1.0);
      accum = accum + scaled * BLUR_KERNEL[kernelIndex];
      kernelIndex = kernelIndex + 1u;
    }
  }
  let blurred = accum / 256.0;
  return clamp(blurred * params.scales.y, 0.0, 1.0);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
  let width = u32(params.size.x + 0.5);
  let height = u32(params.size.y + 0.5);
  if (global_id.x >= width || global_id.y >= height) {
    return;
  }
  let rawChannels = u32(params.size.z + 0.5);
  let channels = max(min(rawChannels, 4u), 1u);
  let dims = vec2<f32>(params.pre.x, params.pre.y);
  let px = f32(global_id.x) + 0.5;
  let py = f32(global_id.y) + 0.5;
  let preCoord = vec2<f32>(px * params.pre.z, py * params.pre.w);

  let combined = combined_value_at(preCoord, dims);
  let shade = shade_value_at(preCoord, dims);

  let texel = textureLoad(inputTex, vec2<i32>(i32(global_id.x), i32(global_id.y)), 0);
  var components = array<f32, 4>(texel.x, texel.y, texel.z, texel.w);

  let pixelIndex = (global_id.y * width + global_id.x) * channels;
  for (var ch : u32 = 0u; ch < channels; ch = ch + 1u) {
    let inputValue = components[ch];
    let darkened = inputValue * (1.0 - shade);
    let blended = darkened + (1.0 - darkened) * combined;
    outBuffer[pixelIndex + ch] = clamp(blended, 0.0, 1.0);
  }
}
