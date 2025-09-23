// Simplified multi-resolution generator compute shader.
//
// This version keeps the same uniform layout and binding model as the original
// work-in-progress port so it can drop into the existing WebGPU pipeline, but
// the implementation focuses on the core octave stacking behaviour.  Procedural
// masks, lattice drift, permutation table indirection, and other advanced
// features are intentionally left out so the shader is easier to reason about
// and cheaper to execute while we continue iterating on the GPU path.

struct FrameUniforms {
  resolution : vec2<f32>,
  time : f32,
  seed : u32,
  frame_index : u32,
  padding0 : u32,
  padding1 : vec2<f32>,
};

struct StageUniforms {
  freq : vec2<f32>,
  speed : f32,
  sin : f32,
  colorParams0 : vec4<f32>,
  colorParams1 : vec4<f32>,
  options0 : vec4<u32>,
  options1 : vec4<u32>,
  options2 : vec4<u32>,
  options3 : vec4<u32>,
};

struct NormalizationState {
  min_value : atomic<u32>,
  max_value : atomic<u32>,
  count : atomic<u32>,
  phase : atomic<u32>,
};

struct SinNormalizationState {
  min_value : atomic<u32>,
  max_value : atomic<u32>,
  count : atomic<u32>,
  phase : atomic<u32>,
};

// Dummy structures for bindings that are still part of the pipeline layout.
struct MaskData {
  values : array<f32>,
};

struct PermutationTableStorage {
  values : array<u32>,
};

const TAU : f32 = 6.283185307179586;
const PI : f32 = 3.141592653589793;

const OCTAVE_BLENDING_FALLOFF : u32 = 0u;
const OCTAVE_BLENDING_REDUCE_MAX : u32 = 10u;
const OCTAVE_BLENDING_ALPHA : u32 = 20u;

const COLOR_SPACE_GRAYSCALE : u32 = 1u;
const COLOR_SPACE_RGB : u32 = 11u;
const COLOR_SPACE_HSV : u32 = 21u;
const COLOR_SPACE_OKLAB : u32 = 31u;

const DISTRIB_NONE : u32 = 0u;
const DISTRIB_SIMPLEX : u32 = 1u;
const DISTRIB_EXP : u32 = 2u;
const DISTRIB_ONES : u32 = 5u;
const DISTRIB_MIDS : u32 = 6u;
const DISTRIB_ZEROS : u32 = 7u;
const DISTRIB_COLUMN_INDEX : u32 = 10u;
const DISTRIB_ROW_INDEX : u32 = 11u;
const DISTRIB_CENTER_CIRCLE : u32 = 20u;
const DISTRIB_CENTER_DIAMOND : u32 = 21u;
const DISTRIB_CENTER_TRIANGLE : u32 = 23u;
const DISTRIB_CENTER_SQUARE : u32 = 24u;
const DISTRIB_CENTER_PENTAGON : u32 = 25u;
const DISTRIB_CENTER_HEXAGON : u32 = 26u;
const DISTRIB_CENTER_HEPTAGON : u32 = 27u;
const DISTRIB_CENTER_OCTAGON : u32 = 28u;
const DISTRIB_CENTER_NONAGON : u32 = 29u;
const DISTRIB_CENTER_DECAGON : u32 = 30u;
const DISTRIB_CENTER_HENDECAGON : u32 = 31u;
const DISTRIB_CENTER_DODECAGON : u32 = 32u;

const INTERPOLATION_CONSTANT : u32 = 0u;
const INTERPOLATION_LINEAR : u32 = 1u;
const INTERPOLATION_COSINE : u32 = 2u;
const INTERPOLATION_BICUBIC : u32 = 3u;

const FLOAT_SIGN_BIT : u32 = 0x80000000u;
const F32_MAX : f32 = 3.402823466e38;

fn bool_from_u32(value : u32) -> bool {
  return value != 0u;
}

fn consume_u32(_value : u32) {
}

fn saturate(value : f32) -> f32 {
  return clamp(value, 0.0, 1.0);
}

fn float_is_valid(value : f32) -> bool {
  return value == value && abs(value) <= F32_MAX;
}

fn float_to_ordered_uint(value : f32) -> u32 {
  let bits : u32 = bitcast<u32>(value);
  if ((bits & FLOAT_SIGN_BIT) != 0u) {
    return ~bits;
  }
  return bits | FLOAT_SIGN_BIT;
}

fn ridge_transform(value : f32) -> f32 {
  return 1.0 - abs(value * 2.0 - 1.0);
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

fn random_from_cell(cell : vec2<i32>, seed : u32) -> f32 {
  let packed : vec3<u32> = vec3<u32>(
    bitcast<u32>(cell.x),
    bitcast<u32>(cell.y),
    seed,
  );
  let noise : vec3<u32> = pcg3d(packed);
  return f32(noise.x) / f32(0xffffffffu);
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

fn interpolation_weight(value : f32, spline_order : u32) -> f32 {
  if (spline_order == INTERPOLATION_COSINE) {
    let clamped : f32 = clamp(value, 0.0, 1.0);
    let angle : f32 = clamped * PI;
    let cos_value : f32 = cos(angle);
    return (1.0 - cos_value) * 0.5;
  }
  return value;
}

fn blend_cubic(a : f32, b : f32, c : f32, d : f32, g : f32) -> f32 {
  let t : f32 = clamp(g, 0.0, 1.0);
  let t2 : f32 = t * t;
  let a0 : f32 = ((d - c) - a) + b;
  let a1 : f32 = (a - b) - a0;
  let a2 : f32 = c - a;
  let a3 : f32 = b;
  let term1 : f32 = (a0 * t) * t2;
  let term2 : f32 = a1 * t2;
  let term3 : f32 = (a2 * t) + a3;
  return (term1 + term2) + term3;
}

fn sample_bicubic_layer(
  cell : vec2<i32>,
  frac : vec2<f32>,
  z_cell : i32,
  base_seed : u32,
) -> f32 {
  let row0 : f32 = blend_cubic(
    random_from_cell_3d(vec3<i32>(cell.x - 1, cell.y - 1, z_cell), base_seed),
    random_from_cell_3d(vec3<i32>(cell.x + 0, cell.y - 1, z_cell), base_seed),
    random_from_cell_3d(vec3<i32>(cell.x + 1, cell.y - 1, z_cell), base_seed),
    random_from_cell_3d(vec3<i32>(cell.x + 2, cell.y - 1, z_cell), base_seed),
    frac.x,
  );
  let row1 : f32 = blend_cubic(
    random_from_cell_3d(vec3<i32>(cell.x - 1, cell.y + 0, z_cell), base_seed),
    random_from_cell_3d(vec3<i32>(cell.x + 0, cell.y + 0, z_cell), base_seed),
    random_from_cell_3d(vec3<i32>(cell.x + 1, cell.y + 0, z_cell), base_seed),
    random_from_cell_3d(vec3<i32>(cell.x + 2, cell.y + 0, z_cell), base_seed),
    frac.x,
  );
  let row2 : f32 = blend_cubic(
    random_from_cell_3d(vec3<i32>(cell.x - 1, cell.y + 1, z_cell), base_seed),
    random_from_cell_3d(vec3<i32>(cell.x + 0, cell.y + 1, z_cell), base_seed),
    random_from_cell_3d(vec3<i32>(cell.x + 1, cell.y + 1, z_cell), base_seed),
    random_from_cell_3d(vec3<i32>(cell.x + 2, cell.y + 1, z_cell), base_seed),
    frac.x,
  );
  let row3 : f32 = blend_cubic(
    random_from_cell_3d(vec3<i32>(cell.x - 1, cell.y + 2, z_cell), base_seed),
    random_from_cell_3d(vec3<i32>(cell.x + 0, cell.y + 2, z_cell), base_seed),
    random_from_cell_3d(vec3<i32>(cell.x + 1, cell.y + 2, z_cell), base_seed),
    random_from_cell_3d(vec3<i32>(cell.x + 2, cell.y + 2, z_cell), base_seed),
    frac.x,
  );
  return blend_cubic(row0, row1, row2, row3, frac.y);
}

fn sample_value_noise(
  uv : vec2<f32>,
  freq : vec2<f32>,
  seed : u32,
  channel : u32,
  octave : u32,
  time_value : f32,
  speed : f32,
  spline_order : u32,
) -> f32 {
  let scaled_freq : vec2<f32> = max(freq, vec2<f32>(1.0, 1.0));
  let scaled_uv : vec2<f32> = uv * scaled_freq;
  let cell_f : vec2<f32> = floor(scaled_uv);
  let cell : vec2<i32> = vec2<i32>(i32(cell_f.x), i32(cell_f.y));
  let frac : vec2<f32> = fract(scaled_uv);
  let salt : u32 = (channel * 0x9e3779b9u) ^ (octave * 0x85ebca6bu);
  let base_seed : u32 = seed ^ salt;

  let angle : f32 = time_value * TAU;
  let time_coord : f32 = cos(angle) * speed;
  let time_floor : f32 = floor(time_coord);
  let time_cell : i32 = i32(time_floor);
  let time_frac : f32 = fract(time_coord);

  if (spline_order == INTERPOLATION_CONSTANT) {
    return random_from_cell_3d(vec3<i32>(cell.x, cell.y, time_cell), base_seed);
  }

  if (spline_order == INTERPOLATION_LINEAR || spline_order == INTERPOLATION_COSINE) {
    let weight_x : f32 = interpolation_weight(frac.x, spline_order);
    let weight_y : f32 = interpolation_weight(frac.y, spline_order);
    let weight_z : f32 = interpolation_weight(time_frac, spline_order);
    let v000 : f32 = random_from_cell_3d(vec3<i32>(cell.x, cell.y, time_cell), base_seed);
    let v100 : f32 = random_from_cell_3d(vec3<i32>(cell.x + 1, cell.y, time_cell), base_seed);
    let v010 : f32 = random_from_cell_3d(vec3<i32>(cell.x, cell.y + 1, time_cell), base_seed);
    let v110 : f32 = random_from_cell_3d(vec3<i32>(cell.x + 1, cell.y + 1, time_cell), base_seed);
    let v001 : f32 = random_from_cell_3d(vec3<i32>(cell.x, cell.y, time_cell + 1), base_seed);
    let v101 : f32 = random_from_cell_3d(vec3<i32>(cell.x + 1, cell.y, time_cell + 1), base_seed);
    let v011 : f32 = random_from_cell_3d(vec3<i32>(cell.x, cell.y + 1, time_cell + 1), base_seed);
    let v111 : f32 = random_from_cell_3d(vec3<i32>(cell.x + 1, cell.y + 1, time_cell + 1), base_seed);

    let x00 : f32 = mix(v000, v100, weight_x);
    let x10 : f32 = mix(v010, v110, weight_x);
    let x01 : f32 = mix(v001, v101, weight_x);
    let x11 : f32 = mix(v011, v111, weight_x);
    let y0 : f32 = mix(x00, x10, weight_y);
    let y1 : f32 = mix(x01, x11, weight_y);
    return mix(y0, y1, weight_z);
  }

  let slice0 : f32 = sample_bicubic_layer(cell, frac, time_cell - 1, base_seed);
  let slice1 : f32 = sample_bicubic_layer(cell, frac, time_cell + 0, base_seed);
  let slice2 : f32 = sample_bicubic_layer(cell, frac, time_cell + 1, base_seed);
  let slice3 : f32 = sample_bicubic_layer(cell, frac, time_cell + 2, base_seed);
  return blend_cubic(slice0, slice1, slice2, slice3, time_frac);
}

fn regular_polygon_weight(centered : vec2<f32>, sides : f32) -> f32 {
  if (sides <= 0.0) {
    return 0.0;
  }
  let angle : f32 = atan2(centered.y, centered.x);
  let radius : f32 = length(centered);
  if (radius == 0.0) {
    return 1.0;
  }
  let sector : f32 = TAU / sides;
  let snapped : f32 = floor(0.5 + angle / sector) * sector;
  let distance : f32 = cos(snapped - angle) * radius;
  return saturate(1.0 - distance * 2.0);
}

fn center_distribution(distrib : u32, uv : vec2<f32>, aspect_ratio : f32) -> f32 {
  let center : vec2<f32> = vec2<f32>(0.5 * aspect_ratio, 0.5);
  let scaled : vec2<f32> = vec2<f32>(uv.x * aspect_ratio, uv.y) - center;
  let circle : f32 = saturate(1.0 - length(scaled) * 2.0);

  if (distrib == DISTRIB_CENTER_CIRCLE) {
    return circle;
  }
  if (distrib == DISTRIB_CENTER_DIAMOND) {
    let diamond : f32 = saturate(1.0 - (abs(scaled.x) + abs(scaled.y)) * 1.5);
    return diamond;
  }
  if (distrib == DISTRIB_CENTER_SQUARE) {
    let square : f32 = saturate(1.0 - max(abs(scaled.x), abs(scaled.y)) * 2.0);
    return square;
  }

  var sides : u32 = 0u;
  if (distrib >= DISTRIB_CENTER_TRIANGLE && distrib <= DISTRIB_CENTER_DODECAGON) {
    sides = 3u + (distrib - DISTRIB_CENTER_TRIANGLE);
  }
  if (sides >= 3u) {
    return regular_polygon_weight(scaled, f32(sides));
  }
  return circle;
}

fn apply_distribution(
  base_value : f32,
  distrib : u32,
  uv : vec2<f32>,
  aspect_ratio : f32,
) -> f32 {
  if (distrib == DISTRIB_NONE || distrib == 0u) {
    return base_value;
  }
  switch (distrib) {
    case DISTRIB_SIMPLEX: {
      return base_value;
    }
    case DISTRIB_EXP: {
      return pow(base_value, 3.0);
    }
    case DISTRIB_ONES: {
      return 1.0;
    }
    case DISTRIB_MIDS: {
      return 0.5;
    }
    case DISTRIB_ZEROS: {
      return 0.0;
    }
    case DISTRIB_COLUMN_INDEX: {
      return saturate(uv.x);
    }
    case DISTRIB_ROW_INDEX: {
      return saturate(uv.y);
    }
    default: {
      if (distrib >= DISTRIB_CENTER_CIRCLE) {
        return center_distribution(distrib, uv, aspect_ratio);
      }
      return base_value;
    }
  }
}

fn sample_distribution_value(
  uv : vec2<f32>,
  freq : vec2<f32>,
  seed : u32,
  distrib : u32,
  octave : u32,
  time_value : f32,
  speed : f32,
  spline_order : u32,
  aspect_ratio : f32,
) -> f32 {
  let noise_value : f32 = sample_value_noise(
    uv,
    freq,
    seed,
    0u,
    octave,
    time_value,
    speed,
    spline_order,
  );
  return apply_distribution(noise_value, distrib, uv, aspect_ratio);
}

fn linear_to_srgb_component(value : f32) -> f32 {
  if (value <= 0.0031308) {
    return value * 12.92;
  }
  return 1.055 * pow(max(value, 0.0), 1.0 / 2.4) - 0.055;
}

fn linear_to_srgb(linear : vec3<f32>) -> vec3<f32> {
  return vec3<f32>(
    linear_to_srgb_component(linear.x),
    linear_to_srgb_component(linear.y),
    linear_to_srgb_component(linear.z),
  );
}

fn rgb_to_hsv(rgb : vec3<f32>) -> vec3<f32> {
  let cmax : f32 = max(max(rgb.x, rgb.y), rgb.z);
  let cmin : f32 = min(min(rgb.x, rgb.y), rgb.z);
  let delta : f32 = cmax - cmin;

  var hue : f32 = 0.0;
  if (delta > 0.0) {
    if (cmax == rgb.x) {
      hue = (rgb.y - rgb.z) / delta;
      if (hue < 0.0) {
        hue = hue + 6.0;
      }
    } else if (cmax == rgb.y) {
      hue = ((rgb.z - rgb.x) / delta) + 2.0;
    } else {
      hue = ((rgb.x - rgb.y) / delta) + 4.0;
    }
    hue = hue / 6.0;
  }

  var saturation : f32 = 0.0;
  if (cmax > 0.0) {
    saturation = delta / cmax;
  }

  return vec3<f32>(hue, saturation, cmax);
}

fn hsv_to_rgb(hsv : vec3<f32>) -> vec3<f32> {
  let hue : f32 = hsv.x;
  let saturation : f32 = hsv.y;
  let value : f32 = hsv.z;

  let dh : f32 = hue * 6.0;
  let dr : f32 = clamp(abs(dh - 3.0) - 1.0, 0.0, 1.0);
  let dg : f32 = clamp(-abs(dh - 2.0) + 2.0, 0.0, 1.0);
  let db : f32 = clamp(-abs(dh - 4.0) + 2.0, 0.0, 1.0);

  let one_minus_s : f32 = 1.0 - saturation;
  let sr : f32 = saturation * dr;
  let sg : f32 = saturation * dg;
  let sb : f32 = saturation * db;

  let r : f32 = (one_minus_s + sr) * value;
  let g : f32 = (one_minus_s + sg) * value;
  let b : f32 = (one_minus_s + sb) * value;

  return vec3<f32>(r, g, b);
}

const OKLAB_FWD_A : mat3x3<f32> = mat3x3<f32>(
  vec3<f32>(1.0, 1.0, 1.0),
  vec3<f32>(0.3963377774, -0.1055613458, -0.0894841775),
  vec3<f32>(0.2158037573, -0.0638541728, -1.2914855480),
);

const OKLAB_FWD_B : mat3x3<f32> = mat3x3<f32>(
  vec3<f32>(4.0767245293, -1.2681437731, -0.0041119885),
  vec3<f32>(-3.3072168827, 2.6093323231, -0.7034763098),
  vec3<f32>(0.2307590544, -0.3411344290, 1.7068625689),
);

fn oklab_to_srgb(lab : vec3<f32>) -> vec3<f32> {
  let lms : vec3<f32> = OKLAB_FWD_A * lab;
  let cubic : vec3<f32> = vec3<f32>(lms.x * lms.x * lms.x, lms.y * lms.y * lms.y, lms.z * lms.z * lms.z);
  return linear_to_srgb(OKLAB_FWD_B * cubic);
}

fn compute_octave_frequency(base_freq : vec2<f32>, octave_index : u32) -> vec2<f32> {
  let multiplier : f32 = pow(2.0, f32(octave_index));
  return base_freq * multiplier;
}

fn combine_alpha(base_color : vec4<f32>, layer : vec4<f32>) -> vec4<f32> {
  let alpha_vec : vec4<f32> = vec4<f32>(layer.w, layer.w, layer.w, layer.w);
  return base_color * (vec4<f32>(1.0, 1.0, 1.0, 1.0) - alpha_vec) + layer * alpha_vec;
}

fn update_normalization(sample : vec4<f32>, with_alpha : bool, state : ptr<storage, NormalizationState, read_write>) {
  var min_value : f32 = 0.0;
  var max_value : f32 = 0.0;
  var has_valid_sample : bool = false;

  let components : array<f32, 4> = array<f32, 4>(sample.x, sample.y, sample.z, sample.w);
  let limit : u32 = select(3u, 4u, with_alpha);

  for (var i : u32 = 0u; i < limit; i = i + 1u) {
    let value : f32 = components[i];
    if (!float_is_valid(value)) {
      continue;
    }
    if (!has_valid_sample) {
      min_value = value;
      max_value = value;
      has_valid_sample = true;
    } else {
      min_value = min(min_value, value);
      max_value = max(max_value, value);
    }
  }

  if (!has_valid_sample) {
    return;
  }

  atomicMin(&(*state).min_value, float_to_ordered_uint(min_value));
  atomicMax(&(*state).max_value, float_to_ordered_uint(max_value));
}

@group(0) @binding(0) var<uniform> stage_uniforms : StageUniforms;
@group(0) @binding(1) var<uniform> frame_uniforms : FrameUniforms;
@group(0) @binding(3) var output_texture : texture_storage_2d<rgba32float, write>;
@group(0) @binding(4) var<storage, read_write> normalization_state : NormalizationState;
@group(0) @binding(5) var<storage, read_write> sin_state : SinNormalizationState;
@group(0) @binding(6) var<storage, read> mask_data : MaskData;
@group(0) @binding(7) var<storage, read> permutation_table_storage : PermutationTableStorage;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
  let width : u32 = u32(frame_uniforms.resolution.x);
  let height : u32 = u32(frame_uniforms.resolution.y);
  if (global_id.x >= width || global_id.y >= height) {
    return;
  }

  // Touch optional bindings so the compiler keeps them alive when the stage
  // descriptor still provides the associated buffers.
  consume_u32(atomicLoad(&sin_state.phase));
  consume_u32(arrayLength(&mask_data.values));
  consume_u32(arrayLength(&permutation_table_storage.values));

  let resolution_vec : vec2<f32> = vec2<f32>(frame_uniforms.resolution.x, frame_uniforms.resolution.y);
  let aspect_ratio : f32 = select(1.0, resolution_vec.x / max(resolution_vec.y, 1.0), resolution_vec.y != 0.0);
  let pixel : vec2<f32> = vec2<f32>(f32(global_id.x) + 0.5, f32(global_id.y) + 0.5);
  let uv : vec2<f32> = pixel / resolution_vec;

  let base_freq : vec2<f32> = max(stage_uniforms.freq, vec2<f32>(1.0, 1.0));
  let octaves : u32 = max(stage_uniforms.options0.x, 1u);
  let octave_blending : u32 = stage_uniforms.options0.y;
  let ridges_enabled : bool = bool_from_u32(stage_uniforms.options0.w);
  let seed_offset : u32 = stage_uniforms.options1.x;
  let distrib : u32 = stage_uniforms.options1.y;
  let color_space : u32 = stage_uniforms.options1.z;
  let with_alpha_output : bool = bool_from_u32(stage_uniforms.options1.w);
  let hue_distrib : u32 = stage_uniforms.options2.x;
  let saturation_distrib : u32 = stage_uniforms.options2.y;
  let brightness_distrib : u32 = stage_uniforms.options2.z;
  let sin_amount : f32 = stage_uniforms.sin;

  let hue_range : f32 = stage_uniforms.colorParams0.x;
  let hue_rotation_degrees : f32 = stage_uniforms.colorParams0.y;
  let saturation_scale : f32 = stage_uniforms.colorParams0.z;
  let spline_order : u32 = u32(
    clamp(stage_uniforms.colorParams1.w, 0.0, f32(INTERPOLATION_BICUBIC)),
  );

  let time_value : f32 = frame_uniforms.time;
  let speed : f32 = stage_uniforms.speed;
  let base_seed : u32 = frame_uniforms.seed + seed_offset;

  var accum : vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);

  for (var octave_index : u32 = 0u; octave_index < octaves; octave_index = octave_index + 1u) {
    let octave_freq : vec2<f32> = compute_octave_frequency(base_freq, octave_index);
    let octave_seed : u32 = base_seed ^ (octave_index * 0x9e3779b9u + 0x7f4a7c15u);

    let c0 : f32 = sample_value_noise(
      uv,
      octave_freq,
      octave_seed,
      0u,
      octave_index,
      time_value,
      speed,
      spline_order,
    );
    let c1 : f32 = sample_value_noise(
      uv,
      octave_freq,
      octave_seed,
      1u,
      octave_index,
      time_value,
      speed,
      spline_order,
    );
    let c2 : f32 = sample_value_noise(
      uv,
      octave_freq,
      octave_seed,
      2u,
      octave_index,
      time_value,
      speed,
      spline_order,
    );
    var layer_color : vec3<f32> = vec3<f32>(c0, c1, c2);

    let override_seed : u32 = octave_seed ^ 0x94d049b4u;
    var hue_value : f32 = fract(layer_color.x * hue_range + hue_rotation_degrees / 360.0);
    var saturation_value : f32 = layer_color.y * saturation_scale;
    var brightness_value : f32 = apply_distribution(layer_color.z, distrib, uv, aspect_ratio);

    if (hue_distrib != 0u) {
      hue_value = sample_distribution_value(
        uv,
        octave_freq,
        override_seed ^ 0x1u,
        hue_distrib,
        octave_index,
        time_value,
        speed,
        spline_order,
        aspect_ratio,
      );
      hue_value = fract(hue_value + hue_rotation_degrees / 360.0);
    }
    if (saturation_distrib != 0u) {
      saturation_value = sample_distribution_value(
        uv,
        octave_freq,
        override_seed ^ 0x2u,
        saturation_distrib,
        octave_index,
        time_value,
        speed,
        spline_order,
        aspect_ratio,
      );
    }
    if (brightness_distrib != 0u) {
      brightness_value = sample_distribution_value(
        uv,
        octave_freq,
        override_seed ^ 0x3u,
        brightness_distrib,
        octave_index,
        time_value,
        speed,
        spline_order,
        aspect_ratio,
      );
    }

    if (ridges_enabled) {
      brightness_value = ridge_transform(brightness_value);
    }

    var rgb_color : vec3<f32>;
    if (color_space == COLOR_SPACE_GRAYSCALE) {
      rgb_color = vec3<f32>(brightness_value, brightness_value, brightness_value);
    } else if (color_space == COLOR_SPACE_RGB) {
      rgb_color = clamp(layer_color, vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(1.0, 1.0, 1.0));
    } else if (color_space == COLOR_SPACE_OKLAB) {
      let oklab_color : vec3<f32> = vec3<f32>(
        layer_color.x,
        layer_color.y * -0.509 + 0.276,
        layer_color.z * -0.509 + 0.198,
      );
      rgb_color = clamp(
        oklab_to_srgb(oklab_color),
        vec3<f32>(0.0, 0.0, 0.0),
        vec3<f32>(1.0, 1.0, 1.0),
      );
    } else {
      rgb_color = hsv_to_rgb(vec3<f32>(hue_value, saturation_value, brightness_value));
    }

    var layer_alpha : f32 = brightness_value;
    if (!with_alpha_output) {
      layer_alpha = 1.0;
    }

    var layer_rgba : vec4<f32> = vec4<f32>(rgb_color, layer_alpha);

    if (octave_blending == OCTAVE_BLENDING_REDUCE_MAX) {
      accum = max(accum, layer_rgba);
    } else if (octave_blending == OCTAVE_BLENDING_ALPHA) {
      accum = combine_alpha(accum, layer_rgba);
    } else {
      let weight : f32 = pow(0.5, f32(octave_index + 1u));
      accum = accum + layer_rgba * weight;
    }
  }

  var final_color : vec4<f32> = accum;
  if (!with_alpha_output) {
    final_color.w = 1.0;
  }

  var hsv_final : vec3<f32> = rgb_to_hsv(final_color.xyz);
  hsv_final.x = fract(hsv_final.x + hue_rotation_degrees / 360.0);
  hsv_final.y = hsv_final.y * saturation_scale;
  if (ridges_enabled) {
    hsv_final.z = ridge_transform(hsv_final.z);
  }
  if (sin_amount != 0.0) {
    hsv_final.z = sin(hsv_final.z * TAU * sin_amount) * 0.5 + 0.5;
  }
  let rgb_final : vec3<f32> = hsv_to_rgb(hsv_final);
  let raw_final_color : vec4<f32> = vec4<f32>(rgb_final, final_color.w);

  update_normalization(raw_final_color, with_alpha_output, &normalization_state);

  textureStore(
    output_texture,
    vec2<i32>(i32(global_id.x), i32(global_id.y)),
    raw_final_color,
  );
}

