// Multi-resolution generator compute shader (work in progress).
//
// This module ports the CPU "multires" generator to WebGPU. The implementation
// is intentionally verbose so that it mirrors the reference JavaScript and
// Python code paths line-for-line. Large helper blocks (e.g. permutation table
// construction and OpenSimplex evaluation) are included here even though the
// entry point has not yet been fully wired into the pipeline. Future passes
// will flesh out the remaining pieces and hook the shader up to the runtime
// program builder.

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

const PERMUTATION_SIZE : u32 = 256u;
const GRADIENTS_2D_LENGTH : u32 = 16u;
const GRADIENTS_3D_LENGTH : u32 = 72u;
const GRADIENTS_3D_COUNT : u32 = GRADIENTS_3D_LENGTH / 3u;

const STRETCH_CONSTANT_2D : f32 = -0.211324865405187;
const SQUISH_CONSTANT_2D : f32 = 0.366025403784439;
const STRETCH_CONSTANT_3D : f32 = -0.16666666666666666;
const SQUISH_CONSTANT_3D : f32 = 0.3333333333333333;
const NORM_CONSTANT_2D : f32 = 47.0;
const NORM_CONSTANT_3D : f32 = 103.0;

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

const INV_SQRT2 : f32 = 0.7071067811865476;

const EPSILON : f32 = 0.000001;

const F32_MAX : f32 = 3.402823466e38;

const LAYER_FLAG_HAS_SIN_HSV : u32 = 1u;

const GRADIENTS_2D : array<i32, 16> = array<i32, 16>(
  5, 2, 2, 5,
  -5, 2, -2, 5,
  5, -2, 2, -5,
  -5, -2, -2, -5,
);

const GRADIENTS_3D : array<i32, 72> = array<i32, 72>(
  -11, 4, 4, -4, 11, 4, -4, 4, 11,
  11, 4, 4, 4, 11, 4, 4, 4, 11,
  -11, -4, 4, -4, -11, 4, -4, -4, 11,
  11, -4, 4, 4, -11, 4, 4, -4, 11,
  -11, 4, -4, -4, 11, -4, -4, 4, -11,
  11, 4, -4, 4, 11, -4, 4, 4, -11,
  -11, -4, -4, -4, -11, -4, -4, -4, -11,
  11, -4, -4, 4, -11, -4, 4, -4, -11,
);

fn bool_from_u32(value : u32) -> bool {
  return value != 0u;
}

fn saturate(value : f32) -> f32 {
  return clamp(value, 0.0, 1.0);
}

fn replicate4(value : f32) -> vec4<f32> {
  return vec4<f32>(value, value, value, value);
}

fn map_to_unit(value : f32) -> f32 {
  return value * 0.5 + 0.5;
}

fn ridge_transform(value : f32) -> f32 {
  return 1.0 - abs(value * 2.0 - 1.0);
}

fn linear_to_srgb_component(value : f32) -> f32 {
  let threshold : f32 = 0.0031308;
  if (value <= threshold) {
    return value * 12.92;
  }
  let clamped_value : f32 = max(value, 0.0);
  return 1.055 * pow(clamped_value, 1.0 / 2.4) - 0.055;
}

fn linear_to_srgb(linear : vec3<f32>) -> vec3<f32> {
  return vec3<f32>(
    linear_to_srgb_component(linear.x),
    linear_to_srgb_component(linear.y),
    linear_to_srgb_component(linear.z)
  );
}

fn rgb_to_hsv(rgb : vec3<f32>) -> vec3<f32> {
  let cmax : f32 = max(max(rgb.x, rgb.y), rgb.z);
  let cmin : f32 = min(min(rgb.x, rgb.y), rgb.z);
  let delta : f32 = cmax - cmin;

  var hue : f32 = 0.0;
  if (delta > EPSILON) {
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
    if (hue < 0.0) {
      hue = hue + 1.0;
    }
  }

  var saturation : f32 = 0.0;
  if (cmax > EPSILON) {
    saturation = delta / cmax;
  }

  return vec3<f32>(hue, saturation, cmax);
}

fn hsv_to_rgb(hsv : vec3<f32>) -> vec3<f32> {
  let hue : f32 = fract(hsv.x);
  let saturation : f32 = clamp(hsv.y, 0.0, 1.0);
  let value : f32 = clamp(hsv.z, 0.0, 1.0);

  let chroma : f32 = value * saturation;
  let h_prime : f32 = hue * 6.0;
  let x : f32 = chroma * (1.0 - abs(fract(h_prime) * 2.0 - 1.0));
  let m : f32 = value - chroma;

  let sector : u32 = u32(floor(h_prime)) % 6u;
  var rgb : vec3<f32>;
  switch (sector) {
    case 0u: {
      rgb = vec3<f32>(chroma, x, 0.0);
    }
    case 1u: {
      rgb = vec3<f32>(x, chroma, 0.0);
    }
    case 2u: {
      rgb = vec3<f32>(0.0, chroma, x);
    }
    case 3u: {
      rgb = vec3<f32>(0.0, x, chroma);
    }
    case 4u: {
      rgb = vec3<f32>(x, 0.0, chroma);
    }
    default: {
      rgb = vec3<f32>(chroma, 0.0, x);
    }
  }

  return rgb + vec3<f32>(m, m, m);
}

fn oklab_to_rgb(lab : vec3<f32>) -> vec3<f32> {
  let L : f32 = lab.x;
  let a : f32 = lab.y;
  let b : f32 = lab.z;

  let l_ : f32 = L + 0.3963377774 * a + 0.2158037573 * b;
  let m_ : f32 = L - 0.1055613458 * a - 0.0638541728 * b;
  let s_ : f32 = L - 0.0894841775 * a - 1.2914855480 * b;

  let l : f32 = l_ * l_ * l_;
  let m : f32 = m_ * m_ * m_;
  let s : f32 = s_ * s_ * s_;

  let linear : vec3<f32> = vec3<f32>(
    4.0767245293 * l - 3.3072168827 * m + 0.2307590544 * s,
    -1.2681437731 * l + 2.6093323231 * m - 0.3411344290 * s,
    -0.0041119885 * l - 0.7034763098 * m + 1.7068625689 * s
  );

  return linear_to_srgb(linear);
}

fn compute_octave_frequency(base_freq : vec2<f32>, octave_index : u32) -> vec2<f32> {
  let multiplier : f32 = pow(2.0, f32(octave_index));
  var freq : vec2<f32> = floor(base_freq * 0.5 * multiplier);
  if (freq.x < 1.0) {
    freq.x = 1.0;
  }
  if (freq.y < 1.0) {
    freq.y = 1.0;
  }
  return freq;
}

fn compute_pin_offset(
  freq : vec2<f32>,
  corners_enabled : bool,
  resolution : vec2<f32>,
) -> vec2<f32> {
  let fy : f32 = max(freq.y, 1.0);
  let fx : f32 = max(freq.x, 1.0);
  let fy_int : u32 = u32(floor(fy));
  var apply_offset : bool = false;
  if (!corners_enabled) {
    apply_offset = (fy_int & 1u) == 0u;
  } else {
    apply_offset = (fy_int & 1u) == 1u;
  }
  if (!apply_offset) {
    return vec2<f32>(0.0, 0.0);
  }
  let cell_width : f32 = resolution.x / fx;
  let cell_height : f32 = resolution.y / fy;
  let offset_x : f32 = floor(cell_width * 0.5);
  let offset_y : f32 = floor(cell_height * 0.5);
  return vec2<f32>(offset_x / resolution.x, offset_y / resolution.y);
}

fn combine_alpha(accum : ptr<function, vec4<f32>>, layer : vec4<f32>) {
  let alpha_value : f32 = saturate(layer.w);
  let alpha_vec : vec4<f32> = replicate4(alpha_value);
  (*accum) = (*accum) * (vec4<f32>(1.0, 1.0, 1.0, 1.0) - alpha_vec) + layer * alpha_vec;
}

struct LayerResult {
  color : vec4<f32>,
  hsv : vec4<f32>,
  flags : u32,
};

fn float_to_ordered_int(value : f32) -> i32 {
  let bits : i32 = bitcast<i32>(value);
  let mask : i32 = bits >> 31;
  return bits ^ mask;
}

fn ordered_int_to_float(value : i32) -> f32 {
  let mask : i32 = value >> 31;
  return bitcast<f32>(value ^ mask);
}

struct PermutationTables {
  perm : array<u32, PERMUTATION_SIZE>,
  perm_grad_index3d : array<u32, PERMUTATION_SIZE>,
};

struct RandomState {
  state : u32,
};

struct SinNormalizationState {
  min_value : atomic<i32>,
  max_value : atomic<i32>,
  count : atomic<u32>,
  phase : atomic<u32>,
};

fn mulberry32_step(state : ptr<function, RandomState>) -> u32 {
  var t : u32 = (*state).state + 0x6D2B79F5u;
  t = (t ^ (t >> 15u)) * (t | 1u);
  let mix : u32 = (t ^ (t >> 7u)) * (t | 61u);
  t = t ^ (t + mix);
  (*state).state = t;
  return t ^ (t >> 14u);
}

fn random_float(state : ptr<function, RandomState>) -> f32 {
  let bits : u32 = mulberry32_step(state);
  return f32(bits) / 4294967296.0;
}

fn random_int_inclusive(state : ptr<function, RandomState>, min_val : i32, max_val : i32) -> i32 {
  var lo : i32 = min_val;
  var hi : i32 = max_val;
  if (hi < lo) {
    let tmp = lo;
    lo = hi;
    hi = tmp;
  }
  let span : f32 = f32(hi - lo + 1);
  let sample : f32 = random_float(state) * span;
  let idx : i32 = i32(floor(sample));
  if (idx > hi - lo) {
    idx = hi - lo;
  }
  return lo + idx;
}

fn build_permutation_tables(seed : u32) -> PermutationTables {
  var tables : PermutationTables;
  var source : array<u32, PERMUTATION_SIZE>;
  var rng : RandomState = RandomState(seed);
  for (var i : u32 = 0u; i < PERMUTATION_SIZE; i = i + 1u) {
    source[i] = i;
  }
  var idx : i32 = i32(PERMUTATION_SIZE) - 1;
  loop {
    if (idx < 0) {
      break;
    }
    let choice : u32 = u32(random_int_inclusive(&rng, 0, idx));
    let dest : u32 = u32(idx);
    let value : u32 = source[choice];
    tables.perm[dest] = value;
    tables.perm_grad_index3d[dest] = (value % GRADIENTS_3D_COUNT) * 3u;
    source[choice] = source[dest];
    idx = idx - 1;
  }
  return tables;
}

fn extrapolate2d(tables : ptr<function, PermutationTables>, xsb : i32, ysb : i32, dx : f32, dy : f32) -> f32 {
  let px : u32 = (*tables).perm[u32(xsb & 255)];
  let index : u32 = ((*tables).perm[(px + u32(ysb & 255)) & 255u] & 0x0eu);
  let g1 : f32 = f32(GRADIENTS_2D[index]);
  let g2 : f32 = f32(GRADIENTS_2D[index + 1u]);
  return g1 * dx + g2 * dy;
}

fn extrapolate3d(tables : ptr<function, PermutationTables>, xsb : i32, ysb : i32, zsb : i32, dx : f32, dy : f32, dz : f32) -> f32 {
  let px : u32 = (*tables).perm[u32(xsb & 255)];
  let py : u32 = (*tables).perm[(px + u32(ysb & 255)) & 255u];
  let index : u32 = (*tables).perm_grad_index3d[(py + u32(zsb & 255)) & 255u];
  let g1 : f32 = f32(GRADIENTS_3D[index]);
  let g2 : f32 = f32(GRADIENTS_3D[index + 1u]);
  let g3 : f32 = f32(GRADIENTS_3D[index + 2u]);
  return g1 * dx + g2 * dy + g3 * dz;
}

fn open_simplex_2d(tables : ptr<function, PermutationTables>, x : f32, y : f32) -> f32 {
  let stretch_offset : f32 = (x + y) * STRETCH_CONSTANT_2D;
  let xs : f32 = x + stretch_offset;
  let ys : f32 = y + stretch_offset;
  var xsb : i32 = i32(floor(xs));
  var ysb : i32 = i32(floor(ys));
  let squish_offset : f32 = f32(xsb + ysb) * SQUISH_CONSTANT_2D;
  let xb : f32 = f32(xsb) + squish_offset;
  let yb : f32 = f32(ysb) + squish_offset;
  let xins : f32 = xs - f32(xsb);
  let yins : f32 = ys - f32(ysb);
  let in_sum : f32 = xins + yins;
  var dx0 : f32 = x - xb;
  var dy0 : f32 = y - yb;
  var value : f32 = 0.0;

  var dx1 : f32 = dx0 - 1.0 - SQUISH_CONSTANT_2D;
  var dy1 : f32 = dy0 - SQUISH_CONSTANT_2D;
  var attn1 : f32 = 2.0 - dx1 * dx1 - dy1 * dy1;
  if (attn1 > 0.0) {
    attn1 = attn1 * attn1;
    value = value + attn1 * attn1 * extrapolate2d(tables, xsb + 1, ysb, dx1, dy1);
  }

  var dx2 : f32 = dx0 - SQUISH_CONSTANT_2D;
  var dy2 : f32 = dy0 - 1.0 - SQUISH_CONSTANT_2D;
  var attn2 : f32 = 2.0 - dx2 * dx2 - dy2 * dy2;
  if (attn2 > 0.0) {
    attn2 = attn2 * attn2;
    value = value + attn2 * attn2 * extrapolate2d(tables, xsb, ysb + 1, dx2, dy2);
  }

  var xsv_ext : i32;
  var ysv_ext : i32;
  var dx_ext : f32;
  var dy_ext : f32;

  if (in_sum <= 1.0) {
    let zins : f32 = 1.0 - in_sum;
    if (zins > xins || zins > yins) {
      if (xins > yins) {
        xsv_ext = xsb + 1;
        ysv_ext = ysb - 1;
        dx_ext = dx0 - 1.0;
        dy_ext = dy0 + 1.0;
      } else {
        xsv_ext = xsb - 1;
        ysv_ext = ysb + 1;
        dx_ext = dx0 + 1.0;
        dy_ext = dy0 - 1.0;
      }
    } else {
      xsv_ext = xsb + 1;
      ysv_ext = ysb + 1;
      dx_ext = dx0 - 1.0 - 2.0 * SQUISH_CONSTANT_2D;
      dy_ext = dy0 - 1.0 - 2.0 * SQUISH_CONSTANT_2D;
    }
  } else {
    let zins : f32 = 2.0 - in_sum;
    if (zins < xins || zins < yins) {
      if (xins > yins) {
        xsv_ext = xsb + 2;
        ysv_ext = ysb;
        dx_ext = dx0 - 2.0 - 2.0 * SQUISH_CONSTANT_2D;
        dy_ext = dy0 - 2.0 * SQUISH_CONSTANT_2D;
      } else {
        xsv_ext = xsb;
        ysv_ext = ysb + 2;
        dx_ext = dx0 - 2.0 * SQUISH_CONSTANT_2D;
        dy_ext = dy0 - 2.0 - 2.0 * SQUISH_CONSTANT_2D;
      }
    } else {
      xsv_ext = xsb;
      ysv_ext = ysb;
      dx_ext = dx0;
      dy_ext = dy0;
    }
    xsb = xsb + 1;
    ysb = ysb + 1;
    dx0 = dx0 - 1.0 - 2.0 * SQUISH_CONSTANT_2D;
    dy0 = dy0 - 1.0 - 2.0 * SQUISH_CONSTANT_2D;
  }

  var attn0 : f32 = 2.0 - dx0 * dx0 - dy0 * dy0;
  if (attn0 > 0.0) {
    attn0 = attn0 * attn0;
    value = value + attn0 * attn0 * extrapolate2d(tables, xsb, ysb, dx0, dy0);
  }

  var attn_ext : f32 = 2.0 - dx_ext * dx_ext - dy_ext * dy_ext;
  if (attn_ext > 0.0) {
    attn_ext = attn_ext * attn_ext;
    value = value + attn_ext * attn_ext * extrapolate2d(tables, xsv_ext, ysv_ext, dx_ext, dy_ext);
  }

  return value / NORM_CONSTANT_2D;
}

fn open_simplex_3d(tables : ptr<function, PermutationTables>, x : f32, y : f32, z : f32) -> f32 {
  let stretch_offset : f32 = (x + y + z) * STRETCH_CONSTANT_3D;
  let xs : f32 = x + stretch_offset;
  let ys : f32 = y + stretch_offset;
  let zs : f32 = z + stretch_offset;
  var xsb : i32 = i32(floor(xs));
  var ysb : i32 = i32(floor(ys));
  var zsb : i32 = i32(floor(zs));
  let squish_offset : f32 = f32(xsb + ysb + zsb) * SQUISH_CONSTANT_3D;
  var dx0 : f32 = x - (f32(xsb) + squish_offset);
  var dy0 : f32 = y - (f32(ysb) + squish_offset);
  var dz0 : f32 = z - (f32(zsb) + squish_offset);
  let xins : f32 = xs - f32(xsb);
  let yins : f32 = ys - f32(ysb);
  let zins : f32 = zs - f32(zsb);
  let in_sum : f32 = xins + yins + zins;

  var value : f32 = 0.0;

  var dx_ext0 : f32 = 0.0;
  var dy_ext0 : f32 = 0.0;
  var dz_ext0 : f32 = 0.0;
  var dx_ext1 : f32 = 0.0;
  var dy_ext1 : f32 = 0.0;
  var dz_ext1 : f32 = 0.0;
  var xsv_ext0 : i32 = 0;
  var ysv_ext0 : i32 = 0;
  var zsv_ext0 : i32 = 0;
  var xsv_ext1 : i32 = 0;
  var ysv_ext1 : i32 = 0;
  var zsv_ext1 : i32 = 0;

  if (in_sum <= 1.0) {
    let a_score : f32 = xins;
    var a_point : u32 = 0x01u;
    var a_is_further_side : bool = false;
    if (yins > a_score) {
      a_score = yins;
      a_point = 0x02u;
    }
    if (zins > a_score) {
      a_score = zins;
      a_point = 0x04u;
    }

    let b_score : f32 = 1.0 - xins;
    var b_point : u32 = 0x01u;
    var b_is_further_side : bool = true;
    if (1.0 - yins > b_score) {
      b_score = 1.0 - yins;
      b_point = 0x02u;
    }
    if (1.0 - zins > b_score) {
      b_score = 1.0 - zins;
      b_point = 0x04u;
    }

    let p1 : vec3<i32> = vec3<i32>(a_point & 0x01u != 0u ? 1 : 0, a_point & 0x02u != 0u ? 1 : 0, a_point & 0x04u != 0u ? 1 : 0);
    let p2 : vec3<i32> = vec3<i32>(b_point & 0x01u != 0u ? 1 : 0, b_point & 0x02u != 0u ? 1 : 0, b_point & 0x04u != 0u ? 1 : 0);

    if (a_is_further_side == b_is_further_side) {
      if (a_is_further_side) {
        xsv_ext0 = xsb + i32(p1.x);
        ysv_ext0 = ysb + i32(p1.y);
        zsv_ext0 = zsb + i32(p1.z);
        dx_ext0 = dx0 - f32(p1.x) + SQUISH_CONSTANT_3D;
        dy_ext0 = dy0 - f32(p1.y) + SQUISH_CONSTANT_3D;
        dz_ext0 = dz0 - f32(p1.z) + SQUISH_CONSTANT_3D;

        xsv_ext1 = xsb + i32(p2.x);
        ysv_ext1 = ysb + i32(p2.y);
        zsv_ext1 = zsb + i32(p2.z);
        dx_ext1 = dx0 - f32(p2.x) + SQUISH_CONSTANT_3D;
        dy_ext1 = dy0 - f32(p2.y) + SQUISH_CONSTANT_3D;
        dz_ext1 = dz0 - f32(p2.z) + SQUISH_CONSTANT_3D;
      } else {
        xsv_ext0 = xsb - 1;
        ysv_ext0 = ysb - 1;
        zsv_ext0 = zsb - 1;
        dx_ext0 = dx0 + 1.0 - SQUISH_CONSTANT_3D;
        dy_ext0 = dy0 + 1.0 - SQUISH_CONSTANT_3D;
        dz_ext0 = dz0 + 1.0 - SQUISH_CONSTANT_3D;

        xsv_ext1 = xsb;
        ysv_ext1 = ysb;
        zsv_ext1 = zsb;
        dx_ext1 = dx0 - SQUISH_CONSTANT_3D;
        dy_ext1 = dy0 - SQUISH_CONSTANT_3D;
        dz_ext1 = dz0 - SQUISH_CONSTANT_3D;
      }
    } else {
      var c1 : u32 = a_is_further_side ? a_point : b_point;
      var c2 : u32 = a_is_further_side ? b_point : a_point;

      if (a_is_further_side) {
        xsv_ext0 = xsb + i32((c1 & 0x01u) != 0u ? 1 : 0);
        ysv_ext0 = ysb + i32((c1 & 0x02u) != 0u ? 1 : 0);
        zsv_ext0 = zsb + i32((c1 & 0x04u) != 0u ? 1 : 0);
        dx_ext0 = dx0 - f32((c1 & 0x01u) != 0u ? 1 : 0) + SQUISH_CONSTANT_3D;
        dy_ext0 = dy0 - f32((c1 & 0x02u) != 0u ? 1 : 0) + SQUISH_CONSTANT_3D;
        dz_ext0 = dz0 - f32((c1 & 0x04u) != 0u ? 1 : 0) + SQUISH_CONSTANT_3D;

        xsv_ext1 = xsb - 1 + i32((c2 & 0x01u) != 0u ? 1 : 0);
        ysv_ext1 = ysb - 1 + i32((c2 & 0x02u) != 0u ? 1 : 0);
        zsv_ext1 = zsb - 1 + i32((c2 & 0x04u) != 0u ? 1 : 0);
        dx_ext1 = dx0 + 1.0 - f32((c2 & 0x01u) != 0u ? 1 : 0) - SQUISH_CONSTANT_3D;
        dy_ext1 = dy0 + 1.0 - f32((c2 & 0x02u) != 0u ? 1 : 0) - SQUISH_CONSTANT_3D;
        dz_ext1 = dz0 + 1.0 - f32((c2 & 0x04u) != 0u ? 1 : 0) - SQUISH_CONSTANT_3D;
      } else {
        xsv_ext0 = xsb - 1 + i32((c2 & 0x01u) != 0u ? 1 : 0);
        ysv_ext0 = ysb - 1 + i32((c2 & 0x02u) != 0u ? 1 : 0);
        zsv_ext0 = zsb - 1 + i32((c2 & 0x04u) != 0u ? 1 : 0);
        dx_ext0 = dx0 + 1.0 - f32((c2 & 0x01u) != 0u ? 1 : 0) - SQUISH_CONSTANT_3D;
        dy_ext0 = dy0 + 1.0 - f32((c2 & 0x02u) != 0u ? 1 : 0) - SQUISH_CONSTANT_3D;
        dz_ext0 = dz0 + 1.0 - f32((c2 & 0x04u) != 0u ? 1 : 0) - SQUISH_CONSTANT_3D;

        xsv_ext1 = xsb + i32((c1 & 0x01u) != 0u ? 1 : 0);
        ysv_ext1 = ysb + i32((c1 & 0x02u) != 0u ? 1 : 0);
        zsv_ext1 = zsb + i32((c1 & 0x04u) != 0u ? 1 : 0);
        dx_ext1 = dx0 - f32((c1 & 0x01u) != 0u ? 1 : 0) + SQUISH_CONSTANT_3D;
        dy_ext1 = dy0 - f32((c1 & 0x02u) != 0u ? 1 : 0) + SQUISH_CONSTANT_3D;
        dz_ext1 = dz0 - f32((c1 & 0x04u) != 0u ? 1 : 0) + SQUISH_CONSTANT_3D;
      }
    }

    let attn0 : f32 = 2.0 - dx0 * dx0 - dy0 * dy0 - dz0 * dz0;
    if (attn0 > 0.0) {
      var attn : f32 = attn0 * attn0;
      value = value + attn * attn * extrapolate3d(tables, xsb, ysb, zsb, dx0, dy0, dz0);
    }

    let dx1 : f32 = dx0 - 1.0 - SQUISH_CONSTANT_3D;
    let dy1 : f32 = dy0 - SQUISH_CONSTANT_3D;
    let dz1 : f32 = dz0 - SQUISH_CONSTANT_3D;
    var attn1 : f32 = 2.0 - dx1 * dx1 - dy1 * dy1 - dz1 * dz1;
    if (attn1 > 0.0) {
      attn1 = attn1 * attn1;
      value = value + attn1 * attn1 * extrapolate3d(tables, xsb + 1, ysb, zsb, dx1, dy1, dz1);
    }

    let dx2 : f32 = dx0 - SQUISH_CONSTANT_3D;
    let dy2 : f32 = dy0 - 1.0 - SQUISH_CONSTANT_3D;
    let dz2 : f32 = dz1;
    var attn2 : f32 = 2.0 - dx2 * dx2 - dy2 * dy2 - dz2 * dz2;
    if (attn2 > 0.0) {
      attn2 = attn2 * attn2;
      value = value + attn2 * attn2 * extrapolate3d(tables, xsb, ysb + 1, zsb, dx2, dy2, dz2);
    }

    let dx3 : f32 = dx2;
    let dy3 : f32 = dy1;
    let dz3 : f32 = dz0 - 1.0 - SQUISH_CONSTANT_3D;
    var attn3 : f32 = 2.0 - dx3 * dx3 - dy3 * dy3 - dz3 * dz3;
    if (attn3 > 0.0) {
      attn3 = attn3 * attn3;
      value = value + attn3 * attn3 * extrapolate3d(tables, xsb, ysb, zsb + 1, dx3, dy3, dz3);
    }
  } else {
    let a_score : f32 = xins;
    var a_point : u32 = 0x01u;
    var a_is_further_side : bool = true;
    if (yins > a_score) {
      a_score = yins;
      a_point = 0x02u;
    }
    if (zins > a_score) {
      a_score = zins;
      a_point = 0x04u;
    }

    let b_score : f32 = 3.0 - in_sum;
    var b_point : u32 = 0x01u;
    var b_is_further_side : bool = true;
    if (2.0 - yins + xins + zins > b_score) {
      b_score = 2.0 - yins + xins + zins;
      b_point = 0x02u | 0x01u;
      b_is_further_side = false;
    }
    if (2.0 - zins + xins + yins > b_score) {
      b_score = 2.0 - zins + xins + yins;
      b_point = 0x04u | 0x01u;
      b_is_further_side = false;
    }

    var c1 : u32 = a_point;
    var c2 : u32 = b_point;
    if (a_is_further_side == b_is_further_side) {
      if (a_is_further_side) {
        xsv_ext0 = xsb + i32((c1 & 0x01u) != 0u ? 1 : 0);
        ysv_ext0 = ysb + i32((c1 & 0x02u) != 0u ? 1 : 0);
        zsv_ext0 = zsb + i32((c1 & 0x04u) != 0u ? 1 : 0);
        dx_ext0 = dx0 - f32((c1 & 0x01u) != 0u ? 1 : 0) + SQUISH_CONSTANT_3D;
        dy_ext0 = dy0 - f32((c1 & 0x02u) != 0u ? 1 : 0) + SQUISH_CONSTANT_3D;
        dz_ext0 = dz0 - f32((c1 & 0x04u) != 0u ? 1 : 0) + SQUISH_CONSTANT_3D;

        xsv_ext1 = xsb + i32((c2 & 0x01u) != 0u ? 1 : 0);
        ysv_ext1 = ysb + i32((c2 & 0x02u) != 0u ? 1 : 0);
        zsv_ext1 = zsb + i32((c2 & 0x04u) != 0u ? 1 : 0);
        dx_ext1 = dx0 - f32((c2 & 0x01u) != 0u ? 1 : 0) + SQUISH_CONSTANT_3D;
        dy_ext1 = dy0 - f32((c2 & 0x02u) != 0u ? 1 : 0) + SQUISH_CONSTANT_3D;
        dz_ext1 = dz0 - f32((c2 & 0x04u) != 0u ? 1 : 0) + SQUISH_CONSTANT_3D;
      } else {
        xsv_ext0 = xsb - 1 + i32((c1 & 0x01u) != 0u ? 1 : 0);
        ysv_ext0 = ysb - 1 + i32((c1 & 0x02u) != 0u ? 1 : 0);
        zsv_ext0 = zsb - 1 + i32((c1 & 0x04u) != 0u ? 1 : 0);
        dx_ext0 = dx0 + 1.0 - f32((c1 & 0x01u) != 0u ? 1 : 0) - SQUISH_CONSTANT_3D;
        dy_ext0 = dy0 + 1.0 - f32((c1 & 0x02u) != 0u ? 1 : 0) - SQUISH_CONSTANT_3D;
        dz_ext0 = dz0 + 1.0 - f32((c1 & 0x04u) != 0u ? 1 : 0) - SQUISH_CONSTANT_3D;

        xsv_ext1 = xsb - 1 + i32((c2 & 0x01u) != 0u ? 1 : 0);
        ysv_ext1 = ysb - 1 + i32((c2 & 0x02u) != 0u ? 1 : 0);
        zsv_ext1 = zsb - 1 + i32((c2 & 0x04u) != 0u ? 1 : 0);
        dx_ext1 = dx0 + 1.0 - f32((c2 & 0x01u) != 0u ? 1 : 0) - SQUISH_CONSTANT_3D;
        dy_ext1 = dy0 + 1.0 - f32((c2 & 0x02u) != 0u ? 1 : 0) - SQUISH_CONSTANT_3D;
        dz_ext1 = dz0 + 1.0 - f32((c2 & 0x04u) != 0u ? 1 : 0) - SQUISH_CONSTANT_3D;
      }
    } else {
      if (a_is_further_side) {
        c1 = a_point;
        c2 = b_point;
      } else {
        c1 = b_point;
        c2 = a_point;
      }

      xsv_ext0 = xsb + i32((c1 & 0x01u) != 0u ? 1 : 0);
      ysv_ext0 = ysb + i32((c1 & 0x02u) != 0u ? 1 : 0);
      zsv_ext0 = zsb + i32((c1 & 0x04u) != 0u ? 1 : 0);
      dx_ext0 = dx0 - f32((c1 & 0x01u) != 0u ? 1 : 0) + SQUISH_CONSTANT_3D;
      dy_ext0 = dy0 - f32((c1 & 0x02u) != 0u ? 1 : 0) + SQUISH_CONSTANT_3D;
      dz_ext0 = dz0 - f32((c1 & 0x04u) != 0u ? 1 : 0) + SQUISH_CONSTANT_3D;

      xsv_ext1 = xsb + 1;
      ysv_ext1 = ysb + 1;
      zsv_ext1 = zsb + 1;
      dx_ext1 = dx0 - 1.0 - 2.0 * SQUISH_CONSTANT_3D;
      dy_ext1 = dy0 - 1.0 - 2.0 * SQUISH_CONSTANT_3D;
      dz_ext1 = dz0 - 1.0 - 2.0 * SQUISH_CONSTANT_3D;
    }

    let attn0 : f32 = 2.0 - dx0 * dx0 - dy0 * dy0 - dz0 * dz0;
    if (attn0 > 0.0) {
      var attn : f32 = attn0 * attn0;
      value = value + attn * attn * extrapolate3d(tables, xsb, ysb, zsb, dx0, dy0, dz0);
    }

    let dx1 : f32 = dx0 - 1.0 - SQUISH_CONSTANT_3D;
    let dy1 : f32 = dy0 - SQUISH_CONSTANT_3D;
    let dz1 : f32 = dz0 - SQUISH_CONSTANT_3D;
    var attn1 : f32 = 2.0 - dx1 * dx1 - dy1 * dy1 - dz1 * dz1;
    if (attn1 > 0.0) {
      attn1 = attn1 * attn1;
      value = value + attn1 * attn1 * extrapolate3d(tables, xsb + 1, ysb, zsb, dx1, dy1, dz1);
    }

    let dx2 : f32 = dx0 - SQUISH_CONSTANT_3D;
    let dy2 : f32 = dy0 - 1.0 - SQUISH_CONSTANT_3D;
    let dz2 : f32 = dz0 - SQUISH_CONSTANT_3D;
    var attn2 : f32 = 2.0 - dx2 * dx2 - dy2 * dy2 - dz2 * dz2;
    if (attn2 > 0.0) {
      attn2 = attn2 * attn2;
      value = value + attn2 * attn2 * extrapolate3d(tables, xsb, ysb + 1, zsb, dx2, dy2, dz2);
    }

    let dx3 : f32 = dx0 - SQUISH_CONSTANT_3D;
    let dy3 : f32 = dy0 - SQUISH_CONSTANT_3D;
    let dz3 : f32 = dz0 - 1.0 - SQUISH_CONSTANT_3D;
    var attn3 : f32 = 2.0 - dx3 * dx3 - dy3 * dy3 - dz3 * dz3;
    if (attn3 > 0.0) {
      attn3 = attn3 * attn3;
      value = value + attn3 * attn3 * extrapolate3d(tables, xsb, ysb, zsb + 1, dx3, dy3, dz3);
    }

    let dx4 : f32 = dx0 - 1.0 - SQUISH_CONSTANT_3D;
    let dy4 : f32 = dy0 - 1.0 - SQUISH_CONSTANT_3D;
    let dz4 : f32 = dz0 - 1.0 - SQUISH_CONSTANT_3D;
    var attn4 : f32 = 2.0 - dx4 * dx4 - dy4 * dy4 - dz4 * dz4;
    if (attn4 > 0.0) {
      attn4 = attn4 * attn4;
      value = value + attn4 * attn4 * extrapolate3d(tables, xsb + 1, ysb + 1, zsb + 1, dx4, dy4, dz4);
    }
  }

  let attn_ext0 : f32 = 2.0 - dx_ext0 * dx_ext0 - dy_ext0 * dy_ext0 - dz_ext0 * dz_ext0;
  if (attn_ext0 > 0.0) {
    var attn : f32 = attn_ext0 * attn_ext0;
    value = value + attn * attn * extrapolate3d(tables, xsv_ext0, ysv_ext0, zsv_ext0, dx_ext0, dy_ext0, dz_ext0);
  }

  let attn_ext1 : f32 = 2.0 - dx_ext1 * dx_ext1 - dy_ext1 * dy_ext1 - dz_ext1 * dz_ext1;
  if (attn_ext1 > 0.0) {
    var attn : f32 = attn_ext1 * attn_ext1;
    value = value + attn * attn * extrapolate3d(tables, xsv_ext1, ysv_ext1, zsv_ext1, dx_ext1, dy_ext1, dz_ext1);
  }

  return value / NORM_CONSTANT_3D;
}

fn apply_basic_distribution(value : f32, distrib : u32) -> f32 {
  if (distrib == DISTRIB_EXP) {
    return pow(clamp(value, 0.0, 1.0), 4.0);
  }
  if (distrib == DISTRIB_ONES) {
    return 1.0;
  }
  if (distrib == DISTRIB_MIDS) {
    return 0.5;
  }
  if (distrib == DISTRIB_ZEROS) {
    return 0.0;
  }
  return clamp(value, 0.0, 1.0);
}

fn compute_sdf_distance(dx : f32, dy : f32, sides : f32) -> f32 {
  let arctan_value : f32 = atan2(dx, -dy) + PI;
  let step : f32 = TAU / sides;
  let offset : f32 = floor(0.5 + arctan_value / step) * step - arctan_value;
  return cos(offset) * sqrt(dx * dx + dy * dy);
}

fn compute_center_distance(dx : f32, dy : f32, distrib : u32) -> f32 {
  let abs_dx : f32 = abs(dx);
  let abs_dy : f32 = abs(dy);
  if (distrib == DISTRIB_CENTER_DIAMOND) {
    return abs_dx + abs_dy;
  }
  if (distrib == DISTRIB_CENTER_SQUARE) {
    return max(abs_dx, abs_dy);
  }
  if (distrib == DISTRIB_CENTER_TRIANGLE) {
    return max(abs_dx - dy * 0.5, dy);
  }
  if (distrib == DISTRIB_CENTER_HEXAGON) {
    let term1 : f32 = max(abs_dx - dy * 0.5, dy);
    let term2 : f32 = max(abs_dx + dy * 0.5, -dy);
    return max(term1, term2);
  }
  if (distrib == DISTRIB_CENTER_OCTAGON) {
    let manhattan_term : f32 = (abs_dx + abs_dy) * INV_SQRT2;
    let chebyshev_term : f32 = max(abs_dx, abs_dy);
    return max(manhattan_term, chebyshev_term);
  }
  if (distrib == DISTRIB_CENTER_CIRCLE) {
    return sqrt(dx * dx + dy * dy);
  }
  if (distrib == DISTRIB_CENTER_PENTAGON) {
    return compute_sdf_distance(dx, dy, 5.0);
  }
  if (distrib == DISTRIB_CENTER_HEPTAGON) {
    return compute_sdf_distance(dx, dy, 7.0);
  }
  if (distrib == DISTRIB_CENTER_NONAGON) {
    return compute_sdf_distance(dx, dy, 9.0);
  }
  if (distrib == DISTRIB_CENTER_DECAGON) {
    return compute_sdf_distance(dx, dy, 10.0);
  }
  if (distrib == DISTRIB_CENTER_HENDECAGON) {
    return compute_sdf_distance(dx, dy, 11.0);
  }
  if (distrib == DISTRIB_CENTER_DODECAGON) {
    return compute_sdf_distance(dx, dy, 12.0);
  }
  return sqrt(dx * dx + dy * dy);
}

fn compute_center_max_distance(distrib : u32) -> f32 {
  let d1 : f32 = compute_center_distance(-0.5, -0.5, distrib);
  let d2 : f32 = compute_center_distance(-0.5, 0.5, distrib);
  let d3 : f32 = compute_center_distance(0.5, -0.5, distrib);
  let d4 : f32 = compute_center_distance(0.5, 0.5, distrib);
  let d5 : f32 = compute_center_distance(-0.5, 0.0, distrib);
  let d6 : f32 = compute_center_distance(0.5, 0.0, distrib);
  let d7 : f32 = compute_center_distance(0.0, -0.5, distrib);
  let d8 : f32 = compute_center_distance(0.0, 0.5, distrib);
  let first_max : f32 = max(max(d1, d2), max(d3, d4));
  let second_max : f32 = max(max(d5, d6), max(d7, d8));
  return max(first_max, second_max);
}

fn compute_center_distribution_value(
  pixel : vec2<f32>,
  resolution : vec2<f32>,
  distrib : u32,
  octave_freq : vec2<f32>,
  time_value : f32,
  speed_value : f32,
) -> f32 {
  let width : f32 = max(resolution.x, 1.0);
  let height : f32 = max(resolution.y, 1.0);
  let dx : f32 = (pixel.x / width) - 0.5;
  let dy : f32 = (pixel.y / height) - 0.5;
  let distance_value : f32 = compute_center_distance(dx, dy, distrib);
  let max_distance : f32 = max(compute_center_max_distance(distrib), EPSILON);
  let normalized_distance : f32 = clamp(distance_value / max_distance, 0.0, 1.0);
  let eased_distance : f32 = sqrt(normalized_distance);
  let freq_scale : f32 = max(max(octave_freq.x, octave_freq.y), 1.0);
  var rounded_speed : f32;
  if (speed_value > 0.0) {
    rounded_speed = floor(1.0 + speed_value);
  } else {
    rounded_speed = ceil(-1.0 + speed_value);
  }
  let phase : f32 = eased_distance * freq_scale * TAU - TAU * time_value * rounded_speed;
  let sine_value : f32 = sin(phase);
  return clamp((sine_value + 1.0) * 0.5, 0.0, 1.0);
}

fn apply_primary_distribution(
  value : f32,
  distrib : u32,
  sample_coord : vec2<f32>,
  octave_freq : vec2<f32>,
  pixel : vec2<f32>,
  resolution : vec2<f32>,
  time_value : f32,
  speed_value : f32,
) -> f32 {
  if (distrib == DISTRIB_COLUMN_INDEX) {
    let freq_y : f32 = max(octave_freq.y, 1.0);
    if (freq_y <= 1.0) {
      return 0.0;
    }
    let capped : f32 = min(sample_coord.y, freq_y - 1.0);
    let denom : f32 = max(freq_y - 1.0, 1.0);
    return clamp(capped / denom, 0.0, 1.0);
  }
  if (distrib == DISTRIB_ROW_INDEX) {
    let freq_x : f32 = max(octave_freq.x, 1.0);
    if (freq_x <= 1.0) {
      return 0.0;
    }
    let capped : f32 = min(sample_coord.x, freq_x - 1.0);
    let denom : f32 = max(freq_x - 1.0, 1.0);
    return clamp(capped / denom, 0.0, 1.0);
  }
  if (distrib >= DISTRIB_CENTER_CIRCLE && distrib <= DISTRIB_CENTER_DODECAGON) {
    return compute_center_distribution_value(pixel, resolution, distrib, octave_freq, time_value, speed_value);
  }
  return apply_basic_distribution(value, distrib);
}

fn sample_simplex_channel(
  coord : vec2<f32>,
  z : f32,
  base_seed : u32,
  channel_index : u32,
) -> f32 {
  var tables : PermutationTables = build_permutation_tables(base_seed + channel_index * 65535u);
  let raw_value : f32 = open_simplex_3d(&tables, coord.x, coord.y, z);
  return map_to_unit(raw_value);
}

fn sample_distribution_value(
  coord : vec2<f32>,
  z : f32,
  seed : u32,
  distrib : u32,
) -> f32 {
  if (distrib == DISTRIB_NONE || distrib == 0u) {
    return 0.0;
  }
  if (distrib == DISTRIB_ONES) {
    return 1.0;
  }
  if (distrib == DISTRIB_MIDS) {
    return 0.5;
  }
  if (distrib == DISTRIB_ZEROS) {
    return 0.0;
  }
  var tables : PermutationTables = build_permutation_tables(seed);
  let raw_value : f32 = open_simplex_3d(&tables, coord.x, coord.y, z);
  let mapped : f32 = map_to_unit(raw_value);
  return apply_basic_distribution(mapped, distrib);
}

fn evaluate_simplex_layer_rgba(
  sample_coord : vec2<f32>,
  override_coord : vec2<f32>,
  brightness_coord : vec2<f32>,
  z : f32,
  base_seed : u32,
  hue_seed : u32,
  saturation_seed : u32,
  brightness_seed : u32,
  channel_count : u32,
  color_space : u32,
  ridges : bool,
  sin_amount : f32,
  distrib : u32,
  hue_range : f32,
  hue_rotation : f32,
  saturation_scale : f32,
  hue_distrib : u32,
  saturation_distrib : u32,
  brightness_distrib : u32,
  octave_freq : vec2<f32>,
  pixel_coord : vec2<f32>,
  resolution : vec2<f32>,
  time_value : f32,
  speed_value : f32,
) -> LayerResult {
  var effective_channels : u32 = channel_count;
  if (effective_channels == 0u) {
    effective_channels = 1u;
  }
  if (effective_channels > 4u) {
    effective_channels = 4u;
  }

  var raw : array<f32, 4> = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
  var idx : u32 = 0u;
  loop {
    if (idx >= effective_channels) {
      break;
    }
    var sample : f32 = sample_simplex_channel(sample_coord, z, base_seed, idx);
    var effective_distrib : u32 = distrib;
    if (effective_distrib == DISTRIB_NONE || effective_distrib == 0u) {
      effective_distrib = DISTRIB_SIMPLEX;
    }
    sample = apply_primary_distribution(
      sample,
      effective_distrib,
      sample_coord,
      octave_freq,
      pixel_coord,
      resolution,
      time_value,
      speed_value,
    );
    raw[idx] = sample;
    idx = idx + 1u;
  }

  var alpha_value : f32 = 1.0;
  if (channel_count >= 4u) {
    alpha_value = saturate(raw[3]);
  } else if (channel_count == 2u) {
    alpha_value = saturate(raw[1]);
  }

  if (color_space == COLOR_SPACE_GRAYSCALE || channel_count <= 2u) {
    var luminance : f32 = raw[0];
    if (ridges) {
      luminance = ridge_transform(luminance);
    }
    if (sin_amount != 0.0) {
      luminance = map_to_unit(sin(sin_amount * luminance));
    }
    let clamped_luminance : f32 = saturate(luminance);
    return LayerResult(
      vec4<f32>(clamped_luminance, clamped_luminance, clamped_luminance, alpha_value),
      vec4<f32>(0.0, 0.0, 0.0, alpha_value),
      0u,
    );
  }

  var working_space : u32 = color_space;
  var color_vec : vec3<f32> = vec3<f32>(raw[0], raw[1], raw[2]);

  if (working_space == COLOR_SPACE_OKLAB) {
    let lab : vec3<f32> = vec3<f32>(
      color_vec.x,
      color_vec.y * -0.509 + 0.276,
      color_vec.z * -0.509 + 0.198
    );
    color_vec = clamp(
      oklab_to_rgb(lab),
      vec3<f32>(0.0, 0.0, 0.0),
      vec3<f32>(1.0, 1.0, 1.0)
    );
    working_space = COLOR_SPACE_RGB;
  }

  if (working_space == COLOR_SPACE_RGB) {
    color_vec = rgb_to_hsv(color_vec);
    working_space = COLOR_SPACE_HSV;
  }

  if (working_space == COLOR_SPACE_HSV) {
    let has_hue_override : bool = hue_distrib != DISTRIB_NONE && hue_distrib != 0u;
    let has_saturation_override : bool = saturation_distrib != DISTRIB_NONE && saturation_distrib != 0u;
    let has_brightness_override : bool = brightness_distrib != DISTRIB_NONE && brightness_distrib != 0u;

    var hue_value : f32;
    if (has_hue_override) {
      hue_value = clamp(sample_distribution_value(override_coord, z, hue_seed, hue_distrib), 0.0, 1.0);
    } else {
      hue_value = color_vec.x * hue_range + hue_rotation;
      hue_value = fract(hue_value);
      if (hue_value < 0.0) {
        hue_value = hue_value + 1.0;
      }
    }

    var saturation_value : f32;
    if (has_saturation_override) {
      saturation_value = clamp(
        sample_distribution_value(override_coord, z, saturation_seed, saturation_distrib),
        0.0,
        1.0,
      );
    } else {
      saturation_value = clamp(color_vec.y, 0.0, 1.0);
    }
    saturation_value = clamp(saturation_value * saturation_scale, 0.0, 1.0);

    var brightness_value : f32;
    if (has_brightness_override) {
      brightness_value = clamp(
        sample_distribution_value(brightness_coord, z, brightness_seed, brightness_distrib),
        0.0,
        1.0,
      );
    } else {
      brightness_value = clamp(color_vec.z, 0.0, 1.0);
    }
    if (ridges) {
      brightness_value = ridge_transform(brightness_value);
    }
    if (sin_amount != 0.0) {
      let sin_value : f32 = sin(sin_amount * brightness_value);
      return LayerResult(
        vec4<f32>(0.0, 0.0, 0.0, 0.0),
        vec4<f32>(hue_value, saturation_value, sin_value, alpha_value),
        LAYER_FLAG_HAS_SIN_HSV,
      );
    }
    brightness_value = clamp(brightness_value, 0.0, 1.0);

    color_vec = hsv_to_rgb(vec3<f32>(hue_value, saturation_value, brightness_value));
  }

  color_vec = clamp(color_vec, vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(1.0, 1.0, 1.0));
  return LayerResult(
    vec4<f32>(color_vec, alpha_value),
    vec4<f32>(0.0, 0.0, 0.0, alpha_value),
    0u,
  );
}

@group(0) @binding(0) var<uniform> stage_uniforms : StageUniforms;
@group(0) @binding(1) var<uniform> frame_uniforms : FrameUniforms;
@group(0) @binding(3) var output_texture : texture_storage_2d<rgba32float, write>;
@group(0) @binding(4) var<storage, read_write> sin_state : SinNormalizationState;

@compute @workgroup_size(8, 8, 1)
fn multires_main(@builtin(global_invocation_id) global_id : vec3<u32>) {
  let width : u32 = u32(frame_uniforms.resolution.x);
  let height : u32 = u32(frame_uniforms.resolution.y);
  if (global_id.x >= width || global_id.y >= height) {
    return;
  }
  let total_invocations : u32 = max(width * height, 1u);

  let base_freq : vec2<f32> = stage_uniforms.freq;
  let octaves : u32 = stage_uniforms.options0.x;
  let octave_blending : u32 = stage_uniforms.options0.y;
  let channel_count : u32 = stage_uniforms.options0.z;
  let ridges_enabled : bool = bool_from_u32(stage_uniforms.options0.w);
  let seed_offset : u32 = stage_uniforms.options1.x;
  let distrib : u32 = stage_uniforms.options1.y;
  let color_space : u32 = stage_uniforms.options1.z;
  let with_alpha_output : bool = bool_from_u32(stage_uniforms.options1.w);

  let hue_range : f32 = stage_uniforms.colorParams0.x;
  let hue_rotation : f32 = stage_uniforms.colorParams0.y;
  let saturation_scale : f32 = stage_uniforms.colorParams0.z;
  let sin_amount : f32 = stage_uniforms.sin;
  let speed : f32 = stage_uniforms.speed;
  let hue_distrib : u32 = stage_uniforms.options2.x;
  let saturation_distrib : u32 = stage_uniforms.options2.y;
  var brightness_distrib : u32 = stage_uniforms.options2.z;
  let brightness_freq_flag : u32 = stage_uniforms.options2.w;
  let brightness_freq_params : vec2<f32> = vec2<f32>(
    stage_uniforms.colorParams1.x,
    stage_uniforms.colorParams1.y,
  );
  let lattice_drift_amount : f32 = stage_uniforms.colorParams1.z;
  let has_hue_override : bool = hue_distrib != 0u;
  let has_saturation_override : bool = saturation_distrib != 0u;
  let has_brightness_freq_override : bool = bool_from_u32(brightness_freq_flag);
  var has_brightness_override : bool = brightness_distrib != 0u || has_brightness_freq_override;
  if (has_brightness_override && brightness_distrib == 0u) {
    brightness_distrib = DISTRIB_SIMPLEX;
  }
  let corners_enabled : bool = bool_from_u32(stage_uniforms.options3.x);

  var calls_per_octave : u32 = 1u;
  if (has_hue_override) {
    calls_per_octave = calls_per_octave + 1u;
  }
  if (has_saturation_override) {
    calls_per_octave = calls_per_octave + 1u;
  }
  if (has_brightness_override) {
    calls_per_octave = calls_per_octave + 1u;
  }
  let has_lattice_drift : bool = lattice_drift_amount != 0.0;
  if (has_lattice_drift) {
    calls_per_octave = calls_per_octave + 2u;
  }

  let angle : f32 = frame_uniforms.time * TAU;
  let z : f32 = cos(angle) * speed;
  let resolution : vec2<f32> = frame_uniforms.resolution;
  let pixel : vec2<f32> = vec2<f32>(f32(global_id.x), f32(global_id.y));
  let uv : vec2<f32> = (pixel + vec2<f32>(0.5, 0.5)) / resolution;

  var accum : vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  var weight : f32 = 0.5;
  var octave_index : u32 = 1u;

  loop {
    if (octave_index > octaves) {
      break;
    }

    let octave_freq : vec2<f32> = compute_octave_frequency(base_freq, octave_index);
    if (octave_freq.x > resolution.x && octave_freq.y > resolution.y) {
      break;
    }

    let pin_offset_uv : vec2<f32> = compute_pin_offset(octave_freq, corners_enabled, resolution);
    let base_uv : vec2<f32> = fract(uv + pin_offset_uv);
    let base_coord : vec2<f32> = base_uv * octave_freq;

    var brightness_freq_vec : vec2<f32> = octave_freq;
    if (has_brightness_override && has_brightness_freq_override) {
      brightness_freq_vec = vec2<f32>(
        max(brightness_freq_params.x, 1.0),
        max(brightness_freq_params.y, 1.0),
      );
    }
    let brightness_pin_offset : vec2<f32> = compute_pin_offset(
      brightness_freq_vec,
      corners_enabled,
      resolution,
    );
    let brightness_uv : vec2<f32> = fract(uv + brightness_pin_offset);
    let brightness_coord : vec2<f32> = brightness_uv * brightness_freq_vec;
    let octave_offset : u32 = (octave_index - 1u) * calls_per_octave;
    let reseeded : bool = frame_uniforms.seed != 0u;
    var base_seed : u32 = frame_uniforms.seed + seed_offset + octave_offset;
    if (reseeded) {
      base_seed = base_seed + 1u;
    }

    var seed_cursor : u32 = base_seed;
    let layer_seed : u32 = seed_cursor;
    seed_cursor = seed_cursor + 1u;

    var hue_seed : u32 = 0u;
    if (has_hue_override) {
      hue_seed = seed_cursor;
      seed_cursor = seed_cursor + 1u;
    }

    var saturation_seed : u32 = 0u;
    if (has_saturation_override) {
      saturation_seed = seed_cursor;
      seed_cursor = seed_cursor + 1u;
    }

    var brightness_seed : u32 = 0u;
    if (has_brightness_override) {
      brightness_seed = seed_cursor;
      seed_cursor = seed_cursor + 1u;
    }

    var refx_seed : u32 = 0u;
    var refy_seed : u32 = 0u;
    if (has_lattice_drift) {
      refx_seed = seed_cursor;
      seed_cursor = seed_cursor + 1u;
      refy_seed = seed_cursor;
      seed_cursor = seed_cursor + 1u;
    }

    let effective_brightness_distrib : u32 = has_brightness_override ? brightness_distrib : DISTRIB_NONE;

    var sample_coord : vec2<f32> = base_coord;
    let override_coord : vec2<f32> = base_coord;
    if (has_lattice_drift) {
      let min_freq_component : f32 = max(min(octave_freq.x, octave_freq.y), 1.0);
      let displacement : f32 = lattice_drift_amount / min_freq_component;
      let refx_value : f32 = sample_simplex_channel(base_coord, z, refx_seed, 0u);
      let refy_value : f32 = sample_simplex_channel(base_coord, z, refy_seed, 0u);
      let lattice_offset_uv : vec2<f32> = vec2<f32>(refx_value, refy_value) * displacement * 2.0;
      let warped_uv : vec2<f32> = fract(base_uv + lattice_offset_uv);
      sample_coord = warped_uv * octave_freq;
    }

    let layer_result : LayerResult = evaluate_simplex_layer_rgba(
      sample_coord,
      override_coord,
      brightness_coord,
      z,
      layer_seed,
      hue_seed,
      saturation_seed,
      brightness_seed,
      channel_count,
      color_space,
      ridges_enabled,
      sin_amount,
      distrib,
      hue_range,
      hue_rotation,
      saturation_scale,
      hue_distrib,
      saturation_distrib,
      effective_brightness_distrib,
      octave_freq,
      pixel,
      resolution,
      frame_uniforms.time,
      speed,
    );

    var layer_rgba : vec4<f32> = layer_result.color;
    if ((layer_result.flags & LAYER_FLAG_HAS_SIN_HSV) != 0u) {
      let iteration_id : u32 = octave_index;
      if (global_id.x == 0u && global_id.y == 0u) {
        atomicStore(&sin_state.count, 0u);
        atomicStore(&sin_state.min_value, float_to_ordered_int(F32_MAX));
        atomicStore(&sin_state.max_value, float_to_ordered_int(-F32_MAX));
        storageBarrier();
        atomicStore(&sin_state.phase, iteration_id);
      }
      loop {
        let current_phase : u32 = atomicLoad(&sin_state.phase);
        if (current_phase == iteration_id) {
          break;
        }
      }

      let brightness_sample : f32 = layer_result.hsv.z;
      let encoded_sample : i32 = float_to_ordered_int(brightness_sample);
      atomicMin(&sin_state.min_value, encoded_sample);
      atomicMax(&sin_state.max_value, encoded_sample);
      atomicAdd(&sin_state.count, 1u);
      loop {
        let current_count : u32 = atomicLoad(&sin_state.count);
        if (current_count >= total_invocations) {
          break;
        }
      }
      storageBarrier();
      let encoded_min : i32 = atomicLoad(&sin_state.min_value);
      let encoded_max : i32 = atomicLoad(&sin_state.max_value);
      let min_value : f32 = ordered_int_to_float(encoded_min);
      let max_value : f32 = ordered_int_to_float(encoded_max);
      let range : f32 = max(max_value - min_value, EPSILON);
      let normalized_brightness : f32 = clamp((brightness_sample - min_value) / range, 0.0, 1.0);
      let hue_value : f32 = layer_result.hsv.x;
      let saturation_value : f32 = clamp(layer_result.hsv.y, 0.0, 1.0);
      let alpha_value : f32 = layer_result.hsv.w;
      let color_vec : vec3<f32> = hsv_to_rgb(vec3<f32>(hue_value, saturation_value, normalized_brightness));
      let clamped_color : vec3<f32> = clamp(color_vec, vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(1.0, 1.0, 1.0));
      layer_rgba = vec4<f32>(clamped_color, alpha_value);
    }

    if (octave_blending == OCTAVE_BLENDING_REDUCE_MAX) {
      accum = max(accum, layer_rgba);
    } else if (octave_blending == OCTAVE_BLENDING_ALPHA) {
      combine_alpha(&accum, layer_rgba);
    } else {
      accum = accum + layer_rgba * weight;
    }

    weight = weight * 0.5;
    octave_index = octave_index + 1u;
  }

  var resolved_color : vec4<f32> = accum;
  if (!with_alpha_output && octave_blending == OCTAVE_BLENDING_ALPHA) {
    let alpha_component : f32 = resolved_color.w;
    resolved_color = vec4<f32>(resolved_color.xyz * alpha_component, 1.0);
  }

  let final_color : vec4<f32> = clamp(
    resolved_color,
    vec4<f32>(0.0, 0.0, 0.0, 0.0),
    vec4<f32>(1.0, 1.0, 1.0, 1.0)
  );
  textureStore(output_texture, vec2<i32>(i32(global_id.x), i32(global_id.y)), final_color);
}
