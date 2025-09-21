// WebGPU shader stubs. The previous pipeline has been removed and will be
// replaced in a future rewrite.  These placeholders satisfy existing imports
// while ensuring any attempted WebGPU execution falls back to CPU code paths.
const SHADER_PLACEHOLDER = null;

export const MULTIRES_WGSL = String.raw`
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
  sin_amount : f32,
  options0 : vec4<u32>,
  options1 : vec4<u32>,
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

const OCTAVE_BLENDING_FALLOFF : u32 = 0u;
const OCTAVE_BLENDING_REDUCE_MAX : u32 = 10u;
const OCTAVE_BLENDING_ALPHA : u32 = 20u;

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

fn combine_alpha(accum : ptr<function, vec4<f32>>, layer : vec4<f32>) {
  let alpha_value : f32 = saturate(layer.w);
  let alpha_vec : vec4<f32> = replicate4(alpha_value);
  (*accum) = (*accum) * (vec4<f32>(1.0, 1.0, 1.0, 1.0) - alpha_vec) + layer * alpha_vec;
}

fn finalize_color(color : vec4<f32>, channel_count : u32) -> vec4<f32> {
  let luminance : f32 = saturate(color.x);
  var alpha : f32 = saturate(color.w);
  if (channel_count <= 1u || channel_count == 3u) {
    alpha = 1.0;
  } else if (channel_count >= 4u) {
    if (alpha <= 0.0) {
      alpha = 1.0;
    }
  } else {
    if (alpha <= 0.0) {
      alpha = 1.0;
    }
  }
  return vec4<f32>(luminance, luminance, luminance, alpha);
}

struct PermutationTables {
  perm : array<u32, PERMUTATION_SIZE>,
  perm_grad_index3d : array<u32, PERMUTATION_SIZE>,
};

struct RandomState {
  state : u32,
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

fn evaluate_simplex_layer(tables : ptr<function, PermutationTables>, coord : vec2<f32>, z : f32, ridges : bool) -> vec4<f32> {
  let raw_value : f32 = open_simplex_3d(tables, coord.x, coord.y, z);
  var normalized : f32 = map_to_unit(raw_value);
  if (ridges) {
    normalized = ridge_transform(normalized);
  }
  normalized = saturate(normalized);
  return vec4<f32>(normalized, normalized, normalized, normalized);
}

@group(0) @binding(0) var<uniform> stage_uniforms : StageUniforms;
@group(0) @binding(1) var<uniform> frame_uniforms : FrameUniforms;
@group(0) @binding(3) var output_texture : texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8, 8, 1)
fn multires_main(@builtin(global_invocation_id) global_id : vec3<u32>) {
  let width : u32 = u32(frame_uniforms.resolution.x);
  let height : u32 = u32(frame_uniforms.resolution.y);
  if (global_id.x >= width || global_id.y >= height) {
    return;
  }

  let base_freq : vec2<f32> = stage_uniforms.freq;
  let octaves : u32 = stage_uniforms.options0.x;
  let octave_blending : u32 = stage_uniforms.options0.y;
  let channel_count : u32 = stage_uniforms.options0.z;
  let ridges_enabled : bool = bool_from_u32(stage_uniforms.options0.w);
  let seed_offset : u32 = stage_uniforms.options1.x;

  let angle : f32 = frame_uniforms.time * TAU;
  let z : f32 = cos(angle) * stage_uniforms.speed;
  let resolution : vec2<f32> = frame_uniforms.resolution;
  let pixel : vec2<f32> = vec2<f32>(f32(global_id.x), f32(global_id.y));
  let uv : vec2<f32> = (pixel + vec2<f32>(0.5, 0.5)) / resolution;

  var accum : vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  var weight_sum : f32 = 0.0;
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

    let coord : vec2<f32> = uv * octave_freq;
    var octave_tables : PermutationTables = build_permutation_tables(frame_uniforms.seed + seed_offset + octave_index);
    let layer : vec4<f32> = evaluate_simplex_layer(&octave_tables, coord, z, ridges_enabled);

    if (octave_blending == OCTAVE_BLENDING_REDUCE_MAX) {
      accum = max(accum, layer);
    } else if (octave_blending == OCTAVE_BLENDING_ALPHA) {
      combine_alpha(&accum, layer);
    } else {
      accum = accum + layer * weight;
      weight_sum = weight_sum + weight;
    }

    weight = weight * 0.5;
    octave_index = octave_index + 1u;
  }

  var final_color : vec4<f32> = accum;
  if (octave_blending == OCTAVE_BLENDING_FALLOFF) {
    if (weight_sum > 0.0) {
      final_color = final_color / weight_sum;
    }
  }

  final_color = clamp(final_color, vec4<f32>(0.0, 0.0, 0.0, 0.0), vec4<f32>(1.0, 1.0, 1.0, 1.0));
  let output_color : vec4<f32> = finalize_color(final_color, channel_count);
  textureStore(output_texture, vec2<i32>(i32(global_id.x), i32(global_id.y)), output_color);
}
`;

export const VALUE_WGSL = SHADER_PLACEHOLDER;
export const RESAMPLE_WGSL = SHADER_PLACEHOLDER;
export const DOWNSAMPLE_WGSL = SHADER_PLACEHOLDER;
export const BLEND_WGSL = SHADER_PLACEHOLDER;
export const BLEND_CONST_WGSL = SHADER_PLACEHOLDER;
export const SOBEL_WGSL = SHADER_PLACEHOLDER;
export const REFRACT_WGSL = SHADER_PLACEHOLDER;
export const CONVOLUTION_WGSL = SHADER_PLACEHOLDER;
export const FXAA_WGSL = SHADER_PLACEHOLDER;
export const NORMALIZE_WGSL = SHADER_PLACEHOLDER;
export const RGB_TO_HSV_WGSL = SHADER_PLACEHOLDER;
export const HSV_TO_RGB_WGSL = SHADER_PLACEHOLDER;
export const OCTAVE_COMBINE_WGSL = SHADER_PLACEHOLDER;
export const UPSAMPLE_WGSL = SHADER_PLACEHOLDER;
export const VORONOI_WGSL = SHADER_PLACEHOLDER;
export const EROSION_WORMS_WGSL = SHADER_PLACEHOLDER;
export const WORMS_WGSL = SHADER_PLACEHOLDER;
export const REINDEX_WGSL = SHADER_PLACEHOLDER;
export const RIPPLE_WGSL = SHADER_PLACEHOLDER;
export const COLOR_MAP_WGSL = SHADER_PLACEHOLDER;
export const VIGNETTE_WGSL = SHADER_PLACEHOLDER;
export const DITHER_WGSL = SHADER_PLACEHOLDER;
export const ADJUST_BRIGHTNESS_WGSL = SHADER_PLACEHOLDER;
export const ADJUST_CONTRAST_WGSL = SHADER_PLACEHOLDER;
export const ROTATE_WGSL = SHADER_PLACEHOLDER;
export const GLYPH_MAP_WGSL = SHADER_PLACEHOLDER;
export const WARP_WGSL = SHADER_PLACEHOLDER;
export const SPATTER_MASK_WGSL = SHADER_PLACEHOLDER;
export const SCRATCHES_MASK_WGSL = SHADER_PLACEHOLDER;
export const SCRATCHES_BLEND_WGSL = SHADER_PLACEHOLDER;
export const GRIME_MASK_WGSL = SHADER_PLACEHOLDER;
export const GRIME_BLEND_WGSL = SHADER_PLACEHOLDER;
export const DERIVATIVE_WGSL = SHADER_PLACEHOLDER;
export const PIXEL_SORT_WGSL = SHADER_PLACEHOLDER;
export const KALEIDO_WGSL = SHADER_PLACEHOLDER;
export const NORMAL_MAP_WGSL = SHADER_PLACEHOLDER;
export const CRT_WGSL = SHADER_PLACEHOLDER;
export const WOBBLE_WGSL = SHADER_PLACEHOLDER;
export const VORTEX_WGSL = SHADER_PLACEHOLDER;
export const WORMHOLE_WGSL = SHADER_PLACEHOLDER;
export const DLA_WGSL = SHADER_PLACEHOLDER;
export const REVERB_WGSL = SHADER_PLACEHOLDER;
export const VASELINE_BLUR_WGSL = SHADER_PLACEHOLDER;
export const VASELINE_MASK_WGSL = SHADER_PLACEHOLDER;
export const LENS_DISTORTION_WGSL = SHADER_PLACEHOLDER;
export const DEGAUSS_WGSL = SHADER_PLACEHOLDER;
export const TINT_WGSL = SHADER_PLACEHOLDER;
export const VHS_WGSL = SHADER_PLACEHOLDER;
export const UNARY_OP_WGSL = SHADER_PLACEHOLDER;
export const BINARY_OP_WGSL = SHADER_PLACEHOLDER;
export const GRAYSCALE_WGSL = SHADER_PLACEHOLDER;
export const EXPAND_CHANNELS_WGSL = SHADER_PLACEHOLDER;
