struct StageUniforms {
  kernel_id : i32,
  with_normalize : u32,
  alpha : f32,
  padding : f32,
};

struct FrameUniforms {
  resolution : vec2<f32>,
  time : f32,
  seed : u32,
  frame_index : u32,
  padding0 : u32,
  padding1 : vec2<f32>,
};

@group(0) @binding(0) var<uniform> stage_uniforms : StageUniforms;
@group(0) @binding(1) var<uniform> frame_uniforms : FrameUniforms;
@group(0) @binding(2) var input_texture : texture_2d<f32>;
@group(0) @binding(3) var output_texture : texture_storage_2d<rgba32float, write>;

const KERNEL_CONV2D_BLUR : i32 = 800;
const KERNEL_CONV2D_DERIV_X : i32 = 801;
const KERNEL_CONV2D_DERIV_Y : i32 = 802;
const KERNEL_CONV2D_EDGES : i32 = 803;
const KERNEL_CONV2D_EMBOSS : i32 = 804;
const KERNEL_CONV2D_INVERT : i32 = 805;
const KERNEL_CONV2D_RAND : i32 = 806;
const KERNEL_CONV2D_SHARPEN : i32 = 807;
const KERNEL_CONV2D_SOBEL_X : i32 = 808;
const KERNEL_CONV2D_SOBEL_Y : i32 = 809;
const KERNEL_CONV2D_BOX_BLUR : i32 = 810;

fn wrap_coord(value : i32, limit : i32) -> i32 {
  if (limit <= 0) {
    return 0;
  }
  var wrapped = value % limit;
  if (wrapped < 0) {
    wrapped = wrapped + limit;
  }
  return wrapped;
}

fn kernel_dimensions(id : i32) -> vec2<i32> {
  switch id {
    case KERNEL_CONV2D_BLUR, KERNEL_CONV2D_RAND: {
      return vec2<i32>(5, 5);
    }
    case KERNEL_CONV2D_BOX_BLUR: {
      return vec2<i32>(3, 3);
    }
    case KERNEL_CONV2D_DERIV_X,
         KERNEL_CONV2D_DERIV_Y,
         KERNEL_CONV2D_EDGES,
         KERNEL_CONV2D_EMBOSS,
         KERNEL_CONV2D_INVERT,
         KERNEL_CONV2D_SHARPEN,
         KERNEL_CONV2D_SOBEL_X,
         KERNEL_CONV2D_SOBEL_Y: {
      return vec2<i32>(3, 3);
    }
    default: {
      return vec2<i32>(1, 1);
    }
  }
}

fn kernel_weight(id : i32, row : i32, col : i32) -> f32 {
  switch id {
    case KERNEL_CONV2D_BLUR: {
      let weights : array<f32, 25> = array<f32, 25>(
        1.0, 4.0, 6.0, 4.0, 1.0,
        4.0, 16.0, 24.0, 16.0, 4.0,
        6.0, 24.0, 36.0, 24.0, 6.0,
        4.0, 16.0, 24.0, 16.0, 4.0,
        1.0, 4.0, 6.0, 4.0, 1.0,
      );
      return weights[row * 5 + col];
    }
    case KERNEL_CONV2D_BOX_BLUR: {
      let weights : array<f32, 9> = array<f32, 9>(
        1.0, 2.0, 1.0,
        2.0, 4.0, 2.0,
        1.0, 2.0, 1.0,
      );
      return weights[row * 3 + col];
    }
    case KERNEL_CONV2D_DERIV_X: {
      let weights : array<f32, 9> = array<f32, 9>(
        0.0, 0.0, 0.0,
        0.0, 1.0, -1.0,
        0.0, 0.0, 0.0,
      );
      return weights[row * 3 + col];
    }
    case KERNEL_CONV2D_DERIV_Y: {
      let weights : array<f32, 9> = array<f32, 9>(
        0.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, -1.0, 0.0,
      );
      return weights[row * 3 + col];
    }
    case KERNEL_CONV2D_EDGES: {
      let weights : array<f32, 9> = array<f32, 9>(
        1.0, 2.0, 1.0,
        2.0, -12.0, 2.0,
        1.0, 2.0, 1.0,
      );
      return weights[row * 3 + col];
    }
    case KERNEL_CONV2D_EMBOSS: {
      let weights : array<f32, 9> = array<f32, 9>(
        0.0, 2.0, 4.0,
        -2.0, 1.0, 2.0,
        -4.0, -2.0, 0.0,
      );
      return weights[row * 3 + col];
    }
    case KERNEL_CONV2D_INVERT: {
      let weights : array<f32, 9> = array<f32, 9>(
        0.0, 0.0, 0.0,
        0.0, -1.0, 0.0,
        0.0, 0.0, 0.0,
      );
      return weights[row * 3 + col];
    }
    case KERNEL_CONV2D_RAND: {
      let weights : array<f32, 25> = array<f32, 25>(
        1.38202617, 0.700078607, 0.989368975, 1.62044656, 1.433778997,
        0.0113610601, 0.975044191, 0.424321383, 0.448390573, 0.705299258,
        0.572021782, 1.22713673, 0.880518854, 0.560837507, 0.721931636,
        0.666837156, 1.24703956, 0.397420883, 0.656533837, 0.0729521295,
        -0.77649492, 0.826809287, 0.932218075, 0.128917485, 1.63487732,
      );
      return weights[row * 5 + col];
    }
    case KERNEL_CONV2D_SHARPEN: {
      let weights : array<f32, 9> = array<f32, 9>(
        0.0, -1.0, 0.0,
        -1.0, 5.0, -1.0,
        0.0, -1.0, 0.0,
      );
      return weights[row * 3 + col];
    }
    case KERNEL_CONV2D_SOBEL_X: {
      let weights : array<f32, 9> = array<f32, 9>(
        1.0, 0.0, -1.0,
        2.0, 0.0, -2.0,
        1.0, 0.0, -1.0,
      );
      return weights[row * 3 + col];
    }
    case KERNEL_CONV2D_SOBEL_Y: {
      let weights : array<f32, 9> = array<f32, 9>(
        1.0, 2.0, 1.0,
        0.0, 0.0, 0.0,
        -1.0, -2.0, -1.0,
      );
      return weights[row * 3 + col];
    }
    default: {
      if (row == 0 && col == 0) {
        return 1.0;
      }
      return 0.0;
    }
  }
}

fn kernel_sum(id : i32) -> f32 {
  let dims = kernel_dimensions(id);
  var total : f32 = 0.0;
  for (var y : i32 = 0; y < dims.y; y = y + 1) {
    for (var x : i32 = 0; x < dims.x; x = x + 1) {
      total = total + kernel_weight(id, y, x);
    }
  }
  return total;
}

fn kernel_abs_sum(id : i32) -> f32 {
  let dims = kernel_dimensions(id);
  var total : f32 = 0.0;
  for (var y : i32 = 0; y < dims.y; y = y + 1) {
    for (var x : i32 = 0; x < dims.x; x = x + 1) {
      total = total + abs(kernel_weight(id, y, x));
    }
  }
  return total;
}

fn apply_normalization(value : vec4<f32>, id : i32, enabled : bool) -> vec4<f32> {
  if (!enabled) {
    return value;
  }
  let sum = kernel_sum(id);
  if (sum != 0.0) {
    return value / vec4<f32>(sum);
  }
  let abs_sum = kernel_abs_sum(id);
  if (abs_sum != 0.0) {
    let scaled = value / vec4<f32>(abs_sum);
    return scaled * 0.5 + vec4<f32>(0.5);
  }
  return value;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let width : u32 = u32(max(frame_uniforms.resolution.x, 0.0));
  let height : u32 = u32(max(frame_uniforms.resolution.y, 0.0));
  if (gid.x >= width || gid.y >= height) {
    return;
  }

  let coords : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
  let dims = kernel_dimensions(stage_uniforms.kernel_id);
  let half_w : i32 = dims.x / 2;
  let half_h : i32 = dims.y / 2;
  let width_i : i32 = i32(width);
  let height_i : i32 = i32(height);

  var accum : vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  for (var ky : i32 = 0; ky < dims.y; ky = ky + 1) {
    for (var kx : i32 = 0; kx < dims.x; kx = kx + 1) {
      let offset_x = wrap_coord(coords.x + (kx - half_w), width_i);
      let offset_y = wrap_coord(coords.y + (ky - half_h), height_i);
      let sample = textureLoad(input_texture, vec2<i32>(offset_x, offset_y), 0);
      let weight = kernel_weight(stage_uniforms.kernel_id, ky, kx);
      accum = accum + sample * weight;
    }
  }

  let normalized = apply_normalization(accum, stage_uniforms.kernel_id, stage_uniforms.with_normalize != 0u);
  let alpha = clamp(stage_uniforms.alpha, 0.0, 1.0);
  let original = textureLoad(input_texture, coords, 0);
  let blended = mix(original, normalized, alpha);
  let clamped = clamp(blended, vec4<f32>(0.0), vec4<f32>(1.0));
  textureStore(output_texture, coords, clamped);
}
