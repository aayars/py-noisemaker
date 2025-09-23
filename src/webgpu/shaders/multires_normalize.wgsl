// Multires normalization resolve pass.
//
// Applies the global min/max range accumulated during the primary multires
// compute dispatch and maps stored colors into the [0, 1] interval. The shader
// mirrors the StageUniforms/FrameUniforms layout from multires.wgsl so the same
// uniform buffers can be reused.

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

const FLOAT_SIGN_BIT : u32 = 0x80000000u;
const FLOAT_MAX : f32 = 3.4028234663852886e+38;

fn bool_from_u32(value : u32) -> bool {
  return value != 0u;
}

fn ordered_uint_to_float(value : u32) -> f32 {
  var bits : u32;
  if ((value & FLOAT_SIGN_BIT) != 0u) {
    bits = value ^ FLOAT_SIGN_BIT;
  } else {
    bits = ~value;
  }
  return bitcast<f32>(bits);
}

fn float_is_finite(value : f32) -> bool {
  return value == value && abs(value) <= FLOAT_MAX;
}

fn sanitize_component(value : f32) -> f32 {
  return select(0.0, value, float_is_finite(value));
}

fn sanitize_sample(value : vec4<f32>) -> vec4<f32> {
  return vec4<f32>(
    sanitize_component(value.x),
    sanitize_component(value.y),
    sanitize_component(value.z),
    sanitize_component(value.w),
  );
}

@group(0) @binding(0) var<uniform> stage_uniforms : StageUniforms;
@group(0) @binding(1) var<uniform> frame_uniforms : FrameUniforms;
@group(0) @binding(2) var input_texture : texture_2d<f32>;
@group(0) @binding(3) var output_texture : texture_storage_2d<rgba32float, write>;
@group(0) @binding(4) var<storage, read_write> normalization_state : NormalizationState;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
  let width : u32 = u32(frame_uniforms.resolution.x);
  let height : u32 = u32(frame_uniforms.resolution.y);
  if (global_id.x >= width || global_id.y >= height) {
    return;
  }

  let encoded_min : u32 = atomicLoad(&normalization_state.min_value);
  let encoded_max : u32 = atomicLoad(&normalization_state.max_value);
  let min_value : f32 = ordered_uint_to_float(encoded_min);
  let max_value : f32 = ordered_uint_to_float(encoded_max);

  let raw_sample : vec4<f32> = textureLoad(
    input_texture,
    vec2<i32>(i32(global_id.x), i32(global_id.y)),
    0,
  );
  let sample : vec4<f32> = sanitize_sample(raw_sample);

  var normalized : vec4<f32> = sample;
  if (float_is_finite(min_value) && float_is_finite(max_value)) {
    let delta : f32 = max_value - min_value;
    if (delta != 0.0) {
      let bias : vec4<f32> = vec4<f32>(min_value, min_value, min_value, min_value);
      let inv_delta : f32 = 1.0 / delta;
      normalized = (sample - bias) * inv_delta;
    }
  }

  if (!bool_from_u32(stage_uniforms.options1.w)) {
    normalized.w = sample.w;
  }

  textureStore(
    output_texture,
    vec2<i32>(i32(global_id.x), i32(global_id.y)),
    normalized,
  );
}
