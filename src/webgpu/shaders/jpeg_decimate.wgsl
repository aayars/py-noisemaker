// GPU approximation of the jpeg_decimate effect.
//
// Simulates repeated JPEG recompression by combining per-iteration quantization
// with coarse pixel averaging sampled from progressively larger blocks. The
// shader mirrors the new pipeline binding model (stage uniforms, frame
// uniforms, storage textures) so it can participate in the WebGPU preset
// pipeline once the runtime enables this effect.

struct StageUniforms {
  iterations : i32,
  _padding0 : vec3<i32>,
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

fn clamp01(value : f32) -> f32 {
  return clamp(value, 0.0, 1.0);
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

fn random_u32(base_seed : u32, iteration : u32, salt : u32) -> u32 {
  let mixed0 : u32 = base_seed ^ (iteration * 0x9e3779b9u) ^ (salt * 0x6c8e9cf5u);
  let mixed1 : u32 = (base_seed + salt * 0x85ebca6bu) ^ (iteration * 0x27d4eb2du);
  let mixed2 : u32 = (base_seed ^ (iteration << 16)) + salt * 0x165667b1u + 0x9e3779b9u;
  let hashed : vec3<u32> = vec3<u32>(mixed0, mixed1, mixed2);
  let noise : vec3<u32> = pcg3d(hashed);
  return noise.x;
}

fn random_float(base_seed : u32, iteration : u32, salt : u32) -> f32 {
  return f32(random_u32(base_seed, iteration, salt)) / 4294967296.0;
}

fn random_int(base_seed : u32, iteration : u32, salt : u32, min_value : i32, max_value : i32) -> i32 {
  var lo : i32 = min_value;
  var hi : i32 = max_value;
  if (hi < lo) {
    let tmp : i32 = lo;
    lo = hi;
    hi = tmp;
  }
  let range : i32 = hi - lo + 1;
  if (range <= 1) {
    return lo;
  }
  let rnd : f32 = random_float(base_seed, iteration, salt);
  var scaled : i32 = i32(floor(rnd * f32(range)));
  if (scaled >= range) {
    scaled = range - 1;
  }
  if (scaled < 0) {
    scaled = 0;
  }
  return lo + scaled;
}

fn quantize_component(value : f32, step : i32) -> f32 {
  let safe_step : i32 = max(step, 1);
  let safe_value : f32 = clamp01(value);
  let scaled : f32 = safe_value * 255.0;
  let rounded : f32 = floor(scaled + 0.5);
  let stepf : f32 = f32(safe_step);
  let quantized : f32 = floor(rounded / stepf + 0.5) * stepf;
  let clamped_byte : f32 = clamp(quantized, 0.0, 255.0);
  return clamped_byte / 255.0;
}

fn quantize_vec(color : vec4<f32>, step : i32) -> vec4<f32> {
  return vec4<f32>(
    quantize_component(color.x, step),
    quantize_component(color.y, step),
    quantize_component(color.z, step),
    quantize_component(color.w, step),
  );
}

fn clamp_coord(value : i32, limit : i32) -> i32 {
  if (limit <= 0) {
    return 0;
  }
  return clamp(value, 0, limit - 1);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let width : u32 = u32(max(frame_uniforms.resolution.x, 0.0));
  let height : u32 = u32(max(frame_uniforms.resolution.y, 0.0));
  if (gid.x >= width || gid.y >= height) {
    return;
  }

  let coords : vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
  let width_i32 : i32 = i32(width);
  let height_i32 : i32 = i32(height);

  var color : vec4<f32> = textureLoad(input_texture, coords, 0);
  var sample_pos : vec2<f32> = vec2<f32>(f32(coords.x), f32(coords.y));

  var iteration_count : u32 = 0u;
  if (stage_uniforms.iterations > 0) {
    iteration_count = u32(stage_uniforms.iterations);
  }
  if (iteration_count > 128u) {
    iteration_count = 128u;
  }

  let base_seed : u32 =
    frame_uniforms.seed ^
    (frame_uniforms.frame_index * 0x9e3779b9u) ^
    (bitcast<u32>(frame_uniforms.time) * 0x85ebca6bu);

  var max_factor : u32 = 2u;
  if (width < height) {
    if (width > max_factor) {
      max_factor = width;
    }
  } else {
    if (height > max_factor) {
      max_factor = height;
    }
  }

  for (var i : u32 = 0u; i < iteration_count; i = i + 1u) {
    let quality : i32 = random_int(base_seed, i, 0u, 5, 50);
    let raw_step : f32 = (60.0 - f32(quality)) / 5.0;
    let step : i32 = max(1, i32(floor(raw_step + 0.5)));

    var factor : i32 = random_int(base_seed, i, 1u, 2, 8);
    let max_factor_i32 : i32 = i32(max_factor);
    if (factor > max_factor_i32) {
      factor = max_factor_i32;
    }
    if (factor < 1) {
      factor = 1;
    }

    if (factor > 1) {
      let factor_f : f32 = f32(factor);
      sample_pos = floor(sample_pos / factor_f) * factor_f + vec2<f32>(factor_f * 0.5, factor_f * 0.5);
      sample_pos.x = clamp(sample_pos.x, 0.0, f32(max(width_i32 - 1, 0)));
      sample_pos.y = clamp(sample_pos.y, 0.0, f32(max(height_i32 - 1, 0)));
      let sample_coord : vec2<i32> = vec2<i32>(
        clamp_coord(i32(floor(sample_pos.x + 0.5)), width_i32),
        clamp_coord(i32(floor(sample_pos.y + 0.5)), height_i32),
      );
      color = textureLoad(input_texture, sample_coord, 0);
    }

    color = quantize_vec(color, step);
  }

  let clamped_color : vec4<f32> = clamp(color, vec4<f32>(0.0), vec4<f32>(1.0));
  textureStore(output_texture, coords, clamped_color);
}
