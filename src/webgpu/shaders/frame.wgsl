struct FrameParams {
  sizeTime : vec4<f32>;   // width, height, channels, time
  motionSeed : vec4<f32>; // speed, seed, grainStrength, frameStrength
};

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
@group(0) @binding(2) var<uniform> params : FrameParams;

fn clamp01(value : f32) -> f32 {
  return clamp(value, 0.0, 1.0);
}

fn as_u32(value : f32) -> u32 {
  if (value <= 0.0) {
    return 0u;
  }
  return u32(value + 0.5);
}

fn luminance(rgb : vec3<f32>) -> f32 {
  return dot(rgb, vec3<f32>(0.299, 0.587, 0.114));
}

fn hash3(value : vec3<f32>) -> f32 {
  let dot_val : f32 = dot(value, vec3<f32>(12.9898, 78.233, 37.719));
  let s : f32 = sin(dot_val);
  return fract(s * 43758.5453);
}

fn edge_distance(uv : vec2<f32>) -> f32 {
  let centered : vec2<f32> = abs(uv - vec2<f32>(0.5, 0.5));
  return max(centered.x, centered.y) * 2.0;
}

fn saturate_color(rgb : vec3<f32>, factor : f32) -> vec3<f32> {
  let gray : f32 = luminance(rgb);
  return mix(vec3<f32>(gray, gray, gray), rgb, clamp01(factor));
}

fn apply_frame_tint(rgb : vec3<f32>, center_mix : f32, edge_mix : f32) -> vec3<f32> {
  let warm : vec3<f32> = vec3<f32>(1.08, 1.02, 0.94);
  let cool : vec3<f32> = vec3<f32>(0.82, 0.86, 0.93);
  let mid : vec3<f32> = mix(cool, warm, clamp01(center_mix));
  let frame_color : vec3<f32> = vec3<f32>(0.82, 0.8, 0.76);
  let tinted : vec3<f32> = mix(rgb, rgb * mid, 0.4);
  return mix(tinted, frame_color, clamp01(edge_mix));
}

fn apply_grain(rgb : vec3<f32>, grain : f32, edge_mask : f32) -> vec3<f32> {
  let strength : f32 = 0.35 * edge_mask + 0.12;
  return clamp(rgb + grain * strength, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn apply_ghosting(rgb : vec3<f32>, shift : f32) -> vec3<f32> {
  let r : f32 = clamp01(rgb.x + shift);
  let b : f32 = clamp01(rgb.z - shift);
  return vec3<f32>(r, rgb.y, b);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
  let width_u : u32 = max(as_u32(params.sizeTime.x), 1u);
  let height_u : u32 = max(as_u32(params.sizeTime.y), 1u);
  if (global_id.x >= width_u || global_id.y >= height_u) {
    return;
  }

  let channels_u : u32 = max(as_u32(params.sizeTime.z), 1u);
  let width_f : f32 = max(params.sizeTime.x, 1.0);
  let height_f : f32 = max(params.sizeTime.y, 1.0);
  let time_value : f32 = params.sizeTime.w;
  let speed_value : f32 = params.motionSeed.x;
  let seed_value : f32 = params.motionSeed.y;
  let grain_strength : f32 = params.motionSeed.z;
  let frame_strength : f32 = params.motionSeed.w;

  let pixel_index : u32 = global_id.y * width_u + global_id.x;
  let base_index : u32 = pixel_index * channels_u;
  let coords : vec2<i32> = vec2<i32>(i32(global_id.x), i32(global_id.y));
  let sample : vec4<f32> = textureLoad(input_texture, coords, 0);
  var color : vec3<f32> = sample.xyz;

  let uv : vec2<f32> = (vec2<f32>(f32(global_id.x), f32(global_id.y)) + vec2<f32>(0.5, 0.5)) /
    vec2<f32>(width_f, height_f);
  let dist : f32 = edge_distance(uv);
  let frame_mask : f32 = smoothstep(0.72, 0.96, dist);
  let outer_mask : f32 = smoothstep(0.88, 1.04, dist);
  let center_mask : f32 = smoothstep(0.05, 0.6, 1.0 - dist);

  color = saturate_color(color, mix(0.58, 0.82, center_mask));
  color = apply_frame_tint(color, center_mask, frame_mask * frame_strength);
  color = color * mix(1.0, 0.68, frame_mask);
  color = color * (1.0 + center_mask * 0.08);

  let wobble_seed : vec3<f32> = vec3<f32>(uv * vec2<f32>(width_f, height_f), time_value * speed_value + seed_value);
  let grain_noise : f32 = (hash3(wobble_seed + vec3<f32>(0.37, 0.11, 0.53)) - 0.5) * grain_strength;
  color = apply_grain(color, grain_noise, frame_mask);

  let streak_cell : vec3<f32> = vec3<f32>(floor(uv * vec2<f32>(width_f * 0.5, height_f * 0.5)), seed_value * 1.37);
  let streak_noise : f32 = hash3(streak_cell + vec3<f32>(time_value * 0.25, 0.0, 0.5));
  let streak_mask : f32 = step(0.985, streak_noise) * frame_mask;
  color = clamp(color + streak_mask * 0.28, vec3<f32>(0.0), vec3<f32>(1.0));

  let dust_cell : vec3<f32> = vec3<f32>(floor(uv * vec2<f32>(width_f, height_f)), seed_value * 2.17 + time_value);
  let dust_noise : f32 = hash3(dust_cell);
  let dust_mask : f32 = smoothstep(0.7, 1.0, dust_noise) * outer_mask * 0.4;
  color = clamp(color + dust_mask * vec3<f32>(0.12, 0.1, 0.08), vec3<f32>(0.0), vec3<f32>(1.0));

  let chroma_shift : f32 = (grain_noise + hash3(wobble_seed + vec3<f32>(1.27, 2.1, 3.8)) - 0.5) * 0.12 * frame_mask;
  color = apply_ghosting(color, chroma_shift);

  let vignette : f32 = smoothstep(0.4, 1.1, dist);
  color = clamp(color * mix(1.0, 0.82, vignette), vec3<f32>(0.0), vec3<f32>(1.0));

  let gray : f32 = clamp01(luminance(color));
  let alpha : f32 = clamp01(sample.w * mix(1.0, 0.88, frame_mask));

  if (channels_u == 1u) {
    output_buffer[base_index] = gray;
    return;
  }

  if (channels_u == 2u) {
    output_buffer[base_index] = gray;
    output_buffer[base_index + 1u] = alpha;
    return;
  }

  output_buffer[base_index] = clamp01(color.x);
  if (channels_u > 1u) {
    output_buffer[base_index + 1u] = clamp01(color.y);
  }
  if (channels_u > 2u) {
    output_buffer[base_index + 2u] = clamp01(color.z);
  }
  if (channels_u > 3u) {
    output_buffer[base_index + 3u] = alpha;
  }
  if (channels_u > 4u) {
    var ch : u32 = 4u;
    loop {
      if (ch >= channels_u) {
        break;
      }
      output_buffer[base_index + ch] = alpha;
      ch = ch + 1u;
    }
  }
}
