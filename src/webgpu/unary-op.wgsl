struct UnaryParams {
  width: u32,
  height: u32,
  channels: u32,
  op: u32,
};

@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: UnaryParams;

fn apply_op(value: f32, op: u32) -> f32 {
  switch op {
    case 0u: { // invert
      return 1.0 - value;
    }
    case 1u: { // square
      return value * value;
    }
    case 2u: { // clamp to [0, 1]
      return clamp(value, 0.0, 1.0);
    }
    default: {
      return value;
    }
  }
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  let w = params.width;
  let h = params.height;
  if (x >= w || y >= h) {
    return;
  }
  let base = (y * w + x) * params.channels;
  let texel = textureLoad(tex, vec2<i32>(i32(x), i32(y)), 0);
  switch params.channels {
    case 1u: {
      out[base] = apply_op(texel.x, params.op);
    }
    case 2u: {
      out[base] = apply_op(texel.x, params.op);
      out[base + 1u] = apply_op(texel.y, params.op);
    }
    case 3u: {
      out[base] = apply_op(texel.x, params.op);
      out[base + 1u] = apply_op(texel.y, params.op);
      out[base + 2u] = apply_op(texel.z, params.op);
    }
    default: {
      out[base] = apply_op(texel.x, params.op);
      out[base + 1u] = apply_op(texel.y, params.op);
      out[base + 2u] = apply_op(texel.z, params.op);
      out[base + 3u] = apply_op(texel.w, params.op);
    }
  }
}
