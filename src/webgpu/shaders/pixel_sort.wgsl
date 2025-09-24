struct PixelSortParams {
  width : u32;
  channels : u32;
  darkest : u32;
  _pad : u32;
};

@group(0) @binding(0) var<storage, read> inputRow : array<f32>;
@group(0) @binding(1) var<storage, read_write> outputRow : array<f32>;
@group(0) @binding(2) var<uniform> params : PixelSortParams;
@group(0) @binding(3) var<storage, read_write> scratch : array<f32>;

const NEG_INFINITY : f32 = -3.402823466e38;

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (gid.x > 0u || gid.y > 0u || gid.z > 0u) {
    return;
  }

  let width : u32 = params.width;
  let channelCount : u32 = max(params.channels, 1u);
  if (width == 0u || channelCount == 0u) {
    return;
  }

  var shift : u32 = 0u;
  var maxValue : f32 = NEG_INFINITY;
  var idx : u32 = 0u;
  loop {
    if (idx >= width) {
      break;
    }
    let v : f32 = scratch[idx];
    if (v > maxValue) {
      maxValue = v;
      shift = idx;
    }
    idx = idx + 1u;
  }

  let totalSize : u32 = width * channelCount;
  var clearIdx : u32 = 0u;
  loop {
    if (clearIdx >= totalSize) {
      break;
    }
    outputRow[clearIdx] = 0.0;
    clearIdx = clearIdx + 1u;
  }

  var channel : u32 = 0u;
  loop {
    if (channel >= channelCount) {
      break;
    }

    var copyIdx : u32 = 0u;
    loop {
      if (copyIdx >= width) {
        break;
      }
      scratch[copyIdx] = inputRow[copyIdx * channelCount + channel];
      copyIdx = copyIdx + 1u;
    }

    var rank : u32 = 0u;
    loop {
      if (rank >= width) {
        break;
      }
      var bestIndex : u32 = 0u;
      var bestValue : f32 = NEG_INFINITY;
      var search : u32 = 0u;
      loop {
        if (search >= width) {
          break;
        }
        let candidate : f32 = scratch[search];
        if (candidate > bestValue) {
          bestValue = candidate;
          bestIndex = search;
        }
        search = search + 1u;
      }
      let targetX : u32 = (rank + shift) % width;
      outputRow[targetX * channelCount + channel] = bestValue;
      scratch[bestIndex] = NEG_INFINITY;
      rank = rank + 1u;
    }

    channel = channel + 1u;
  }
}
