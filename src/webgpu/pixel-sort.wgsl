const MAX_WIDTH: u32 = 2048u;

struct PixelSortParams {
  width: u32,
  channels: u32,
  darkest: u32,
};

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: PixelSortParams;

var<workgroup> values: array<f32, MAX_WIDTH>;
var<workgroup> brightness: array<f32, MAX_WIDTH>;
var<workgroup> pivot: u32;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  let width = params.width;
  let channels = params.channels;
  let darkest = params.darkest;

  // Load brightness per pixel
  var i = tid;
  while (i < width) {
    var b: f32 = 0.0;
    let base = i * channels;
    var k: u32 = 0u;
    loop {
      if (k >= channels) { break; }
      let v = src[base + k];
      b = b + (darkest == 1u ? (1.0 - v) : v);
      k = k + 1u;
    }
    brightness[i] = b / f32(channels);
    i = i + 256u;
  }
  workgroupBarrier();

  // Find max brightness index
  if (tid == 0u) {
    var maxB = brightness[0];
    var maxI: u32 = 0u;
    var j: u32 = 1u;
    loop {
      if (j >= width) { break; }
      let v = brightness[j];
      if (v > maxB) {
        maxB = v;
        maxI = j;
      }
      j = j + 1u;
    }
    pivot = maxI;
  }
  workgroupBarrier();
  let maxIdx = pivot;

  // Next power of two
  var n = 1u;
  loop {
    if (n >= width) { break; }
    n = n << 1u;
  }

  var chan: u32 = 0u;
  loop {
    if (chan >= channels) { break; }

    // Load channel values, apply darkest inversion
    var idx = tid;
    while (idx < width) {
      let orig = src[idx * channels + chan];
      values[idx] = darkest == 1u ? (1.0 - orig) : orig;
      idx = idx + 256u;
    }
    var padIdx = width + tid;
    while (padIdx < n) {
      values[padIdx] = -1e9;
      padIdx = padIdx + 256u;
    }
    workgroupBarrier();

    // Bitonic sort descending
    var size = 2u;
    loop {
      if (size > n) { break; }
      var stride = size / 2u;
      loop {
        if (stride == 0u) { break; }
        var j = tid;
        while (j < n) {
          let ixj = j ^ stride;
          if (ixj > j) {
            let a = values[j];
            let b = values[ixj];
            let ascending = (j & size) == 0u;
            var swap = false;
            if (ascending) {
              if (a < b) { swap = true; }
            } else {
              if (a > b) { swap = true; }
            }
            if (swap) {
              values[j] = b;
              values[ixj] = a;
            }
          }
          j = j + 256u;
        }
        workgroupBarrier();
        stride = stride / 2u;
      }
      size = size * 2u;
    }

    // Rotate and write
    var x = tid;
    while (x < width) {
      let xx = (x + width - maxIdx) % width;
      out[x * channels + chan] = values[xx];
      x = x + 256u;
    }
    workgroupBarrier();

    chan = chan + 1u;
  }
}
