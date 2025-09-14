struct Params {
  halfWidth: u32,
  halfHeight: u32,
  count: u32,
  iterations: u32,
};

@group(0) @binding(0) var<storage, read_write> walkers: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read_write> cluster: array<u32>;
@group(0) @binding(2) var<storage, read_write> neighborhood: array<u32>;
@group(0) @binding(3) var<storage, read_write> expanded: array<u32>;
@group(0) @binding(4) var<storage, read> randY: array<f32>;
@group(0) @binding(5) var<storage, read> randX: array<f32>;
struct Counter { value: atomic<u32>; };
@group(0) @binding(6) var<storage, read_write> counter: Counter;
@group(0) @binding(7) var<uniform> params: Params;

const offsets: array<i32,3> = array<i32,3>(-1,0,1);
const expandedOffsets: array<i32,17> = array<i32,17>(-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8);

@compute @workgroup_size(64,1,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.count) { return; }
  var y = walkers[idx].x;
  var x = walkers[idx].y;
  let hw = params.halfWidth;
  let hh = params.halfHeight;
  let cnt = params.count;
  let iterMax = params.iterations;

  for (var iter: u32 = 0u; iter < iterMax; iter = iter + 1u) {
    let key = y * hw + x;
    if (neighborhood[key] != 0u) {
      let order = atomicAdd(&counter.value, 1u) + 1u;
      cluster[key] = order;
      for (var oy: u32 = 0u; oy < 3u; oy = oy + 1u) {
        let offY = offsets[i32(oy)];
        for (var ox: u32 = 0u; ox < 3u; ox = ox + 1u) {
          let offX = offsets[i32(ox)];
          let ny = u32((i32(y) + offY + i32(hh)) % i32(hh));
          let nx = u32((i32(x) + offX + i32(hw)) % i32(hw));
          let nkey = ny * hw + nx;
          neighborhood[nkey] = 1u;
        }
      }
      for (var oy: u32 = 0u; oy < 17u; oy = oy + 1u) {
        let offY = expandedOffsets[i32(oy)];
        for (var ox: u32 = 0u; ox < 17u; ox = ox + 1u) {
          let offX = expandedOffsets[i32(ox)];
          let ny = u32((i32(y) + offY + i32(hh)) % i32(hh));
          let nx = u32((i32(x) + offX + i32(hw)) % i32(hw));
          let nkey = ny * hw + nx;
          expanded[nkey] = 1u;
        }
      }
      break;
    }
    let rIndex = iter * cnt + idx;
    var yo: i32;
    var xo: i32;
    if (expanded[key] != 0u) {
      yo = offsets[i32(floor(randY[rIndex] * 3.0))];
      xo = offsets[i32(floor(randX[rIndex] * 3.0))];
    } else {
      yo = expandedOffsets[i32(floor(randY[rIndex] * 17.0))];
      xo = expandedOffsets[i32(floor(randX[rIndex] * 17.0))];
    }
    y = u32((i32(y) + yo + i32(hh)) % i32(hh));
    x = u32((i32(x) + xo + i32(hw)) % i32(hw));
  }
  walkers[idx] = vec2<u32>(y, x);
}
