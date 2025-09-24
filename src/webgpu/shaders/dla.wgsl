struct SimulationParams {
  halfWidth : u32,
  halfHeight : u32,
  walkerCount : u32,
  iterationCount : u32,
};

struct CounterState {
  value : atomic<u32>,
};

const SMALL_OFFSETS : array<i32, 3> = array<i32, 3>(-1, 0, 1);
const EXPANDED_RANGE : i32 = 8;
const EXPANDED_WIDTH : u32 = u32(EXPANDED_RANGE * 2 + 1);

@group(0) @binding(0) var<storage, read_write> walkers : array<u32>;
@group(0) @binding(1) var<storage, read_write> cluster : array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> neighborhood : array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> expanded : array<atomic<u32>>;
@group(0) @binding(4) var<storage, read> randY : array<f32>;
@group(0) @binding(5) var<storage, read> randX : array<f32>;
@group(0) @binding(6) var<storage, read_write> counter : CounterState;
@group(0) @binding(7) var<uniform> params : SimulationParams;

fn clamp_index(value : f32, maxIndex : u32) -> u32 {
  let scaled : f32 = value * f32(maxIndex + 1u);
  let clamped : u32 = u32(scaled);
  if (clamped > maxIndex) {
    return maxIndex;
  }
  return clamped;
}

fn wrap_coord(coord : u32, delta : i32, limit : u32) -> u32 {
  if (limit == 0u) {
    return 0u;
  }
  let size : i32 = i32(limit);
  var sum : i32 = i32(coord) + delta;
  sum = sum % size;
  if (sum < 0) {
    sum = sum + size;
  }
  return u32(sum);
}

fn select_small_offset(value : f32) -> i32 {
  let index : u32 = clamp_index(value, 2u);
  return SMALL_OFFSETS[index];
}

fn select_expanded_offset(value : f32) -> i32 {
  let index : u32 = clamp_index(value, EXPANDED_WIDTH - 1u);
  return i32(index) - EXPANDED_RANGE;
}

fn update_fields(y : u32, x : u32, height : u32, width : u32) {
  for (var dy : i32 = -1; dy <= 1; dy = dy + 1) {
    for (var dx : i32 = -1; dx <= 1; dx = dx + 1) {
      let ny : u32 = wrap_coord(y, dy, height);
      let nx : u32 = wrap_coord(x, dx, width);
      let idx : u32 = ny * width + nx;
      atomicStore(&neighborhood[idx], 1u);
    }
  }

  for (var dy : i32 = -EXPANDED_RANGE; dy <= EXPANDED_RANGE; dy = dy + 1) {
    for (var dx : i32 = -EXPANDED_RANGE; dx <= EXPANDED_RANGE; dx = dx + 1) {
      let ny : u32 = wrap_coord(y, dy, height);
      let nx : u32 = wrap_coord(x, dx, width);
      let idx : u32 = ny * width + nx;
      atomicStore(&expanded[idx], 1u);
    }
  }
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let index : u32 = gid.x;
  if (index >= params.walkerCount) {
    return;
  }

  let width : u32 = params.halfWidth;
  let height : u32 = params.halfHeight;
  if (width == 0u || height == 0u) {
    return;
  }

  let iterations : u32 = params.iterationCount;
  let walkerBase : u32 = index * 2u;
  var wy : u32 = walkers[walkerBase];
  var wx : u32 = walkers[walkerBase + 1u];

  if (wy >= height || wx >= width) {
    return;
  }

  for (var iter : u32 = 0u; iter < iterations; iter = iter + 1u) {
    if (wy >= height || wx >= width) {
      break;
    }

    let baseIndex : u32 = wy * width + wx;
    if (atomicLoad(&neighborhood[baseIndex]) != 0u) {
      var created_new_cell : bool = false;
      if (atomicLoad(&cluster[baseIndex]) == 0u) {
        let order : u32 = atomicAdd(&counter.value, 1u) + 1u;
        let exchange = atomicCompareExchangeWeak(&cluster[baseIndex], 0u, order);
        if (exchange.exchanged) {
          created_new_cell = true;
        }
      }

      if (created_new_cell) {
        update_fields(wy, wx, height, width);
      }

      wy = height;
      wx = width;
      break;
    }

    let randIndex : u32 = iter * params.walkerCount + index;
    let yRand : f32 = randY[randIndex];
    let xRand : f32 = randX[randIndex];
    let isExpanded : bool = atomicLoad(&expanded[baseIndex]) != 0u;

    var yOffset : i32;
    var xOffset : i32;
    if (isExpanded) {
      yOffset = select_small_offset(yRand);
      xOffset = select_small_offset(xRand);
    } else {
      yOffset = select_expanded_offset(yRand);
      xOffset = select_expanded_offset(xRand);
    }

    wy = wrap_coord(wy, yOffset, height);
    wx = wrap_coord(wx, xOffset, width);
  }

  walkers[walkerBase] = wy;
  walkers[walkerBase + 1u] = wx;
}
