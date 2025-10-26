import { Tensor } from './tensor.js';
import { random as rngRandom, randomInt as rngRandomInt, Random } from './rng.js';

// Maintain our own seed state just like the Python implementation.  The
// previous port simply re-exported the RNG's ``setSeed``/``getSeed`` helpers,
// so calls to ``simplex.setSeed`` would inadvertently reset the global RNG
// instead of the simplex module's internal counter.  This caused divergences in
// functions that rely on ``simplex.getSeed`` to produce deterministic seeds.
//
// ``_seed`` mirrors ``noisemaker.simplex``'s behaviour: when explicitly set it
// is incremented on each ``getSeed`` call; otherwise ``getSeed`` consumes a
// value from the global RNG.
let _seed = null;

export function setSeed(s) {
  _seed = s >>> 0;
}

export function getSeed() {
  if (_seed === null) {
    _seed = rngRandomInt(1, 65536); // RNG call
  } else {
    _seed = (_seed + 1) >>> 0;
  }
  return _seed;
}

const STRETCH_CONSTANT_2D = -0.211324865405187;
const SQUISH_CONSTANT_2D = 0.366025403784439;
const STRETCH_CONSTANT_3D = -1.0 / 6;
const SQUISH_CONSTANT_3D = 1.0 / 3;
const NORM_CONSTANT_2D = 47;
const NORM_CONSTANT_3D = 103;

const GRADIENTS_2D = [
  5, 2, 2, 5,
  -5, 2, -2, 5,
  5, -2, 2, -5,
  -5, -2, -2, -5,
];

const GRADIENTS_3D = [
  -11, 4, 4, -4, 11, 4, -4, 4, 11,
  11, 4, 4, 4, 11, 4, 4, 4, 11,
  -11, -4, 4, -4, -11, 4, -4, -4, 11,
  11, -4, 4, 4, -11, 4, 4, -4, 11,
  -11, 4, -4, -4, 11, -4, -4, 4, -11,
  11, 4, -4, 4, 11, -4, 4, 4, -11,
  -11, -4, -4, -4, -11, -4, -4, -4, -11,
  11, -4, -4, 4, -11, -4, 4, -4, -11,
];

function _buildPermutations(seed) {
  const perm = new Uint8Array(256);
  const permGradIndex3D = new Uint8Array(256);
  const source = new Uint8Array(256);
  for (let i = 0; i < 256; i++) source[i] = i;
  const r = new Random(seed);
  const gradLen = GRADIENTS_3D.length / 3;
  for (let i = 255; i >= 0; i--) {
    const idx = r.randomInt(0, i);
    perm[i] = source[idx];
    permGradIndex3D[i] = (perm[i] % gradLen) * 3;
    source[idx] = source[i];
  }
  return { perm, permGradIndex3D };
}

export class OpenSimplex {
  constructor(seed = 0) {
    const { perm, permGradIndex3D } = _buildPermutations(seed);
    this.perm = perm;
    this.permGradIndex3D = permGradIndex3D;
  }

  _extrapolate2d(xsb, ysb, dx, dy) {
    const perm = this.perm;
    const index = perm[(perm[xsb & 0xff] + ysb) & 0xff] & 0x0e;
    const g1 = GRADIENTS_2D[index];
    const g2 = GRADIENTS_2D[index + 1];
    return g1 * dx + g2 * dy;
  }

  _extrapolate3d(xsb, ysb, zsb, dx, dy, dz) {
    const perm = this.perm;
    const index = this.permGradIndex3D[(perm[(perm[xsb & 0xff] + ysb) & 0xff] + zsb) & 0xff];
    const g1 = GRADIENTS_3D[index];
    const g2 = GRADIENTS_3D[index + 1];
    const g3 = GRADIENTS_3D[index + 2];
    return g1 * dx + g2 * dy + g3 * dz;
  }

  noise2D(x, y) {
    const stretchOffset = (x + y) * STRETCH_CONSTANT_2D;
    const xs = x + stretchOffset;
    const ys = y + stretchOffset;
    let xsb = Math.floor(xs);
    let ysb = Math.floor(ys);
    const squishOffset = (xsb + ysb) * SQUISH_CONSTANT_2D;
    const xb = xsb + squishOffset;
    const yb = ysb + squishOffset;
    const xins = xs - xsb;
    const yins = ys - ysb;
    const inSum = xins + yins;
    let dx0 = x - xb;
    let dy0 = y - yb;
    let value = 0;

    let dx1 = dx0 - 1 - SQUISH_CONSTANT_2D;
    let dy1 = dy0 - SQUISH_CONSTANT_2D;
    let attn1 = 2 - dx1 * dx1 - dy1 * dy1;
    if (attn1 > 0) {
      attn1 *= attn1;
      value += attn1 * attn1 * this._extrapolate2d(xsb + 1, ysb, dx1, dy1);
    }

    let dx2 = dx0 - SQUISH_CONSTANT_2D;
    let dy2 = dy0 - 1 - SQUISH_CONSTANT_2D;
    let attn2 = 2 - dx2 * dx2 - dy2 * dy2;
    if (attn2 > 0) {
      attn2 *= attn2;
      value += attn2 * attn2 * this._extrapolate2d(xsb, ysb + 1, dx2, dy2);
    }

    let xsv_ext, ysv_ext, dx_ext, dy_ext;
    if (inSum <= 1) {
      let zins = 1 - inSum;
      if (zins > xins || zins > yins) {
        if (xins > yins) {
          xsv_ext = xsb + 1;
          ysv_ext = ysb - 1;
          dx_ext = dx0 - 1;
          dy_ext = dy0 + 1;
        } else {
          xsv_ext = xsb - 1;
          ysv_ext = ysb + 1;
          dx_ext = dx0 + 1;
          dy_ext = dy0 - 1;
        }
      } else {
        xsv_ext = xsb + 1;
        ysv_ext = ysb + 1;
        dx_ext = dx0 - 1 - 2 * SQUISH_CONSTANT_2D;
        dy_ext = dy0 - 1 - 2 * SQUISH_CONSTANT_2D;
      }
    } else {
      let zins = 2 - inSum;
      if (zins < xins || zins < yins) {
        if (xins > yins) {
          xsv_ext = xsb + 2;
          ysv_ext = ysb;
          dx_ext = dx0 - 2 - 2 * SQUISH_CONSTANT_2D;
          dy_ext = dy0 - 2 * SQUISH_CONSTANT_2D;
        } else {
          xsv_ext = xsb;
          ysv_ext = ysb + 2;
          dx_ext = dx0 - 2 * SQUISH_CONSTANT_2D;
          dy_ext = dy0 - 2 - 2 * SQUISH_CONSTANT_2D;
        }
      } else {
        xsv_ext = xsb;
        ysv_ext = ysb;
        dx_ext = dx0;
        dy_ext = dy0;
      }
      xsb += 1;
      ysb += 1;
      dx0 = dx0 - 1 - 2 * SQUISH_CONSTANT_2D;
      dy0 = dy0 - 1 - 2 * SQUISH_CONSTANT_2D;
    }

    let attn0 = 2 - dx0 * dx0 - dy0 * dy0;
    if (attn0 > 0) {
      attn0 *= attn0;
      value += attn0 * attn0 * this._extrapolate2d(xsb, ysb, dx0, dy0);
    }

    let attn_ext = 2 - dx_ext * dx_ext - dy_ext * dy_ext;
    if (attn_ext > 0) {
      attn_ext *= attn_ext;
      value += attn_ext * attn_ext * this._extrapolate2d(xsv_ext, ysv_ext, dx_ext, dy_ext);
    }

    return value / NORM_CONSTANT_2D;
  }

  noise3D(x, y, z) {
    const stretchOffset = (x + y + z) * STRETCH_CONSTANT_3D;
    const xs = x + stretchOffset;
    const ys = y + stretchOffset;
    const zs = z + stretchOffset;
    let xsb = Math.floor(xs);
    let ysb = Math.floor(ys);
    let zsb = Math.floor(zs);
    const squishOffset = (xsb + ysb + zsb) * SQUISH_CONSTANT_3D;
    let dx0 = x - (xsb + squishOffset);
    let dy0 = y - (ysb + squishOffset);
    let dz0 = z - (zsb + squishOffset);
    const xins = xs - xsb;
    const yins = ys - ysb;
    const zins = zs - zsb;
    const inSum = xins + yins + zins;
    let value = 0;

    let xsv_ext0, ysv_ext0, zsv_ext0;
    let xsv_ext1, ysv_ext1, zsv_ext1;
    let dx_ext0, dy_ext0, dz_ext0;
    let dx_ext1, dy_ext1, dz_ext1;

    if (inSum <= 1) {
      let aPoint = 0x01;
      let aScore = xins;
      let bPoint = 0x02;
      let bScore = yins;
      if (aScore >= bScore && zins > bScore) {
        bScore = zins;
        bPoint = 0x04;
      } else if (aScore < bScore && zins > aScore) {
        aScore = zins;
        aPoint = 0x04;
      }
      const wins = 1 - inSum;
      if (wins > aScore || wins > bScore) {
        const c = bScore > aScore ? bPoint : aPoint;
        if ((c & 0x01) === 0) {
          xsv_ext0 = xsb - 1;
          xsv_ext1 = xsb;
          dx_ext0 = dx0 + 1;
          dx_ext1 = dx0;
        } else {
          xsv_ext0 = xsv_ext1 = xsb + 1;
          dx_ext0 = dx_ext1 = dx0 - 1;
        }
        if ((c & 0x02) === 0) {
          ysv_ext0 = ysv_ext1 = ysb;
          dy_ext0 = dy_ext1 = dy0;
          if ((c & 0x01) === 0) {
            ysv_ext1 -= 1;
            dy_ext1 += 1;
          } else {
            ysv_ext0 -= 1;
            dy_ext0 += 1;
          }
        } else {
          ysv_ext0 = ysv_ext1 = ysb + 1;
          dy_ext0 = dy_ext1 = dy0 - 1;
        }
        if ((c & 0x04) === 0) {
          zsv_ext0 = zsb;
          zsv_ext1 = zsb - 1;
          dz_ext0 = dz0;
          dz_ext1 = dz0 + 1;
        } else {
          zsv_ext0 = zsv_ext1 = zsb + 1;
          dz_ext0 = dz_ext1 = dz0 - 1;
        }
      } else {
        const c = aPoint | bPoint;
        if ((c & 0x01) === 0) {
          xsv_ext0 = xsb;
          xsv_ext1 = xsb - 1;
          dx_ext0 = dx0 - 2 * SQUISH_CONSTANT_3D;
          dx_ext1 = dx0 + 1 - SQUISH_CONSTANT_3D;
        } else {
          xsv_ext0 = xsv_ext1 = xsb + 1;
          dx_ext0 = dx0 - 1 - 2 * SQUISH_CONSTANT_3D;
          dx_ext1 = dx0 - 1 - SQUISH_CONSTANT_3D;
        }
        if ((c & 0x02) === 0) {
          ysv_ext0 = ysb;
          ysv_ext1 = ysb - 1;
          dy_ext0 = dy0 - 2 * SQUISH_CONSTANT_3D;
          dy_ext1 = dy0 + 1 - SQUISH_CONSTANT_3D;
        } else {
          ysv_ext0 = ysv_ext1 = ysb + 1;
          dy_ext0 = dy0 - 1 - 2 * SQUISH_CONSTANT_3D;
          dy_ext1 = dy0 - 1 - SQUISH_CONSTANT_3D;
        }
        if ((c & 0x04) === 0) {
          zsv_ext0 = zsb;
          zsv_ext1 = zsb - 1;
          dz_ext0 = dz0 - 2 * SQUISH_CONSTANT_3D;
          dz_ext1 = dz0 + 1 - SQUISH_CONSTANT_3D;
        } else {
          zsv_ext0 = zsv_ext1 = zsb + 1;
          dz_ext0 = dz0 - 1 - 2 * SQUISH_CONSTANT_3D;
          dz_ext1 = dz0 - 1 - SQUISH_CONSTANT_3D;
        }
      }
      let attn0 = 2 - dx0 * dx0 - dy0 * dy0 - dz0 * dz0;
      if (attn0 > 0) {
        attn0 *= attn0;
        value += attn0 * attn0 * this._extrapolate3d(xsb, ysb, zsb, dx0, dy0, dz0);
      }
      let dx1 = dx0 - 1 - SQUISH_CONSTANT_3D;
      let dy1 = dy0 - SQUISH_CONSTANT_3D;
      let dz1 = dz0 - SQUISH_CONSTANT_3D;
      let attn1 = 2 - dx1 * dx1 - dy1 * dy1 - dz1 * dz1;
      if (attn1 > 0) {
        attn1 *= attn1;
        value += attn1 * attn1 * this._extrapolate3d(xsb + 1, ysb, zsb, dx1, dy1, dz1);
      }
      let dx2 = dx0 - SQUISH_CONSTANT_3D;
      let dy2 = dy0 - 1 - SQUISH_CONSTANT_3D;
      let dz2 = dz1;
      let attn2 = 2 - dx2 * dx2 - dy2 * dy2 - dz2 * dz2;
      if (attn2 > 0) {
        attn2 *= attn2;
        value += attn2 * attn2 * this._extrapolate3d(xsb, ysb + 1, zsb, dx2, dy2, dz2);
      }
      let dx3 = dx2;
      let dy3 = dy1;
      let dz3 = dz0 - 1 - SQUISH_CONSTANT_3D;
      let attn3 = 2 - dx3 * dx3 - dy3 * dy3 - dz3 * dz3;
      if (attn3 > 0) {
        attn3 *= attn3;
        value += attn3 * attn3 * this._extrapolate3d(xsb, ysb, zsb + 1, dx3, dy3, dz3);
      }

      let dx4 = dx0 - 1 - 2 * SQUISH_CONSTANT_3D;
      let dy4 = dy0 - 1 - 2 * SQUISH_CONSTANT_3D;
      let dz4 = dz0 - 2 * SQUISH_CONSTANT_3D;
      let attn4 = 2 - dx4 * dx4 - dy4 * dy4 - dz4 * dz4;
      if (attn4 > 0) {
        attn4 *= attn4;
        value += attn4 * attn4 * this._extrapolate3d(xsb + 1, ysb + 1, zsb, dx4, dy4, dz4);
      }
      let dx5 = dx4;
      let dy5 = dy0 - 2 * SQUISH_CONSTANT_3D;
      let dz5 = dz0 - 1 - 2 * SQUISH_CONSTANT_3D;
      let attn5 = 2 - dx5 * dx5 - dy5 * dy5 - dz5 * dz5;
      if (attn5 > 0) {
        attn5 *= attn5;
        value += attn5 * attn5 * this._extrapolate3d(xsb + 1, ysb, zsb + 1, dx5, dy5, dz5);
      }
      let dx6 = dx0 - 2 * SQUISH_CONSTANT_3D;
      let dy6 = dy4;
      let dz6 = dz5;
      let attn6 = 2 - dx6 * dx6 - dy6 * dy6 - dz6 * dz6;
      if (attn6 > 0) {
        attn6 *= attn6;
        value += attn6 * attn6 * this._extrapolate3d(xsb, ysb + 1, zsb + 1, dx6, dy6, dz6);
      }
    } else if (inSum >= 2) {
      let aPoint = 0x06;
      let aScore = xins;
      let bPoint = 0x05;
      let bScore = yins;
      if (aScore <= bScore && zins < bScore) {
        bScore = zins;
        bPoint = 0x03;
      } else if (aScore > bScore && zins < aScore) {
        aScore = zins;
        aPoint = 0x03;
      }
      const wins = 3 - inSum;
      if (wins < aScore || wins < bScore) {
        const c = bScore < aScore ? bPoint : aPoint;
        if ((c & 0x01) !== 0) {
          xsv_ext0 = xsb + 1;
          xsv_ext1 = xsb + 2;
          dx_ext0 = dx0 - 1 - SQUISH_CONSTANT_3D;
          dx_ext1 = dx0 - 2 - 2 * SQUISH_CONSTANT_3D;
        } else {
          xsv_ext0 = xsv_ext1 = xsb;
          dx_ext0 = dx0 - SQUISH_CONSTANT_3D;
          dx_ext1 = dx0 - 2 * SQUISH_CONSTANT_3D;
        }
        if ((c & 0x02) !== 0) {
          ysv_ext0 = ysb + 1;
          ysv_ext1 = ysb + 2;
          dy_ext0 = dy0 - 1 - SQUISH_CONSTANT_3D;
          dy_ext1 = dy0 - 2 - 2 * SQUISH_CONSTANT_3D;
        } else {
          ysv_ext0 = ysv_ext1 = ysb;
          dy_ext0 = dy0 - SQUISH_CONSTANT_3D;
          dy_ext1 = dy0 - 2 * SQUISH_CONSTANT_3D;
        }
        if ((c & 0x04) !== 0) {
          zsv_ext0 = zsb + 1;
          zsv_ext1 = zsb + 2;
          dz_ext0 = dz0 - 1 - SQUISH_CONSTANT_3D;
          dz_ext1 = dz0 - 2 - 2 * SQUISH_CONSTANT_3D;
        } else {
          zsv_ext0 = zsv_ext1 = zsb;
          dz_ext0 = dz0 - SQUISH_CONSTANT_3D;
          dz_ext1 = dz0 - 2 * SQUISH_CONSTANT_3D;
        }
      } else {
        const c = aPoint & bPoint;
        if ((c & 0x01) !== 0) {
          xsv_ext0 = xsb + 1;
          xsv_ext1 = xsb + 2;
          dx_ext0 = dx0 - 1 - SQUISH_CONSTANT_3D;
          dx_ext1 = dx0 - 2 - 2 * SQUISH_CONSTANT_3D;
        } else {
          xsv_ext0 = xsv_ext1 = xsb;
          dx_ext0 = dx0 - SQUISH_CONSTANT_3D;
          dx_ext1 = dx0 - 2 * SQUISH_CONSTANT_3D;
        }
        if ((c & 0x02) !== 0) {
          ysv_ext0 = ysb + 1;
          ysv_ext1 = ysb + 2;
          dy_ext0 = dy0 - 1 - SQUISH_CONSTANT_3D;
          dy_ext1 = dy0 - 2 - 2 * SQUISH_CONSTANT_3D;
        } else {
          ysv_ext0 = ysv_ext1 = ysb;
          dy_ext0 = dy0 - SQUISH_CONSTANT_3D;
          dy_ext1 = dy0 - 2 * SQUISH_CONSTANT_3D;
        }
        if ((c & 0x04) !== 0) {
          zsv_ext0 = zsb + 1;
          zsv_ext1 = zsb + 2;
          dz_ext0 = dz0 - 1 - SQUISH_CONSTANT_3D;
          dz_ext1 = dz0 - 2 - 2 * SQUISH_CONSTANT_3D;
        } else {
          zsv_ext0 = zsv_ext1 = zsb;
          dz_ext0 = dz0 - SQUISH_CONSTANT_3D;
          dz_ext1 = dz0 - 2 * SQUISH_CONSTANT_3D;
        }
      }
      let dx3 = dx0 - 1 - 2 * SQUISH_CONSTANT_3D;
      let dy3 = dy0 - 1 - 2 * SQUISH_CONSTANT_3D;
      let dz3 = dz0 - 0 - 2 * SQUISH_CONSTANT_3D;
      let attn3 = 2 - dx3 * dx3 - dy3 * dy3 - dz3 * dz3;
      if (attn3 > 0) {
        attn3 *= attn3;
        value += attn3 * attn3 * this._extrapolate3d(xsb + 1, ysb + 1, zsb, dx3, dy3, dz3);
      }
      let dx2 = dx3;
      let dy2 = dy0 - 0 - 2 * SQUISH_CONSTANT_3D;
      let dz2 = dz0 - 1 - 2 * SQUISH_CONSTANT_3D;
      let attn2 = 2 - dx2 * dx2 - dy2 * dy2 - dz2 * dz2;
      if (attn2 > 0) {
        attn2 *= attn2;
        value += attn2 * attn2 * this._extrapolate3d(xsb + 1, ysb, zsb + 1, dx2, dy2, dz2);
      }
      let dx1 = dx0 - 0 - 2 * SQUISH_CONSTANT_3D;
      let dy1 = dy3;
      let dz1 = dz2;
      let attn1 = 2 - dx1 * dx1 - dy1 * dy1 - dz1 * dz1;
      if (attn1 > 0) {
        attn1 *= attn1;
        value += attn1 * attn1 * this._extrapolate3d(xsb, ysb + 1, zsb + 1, dx1, dy1, dz1);
      }
      dx0 = dx0 - 1 - 3 * SQUISH_CONSTANT_3D;
      dy0 = dy0 - 1 - 3 * SQUISH_CONSTANT_3D;
      dz0 = dz0 - 1 - 3 * SQUISH_CONSTANT_3D;
      let attn0 = 2 - dx0 * dx0 - dy0 * dy0 - dz0 * dz0;
      if (attn0 > 0) {
        attn0 *= attn0;
        value += attn0 * attn0 * this._extrapolate3d(xsb + 1, ysb + 1, zsb + 1, dx0, dy0, dz0);
      }
    } else {
      let aPoint = 0x01;
      let aScore = xins;
      let bPoint = 0x02;
      let bScore = yins;
      let aIsFurtherSide, bIsFurtherSide;
      if (aScore >= bScore && zins > bScore) {
        bScore = zins;
        bPoint = 0x04;
        bIsFurtherSide = true;
      } else if (aScore < bScore && zins > aScore) {
        aScore = zins;
        aPoint = 0x04;
        aIsFurtherSide = true;
      }
      if (aScore >= bScore && xins + zins > 1) {
        aScore = xins + zins - 1;
        aPoint = 0x03;
        aIsFurtherSide = true;
      } else if (aScore < bScore && yins + zins > 1) {
        bScore = yins + zins - 1;
        bPoint = 0x03;
        bIsFurtherSide = true;
      }
      if (aIsFurtherSide === bIsFurtherSide) {
        if (aIsFurtherSide) {
          xsv_ext0 = xsb + 1;
          xsv_ext1 = xsb + 2;
          dx_ext0 = dx0 - 1 - SQUISH_CONSTANT_3D;
          dx_ext1 = dx0 - 2 - 2 * SQUISH_CONSTANT_3D;
          ysv_ext0 = ysb + 1;
          ysv_ext1 = ysb + 2;
          dy_ext0 = dy0 - 1 - SQUISH_CONSTANT_3D;
          dy_ext1 = dy0 - 2 - 2 * SQUISH_CONSTANT_3D;
          zsv_ext0 = zsb + 1;
          zsv_ext1 = zsb + 2;
          dz_ext0 = dz0 - 1 - SQUISH_CONSTANT_3D;
          dz_ext1 = dz0 - 2 - 2 * SQUISH_CONSTANT_3D;
        } else {
          xsv_ext0 = xsb;
          xsv_ext1 = xsb - 1;
          dx_ext0 = dx0 - SQUISH_CONSTANT_3D;
          dx_ext1 = dx0 + 1 - 2 * SQUISH_CONSTANT_3D;
          ysv_ext0 = ysb;
          ysv_ext1 = ysb - 1;
          dy_ext0 = dy0 - SQUISH_CONSTANT_3D;
          dy_ext1 = dy0 + 1 - 2 * SQUISH_CONSTANT_3D;
          zsv_ext0 = zsb;
          zsv_ext1 = zsb - 1;
          dz_ext0 = dz0 - SQUISH_CONSTANT_3D;
          dz_ext1 = dz0 + 1 - 2 * SQUISH_CONSTANT_3D;
        }
      } else {
        let c1, c2;
        if (aIsFurtherSide) {
          c1 = aPoint;
          c2 = bPoint;
        } else {
          c1 = bPoint;
          c2 = aPoint;
        }
        if ((c1 & 0x01) === 0) {
          xsv_ext0 = xsb;
          dx_ext0 = dx0 - SQUISH_CONSTANT_3D;
        } else {
          xsv_ext0 = xsb + 1;
          dx_ext0 = dx0 - 1 - SQUISH_CONSTANT_3D;
        }
        if ((c1 & 0x02) === 0) {
          ysv_ext0 = ysb;
          dy_ext0 = dy0 - SQUISH_CONSTANT_3D;
        } else {
          ysv_ext0 = ysb + 1;
          dy_ext0 = dy0 - 1 - SQUISH_CONSTANT_3D;
        }
        if ((c1 & 0x04) === 0) {
          zsv_ext0 = zsb;
          dz_ext0 = dz0 - SQUISH_CONSTANT_3D;
        } else {
          zsv_ext0 = zsb + 1;
          dz_ext0 = dz0 - 1 - SQUISH_CONSTANT_3D;
        }
        xsv_ext1 = xsb + 1;
        ysv_ext1 = ysb + 1;
        zsv_ext1 = zsb + 1;
        dx_ext1 = dx0 - 1 - 2 * SQUISH_CONSTANT_3D;
        dy_ext1 = dy0 - 1 - 2 * SQUISH_CONSTANT_3D;
        dz_ext1 = dz0 - 1 - 2 * SQUISH_CONSTANT_3D;
      }
      let attn0 = 2 - dx0 * dx0 - dy0 * dy0 - dz0 * dz0;
      if (attn0 > 0) {
        attn0 *= attn0;
        value += attn0 * attn0 * this._extrapolate3d(xsb, ysb, zsb, dx0, dy0, dz0);
      }
      let dx1 = dx0 - 1 - SQUISH_CONSTANT_3D;
      let dy1 = dy0 - SQUISH_CONSTANT_3D;
      let dz1 = dz0 - SQUISH_CONSTANT_3D;
      let attn1 = 2 - dx1 * dx1 - dy1 * dy1 - dz1 * dz1;
      if (attn1 > 0) {
        attn1 *= attn1;
        value += attn1 * attn1 * this._extrapolate3d(xsb + 1, ysb, zsb, dx1, dy1, dz1);
      }
      let dx2 = dx0 - SQUISH_CONSTANT_3D;
      let dy2 = dy0 - 1 - SQUISH_CONSTANT_3D;
      let dz2 = dz1;
      let attn2 = 2 - dx2 * dx2 - dy2 * dy2 - dz2 * dz2;
      if (attn2 > 0) {
        attn2 *= attn2;
        value += attn2 * attn2 * this._extrapolate3d(xsb, ysb + 1, zsb, dx2, dy2, dz2);
      }
      let dx3 = dx2;
      let dy3 = dy1;
      let dz3 = dz0 - 1 - SQUISH_CONSTANT_3D;
      let attn3 = 2 - dx3 * dx3 - dy3 * dy3 - dz3 * dz3;
      if (attn3 > 0) {
        attn3 *= attn3;
        value += attn3 * attn3 * this._extrapolate3d(xsb, ysb, zsb + 1, dx3, dy3, dz3);
      }
      let attn_ext0 = 2 - dx_ext0 * dx_ext0 - dy_ext0 * dy_ext0 - dz_ext0 * dz_ext0;
      if (attn_ext0 > 0) {
        attn_ext0 *= attn_ext0;
        value += attn_ext0 * attn_ext0 * this._extrapolate3d(xsv_ext0, ysv_ext0, zsv_ext0, dx_ext0, dy_ext0, dz_ext0);
      }
      let attn_ext1 = 2 - dx_ext1 * dx_ext1 - dy_ext1 * dy_ext1 - dz_ext1 * dz_ext1;
      if (attn_ext1 > 0) {
        attn_ext1 *= attn_ext1;
        value += attn_ext1 * attn_ext1 * this._extrapolate3d(xsv_ext1, ysv_ext1, zsv_ext1, dx_ext1, dy_ext1, dz_ext1);
      }
    }
    return value / NORM_CONSTANT_3D;
  }
}

export function fromSeed(seed) {
  const os = new OpenSimplex(seed);
  return { os, data: { perm: Array.from(os.perm), perm_grad: Array.from(os.permGradIndex3D) } };
}

export function random(time = 0, seed, speed = 1) {
  const angle = Math.PI * 2 * time;
  const z = Math.cos(angle) * speed;
  const w = Math.sin(angle) * speed;
  const s = seed ?? rngRandomInt(1, 65536); // RNG call
  const { os } = fromSeed(s);
  const value = os.noise2D(z, w);
  return (value + 1) * 0.5;
}

export function simplex(shape, { time = 0, seed, speed = 1 } = {}) {
  const [height, width, channels = 1] = shape;
  const baseSeed = seed ?? getSeed(); // RNG call if seed absent
  const angle = Math.PI * 2 * time;
  const z = Math.cos(angle) * speed;
  const data = new Float32Array(height * width * channels);
  for (let c = 0; c < channels; c++) {
    const { os } = fromSeed(baseSeed + c * 65535);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const val = os.noise3D(x, y, z);
        data[(y * width + x) * channels + c] = (val + 1) * 0.5;
      }
    }
  }
  return Tensor.fromArray(null, data, [height, width, channels]);
}

