// Grime mask generator. Produces the base FBM mask used by the grime effect
// along with an offset copy used during the refraction step. The shader mirrors
// the CPU implementation in `effects.js` by sampling OpenSimplex noise across
// multiple octaves, performing bicubic interpolation to match the resampling
// behaviour of the CPU path, and writing both the base and shifted values to
// dedicated single-channel storage textures.

struct GrimeMaskParams {
  sizeFreq : vec4<f32>;   // width, height, freqX, freqY
  timeGain : vec4<f32>;   // time, speed, gain, lacunarity
  offsets : vec4<f32>;    // octaves, offsetX, offsetY, unused
};

@group(0) @binding(0) var baseTexture : texture_storage_2d<r32float, write>;
@group(0) @binding(1) var offsetTexture : texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<uniform> params : GrimeMaskParams;
@group(0) @binding(3) var<storage, read> permTable : array<u32>;
@group(0) @binding(4) var<storage, read> permGradIndex3D : array<u32>;

const STRETCH_CONSTANT_3D : f32 = -1.0 / 6.0;
const SQUISH_CONSTANT_3D : f32 = 1.0 / 3.0;
const NORM_CONSTANT_3D : f32 = 103.0;

const GRADIENTS_3D : array<f32, 72> = array<f32, 72>(
  -11.0, 4.0, 4.0, -4.0, 11.0, 4.0, -4.0, 4.0, 11.0,
  11.0, 4.0, 4.0, 4.0, 11.0, 4.0, 4.0, 4.0, 11.0,
  -11.0, -4.0, 4.0, -4.0, -11.0, 4.0, -4.0, -4.0, 11.0,
  11.0, -4.0, 4.0, 4.0, -11.0, 4.0, 4.0, -4.0, 11.0,
  -11.0, 4.0, -4.0, -4.0, 11.0, -4.0, -4.0, 4.0, -11.0,
  11.0, 4.0, -4.0, 4.0, 11.0, -4.0, 4.0, 4.0, -11.0,
  -11.0, -4.0, -4.0, -4.0, -11.0, -4.0, -4.0, -4.0, -11.0,
  11.0, -4.0, -4.0, 4.0, -11.0, -4.0, 4.0, -4.0, -11.0
);

fn wrap_i32(value : i32, limit : i32) -> i32 {
  if (limit == 0) {
    return 0;
  }
  let modVal = value % limit;
  return select(modVal, modVal + limit, modVal < 0);
}

fn extrapolate3d(xsb : i32, ysb : i32, zsb : i32, dx : f32, dy : f32, dz : f32) -> f32 {
  let xIndex = permTable[u32(xsb & 0xff)];
  let yIndex = permTable[(xIndex + u32(ysb & 0xff)) & 0xffu];
  let gradIndex = permGradIndex3D[(yIndex + u32(zsb & 0xff)) & 0xffu];
  let g = gradIndex;
  let g1 = GRADIENTS_3D[g];
  let g2 = GRADIENTS_3D[g + 1u];
  let g3 = GRADIENTS_3D[g + 2u];
  return g1 * dx + g2 * dy + g3 * dz;
}

fn noise3d(x : f32, y : f32, z : f32) -> f32 {
  let stretchOffset = (x + y + z) * STRETCH_CONSTANT_3D;
  let xs = x + stretchOffset;
  let ys = y + stretchOffset;
  let zs = z + stretchOffset;

  var xsb = i32(floor(xs));
  var ysb = i32(floor(ys));
  var zsb = i32(floor(zs));

  let squishOffset = f32(xsb + ysb + zsb) * SQUISH_CONSTANT_3D;
  var dx0 = x - (f32(xsb) + squishOffset);
  var dy0 = y - (f32(ysb) + squishOffset);
  var dz0 = z - (f32(zsb) + squishOffset);

  let xins = xs - f32(xsb);
  let yins = ys - f32(ysb);
  let zins = zs - f32(zsb);
  let inSum = xins + yins + zins;

  var value = 0.0;

  var xsv_ext0 : i32 = 0;
  var ysv_ext0 : i32 = 0;
  var zsv_ext0 : i32 = 0;
  var xsv_ext1 : i32 = 0;
  var ysv_ext1 : i32 = 0;
  var zsv_ext1 : i32 = 0;
  var dx_ext0 = 0.0;
  var dy_ext0 = 0.0;
  var dz_ext0 = 0.0;
  var dx_ext1 = 0.0;
  var dy_ext1 = 0.0;
  var dz_ext1 = 0.0;

  if (inSum <= 1.0) {
    var aPoint : i32 = 0x01;
    var aScore = xins;
    var bPoint : i32 = 0x02;
    var bScore = yins;
    if (aScore >= bScore && zins > bScore) {
      bScore = zins;
      bPoint = 0x04;
    } else if (aScore < bScore && zins > aScore) {
      aScore = zins;
      aPoint = 0x04;
    }
    let wins = 1.0 - inSum;
    if (wins > aScore || wins > bScore) {
      let c = select(aPoint, bPoint, bScore > aScore);
      if ((c & 0x01) == 0) {
        xsv_ext0 = xsb - 1;
        xsv_ext1 = xsb;
        dx_ext0 = dx0 + 1.0;
        dx_ext1 = dx0;
      } else {
        xsv_ext0 = xsb + 1;
        xsv_ext1 = xsb + 1;
        dx_ext0 = dx0 - 1.0;
        dx_ext1 = dx0 - 1.0;
      }
      if ((c & 0x02) == 0) {
        ysv_ext0 = ysb;
        ysv_ext1 = ysb;
        dy_ext0 = dy0;
        dy_ext1 = dy0;
        if ((c & 0x01) == 0) {
          ysv_ext1 -= 1;
          dy_ext1 += 1.0;
        } else {
          ysv_ext0 -= 1;
          dy_ext0 += 1.0;
        }
      } else {
        ysv_ext0 = ysb + 1;
        ysv_ext1 = ysb + 1;
        dy_ext0 = dy0 - 1.0;
        dy_ext1 = dy0 - 1.0;
      }
      if ((c & 0x04) == 0) {
        zsv_ext0 = zsb;
        zsv_ext1 = zsb - 1;
        dz_ext0 = dz0;
        dz_ext1 = dz0 + 1.0;
      } else {
        zsv_ext0 = zsb + 1;
        zsv_ext1 = zsb + 1;
        dz_ext0 = dz0 - 1.0;
        dz_ext1 = dz0 - 1.0;
      }
    } else {
      let c = aPoint | bPoint;
      if ((c & 0x01) == 0) {
        xsv_ext0 = xsb;
        xsv_ext1 = xsb - 1;
        dx_ext0 = dx0 - 2.0 * SQUISH_CONSTANT_3D;
        dx_ext1 = dx0 + 1.0 - SQUISH_CONSTANT_3D;
      } else {
        xsv_ext0 = xsb + 1;
        xsv_ext1 = xsb + 1;
        dx_ext0 = dx0 - 1.0 - 2.0 * SQUISH_CONSTANT_3D;
        dx_ext1 = dx0 - 1.0 - SQUISH_CONSTANT_3D;
      }
      if ((c & 0x02) == 0) {
        ysv_ext0 = ysb;
        ysv_ext1 = ysb - 1;
        dy_ext0 = dy0 - 2.0 * SQUISH_CONSTANT_3D;
        dy_ext1 = dy0 + 1.0 - SQUISH_CONSTANT_3D;
      } else {
        ysv_ext0 = ysb + 1;
        ysv_ext1 = ysb + 1;
        dy_ext0 = dy0 - 1.0 - 2.0 * SQUISH_CONSTANT_3D;
        dy_ext1 = dy0 - 1.0 - SQUISH_CONSTANT_3D;
      }
      if ((c & 0x04) == 0) {
        zsv_ext0 = zsb;
        zsv_ext1 = zsb - 1;
        dz_ext0 = dz0 - 2.0 * SQUISH_CONSTANT_3D;
        dz_ext1 = dz0 + 1.0 - SQUISH_CONSTANT_3D;
      } else {
        zsv_ext0 = zsb + 1;
        zsv_ext1 = zsb + 1;
        dz_ext0 = dz0 - 1.0 - 2.0 * SQUISH_CONSTANT_3D;
        dz_ext1 = dz0 - 1.0 - SQUISH_CONSTANT_3D;
      }
    }

    var attn0 = 2.0 - dx0 * dx0 - dy0 * dy0 - dz0 * dz0;
    if (attn0 > 0.0) {
      attn0 *= attn0;
      value += attn0 * attn0 * extrapolate3d(xsb, ysb, zsb, dx0, dy0, dz0);
    }
    var dx1 = dx0 - 1.0 - SQUISH_CONSTANT_3D;
    var dy1 = dy0 - SQUISH_CONSTANT_3D;
    var dz1 = dz0 - SQUISH_CONSTANT_3D;
    var attn1 = 2.0 - dx1 * dx1 - dy1 * dy1 - dz1 * dz1;
    if (attn1 > 0.0) {
      attn1 *= attn1;
      value += attn1 * attn1 * extrapolate3d(xsb + 1, ysb, zsb, dx1, dy1, dz1);
    }
    var dx2 = dx0 - SQUISH_CONSTANT_3D;
    var dy2 = dy0 - 1.0 - SQUISH_CONSTANT_3D;
    var dz2 = dz1;
    var attn2 = 2.0 - dx2 * dx2 - dy2 * dy2 - dz2 * dz2;
    if (attn2 > 0.0) {
      attn2 *= attn2;
      value += attn2 * attn2 * extrapolate3d(xsb, ysb + 1, zsb, dx2, dy2, dz2);
    }
    var dx3 = dx2;
    var dy3 = dy1;
    var dz3 = dz0 - 1.0 - SQUISH_CONSTANT_3D;
    var attn3 = 2.0 - dx3 * dx3 - dy3 * dy3 - dz3 * dz3;
    if (attn3 > 0.0) {
      attn3 *= attn3;
      value += attn3 * attn3 * extrapolate3d(xsb, ysb, zsb + 1, dx3, dy3, dz3);
    }

    var dx4 = dx0 - 1.0 - 2.0 * SQUISH_CONSTANT_3D;
    var dy4 = dy0 - 1.0 - 2.0 * SQUISH_CONSTANT_3D;
    var dz4 = dz0 - 2.0 * SQUISH_CONSTANT_3D;
    var attn4 = 2.0 - dx4 * dx4 - dy4 * dy4 - dz4 * dz4;
    if (attn4 > 0.0) {
      attn4 *= attn4;
      value += attn4 * attn4 * extrapolate3d(xsb + 1, ysb + 1, zsb, dx4, dy4, dz4);
    }
    var dx5 = dx4;
    var dy5 = dy0 - 2.0 * SQUISH_CONSTANT_3D;
    var dz5 = dz0 - 1.0 - 2.0 * SQUISH_CONSTANT_3D;
    var attn5 = 2.0 - dx5 * dx5 - dy5 * dy5 - dz5 * dz5;
    if (attn5 > 0.0) {
      attn5 *= attn5;
      value += attn5 * attn5 * extrapolate3d(xsb + 1, ysb, zsb + 1, dx5, dy5, dz5);
    }
    var dx6 = dx0 - 2.0 * SQUISH_CONSTANT_3D;
    var dy6 = dy4;
    var dz6 = dz5;
    var attn6 = 2.0 - dx6 * dx6 - dy6 * dy6 - dz6 * dz6;
    if (attn6 > 0.0) {
      attn6 *= attn6;
      value += attn6 * attn6 * extrapolate3d(xsb, ysb + 1, zsb + 1, dx6, dy6, dz6);
    }
  } else if (inSum >= 2.0) {
    var aPoint : i32 = 0x06;
    var aScore = xins;
    var bPoint : i32 = 0x05;
    var bScore = yins;
    if (aScore <= bScore && zins < bScore) {
      bScore = zins;
      bPoint = 0x03;
    } else if (aScore > bScore && zins < aScore) {
      aScore = zins;
      aPoint = 0x03;
    }
    let wins = 3.0 - inSum;
    if (wins < aScore || wins < bScore) {
      let c = select(aPoint, bPoint, bScore < aScore);
      if ((c & 0x01) != 0) {
        xsv_ext0 = xsb + 1;
        xsv_ext1 = xsb + 2;
        dx_ext0 = dx0 - 1.0 - SQUISH_CONSTANT_3D;
        dx_ext1 = dx0 - 2.0 - 2.0 * SQUISH_CONSTANT_3D;
      } else {
        xsv_ext0 = xsb;
        xsv_ext1 = xsb;
        dx_ext0 = dx0 - SQUISH_CONSTANT_3D;
        dx_ext1 = dx0 - 2.0 * SQUISH_CONSTANT_3D;
      }
      if ((c & 0x02) != 0) {
        ysv_ext0 = ysb + 1;
        ysv_ext1 = ysb + 2;
        dy_ext0 = dy0 - 1.0 - SQUISH_CONSTANT_3D;
        dy_ext1 = dy0 - 2.0 - 2.0 * SQUISH_CONSTANT_3D;
      } else {
        ysv_ext0 = ysb;
        ysv_ext1 = ysb;
        dy_ext0 = dy0 - SQUISH_CONSTANT_3D;
        dy_ext1 = dy0 - 2.0 * SQUISH_CONSTANT_3D;
      }
      if ((c & 0x04) != 0) {
        zsv_ext0 = zsb + 1;
        zsv_ext1 = zsb + 2;
        dz_ext0 = dz0 - 1.0 - SQUISH_CONSTANT_3D;
        dz_ext1 = dz0 - 2.0 - 2.0 * SQUISH_CONSTANT_3D;
      } else {
        zsv_ext0 = zsb;
        zsv_ext1 = zsb;
        dz_ext0 = dz0 - SQUISH_CONSTANT_3D;
        dz_ext1 = dz0 - 2.0 * SQUISH_CONSTANT_3D;
      }
    } else {
      let c = aPoint & bPoint;
      if ((c & 0x01) != 0) {
        xsv_ext0 = xsb + 1;
        xsv_ext1 = xsb + 2;
        dx_ext0 = dx0 - 1.0 - SQUISH_CONSTANT_3D;
        dx_ext1 = dx0 - 2.0 - 2.0 * SQUISH_CONSTANT_3D;
      } else {
        xsv_ext0 = xsb;
        xsv_ext1 = xsb;
        dx_ext0 = dx0 - SQUISH_CONSTANT_3D;
        dx_ext1 = dx0 - 2.0 * SQUISH_CONSTANT_3D;
      }
      if ((c & 0x02) != 0) {
        ysv_ext0 = ysb + 1;
        ysv_ext1 = ysb + 2;
        dy_ext0 = dy0 - 1.0 - SQUISH_CONSTANT_3D;
        dy_ext1 = dy0 - 2.0 - 2.0 * SQUISH_CONSTANT_3D;
      } else {
        ysv_ext0 = ysb;
        ysv_ext1 = ysb;
        dy_ext0 = dy0 - SQUISH_CONSTANT_3D;
        dy_ext1 = dy0 - 2.0 * SQUISH_CONSTANT_3D;
      }
      if ((c & 0x04) != 0) {
        zsv_ext0 = zsb + 1;
        zsv_ext1 = zsb + 2;
        dz_ext0 = dz0 - 1.0 - SQUISH_CONSTANT_3D;
        dz_ext1 = dz0 - 2.0 - 2.0 * SQUISH_CONSTANT_3D;
      } else {
        zsv_ext0 = zsb;
        zsv_ext1 = zsb;
        dz_ext0 = dz0 - SQUISH_CONSTANT_3D;
        dz_ext1 = dz0 - 2.0 * SQUISH_CONSTANT_3D;
      }
    }

    var dx3 = dx0 - 1.0 - 2.0 * SQUISH_CONSTANT_3D;
    var dy3 = dy0 - 1.0 - 2.0 * SQUISH_CONSTANT_3D;
    var dz3 = dz0 - 2.0 * SQUISH_CONSTANT_3D;
    var attn3 = 2.0 - dx3 * dx3 - dy3 * dy3 - dz3 * dz3;
    if (attn3 > 0.0) {
      attn3 *= attn3;
      value += attn3 * attn3 * extrapolate3d(xsb + 1, ysb + 1, zsb, dx3, dy3, dz3);
    }
    var dx2 = dx3;
    var dy2 = dy0 - 2.0 * SQUISH_CONSTANT_3D;
    var dz2 = dz0 - 1.0 - 2.0 * SQUISH_CONSTANT_3D;
    var attn2 = 2.0 - dx2 * dx2 - dy2 * dy2 - dz2 * dz2;
    if (attn2 > 0.0) {
      attn2 *= attn2;
      value += attn2 * attn2 * extrapolate3d(xsb + 1, ysb, zsb + 1, dx2, dy2, dz2);
    }
    var dx1 = dx0 - 2.0 * SQUISH_CONSTANT_3D;
    var dy1 = dy3;
    var dz1 = dz2;
    var attn1 = 2.0 - dx1 * dx1 - dy1 * dy1 - dz1 * dz1;
    if (attn1 > 0.0) {
      attn1 *= attn1;
      value += attn1 * attn1 * extrapolate3d(xsb, ysb + 1, zsb + 1, dx1, dy1, dz1);
    }
    dx0 = dx0 - 1.0 - 3.0 * SQUISH_CONSTANT_3D;
    dy0 = dy0 - 1.0 - 3.0 * SQUISH_CONSTANT_3D;
    dz0 = dz0 - 1.0 - 3.0 * SQUISH_CONSTANT_3D;
    var attn0 = 2.0 - dx0 * dx0 - dy0 * dy0 - dz0 * dz0;
    if (attn0 > 0.0) {
      attn0 *= attn0;
      value += attn0 * attn0 * extrapolate3d(xsb + 1, ysb + 1, zsb + 1, dx0, dy0, dz0);
    }
  } else {
    var aPoint : i32 = 0x01;
    var aScore = xins;
    var bPoint : i32 = 0x02;
    var bScore = yins;
    var aIsFurtherSide = false;
    var bIsFurtherSide = false;
    if (aScore >= bScore && zins > bScore) {
      bScore = zins;
      bPoint = 0x04;
      bIsFurtherSide = true;
    } else if (aScore < bScore && zins > aScore) {
      aScore = zins;
      aPoint = 0x04;
      aIsFurtherSide = true;
    }
    if (aScore >= bScore && xins + zins > 1.0) {
      aScore = xins + zins - 1.0;
      aPoint = 0x03;
      aIsFurtherSide = true;
    } else if (aScore < bScore && yins + zins > 1.0) {
      bScore = yins + zins - 1.0;
      bPoint = 0x03;
      bIsFurtherSide = true;
    }
    if (aIsFurtherSide == bIsFurtherSide) {
      if (aIsFurtherSide) {
        xsv_ext0 = xsb + 1;
        xsv_ext1 = xsb + 2;
        dx_ext0 = dx0 - 1.0 - SQUISH_CONSTANT_3D;
        dx_ext1 = dx0 - 2.0 - 2.0 * SQUISH_CONSTANT_3D;
        ysv_ext0 = ysb + 1;
        ysv_ext1 = ysb + 2;
        dy_ext0 = dy0 - 1.0 - SQUISH_CONSTANT_3D;
        dy_ext1 = dy0 - 2.0 - 2.0 * SQUISH_CONSTANT_3D;
        zsv_ext0 = zsb + 1;
        zsv_ext1 = zsb + 2;
        dz_ext0 = dz0 - 1.0 - SQUISH_CONSTANT_3D;
        dz_ext1 = dz0 - 2.0 - 2.0 * SQUISH_CONSTANT_3D;
      } else {
        xsv_ext0 = xsb;
        xsv_ext1 = xsb - 1;
        dx_ext0 = dx0 - SQUISH_CONSTANT_3D;
        dx_ext1 = dx0 + 1.0 - 2.0 * SQUISH_CONSTANT_3D;
        ysv_ext0 = ysb;
        ysv_ext1 = ysb - 1;
        dy_ext0 = dy0 - SQUISH_CONSTANT_3D;
        dy_ext1 = dy0 + 1.0 - 2.0 * SQUISH_CONSTANT_3D;
        zsv_ext0 = zsb;
        zsv_ext1 = zsb - 1;
        dz_ext0 = dz0 - SQUISH_CONSTANT_3D;
        dz_ext1 = dz0 + 1.0 - 2.0 * SQUISH_CONSTANT_3D;
      }
    } else {
      var c1 : i32;
      var c2 : i32;
      if (aIsFurtherSide) {
        c1 = aPoint;
        c2 = bPoint;
      } else {
        c1 = bPoint;
        c2 = aPoint;
      }
      if ((c1 & 0x01) == 0) {
        xsv_ext0 = xsb;
        dx_ext0 = dx0 - SQUISH_CONSTANT_3D;
      } else {
        xsv_ext0 = xsb + 1;
        dx_ext0 = dx0 - 1.0 - SQUISH_CONSTANT_3D;
      }
      if ((c1 & 0x02) == 0) {
        ysv_ext0 = ysb;
        dy_ext0 = dy0 - SQUISH_CONSTANT_3D;
      } else {
        ysv_ext0 = ysb + 1;
        dy_ext0 = dy0 - 1.0 - SQUISH_CONSTANT_3D;
      }
      if ((c1 & 0x04) == 0) {
        zsv_ext0 = zsb;
        dz_ext0 = dz0 - SQUISH_CONSTANT_3D;
      } else {
        zsv_ext0 = zsb + 1;
        dz_ext0 = dz0 - 1.0 - SQUISH_CONSTANT_3D;
      }
      xsv_ext1 = xsb + 1;
      ysv_ext1 = ysb + 1;
      zsv_ext1 = zsb + 1;
      dx_ext1 = dx0 - 1.0 - 2.0 * SQUISH_CONSTANT_3D;
      dy_ext1 = dy0 - 1.0 - 2.0 * SQUISH_CONSTANT_3D;
      dz_ext1 = dz0 - 1.0 - 2.0 * SQUISH_CONSTANT_3D;
    }

    var attn0 = 2.0 - dx0 * dx0 - dy0 * dy0 - dz0 * dz0;
    if (attn0 > 0.0) {
      attn0 *= attn0;
      value += attn0 * attn0 * extrapolate3d(xsb, ysb, zsb, dx0, dy0, dz0);
    }
    var dx1 = dx0 - 1.0 - SQUISH_CONSTANT_3D;
    var dy1 = dy0 - SQUISH_CONSTANT_3D;
    var dz1 = dz0 - SQUISH_CONSTANT_3D;
    var attn1 = 2.0 - dx1 * dx1 - dy1 * dy1 - dz1 * dz1;
    if (attn1 > 0.0) {
      attn1 *= attn1;
      value += attn1 * attn1 * extrapolate3d(xsb + 1, ysb, zsb, dx1, dy1, dz1);
    }
    var dx2 = dx0 - SQUISH_CONSTANT_3D;
    var dy2 = dy0 - 1.0 - SQUISH_CONSTANT_3D;
    var dz2 = dz1;
    var attn2 = 2.0 - dx2 * dx2 - dy2 * dy2 - dz2 * dz2;
    if (attn2 > 0.0) {
      attn2 *= attn2;
      value += attn2 * attn2 * extrapolate3d(xsb, ysb + 1, zsb, dx2, dy2, dz2);
    }
    var dx3 = dx2;
    var dy3 = dy1;
    var dz3 = dz0 - 1.0 - SQUISH_CONSTANT_3D;
    var attn3 = 2.0 - dx3 * dx3 - dy3 * dy3 - dz3 * dz3;
    if (attn3 > 0.0) {
      attn3 *= attn3;
      value += attn3 * attn3 * extrapolate3d(xsb, ysb, zsb + 1, dx3, dy3, dz3);
    }
    var attn_ext0 = 2.0 - dx_ext0 * dx_ext0 - dy_ext0 * dy_ext0 - dz_ext0 * dz_ext0;
    if (attn_ext0 > 0.0) {
      attn_ext0 *= attn_ext0;
      value += attn_ext0 * attn_ext0 * extrapolate3d(xsv_ext0, ysv_ext0, zsv_ext0, dx_ext0, dy_ext0, dz_ext0);
    }
    var attn_ext1 = 2.0 - dx_ext1 * dx_ext1 - dy_ext1 * dy_ext1 - dz_ext1 * dz_ext1;
    if (attn_ext1 > 0.0) {
      attn_ext1 *= attn_ext1;
      value += attn_ext1 * attn_ext1 * extrapolate3d(xsv_ext1, ysv_ext1, zsv_ext1, dx_ext1, dy_ext1, dz_ext1);
    }
  }

  return value / NORM_CONSTANT_3D;
}

fn cubicInterpolate(a : f32, b : f32, c : f32, d : f32, t : f32) -> f32 {
  let t2 = t * t;
  let a0 = d - c - a + b;
  let a1 = a - b - a0;
  let a2 = c - a;
  let a3 = b;
  return a0 * t * t2 + a1 * t2 + a2 * t + a3;
}

fn sample_noise(ix : i32, iy : i32, z : f32) -> f32 {
  let val = noise3d(f32(ix), f32(iy), z);
  return (val + 1.0) * 0.5;
}

fn fbm_value(x : u32, y : u32, z : f32) -> f32 {
  let width = max(params.sizeFreq.x, 1.0);
  let height = max(params.sizeFreq.y, 1.0);
  let baseFreqX = max(params.sizeFreq.z, 1.0);
  let baseFreqY = max(params.sizeFreq.w, 1.0);
  let gain = params.timeGain.z;
  let lacunarity = params.timeGain.w;
  let octaves = u32(max(params.offsets.x, 1.0));

  let widthF = width;
  let heightF = height;

  var accum = 0.0;
  for (var octave : u32 = 0u; octave < octaves; octave = octave + 1u) {
    let powFactor = pow(lacunarity, f32(octave));
    var freqX = floor(baseFreqX * powFactor + 0.00001);
    var freqY = floor(baseFreqY * powFactor + 0.00001);
    if (freqX < 1.0) {
      freqX = 1.0;
    }
    if (freqY < 1.0) {
      freqY = 1.0;
    }
    if (freqY > heightF && freqX > widthF) {
      break;
    }

    let freqXi = max(i32(freqX), 1);
    let freqYi = max(i32(freqY), 1);

    let scaleX = f32(freqXi) / widthF;
    let scaleY = f32(freqYi) / heightF;

    let gx = f32(x) * scaleX;
    let gy = f32(y) * scaleY;
    let baseX = i32(floor(gx));
    let baseY = i32(floor(gy));
    let fx = gx - f32(baseX);
    let fy = gy - f32(baseY);

    var rows : array<f32, 4>;
    for (var m : i32 = -1; m <= 2; m = m + 1) {
      var rowVals : array<f32, 4>;
      for (var n : i32 = -1; n <= 2; n = n + 1) {
        let sampleX = wrap_i32(baseX + n, freqXi);
        let sampleY = wrap_i32(baseY + m, freqYi);
        rowVals[u32(n + 1)] = sample_noise(sampleX, sampleY, z);
      }
      rows[u32(m + 1)] = cubicInterpolate(rowVals[0], rowVals[1], rowVals[2], rowVals[3], fx);
    }
    let sample = cubicInterpolate(rows[0], rows[1], rows[2], rows[3], fy);
    let weight = pow(gain, f32(octave + 1u));
    accum = accum + sample * weight;
  }
  return clamp(accum, 0.0, 1.0);
}

fn grime_value(x : u32, y : u32) -> vec2<f32> {
  let angle = params.timeGain.x * 6.283185307179586;
  let z = cos(angle) * params.timeGain.y;
  let baseVal = fbm_value(x, y, z);
  let offsetX = u32(max(params.offsets.y, 0.0));
  let offsetY = u32(max(params.offsets.z, 0.0));
  let width = u32(max(params.sizeFreq.x, 1.0));
  let height = u32(max(params.sizeFreq.y, 1.0));
  let shiftedX = (x + offsetX) % width;
  let shiftedY = (y + offsetY) % height;
  let offsetVal = fbm_value(shiftedX, shiftedY, z);
  return vec2<f32>(baseVal, offsetVal);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let width = u32(max(params.sizeFreq.x, 1.0) + 0.5);
  let height = u32(max(params.sizeFreq.y, 1.0) + 0.5);
  if (gid.x >= width || gid.y >= height) {
    return;
  }
  let values = grime_value(gid.x, gid.y);
  let coords = vec2<i32>(i32(gid.x), i32(gid.y));
  textureStore(baseTexture, coords, vec4<f32>(values.x, 0.0, 0.0, 0.0));
  textureStore(offsetTexture, coords, vec4<f32>(values.y, 0.0, 0.0, 0.0));
}
