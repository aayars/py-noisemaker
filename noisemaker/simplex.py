from __future__ import annotations

import math

import numpy as np
from typing import Optional

import tensorflow as tf

import noisemaker.rng as rng


_seed = None


def set_seed(s: int) -> None:
    """
    Set the global simplex noise seed.

    Args:
        s: Random seed value
    """
    global _seed
    _seed = s & 0xFFFFFFFF


def get_seed() -> int:
    """
    Get the current simplex noise seed, generating one if needed.

    Returns:
        Current seed value
    """
    global _seed
    if _seed is None:
        _seed = rng.random_int(1, 65536)
    else:
        _seed = (_seed + 1) & 0xFFFFFFFF
    return _seed


STRETCH_CONSTANT_2D = -0.211324865405187
SQUISH_CONSTANT_2D = 0.366025403784439
STRETCH_CONSTANT_3D = -1.0 / 6.0
SQUISH_CONSTANT_3D = 1.0 / 3.0
NORM_CONSTANT_2D = 47
NORM_CONSTANT_3D = 103

GRADIENTS_2D = [
    5, 2, 2, 5,
    -5, 2, -2, 5,
    5, -2, 2, -5,
    -5, -2, -2, -5,
]

GRADIENTS_3D = [
    -11, 4, 4, -4, 11, 4, -4, 4, 11,
    11, 4, 4, 4, 11, 4, 4, 4, 11,
    -11, -4, 4, -4, -11, 4, -4, -4, 11,
    11, -4, 4, 4, -11, 4, 4, -4, 11,
    -11, 4, -4, -4, 11, -4, -4, 4, -11,
    11, 4, -4, 4, 11, -4, 4, 4, -11,
    -11, -4, -4, -4, -11, -4, -4, -4, -11,
    11, -4, -4, 4, -11, -4, 4, -4, -11,
]


def _build_permutations(seed: int) -> tuple[list[int], list[int]]:
    """
    Build permutation tables for simplex noise.

    Args:
        seed: Random seed for generating permutations

    Returns:
        Tuple of (permutation_table, gradient_index_table)
    """
    perm = [0] * 256
    perm_grad_index_3D = [0] * 256
    source = list(range(256))
    r = rng.Random(seed)
    grad_len = len(GRADIENTS_3D) // 3
    for i in range(255, -1, -1):
        idx = r.random_int(0, i)
        perm[i] = source[idx]
        perm_grad_index_3D[i] = (perm[i] % grad_len) * 3
        source[idx] = source[i]
    return perm, perm_grad_index_3D


class OpenSimplex:
    def __init__(self, seed: int = 0) -> None:
        """
        Initialize OpenSimplex noise generator.

        Args:
            seed: Random seed value, default 0
        """
        perm, perm_grad_index_3D = _build_permutations(seed)
        self.perm = perm
        self.perm_grad_index_3D = perm_grad_index_3D

    def _extrapolate2d(self, xsb: int, ysb: int, dx: float, dy: float) -> float:
        """
        Extrapolate 2D simplex noise gradient contribution.

        Args:
            xsb: X simplex base coordinate
            ysb: Y simplex base coordinate
            dx: X distance from base
            dy: Y distance from base

        Returns:
            Gradient contribution value
        """
        perm = self.perm
        index = perm[(perm[xsb & 0xff] + ysb) & 0xff] & 0x0e
        g1 = GRADIENTS_2D[index]
        g2 = GRADIENTS_2D[index + 1]
        return g1 * dx + g2 * dy

    def _extrapolate3d(self, xsb: int, ysb: int, zsb: int, dx: float, dy: float, dz: float) -> float:
        """
        Extrapolate 3D simplex noise gradient contribution.

        Args:
            xsb: X simplex base coordinate
            ysb: Y simplex base coordinate
            zsb: Z simplex base coordinate
            dx: X distance from base
            dy: Y distance from base
            dz: Z distance from base

        Returns:
            Gradient contribution value
        """
        perm = self.perm
        index = self.perm_grad_index_3D[(perm[(perm[xsb & 0xff] + ysb) & 0xff] + zsb) & 0xff]
        g1 = GRADIENTS_3D[index]
        g2 = GRADIENTS_3D[index + 1]
        g3 = GRADIENTS_3D[index + 2]
        return g1 * dx + g2 * dy + g3 * dz

    def noise2d(self, x: float, y: float) -> float:
        """
        Generate 2D OpenSimplex noise value.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            Noise value in range approximately [-1, 1]
        """
        stretch_offset = (x + y) * STRETCH_CONSTANT_2D
        xs = x + stretch_offset
        ys = y + stretch_offset
        xsb = math.floor(xs)
        ysb = math.floor(ys)
        squish_offset = (xsb + ysb) * SQUISH_CONSTANT_2D
        xb = xsb + squish_offset
        yb = ysb + squish_offset
        xins = xs - xsb
        yins = ys - ysb
        in_sum = xins + yins
        dx0 = x - xb
        dy0 = y - yb
        value = 0.0

        dx1 = dx0 - 1 - SQUISH_CONSTANT_2D
        dy1 = dy0 - SQUISH_CONSTANT_2D
        attn1 = 2 - dx1 * dx1 - dy1 * dy1
        if attn1 > 0:
            attn1 *= attn1
            value += attn1 * attn1 * self._extrapolate2d(xsb + 1, ysb, dx1, dy1)

        dx2 = dx0 - SQUISH_CONSTANT_2D
        dy2 = dy0 - 1 - SQUISH_CONSTANT_2D
        attn2 = 2 - dx2 * dx2 - dy2 * dy2
        if attn2 > 0:
            attn2 *= attn2
            value += attn2 * attn2 * self._extrapolate2d(xsb, ysb + 1, dx2, dy2)

        if in_sum <= 1:
            zins = 1 - in_sum
            if zins > xins or zins > yins:
                if xins > yins:
                    xsv_ext = xsb + 1
                    ysv_ext = ysb - 1
                    dx_ext = dx0 - 1
                    dy_ext = dy0 + 1
                else:
                    xsv_ext = xsb - 1
                    ysv_ext = ysb + 1
                    dx_ext = dx0 + 1
                    dy_ext = dy0 - 1
            else:
                xsv_ext = xsb + 1
                ysv_ext = ysb + 1
                dx_ext = dx0 - 1 - 2 * SQUISH_CONSTANT_2D
                dy_ext = dy0 - 1 - 2 * SQUISH_CONSTANT_2D
        else:
            zins = 2 - in_sum
            if zins < xins or zins < yins:
                if xins > yins:
                    xsv_ext = xsb + 2
                    ysv_ext = ysb
                    dx_ext = dx0 - 2 - 2 * SQUISH_CONSTANT_2D
                    dy_ext = dy0 - 2 * SQUISH_CONSTANT_2D
                else:
                    xsv_ext = xsb
                    ysv_ext = ysb + 2
                    dx_ext = dx0 - 2 * SQUISH_CONSTANT_2D
                    dy_ext = dy0 - 2 - 2 * SQUISH_CONSTANT_2D
            else:
                xsv_ext = xsb
                ysv_ext = ysb
                dx_ext = dx0
                dy_ext = dy0
            xsb += 1
            ysb += 1
            dx0 = dx0 - 1 - 2 * SQUISH_CONSTANT_2D
            dy0 = dy0 - 1 - 2 * SQUISH_CONSTANT_2D

        attn0 = 2 - dx0 * dx0 - dy0 * dy0
        if attn0 > 0:
            attn0 *= attn0
            value += attn0 * attn0 * self._extrapolate2d(xsb, ysb, dx0, dy0)

        attn_ext = 2 - dx_ext * dx_ext - dy_ext * dy_ext
        if attn_ext > 0:
            attn_ext *= attn_ext
            value += attn_ext * attn_ext * self._extrapolate2d(xsv_ext, ysv_ext, dx_ext, dy_ext)

        return value / NORM_CONSTANT_2D

    def noise3d(self, x: float, y: float, z: float) -> float:
        """
        Generate 3D OpenSimplex noise value.

        Args:
            x: X coordinate
            y: Y coordinate
            z: Z coordinate

        Returns:
            Noise value in range approximately [-1, 1]
        """
        stretch_offset = (x + y + z) * STRETCH_CONSTANT_3D
        xs = x + stretch_offset
        ys = y + stretch_offset
        zs = z + stretch_offset
        xsb = math.floor(xs)
        ysb = math.floor(ys)
        zsb = math.floor(zs)
        squish_offset = (xsb + ysb + zsb) * SQUISH_CONSTANT_3D
        dx0 = x - (xsb + squish_offset)
        dy0 = y - (ysb + squish_offset)
        dz0 = z - (zsb + squish_offset)
        xins = xs - xsb
        yins = ys - ysb
        zins = zs - zsb
        in_sum = xins + yins + zins
        value = 0.0

        if in_sum <= 1:
            a_point = 0x01
            a_score = xins
            b_point = 0x02
            b_score = yins
            if a_score >= b_score and zins > b_score:
                b_score = zins
                b_point = 0x04
            elif a_score < b_score and zins > a_score:
                a_score = zins
                a_point = 0x04
            wins = 1 - in_sum
            if wins > a_score or wins > b_score:
                c = b_point if b_score > a_score else a_point
                if c & 0x01 == 0:
                    xsv_ext0 = xsb - 1
                    xsv_ext1 = xsb
                    dx_ext0 = dx0 + 1
                    dx_ext1 = dx0
                else:
                    xsv_ext0 = xsv_ext1 = xsb + 1
                    dx_ext0 = dx_ext1 = dx0 - 1
                if c & 0x02 == 0:
                    ysv_ext0 = ysv_ext1 = ysb
                    dy_ext0 = dy_ext1 = dy0
                    if c & 0x01 == 0:
                        ysv_ext1 -= 1
                        dy_ext1 += 1
                    else:
                        ysv_ext0 -= 1
                        dy_ext0 += 1
                else:
                    ysv_ext0 = ysv_ext1 = ysb + 1
                    dy_ext0 = dy_ext1 = dy0 - 1
                if c & 0x04 == 0:
                    zsv_ext0 = zsb
                    zsv_ext1 = zsb - 1
                    dz_ext0 = dz0
                    dz_ext1 = dz0 + 1
                else:
                    zsv_ext0 = zsv_ext1 = zsb + 1
                    dz_ext0 = dz_ext1 = dz0 - 1
            else:
                c = a_point | b_point
                if c & 0x01 == 0:
                    xsv_ext0 = xsb
                    xsv_ext1 = xsb - 1
                    dx_ext0 = dx0 - 2 * SQUISH_CONSTANT_3D
                    dx_ext1 = dx0 + 1 - SQUISH_CONSTANT_3D
                else:
                    xsv_ext0 = xsv_ext1 = xsb + 1
                    dx_ext0 = dx0 - 1 - 2 * SQUISH_CONSTANT_3D
                    dx_ext1 = dx0 - 1 - SQUISH_CONSTANT_3D
                if c & 0x02 == 0:
                    ysv_ext0 = ysb
                    ysv_ext1 = ysb - 1
                    dy_ext0 = dy0 - 2 * SQUISH_CONSTANT_3D
                    dy_ext1 = dy0 + 1 - SQUISH_CONSTANT_3D
                else:
                    ysv_ext0 = ysv_ext1 = ysb + 1
                    dy_ext0 = dy0 - 1 - 2 * SQUISH_CONSTANT_3D
                    dy_ext1 = dy0 - 1 - SQUISH_CONSTANT_3D
                if c & 0x04 == 0:
                    zsv_ext0 = zsb
                    zsv_ext1 = zsb - 1
                    dz_ext0 = dz0 - 2 * SQUISH_CONSTANT_3D
                    dz_ext1 = dz0 + 1 - SQUISH_CONSTANT_3D
                else:
                    zsv_ext0 = zsv_ext1 = zsb + 1
                    dz_ext0 = dz0 - 1 - 2 * SQUISH_CONSTANT_3D
                    dz_ext1 = dz0 - 1 - SQUISH_CONSTANT_3D

            attn0 = 2 - dx0 * dx0 - dy0 * dy0 - dz0 * dz0
            if attn0 > 0:
                attn0 *= attn0
                value += attn0 * attn0 * self._extrapolate3d(xsb, ysb, zsb, dx0, dy0, dz0)

            dx1 = dx0 - 1 - SQUISH_CONSTANT_3D
            dy1 = dy0 - SQUISH_CONSTANT_3D
            dz1 = dz0 - SQUISH_CONSTANT_3D
            attn1 = 2 - dx1 * dx1 - dy1 * dy1 - dz1 * dz1
            if attn1 > 0:
                attn1 *= attn1
                value += attn1 * attn1 * self._extrapolate3d(xsb + 1, ysb, zsb, dx1, dy1, dz1)

            dx2 = dx0 - SQUISH_CONSTANT_3D
            dy2 = dy0 - 1 - SQUISH_CONSTANT_3D
            dz2 = dz1
            attn2 = 2 - dx2 * dx2 - dy2 * dy2 - dz2 * dz2
            if attn2 > 0:
                attn2 *= attn2
                value += attn2 * attn2 * self._extrapolate3d(xsb, ysb + 1, zsb, dx2, dy2, dz2)

            dx3 = dx2
            dy3 = dy1
            dz3 = dz0 - 1 - SQUISH_CONSTANT_3D
            attn3 = 2 - dx3 * dx3 - dy3 * dy3 - dz3 * dz3
            if attn3 > 0:
                attn3 *= attn3
                value += attn3 * attn3 * self._extrapolate3d(xsb, ysb, zsb + 1, dx3, dy3, dz3)

            dx4 = dx0 - 1 - 2 * SQUISH_CONSTANT_3D
            dy4 = dy0 - 1 - 2 * SQUISH_CONSTANT_3D
            dz4 = dz0 - 2 * SQUISH_CONSTANT_3D
            attn4 = 2 - dx4 * dx4 - dy4 * dy4 - dz4 * dz4
            if attn4 > 0:
                attn4 *= attn4
                value += attn4 * attn4 * self._extrapolate3d(xsb + 1, ysb + 1, zsb, dx4, dy4, dz4)

            dx5 = dx4
            dy5 = dy0 - 2 * SQUISH_CONSTANT_3D
            dz5 = dz0 - 1 - 2 * SQUISH_CONSTANT_3D
            attn5 = 2 - dx5 * dx5 - dy5 * dy5 - dz5 * dz5
            if attn5 > 0:
                attn5 *= attn5
                value += attn5 * attn5 * self._extrapolate3d(xsb + 1, ysb, zsb + 1, dx5, dy5, dz5)

            dx6 = dx0 - 2 * SQUISH_CONSTANT_3D
            dy6 = dy4
            dz6 = dz5
            attn6 = 2 - dx6 * dx6 - dy6 * dy6 - dz6 * dz6
            if attn6 > 0:
                attn6 *= attn6
                value += attn6 * attn6 * self._extrapolate3d(xsb, ysb + 1, zsb + 1, dx6, dy6, dz6)
        elif in_sum >= 2:
            a_point = 0x06
            a_score = xins
            b_point = 0x05
            b_score = yins
            if a_score <= b_score and zins < b_score:
                b_score = zins
                b_point = 0x03
            elif a_score > b_score and zins < a_score:
                a_score = zins
                a_point = 0x03
            wins = 3 - in_sum
            if wins < a_score or wins < b_score:
                c = b_point if b_score < a_score else a_point
                if c & 0x01:
                    xsv_ext0 = xsb + 1
                    xsv_ext1 = xsb + 2
                    dx_ext0 = dx0 - 1 - SQUISH_CONSTANT_3D
                    dx_ext1 = dx0 - 2 - 2 * SQUISH_CONSTANT_3D
                else:
                    xsv_ext0 = xsv_ext1 = xsb
                    dx_ext0 = dx0 - SQUISH_CONSTANT_3D
                    dx_ext1 = dx0 - 2 * SQUISH_CONSTANT_3D
                if c & 0x02:
                    ysv_ext0 = ysb + 1
                    ysv_ext1 = ysb + 2
                    dy_ext0 = dy0 - 1 - SQUISH_CONSTANT_3D
                    dy_ext1 = dy0 - 2 - 2 * SQUISH_CONSTANT_3D
                else:
                    ysv_ext0 = ysv_ext1 = ysb
                    dy_ext0 = dy0 - SQUISH_CONSTANT_3D
                    dy_ext1 = dy0 - 2 * SQUISH_CONSTANT_3D
                if c & 0x04:
                    zsv_ext0 = zsb + 1
                    zsv_ext1 = zsb + 2
                    dz_ext0 = dz0 - 1 - SQUISH_CONSTANT_3D
                    dz_ext1 = dz0 - 2 - 2 * SQUISH_CONSTANT_3D
                else:
                    zsv_ext0 = zsv_ext1 = zsb
                    dz_ext0 = dz0 - SQUISH_CONSTANT_3D
                    dz_ext1 = dz0 - 2 * SQUISH_CONSTANT_3D
            else:
                c = a_point & b_point
                if c & 0x01:
                    xsv_ext0 = xsb + 1
                    xsv_ext1 = xsb + 2
                    dx_ext0 = dx0 - 1 - SQUISH_CONSTANT_3D
                    dx_ext1 = dx0 - 2 - 2 * SQUISH_CONSTANT_3D
                else:
                    xsv_ext0 = xsv_ext1 = xsb
                    dx_ext0 = dx0 - SQUISH_CONSTANT_3D
                    dx_ext1 = dx0 - 2 * SQUISH_CONSTANT_3D
                if c & 0x02:
                    ysv_ext0 = ysb + 1
                    ysv_ext1 = ysb + 2
                    dy_ext0 = dy0 - 1 - SQUISH_CONSTANT_3D
                    dy_ext1 = dy0 - 2 - 2 * SQUISH_CONSTANT_3D
                else:
                    ysv_ext0 = ysv_ext1 = ysb
                    dy_ext0 = dy0 - SQUISH_CONSTANT_3D
                    dy_ext1 = dy0 - 2 * SQUISH_CONSTANT_3D
                if c & 0x04:
                    zsv_ext0 = zsb + 1
                    zsv_ext1 = zsb + 2
                    dz_ext0 = dz0 - 1 - SQUISH_CONSTANT_3D
                    dz_ext1 = dz0 - 2 - 2 * SQUISH_CONSTANT_3D
                else:
                    zsv_ext0 = zsv_ext1 = zsb
                    dz_ext0 = dz0 - SQUISH_CONSTANT_3D
                    dz_ext1 = dz0 - 2 * SQUISH_CONSTANT_3D

            dx3 = dx0 - 1 - 2 * SQUISH_CONSTANT_3D
            dy3 = dy0 - 1 - 2 * SQUISH_CONSTANT_3D
            dz3 = dz0 - 2 * SQUISH_CONSTANT_3D
            attn3 = 2 - dx3 * dx3 - dy3 * dy3 - dz3 * dz3
            if attn3 > 0:
                attn3 *= attn3
                value += attn3 * attn3 * self._extrapolate3d(xsb + 1, ysb + 1, zsb, dx3, dy3, dz3)

            dx2 = dx3
            dy2 = dy0 - 2 * SQUISH_CONSTANT_3D
            dz2 = dz0 - 1 - 2 * SQUISH_CONSTANT_3D
            attn2 = 2 - dx2 * dx2 - dy2 * dy2 - dz2 * dz2
            if attn2 > 0:
                attn2 *= attn2
                value += attn2 * attn2 * self._extrapolate3d(xsb + 1, ysb, zsb + 1, dx2, dy2, dz2)

            dx1 = dx0 - 2 * SQUISH_CONSTANT_3D
            dy1 = dy3
            dz1 = dz2
            attn1 = 2 - dx1 * dx1 - dy1 * dy1 - dz1 * dz1
            if attn1 > 0:
                attn1 *= attn1
                value += attn1 * attn1 * self._extrapolate3d(xsb, ysb + 1, zsb + 1, dx1, dy1, dz1)

            dx0 = dx0 - 1 - 3 * SQUISH_CONSTANT_3D
            dy0 = dy0 - 1 - 3 * SQUISH_CONSTANT_3D
            dz0 = dz0 - 1 - 3 * SQUISH_CONSTANT_3D
            attn0 = 2 - dx0 * dx0 - dy0 * dy0 - dz0 * dz0
            if attn0 > 0:
                attn0 *= attn0
                value += attn0 * attn0 * self._extrapolate3d(xsb + 1, ysb + 1, zsb + 1, dx0, dy0, dz0)
        else:
            a_point = 0x01
            a_score = xins
            b_point = 0x02
            b_score = yins
            a_is_further_side = False
            b_is_further_side = False
            if a_score >= b_score and zins > b_score:
                b_score = zins
                b_point = 0x04
                b_is_further_side = True
            elif a_score < b_score and zins > a_score:
                a_score = zins
                a_point = 0x04
                a_is_further_side = True
            if a_score >= b_score and xins + zins > 1:
                a_score = xins + zins - 1
                a_point = 0x03
                a_is_further_side = True
            elif a_score < b_score and yins + zins > 1:
                b_score = yins + zins - 1
                b_point = 0x03
                b_is_further_side = True
            if a_is_further_side == b_is_further_side:
                if a_is_further_side:
                    xsv_ext0 = xsb + 1
                    xsv_ext1 = xsb + 2
                    dx_ext0 = dx0 - 1 - SQUISH_CONSTANT_3D
                    dx_ext1 = dx0 - 2 - 2 * SQUISH_CONSTANT_3D
                    ysv_ext0 = ysb + 1
                    ysv_ext1 = ysb + 2
                    dy_ext0 = dy0 - 1 - SQUISH_CONSTANT_3D
                    dy_ext1 = dy0 - 2 - 2 * SQUISH_CONSTANT_3D
                    zsv_ext0 = zsb + 1
                    zsv_ext1 = zsb + 2
                    dz_ext0 = dz0 - 1 - SQUISH_CONSTANT_3D
                    dz_ext1 = dz0 - 2 - 2 * SQUISH_CONSTANT_3D
                else:
                    xsv_ext0 = xsb
                    xsv_ext1 = xsb - 1
                    dx_ext0 = dx0 - SQUISH_CONSTANT_3D
                    dx_ext1 = dx0 + 1 - 2 * SQUISH_CONSTANT_3D
                    ysv_ext0 = ysb
                    ysv_ext1 = ysb - 1
                    dy_ext0 = dy0 - SQUISH_CONSTANT_3D
                    dy_ext1 = dy0 + 1 - 2 * SQUISH_CONSTANT_3D
                    zsv_ext0 = zsb
                    zsv_ext1 = zsb - 1
                    dz_ext0 = dz0 - SQUISH_CONSTANT_3D
                    dz_ext1 = dz0 + 1 - 2 * SQUISH_CONSTANT_3D
            else:
                if a_is_further_side:
                    c1 = a_point
                    c2 = b_point
                else:
                    c1 = b_point
                    c2 = a_point
                if c1 & 0x01 == 0:
                    xsv_ext0 = xsb
                    dx_ext0 = dx0 - SQUISH_CONSTANT_3D
                else:
                    xsv_ext0 = xsb + 1
                    dx_ext0 = dx0 - 1 - SQUISH_CONSTANT_3D
                if c1 & 0x02 == 0:
                    ysv_ext0 = ysb
                    dy_ext0 = dy0 - SQUISH_CONSTANT_3D
                else:
                    ysv_ext0 = ysb + 1
                    dy_ext0 = dy0 - 1 - SQUISH_CONSTANT_3D
                if c1 & 0x04 == 0:
                    zsv_ext0 = zsb
                    dz_ext0 = dz0 - SQUISH_CONSTANT_3D
                else:
                    zsv_ext0 = zsb + 1
                    dz_ext0 = dz0 - 1 - SQUISH_CONSTANT_3D
                xsv_ext1 = xsb + 1
                ysv_ext1 = ysb + 1
                zsv_ext1 = zsb + 1
                dx_ext1 = dx0 - 1 - 2 * SQUISH_CONSTANT_3D
                dy_ext1 = dy0 - 1 - 2 * SQUISH_CONSTANT_3D
                dz_ext1 = dz0 - 1 - 2 * SQUISH_CONSTANT_3D

            attn0 = 2 - dx0 * dx0 - dy0 * dy0 - dz0 * dz0
            if attn0 > 0:
                attn0 *= attn0
                value += attn0 * attn0 * self._extrapolate3d(xsb, ysb, zsb, dx0, dy0, dz0)

            dx1 = dx0 - 1 - SQUISH_CONSTANT_3D
            dy1 = dy0 - SQUISH_CONSTANT_3D
            dz1 = dz0 - SQUISH_CONSTANT_3D
            attn1 = 2 - dx1 * dx1 - dy1 * dy1 - dz1 * dz1
            if attn1 > 0:
                attn1 *= attn1
                value += attn1 * attn1 * self._extrapolate3d(xsb + 1, ysb, zsb, dx1, dy1, dz1)

            dx2 = dx0 - SQUISH_CONSTANT_3D
            dy2 = dy0 - 1 - SQUISH_CONSTANT_3D
            dz2 = dz1
            attn2 = 2 - dx2 * dx2 - dy2 * dy2 - dz2 * dz2
            if attn2 > 0:
                attn2 *= attn2
                value += attn2 * attn2 * self._extrapolate3d(xsb, ysb + 1, zsb, dx2, dy2, dz2)

            dx3 = dx2
            dy3 = dy1
            dz3 = dz0 - 1 - SQUISH_CONSTANT_3D
            attn3 = 2 - dx3 * dx3 - dy3 * dy3 - dz3 * dz3
            if attn3 > 0:
                attn3 *= attn3
                value += attn3 * attn3 * self._extrapolate3d(xsb, ysb, zsb + 1, dx3, dy3, dz3)

            attn_ext0 = 2 - dx_ext0 * dx_ext0 - dy_ext0 * dy_ext0 - dz_ext0 * dz_ext0
            if attn_ext0 > 0:
                attn_ext0 *= attn_ext0
                value += attn_ext0 * attn_ext0 * self._extrapolate3d(xsv_ext0, ysv_ext0, zsv_ext0, dx_ext0, dy_ext0, dz_ext0)

            attn_ext1 = 2 - dx_ext1 * dx_ext1 - dy_ext1 * dy_ext1 - dz_ext1 * dz_ext1
            if attn_ext1 > 0:
                attn_ext1 *= attn_ext1
                value += attn_ext1 * attn_ext1 * self._extrapolate3d(xsv_ext1, ysv_ext1, zsv_ext1, dx_ext1, dy_ext1, dz_ext1)

        return value / NORM_CONSTANT_3D


def from_seed(seed: int) -> OpenSimplex:
    """
    Create OpenSimplex noise generator from seed.

    Args:
        seed: Random seed value

    Returns:
        Initialized OpenSimplex generator
    """
    os = OpenSimplex(seed)
    return os, {"perm": list(os.perm), "perm_grad": list(os.perm_grad_index_3D)}


def random(time: int = 0, seed: Optional[int] = None, speed: int = 1) -> OpenSimplex:
    """
    Create time-evolving OpenSimplex noise generator.

    Args:
        time: Time offset for seed evolution, default 0
        seed: Optional random seed (uses global if None), default None
        speed: Seed evolution speed multiplier, default 1

    Returns:
        Initialized OpenSimplex generator with evolved seed
    """
    angle = math.pi * 2 * time
    z = math.cos(angle) * speed
    w = math.sin(angle) * speed
    s = seed if seed is not None else rng.random_int(1, 65536)
    os, _ = from_seed(s)
    value = os.noise2d(z, w)
    return (value + 1) * 0.5


def simplex(shape, time: int = 0, seed: Optional[int] = None, speed: int = 1, as_np: bool = False) -> tf.Tensor | np.ndarray:
    """
    Generate simplex noise tensor.

    Args:
        shape: Output tensor shape
        time: Time offset for noise evolution, default 0
        seed: Optional random seed (uses global if None), default None
        speed: Noise evolution speed multiplier, default 1
        as_np: Return as NumPy array instead of Tensor, default False

    Returns:
        Simplex noise tensor or array with values in range approximately [-1, 1]
    """
    height, width = shape[0], shape[1]
    channels = shape[2] if len(shape) > 2 else 1
    base_seed = seed if seed is not None else get_seed()
    angle = math.pi * 2 * time
    z = math.cos(angle) * speed
    data = np.empty((height, width, channels), dtype=np.float32)
    for c in range(channels):
        os, _ = from_seed(base_seed + c * 65535)
        for y in range(height):
            for x in range(width):
                val = os.noise3d(x, y, z)
                data[y][x][c] = (val + 1) * 0.5
    if not as_np:
        data = tf.stack(data)
    return data

