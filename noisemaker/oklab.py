"""
Oklab color space conversion.

Based on https://bottosson.github.io/posts/oklab/
"""

from __future__ import annotations

import tensorflow as tf

from noisemaker.util import from_linear_rgb, from_srgb


def rgb_to_oklab(tensor: tf.Tensor) -> tf.Tensor:
    """
    Convert RGB color space to Oklab.

    Args:
        tensor: RGB tensor with shape [height, width, channels]

    Returns:
        Oklab tensor with shape [height, width, 3]
    """
    tensor = from_srgb(tensor)

    r = tensor[:, :, 0]
    g = tensor[:, :, 1]
    b = tensor[:, :, 2]

    l = 0.4121656120 * r + 0.5362752080 * g + 0.0514575653 * b
    m = 0.2118591070 * r + 0.6807189584 * g + 0.1074065790 * b
    s = 0.0883097947 * r + 0.2818474174 * g + 0.6302613616 * b

    l_ = l ** (1 / 3.0)
    m_ = m ** (1 / 3.0)
    s_ = s ** (1 / 3.0)

    return tf.stack(
        [
            0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
            1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
            0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
        ],
        2,
    )


def oklab_to_rgb(tensor: tf.Tensor) -> tf.Tensor:
    """
    Convert Oklab color space to RGB.

    Args:
        tensor: Oklab tensor with shape [height, width, 3]

    Returns:
        RGB tensor with shape [height, width, channels]
    """
    L = tensor[:, :, 0]
    a = tensor[:, :, 1]
    b = tensor[:, :, 2]

    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b

    l = l_ * l_ * l_
    m = m_ * m_ * m_
    s = s_ * s_ * s_

    tensor = tf.stack(
        [
            +4.0767245293 * l - 3.3072168827 * m + 0.2307590544 * s,
            -1.2681437731 * l + 2.6093323231 * m - 0.3411344290 * s,
            -0.0041119885 * l - 0.7034763098 * m + 1.7068625689 * s,
        ],
        2,
    )

    return from_linear_rgb(tensor)
