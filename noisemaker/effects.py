from enum import Enum

import math
import random

import numpy as np
import tensorflow as tf

from skimage.transform import resize
from skimage.util import crop, pad


class ConvKernel(Enum):
    """
    A collection of convolution kernels for image post-processing, based on well-known recipes.

    Pass the desired kernel as an argument to :py:func:`convolve`.

    .. code-block:: python

       # Make it pop
       image = convolve(ConvKernel.shadow, image)
    """

    emboss = [
        [   0,   2,   4   ],
        [  -2,   1,   2   ],
        [  -4,  -2,   0   ]
    ]

    shadow = [
        # yeah, one of the really big fuckers
        [  0,   1,   1,   1,   1,   1,   1  ],
        [ -1,   0,   2,   2,   1,   1,   1  ],
        [ -1,  -2,   0,   4,   2,   1,   1  ],
        [ -1,  -2,  -4,   8,   4,   2,   1  ],
        [ -1,  -1,  -2,  -4,   0,   2,   1  ],
        [ -1,  -1,  -1,  -2,  -2,   0,   1  ],
        [ -1,  -1,  -1,  -1,  -1,  -1,   0  ]

        # [  0,  1,  1,  1, 0 ],
        # [ -1, -2,  4,  2, 1 ],
        # [ -1, -4,  2,  4, 1 ],
        # [ -1, -2, -4,  2, 1 ],
        # [  0, -1, -1, -1, 0 ]

        # [  0,  1,  1,  1, 0 ],
        # [ -1, -2,  4,  2, 1 ],
        # [ -1, -4,  2,  4, 1 ],
        # [ -1, -2, -4,  2, 1 ],
        # [  0, -1, -1, -1, 0 ]
    ]

    edges = [
        [   1,   2,  1   ],
        [   2, -12,  2   ],
        [   1,   2,  1   ]
    ]

    sharpen = [
        [   0, -1,  0 ],
        [  -1,  5, -1 ],
        [   0, -1,  0 ]
    ]

    unsharp_mask = [
        [ 1,  4,     6,   4, 1 ],
        [ 4,  16,   24,  16, 4 ],
        [ 6,  24, -476,  24, 6 ],
        [ 4,  16,   24,  16, 4 ],
        [ 1,  4,     6,   4, 1 ]
    ]


def _conform_kernel_to_tensor(kernel, tensor):
    """ Re-shape a convolution kernel to match the given tensor's color dimensions. """

    l = len(kernel)

    channels = tf.shape(tensor).eval()[2]

    temp = np.repeat(kernel, channels)

    temp = tf.reshape(temp, (l, l, channels, 1))

    temp = tf.image.convert_image_dtype(temp, tf.float32, saturate=True)

    return temp


def convolve(kernel, tensor):
    """
    Apply a convolution kernel to an image tensor.

    :param ConvKernel kernel: See ConvKernel enum
    :param Tensor tensor: An image tensor.
    :return: Tensor
    """

    height, width, channels = tf.shape(tensor).eval()

    kernel = _conform_kernel_to_tensor(kernel.value, tensor)

    # Give the conv kernel some room to play on the edges
    pad_height = int(height * .25)
    pad_width = int(width * .25)
    padding = ((pad_height, pad_height), (pad_width, pad_width), (0, 0))
    tensor = tf.stack(pad(tensor.eval(), padding, "wrap"))

    tensor = tf.nn.depthwise_conv2d([tensor], kernel, [1,1,1,1], "VALID")[0]

    # Playtime... is... over!
    post_height, post_width, channels = tf.shape(tensor).eval()
    crop_height = int((post_height - height) * .5)
    crop_width = int((post_width - width) * .5)
    tensor = crop(tensor.eval(), ((crop_height, crop_height), (crop_width, crop_width), (0, 0)))

    tensor = normalize(tensor)

    return tensor


def normalize(tensor):
    """
    Squeeze the given Tensor into a range between 0 and 1.

    :param Tensor tensor: An image tensor.
    :return: Tensor
    """

    return tf.divide(
        tf.subtract(tensor, tf.reduce_min(tensor)),
        tf.subtract(tf.reduce_max(tensor), tf.reduce_min(tensor))
    )


def resample(tensor, width, height, spline_order=3):
    """
    Resize the given image Tensor to the given dimensions.

    :param Tensor tensor: An image tensor.
    :param int width: Output width.
    :param int height: Output height.
    :param int spline_order: Spline point count. 0=Constant, 1=Linear, 3=Bicubic, others may not work.
    :return: Tensor
    """

    _height, _width, channels = tf.shape(tensor).eval()

    if isinstance(tensor, tf.Tensor):  # Sometimes you feel like a Tensor
        downcast = tensor.eval()

    else:  # Sometimes you feel a little more numpy
        downcast = tensor

    downcast = resize(downcast, (height, width, channels), mode="wrap", order=spline_order, preserve_range=True)

    return tf.image.convert_image_dtype(downcast, tf.float32, saturate=True)

    ### TensorFlow doesn't handily let us wrap around edges when resampling.
    # temp = tf.image.resize_images(tensor, [height, width], align_corners=True, method=tf.image.ResizeMethod.BICUBIC)
    # temp = tf.image.convert_image_dtype(temp, tf.float32, saturate=True)
    # return temp


def crease(tensor):
    """
    Create a "crease" (ridge) at midpoint values. (1 - unsigned((n-.5)*2))

    .. image:: images/crease.jpg
       :width: 1024
       :height: 256
       :alt: Noisemaker example output (CC0)

    :param Tensor tensor: An image tensor.
    :return: Tensor
    """

    temp = tf.subtract(tensor, .5)
    temp = tf.multiply(temp, 2)
    temp = tf.maximum(temp, temp*-1)

    temp = tf.subtract(tf.ones(tf.shape(temp)), temp)

    return temp


def reindex(tensor, displacement=1.0):
    """
    Apply self-displacement along Z (pixel value) axes, based on each pixel value.

    :param Tensor tensor: An image tensor.
    :param float displacement:
    :return: Tensor
    """

    shape = tf.shape(tensor).eval()
    height, width, channels = shape

    # TODO: Reduce tensor to single channel more reliably (use a reduce function?)
    reference = tf.image.rgb_to_grayscale(tensor) if channels > 2 else tensor

    mod = min(width, height)
    offset = tf.cast(tf.mod(tf.add(tf.multiply(reference, displacement * mod), reference), mod), tf.int32)

    temp = tf.reshape(tensor.eval()[offset.eval(), 0], shape)
    temp = tf.image.convert_image_dtype(temp, tf.float32, saturate=True)

    return temp


def distort(tensor, displacement=1.0):
    """
    Apply self-displacement along X and Y axes, based on each pixel value.

    .. image:: images/displacement.jpg
       :width: 1024
       :height: 256
       :alt: Noisemaker example output (CC0)

    :param Tensor tensor: An image tensor.
    :param float displacement:
    :return: Tensor
    """

    shape = tf.shape(tensor).eval()
    height, width, channels = shape

    # TODO: Reduce tensor to single channel more reliably (use a reduce function?)
    reference = tf.image.rgb_to_grayscale(tensor) if channels > 2 else tensor

    # Create two channels for X and Y
    reference = tf.reshape(reference, [width * height])
    reference = np.repeat((reference.eval() - .5) * 2 * displacement * min(width, height), 2)
    reference = tf.reshape(reference, [height, width, 2]).eval()

    # Offset X and Y to eliminate diagonal artifacts
    reference[:,:,0] = np.roll(reference[:,:,0], int(random.random() * height * .5 + height * .5))
    reference[:,:,1] = np.roll(reference[:,:,1], int(random.random() * width * .5 + width * .5))

    # Create an "identify" index [ 0 .. width-1 ] * height, apply reference offsets
    row = tf.cumsum(tf.ones((width * 2), dtype=tf.int32), exclusive=True)
    index = tf.reshape(tf.tile(row, [height]), (height, width, 2))
    index = tf.cast(tf.mod(reference + index, min(width, height)), tf.int32)

    return tf.gather_nd(tensor, index)


def wavelet(tensor):
    """
    Convert regular noise into 2-D wavelet noise.

    Completely useless. Maybe useful if Noisemaker supports higher dimensions later.

    .. image:: images/wavelet.jpg
       :width: 1024
       :height: 256
       :alt: Noisemaker example output (CC0)

    :param Tensor tensor: An image tensor.
    :return: Tensor
    """

    shape = tf.shape(tensor).eval()

    height, width, channels = shape

    return tensor - resample(resample(tensor, int(width * .5), int(height * .5)), width, height)
