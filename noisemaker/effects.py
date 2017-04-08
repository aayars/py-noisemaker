from enum import Enum

import math
import random

import numpy as np
import tensorflow as tf

from skimage.transform import resize, rotate
from skimage.util import crop, pad


def post_process(tensor, refract_amount, reindex_amount, clut, horizontal, clut_range, with_worms):
    """
    Apply post-processing filters.

    :param Tensor tensor:
    :param float refract_amount:
    :param float reindex_amount:
    :param str clut:
    :param bool horizontal:
    :param float clut_range:
    :param bool with_worms:
    :return: Tensor
    """

    if refract_amount != 0:
        tensor = refract(tensor, displacement=refract_amount)

    if reindex_amount != 0:
        tensor = reindex(tensor, displacement=reindex_amount)

    if clut:
        tensor = color_map(tensor, clut, horizontal=horizontal, displacement=clut_range)

    else:
        tensor = normalize(tensor)

    if with_worms:
        tensor = worms(tensor)

    return tensor


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

    rand = np.random.normal(.5, .5, (5, 5)).tolist()

    shadow = [
        # yeah, one of the really big fuckers
        [  0,   1,   1,   1,   1,   1,   1  ],
        [ -1,   0,   2,   2,   1,   1,   1  ],
        [ -1,  -2,   0,   4,   2,   1,   1  ],
        [ -1,  -2,  -4,   12,   4,   2,   1  ],
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

    invert = [
        [ 0,  0,  0 ],
        [ 0, -1,  0 ],
        [ 0,  0,  0 ]
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

    tensor = tf.divide(
        tf.subtract(tensor, tf.reduce_min(tensor)),
        tf.subtract(tf.reduce_max(tensor), tf.reduce_min(tensor))
    )

    return tf.image.convert_image_dtype(tensor, tf.float32, saturate=True)


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


def reindex(tensor, displacement=.5):
    """
    Re-color the given tensor, by sampling along one axis at a specified frequency.

    .. image:: images/reindex.jpg
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

    mod = min(height, width)
    offset = tf.cast(tf.mod(tf.add(tf.multiply(reference, displacement * mod), reference), mod), tf.int32)

    temp = tf.reshape(tensor.eval()[offset.eval(), 0], shape)

    temp = tf.image.convert_image_dtype(temp, tf.float32, saturate=True)

    return temp


def refract(tensor, displacement=.5):
    """
    Apply self-displacement along X and Y axes, based on each pixel value.

    .. image:: images/refract.jpg
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
    index = tf.image.rgb_to_grayscale(tensor) if channels > 2 else tensor

    # Create two channels for X and Y
    index = tf.reshape(index, [width * height])
    index = np.repeat((index.eval() - .5) * 2 * displacement * min(width, height), 2)
    index = tf.reshape(index, [height, width, 2]).eval()

    index = _offset_index(index)

    row_identity = _row_index(tensor)
    column_identity = _column_index(tensor)

    index[:,:,0] = (index[:,:,0] + column_identity) % height
    index[:,:,1] = (index[:,:,1] + row_identity) % width

    index = tf.cast(index, tf.int32)

    return tf.gather_nd(tensor, index)


def color_map(tensor, clut, horizontal=False, displacement=.5):
    """
    Apply a color map to an image tensor.

    The color map can be a photo or whatever else.

    .. image:: images/color_map.jpg
       :width: 1024
       :height: 256
       :alt: Noisemaker example output (CC0)

    :param Tensor tensor:
    :param Tensor|str clut: An image tensor or filename (png/jpg only) to use as a color palette
    :param bool horizontal: Scan horizontally
    :param float displacement: Gather distance for clut
    """

    if isinstance(clut, str):
        with open(clut, "rb") as fh:
            if clut.endswith(".png"):
                clut = tf.image.decode_png(fh.read(), channels=3)

            elif clut.endswith(".jpg"):
                clut = tf.image.decode_jpg(fh.read(), channels=3)

    shape = tf.shape(tensor).eval()
    height, width, channels = shape

    # TODO: Reduce tensor to single channel more reliably (use a reduce function?)
    orig_index = tf.image.rgb_to_grayscale(tensor) if channels > 2 else tensor

    index = tf.reshape(orig_index, [width * height])
    index *= displacement
    index = np.repeat(index.eval(), 2)
    index = tf.reshape(index, [height, width, 2]).eval()

    row_identity = _row_index(tensor)
    index[:,:,1] = (index[:,:,1] * (width - 1) + row_identity) % width

    column_identity = _column_index(tensor)

    if horizontal:
        index[:,:,0] = column_identity

    else:
        index[:,:,0] = (index[:,:,0] * (height - 1) + column_identity) % height
        index = _offset_index(index)

    index = tf.cast(index, tf.int32)

    clut = resample(clut, width, height, 3)

    clut = tf.image.convert_image_dtype(clut, tf.float32, saturate=True)

    output = tf.gather_nd(clut, index)
    output = tf.image.convert_image_dtype(output, tf.float32, saturate=True)

    return output


def worms(tensor):
    """
    Make a furry patch of worms which follow field flow rules.

    :param Tensor tensor:
    :return: Tensor
    """

    shape = tf.shape(tensor).eval()
    height, width, channels = shape

    reference = tensor.eval()

    mod = min(width, height)

    worm_count = mod * 4
    worms = normalize(tf.random_uniform((worm_count, 2))).eval()
    worms[:, 0] *= height
    worms[:, 1] *= width

    worm_colors = tf.zeros((worm_count, channels)).eval()
    for i, worm in enumerate(worms):
        worm_colors[i] = reference[int(worm[0]) % height, int(worm[1]) % width]

    index = tf.image.rgb_to_grayscale(reference) if channels > 2 else reference
    index = tf.reshape(index, (height, width))
    index = normalize(index) * 360.0 * math.radians(1)
    index = index.eval()

    thread_len = int(mod / 16)

    out = reference * .5

    for i in range(thread_len):
        coords = worms.astype(int)

        coords[:, 0] = np.mod(coords[:, 0], height)
        coords[:, 1] = np.mod(coords[:, 1], width)

        value = 1 - abs(1 - i / (thread_len - 1) * 2)

        for j, coord in enumerate(coords):
            out[coord[0], coord[1]] = ( 1 + value ) * worm_colors[j]

        worms[:, 0] += np.cos(index[coords[:, 0], coords[:, 1]])
        worms[:, 1] += np.sin(index[coords[:, 0], coords[:, 1]])

    out = tf.image.convert_image_dtype(out, tf.float32, saturate=True)

    return out


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


def _row_index(tensor):
    """
    Generate an X index for the given tensor.

    [
      [ 0, 1, 2, ... width-1 ],
      [ 0, 1, 2, ... width-1 ],
      ... (x height)
    ]

    :param Tensor tensor:
    :return: Tensor
    """

    shape = tf.shape(tensor).eval()
    height, width, channels = shape

    row_identity = tf.cumsum(tf.ones(width, dtype=tf.int32), exclusive=True)
    row_identity = tf.reshape(tf.tile(row_identity, [height]), [height, width]).eval()

    return row_identity


def _column_index(tensor):
    """
    Generate a Y index for the given tensor.

    [
      [ 0, 0, 0, ... ],
      [ 1, 1, 1, ... ],
      [ n, n, n, ... ],
      ...
      [ height-1, height-1, height-1, ... ]
    ]

    :param Tensor tensor:
    :return: Tensor
    """

    shape = tf.shape(tensor).eval()
    height, width, channels = shape

    column_identity = tf.ones(width, dtype=tf.int32)
    column_identity = tf.tile(column_identity, [height])
    column_identity = tf.reshape(column_identity, [height, width])
    column_identity = tf.cumsum(column_identity, exclusive=True).eval()

    return column_identity


def _offset_index(tensor):
    """
    Offset X and Y displacement channels from each other, to help with diagonal banding.

    :param Tensor tensor: Tensor of shape (height, width, 2)
    :return: Tensor
    """

    shape = tf.shape(tensor).eval()
    height, width, channels = shape

    tensor[:,:,0] = np.roll(tensor[:,:,0], int(random.random() * height * .5 + height * .5))

    temp = np.rot90(tensor[:,:,1])
    temp = np.roll(temp, int(random.random() * width * .5 + width * .5))
    tensor[:,:,1] = np.rot90(temp, 3)

    return tensor
